#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import time
from statistics import mean
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import VerboseLogger, compute_size_and_file_count, print_json_block
from metrics_utils import write_metrics


def read_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    k = (len(values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(values[int(k)])
    return float(values[f] * (c - k) + values[c] * (k - f))


def summarize_lengths(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {"min": 0.0, "mean": 0.0, "max": 0.0, "p95": 0.0}
    sorted_vals = sorted(lengths)
    return {
        "min": float(sorted_vals[0]),
        "mean": float(mean(sorted_vals)),
        "max": float(sorted_vals[-1]),
        "p95": percentile(sorted_vals, 0.95),
    }


def build_examples(records: List[Dict[str, str]], tokenizer, max_length: int, logger: VerboseLogger):
    t0 = time.perf_counter()
    formatted = [f"Prompt: {r['prompt']}\nResponse: {r['target']}" for r in records]
    enc = tokenizer(
        formatted,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    examples = []
    input_lens = []
    target_lens = []
    for i, ids in enumerate(enc["input_ids"]):
        labels = ids.copy()
        examples.append({"input_ids": ids, "labels": labels})
        input_lens.append(len(ids))
        target_text = records[i]["target"]
        target_ids = tokenizer(target_text, truncation=True, max_length=max_length, padding=False)["input_ids"]
        target_lens.append(len(target_ids))
    dt = time.perf_counter() - t0
    logger.log(f"DATA PREPROCESS tokenization_seconds={dt:.4f} examples={len(examples)}")
    return examples, input_lens, target_lens, dt


def collate(batch, pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        ids = item["input_ids"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad)
        labels.append(item["labels"] + [-100] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/tiny_train.jsonl")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--output-dir", default="outputs/checkpoints/tiny_adapter")
    parser.add_argument("--metrics-path", default="outputs/metrics/train_metrics.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = VerboseLogger("TRAIN")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = logger.env_report()
    logger.snapshot_memory("process_start")

    data_load_t0 = time.perf_counter()
    records = read_jsonl(args.train_file)
    data_load_seconds = time.perf_counter() - data_load_t0
    logger.log(f"DATA LOAD train_records={len(records)} data_load_seconds={data_load_seconds:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(args.model_name)
    student.train()
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    logger.snapshot_memory("after_model_load")

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn"],
    )
    student = get_peft_model(student, lora_config)
    logger.snapshot_memory("after_lora_wrapping")

    examples, input_lens, target_lens, preprocess_seconds = build_examples(records, tokenizer, args.max_length, logger)

    data_loader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    optim = torch.optim.AdamW(student.parameters(), lr=args.lr)

    optimizer_step = 0
    cumulative_samples = 0
    step_durations = []
    tokens_per_step = []
    train_loss_values = []
    first_backward_done = False

    per_device_batch_size = args.batch_size
    grad_accum_steps = args.grad_accum
    effective_samples_per_step = per_device_batch_size * grad_accum_steps
    logger.log(
        "TRAIN SHAPE "
        f"per_device_batch_size={per_device_batch_size} grad_accum_steps={grad_accum_steps} "
        f"effective_samples_per_optimizer_step={effective_samples_per_step}"
    )

    overall_train_start = time.perf_counter()
    for epoch in range(args.epochs):
        logger.log(f"EPOCH START epoch={epoch + 1}/{args.epochs}")
        optim.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(data_loader):
            step_start = time.perf_counter()
            cumulative_samples += batch["input_ids"].shape[0]
            active_tokens = int(batch["attention_mask"].sum().item())

            student_out = student(**batch)
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

            lm_loss = student_out.loss
            kl_loss = F.kl_div(
                F.log_softmax(student_out.logits, dim=-1),
                F.softmax(teacher_out.logits, dim=-1),
                reduction="batchmean",
            )
            loss = 0.5 * lm_loss + 0.5 * kl_loss
            (loss / grad_accum_steps).backward()

            if not first_backward_done:
                logger.snapshot_memory("after_first_forward_backward")
                first_backward_done = True

            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1 == len(data_loader)):
                optim.step()
                optim.zero_grad(set_to_none=True)
                optimizer_step += 1
                step_seconds = time.perf_counter() - step_start
                step_durations.append(step_seconds)
                tokens_per_step.append(active_tokens)
                train_loss_values.append(float(loss.item()))
                logger.snapshot_memory(f"optimizer_step_{optimizer_step}")
                logger.log(
                    "OPT STEP "
                    f"idx={optimizer_step} elapsed_since_start={time.perf_counter() - logger.start_time:.3f}s "
                    f"step_duration={step_seconds:.3f}s cumulative_samples={cumulative_samples} "
                    f"tokens_this_step={active_tokens} loss={loss.item():.6f}"
                )

    train_duration = time.perf_counter() - overall_train_start

    logger.snapshot_memory("before_save")
    save_start_utc = time.time()
    save_start = time.perf_counter()
    os.makedirs(args.output_dir, exist_ok=True)
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_end = time.perf_counter()
    save_end_utc = time.time()
    logger.snapshot_memory("after_save")

    ckpt_stats = compute_size_and_file_count(args.output_dir)
    save_metrics = {
        "save_start_utc": save_start_utc,
        "save_end_utc": save_end_utc,
        "save_duration_seconds": save_end - save_start,
        "checkpoint_bytes": ckpt_stats["bytes"],
        "checkpoint_file_count": ckpt_stats["file_count"],
    }
    print_json_block("CHECKPOINT_METRICS", save_metrics)

    dominant = {
        "training_seconds": train_duration,
        "preprocess_seconds": preprocess_seconds,
        "checkpoint_save_seconds": save_metrics["save_duration_seconds"],
        "mean_step_seconds": mean(step_durations) if step_durations else 0.0,
    }
    bottleneck = max(dominant.items(), key=lambda x: x[1])[0] if dominant else "unknown"
    logger.log(f"FINAL BOTTLENECK SUMMARY candidate={bottleneck} details={dominant}")

    metrics = {
        "run_type": "train",
        "model_name": args.model_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_samples_per_optimizer_step": effective_samples_per_step,
        "data_load_seconds": data_load_seconds,
        "preprocess_seconds": preprocess_seconds,
        "train_duration_seconds": train_duration,
        "optimizer_steps": optimizer_step,
        "avg_optimizer_step_seconds": mean(step_durations) if step_durations else 0.0,
        "tokens_per_optimizer_step": tokens_per_step,
        "input_length_stats": summarize_lengths(input_lens),
        "target_length_stats": summarize_lengths(target_lens),
        "train_loss_values": train_loss_values,
        "checkpoint": save_metrics,
        "memory_snapshots": logger.memory_as_dicts(),
        "environment": env,
        "bottleneck_candidate": bottleneck,
    }
    write_metrics(args.metrics_path, metrics)
    logger.log(f"WROTE METRICS path={args.metrics_path}")


if __name__ == "__main__":
    main()
