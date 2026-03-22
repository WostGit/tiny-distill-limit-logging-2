import argparse
import json
import math
import random
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_tiny_distill import evaluate_adapter
from logging_utils import EventLogger, checkpoint_dir_stats, environment_snapshot, memory_snapshot
from metrics_utils import append_jsonl, summarize_times, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny verbose distillation smoke test")
    parser.add_argument("--train-data", default="data/tiny_train.jsonl")
    parser.add_argument("--eval-data", default="data/tiny_eval.jsonl")
    parser.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--output-dir", default="outputs/checkpoints/tiny_adapter")
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="CE/distillation blend")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def p95(values: List[int]) -> float:
    if not values:
        return 0.0
    idx = int(math.ceil(0.95 * len(values))) - 1
    idx = min(max(idx, 0), len(values) - 1)
    return float(sorted(values)[idx])


def preprocess(rows, tokenizer, max_length: int):
    prompts, targets = [], []
    input_lens, target_lens = [], []
    for row in rows:
        prompt = row["prompt"].strip()
        target = row["target"].strip()
        full_text = prompt + "\n" + target

        full_enc = tokenizer(full_text, truncation=True, max_length=max_length)
        prompt_enc = tokenizer(prompt + "\n", truncation=True, max_length=max_length)

        ids = full_enc["input_ids"]
        labels = ids.copy()
        prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        prompts.append(ids)
        targets.append(labels)
        input_lens.append(len(ids))
        target_lens.append(sum(1 for x in labels if x != -100))

    stats = {
        "input_token_len": {
            "min": min(input_lens),
            "mean": round(mean(input_lens), 3),
            "max": max(input_lens),
            "p95": p95(input_lens),
        },
        "target_token_len": {
            "min": min(target_lens),
            "mean": round(mean(target_lens), 3),
            "max": max(target_lens),
            "p95": p95(target_lens),
        },
    }
    return list(zip(prompts, targets)), stats


def collate_fn(batch, pad_token_id: int):
    max_len = max(len(x[0]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for ids, lbls in batch:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(lbls + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> None:
    args = parse_args()
    Path(args.metrics_dir).mkdir(parents=True, exist_ok=True)
    logger = EventLogger()
    set_seed(args.seed)

    run_metrics = {
        "environment": environment_snapshot(),
        "memory": [memory_snapshot("process_start")],
        "args": vars(args),
    }

    time_buckets = {}

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_jsonl(args.train_data)
    eval_rows = load_jsonl(args.eval_data)
    train_data, length_stats = preprocess(train_rows, tokenizer, args.max_length)
    time_buckets["preprocessing"] = time.perf_counter() - t0
    logger.log("preprocessing_done", rows=len(train_data), length_stats=length_stats)

    t0 = time.perf_counter()
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    time_buckets["data_loader_setup"] = time.perf_counter() - t0

    teacher = AutoModelForCausalLM.from_pretrained(args.base_model)
    student = AutoModelForCausalLM.from_pretrained(args.base_model)
    run_metrics["memory"].append(memory_snapshot("after_model_load"))

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, lora_cfg)
    student.train()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    run_metrics["memory"].append(memory_snapshot("after_lora_wrapping"))

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    eff_samples_per_step = args.batch_size * args.grad_accum
    logger.log(
        "training_shape",
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        effective_samples_per_optimizer_step=eff_samples_per_step,
    )

    global_step = 0
    seen_samples = 0
    first_backward_done = False
    tokens_per_opt_step = []

    train_start = time.perf_counter()
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        logger.log("epoch_start", epoch=epoch)
        micro_tokens = 0
        step_start = time.perf_counter()
        for i, batch in enumerate(loader):
            out_s = student(**batch)
            with torch.no_grad():
                out_t = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            ce_loss = out_s.loss
            T = args.temperature
            s_logp = F.log_softmax(out_s.logits / T, dim=-1)
            t_p = F.softmax(out_t.logits / T, dim=-1)
            kd_loss = F.kl_div(s_logp, t_p, reduction="batchmean") * (T * T)
            loss = args.alpha * ce_loss + (1.0 - args.alpha) * kd_loss
            (loss / args.grad_accum).backward()

            if not first_backward_done:
                run_metrics["memory"].append(memory_snapshot("after_first_forward_backward"))
                first_backward_done = True

            micro_tokens += int(batch["attention_mask"].sum().item())
            seen_samples += batch["input_ids"].shape[0]

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                optimizer.step()
                optimizer.zero_grad()
                step_duration = time.perf_counter() - step_start
                global_step += 1
                tokens_per_opt_step.append(micro_tokens)
                logger.log(
                    "optimizer_step",
                    optimizer_step=global_step,
                    step_duration_s=round(step_duration, 6),
                    elapsed_since_start_s=round(logger.since_start(), 6),
                    cumulative_samples=seen_samples,
                    tokens_this_step=micro_tokens,
                    ce_loss=float(ce_loss.detach().cpu().item()),
                    kd_loss=float(kd_loss.detach().cpu().item()),
                    total_loss=float(loss.detach().cpu().item()),
                )
                run_metrics["memory"].append(memory_snapshot(f"optimizer_step_{global_step}"))
                append_jsonl(
                    str(Path(args.metrics_dir) / "train_step_metrics.jsonl"),
                    {
                        "step": global_step,
                        "step_duration_s": step_duration,
                        "cumulative_samples": seen_samples,
                        "tokens_this_step": micro_tokens,
                        "loss": float(loss.detach().cpu().item()),
                    },
                )
                micro_tokens = 0
                step_start = time.perf_counter()

    time_buckets["training"] = time.perf_counter() - train_start

    run_metrics["memory"].append(memory_snapshot("before_save"))
    save_start = time.perf_counter()
    logger.log("checkpoint_save_start", output_dir=args.output_dir)
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_dur = time.perf_counter() - save_start
    save_stats = checkpoint_dir_stats(args.output_dir)
    logger.log("checkpoint_save_end", save_duration_s=round(save_dur, 6), **save_stats)
    time_buckets["checkpoint_save"] = save_dur
    run_metrics["memory"].append(memory_snapshot("after_save"))

    run_metrics["memory"].append(memory_snapshot("before_eval"))
    eval_start = time.perf_counter()
    eval_metrics = evaluate_adapter(
        base_model=args.base_model,
        adapter_dir=args.output_dir,
        eval_data=args.eval_data,
        max_length=args.max_length,
        metrics_dir=args.metrics_dir,
        prefix="post_train_",
    )
    eval_dur = time.perf_counter() - eval_start
    time_buckets["eval"] = eval_dur
    run_metrics["memory"].append(memory_snapshot("after_eval"))

    run_metrics["time_summary"] = summarize_times(time_buckets)
    run_metrics["sequence_length_stats"] = length_stats
    run_metrics["tokens_per_optimizer_step"] = {
        "values": tokens_per_opt_step,
        "min": min(tokens_per_opt_step) if tokens_per_opt_step else 0,
        "mean": round(mean(tokens_per_opt_step), 3) if tokens_per_opt_step else 0.0,
        "max": max(tokens_per_opt_step) if tokens_per_opt_step else 0,
        "p95": p95(tokens_per_opt_step) if tokens_per_opt_step else 0.0,
    }
    run_metrics["eval_metrics"] = eval_metrics

    likely_bottleneck = max(time_buckets, key=time_buckets.get)
    summary = {
        "likely_first_limit": likely_bottleneck,
        "time_breakdown_seconds": {k: round(v, 6) for k, v in time_buckets.items()},
        "peak_rss_mib": max(m["rss_mib"] for m in run_metrics["memory"]),
        "max_input_tokens": length_stats["input_token_len"]["max"],
        "max_target_tokens": length_stats["target_token_len"]["max"],
    }
    run_metrics["bottleneck_summary"] = summary

    print("\n=== FINAL BOTTLENECK SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    write_json(str(Path(args.metrics_dir) / "train_run_metrics.json"), run_metrics)


if __name__ == "__main__":
    main()
