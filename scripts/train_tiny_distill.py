import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_tiny_distill import run_eval
from logging_utils import (
    RunClock,
    VerboseLogger,
    directory_metrics,
    get_environment_snapshot,
    get_memory_snapshot,
    sequence_stats,
    summarize_bottleneck,
)
from metrics_utils import MetricsWriter


def load_jsonl(path: str):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def collate(examples, tokenizer, max_length):
    inputs = [f"{x['input']}\n" for x in examples]
    targets = [x["target"] for x in examples]
    pairs = [f"{inp}{tgt}" for inp, tgt in zip(inputs, targets)]

    tok_all = tokenizer(pairs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tok_in = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tok_tgt = tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

    labels = tok_all["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    tok_all["labels"] = labels

    tok_all["input_lengths"] = tok_in["attention_mask"].sum(dim=1)
    tok_all["target_lengths"] = tok_tgt["attention_mask"].sum(dim=1)
    return tok_all


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    p.add_argument("--train-data", default="data/tiny_train.jsonl")
    p.add_argument("--eval-data", default="data/tiny_eval.jsonl")
    p.add_argument("--output-dir", default="outputs/checkpoints/tiny_lora_adapter")
    p.add_argument("--metrics-out", default="outputs/metrics/train_metrics.json")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--max-length", type=int, default=96)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(123)

    clock = RunClock.start()
    logger = VerboseLogger("tiny-distill-train", clock)
    metrics = MetricsWriter()

    logger.log("env", **get_environment_snapshot(torch))
    snap = get_memory_snapshot("process_start")
    logger.log("memory", **snap)
    metrics.add_snapshot(snap)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    load_start = time.perf_counter()
    teacher = AutoModelForCausalLM.from_pretrained(args.base_model)
    student = AutoModelForCausalLM.from_pretrained(args.base_model)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    model_load_s = time.perf_counter() - load_start
    logger.log("model_loaded", model_load_s=round(model_load_s, 6))

    snap = get_memory_snapshot("after_model_load")
    logger.log("memory", **snap)
    metrics.add_snapshot(snap)

    lora_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    student = get_peft_model(student, lora_cfg)
    student.print_trainable_parameters()

    snap = get_memory_snapshot("after_lora_wrap")
    logger.log("memory", **snap)
    metrics.add_snapshot(snap)

    raw = load_jsonl(args.train_data)
    prep_start = time.perf_counter()
    ds = []
    in_lens, tgt_lens = [], []
    for row in raw:
        in_tok = tokenizer(row["input"], truncation=True, max_length=args.max_length)
        tgt_tok = tokenizer(row["target"], truncation=True, max_length=args.max_length)
        in_lens.append(len(in_tok["input_ids"]))
        tgt_lens.append(len(tgt_tok["input_ids"]))
        ds.append(row)
    preprocess_s = time.perf_counter() - prep_start

    input_stats = sequence_stats(in_lens)
    target_stats = sequence_stats(tgt_lens)
    logger.log("sequence_stats", input=input_stats, target=target_stats)
    logger.log("preprocess_done", preprocess_s=round(preprocess_s, 6), records=len(ds))

    data_load_start = time.perf_counter()
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer, args.max_length),
    )
    data_load_s = time.perf_counter() - data_load_start
    logger.log("dataloader_ready", dataloader_setup_s=round(data_load_s, 6), batches=len(loader))

    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    effective_samples = args.batch_size * args.grad_accum
    logger.log(
        "training_shape",
        per_device_batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        effective_samples_per_optimizer_step=effective_samples,
    )

    global_step = 0
    opt_step = 0
    cumulative_samples = 0
    first_fb_logged = False
    optimizer_step_durations = []

    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            global_step += 1
            cumulative_samples += batch["input_ids"].shape[0]
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            with torch.no_grad():
                teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
            student_out = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            t = args.temperature
            distill = F.kl_div(
                F.log_softmax(student_out.logits / t, dim=-1),
                F.softmax(teacher_logits / t, dim=-1),
                reduction="batchmean",
            ) * (t * t)

            loss = args.alpha * student_out.loss + (1 - args.alpha) * distill
            (loss / args.grad_accum).backward()

            if not first_fb_logged:
                snap = get_memory_snapshot("after_first_forward_backward")
                logger.log("memory", **snap)
                metrics.add_snapshot(snap)
                first_fb_logged = True

            if global_step % args.grad_accum == 0 or i == len(loader) - 1:
                step_start = time.perf_counter()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                step_s = time.perf_counter() - step_start
                optimizer_step_durations.append(step_s)

                tokens_this_step = int(attention_mask.sum().item())
                event = {
                    "optimizer_step": opt_step,
                    "step_duration_s": round(step_s, 6),
                    "elapsed_s": round(clock.elapsed_seconds(), 6),
                    "cumulative_samples": cumulative_samples,
                    "loss": round(loss.item(), 6),
                    "tokens_this_step": tokens_this_step,
                }
                logger.log("optimizer_step", **event)
                metrics.add_event(event)

                snap = get_memory_snapshot(f"optimizer_step_{opt_step}")
                logger.log("memory", **snap)
                metrics.add_snapshot(snap)

    train_s = time.perf_counter() - train_start
    logger.log("train_finished", train_s=round(train_s, 6), optimizer_steps=opt_step)

    save_t0 = time.perf_counter()
    logger.log("checkpoint_save_start", save_start_ts=save_t0)
    snap = get_memory_snapshot("before_save")
    logger.log("memory", **snap)
    metrics.add_snapshot(snap)

    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    save_t1 = time.perf_counter()
    ckpt = directory_metrics(args.output_dir)
    logger.log(
        "checkpoint_save_end",
        save_end_ts=save_t1,
        save_duration_s=round(save_t1 - save_t0, 6),
        **ckpt,
    )
    snap = get_memory_snapshot("after_save")
    logger.log("memory", **snap)
    metrics.add_snapshot(snap)

    logger.log("eval_transition", note="running immediate post-train eval")
    snap = get_memory_snapshot("before_eval")
    logger.log("memory", **snap)
    metrics.add_snapshot(snap)

    eval_metrics_path = "outputs/metrics/post_train_eval_metrics.json"
    run_eval(args.base_model, args.output_dir, args.eval_data, args.max_length, eval_metrics_path)

    snap = get_memory_snapshot("after_eval")
    logger.log("memory", **snap)
    metrics.add_snapshot(snap)

    max_step_s = max(optimizer_step_durations) if optimizer_step_durations else 0.0
    summary = {
        "train_s": round(train_s, 6),
        "preprocess_s": round(preprocess_s, 6),
        "dataloader_setup_s": round(data_load_s, 6),
        "max_optimizer_step_s": round(max_step_s, 6),
        "checkpoint_save_s": round(save_t1 - save_t0, 6),
        "input_length_stats": input_stats,
        "target_length_stats": target_stats,
        "bottleneck_guess": summarize_bottleneck(
            {
                "wall_clock_training": train_s,
                "preprocessing_overhead": preprocess_s,
                "checkpoint_io": save_t1 - save_t0,
                "per_step_latency": max_step_s,
            }
        ),
    }
    logger.log("final_bottleneck_summary", **summary)
    metrics.set_summary(summary)
    metrics.write_json(args.metrics_out)


if __name__ == "__main__":
    main()
