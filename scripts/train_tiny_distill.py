import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import RunTimer, format_seconds, log_section, print_environment_info, print_memory_snapshot, summarize_lengths
from metrics_utils import directory_size_and_count, write_metrics_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_text(row: Dict[str, str]) -> str:
    return (
        f"Instruction: {row['instruction']}\n"
        f"Input: {row['input']}\n"
        f"Response: {row['output']}"
    )


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "token_count": torch.tensor([item["token_count"] for item in batch], dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny LoRA distillation smoke test with verbose logs")
    parser.add_argument("--train-file", default="data/tiny_train.jsonl")
    parser.add_argument("--model-name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--output-dir", default="outputs/checkpoint")
    parser.add_argument("--metrics-path", default="outputs/metrics/train_metrics.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--alpha-kl", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_timer = RunTimer.start()
    set_seed(args.seed)

    env_info = print_environment_info()
    memory_snaps = [print_memory_snapshot("process_start")]

    log_section("Data Load")
    load_t0 = time.perf_counter()
    rows = load_jsonl(args.train_file)
    load_t1 = time.perf_counter()
    print(f"[DATA] Loaded {len(rows)} rows from {args.train_file} in {format_seconds(load_t1 - load_t0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_t0 = time.perf_counter()
    processed = []
    input_lens: List[int] = []
    target_lens: List[int] = []
    for row in rows:
        text = make_text(row)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        non_pad_len = int(encoded["attention_mask"].sum().item())
        target_len = len(tokenizer(row["output"], truncation=True, max_length=args.max_length)["input_ids"])
        input_lens.append(non_pad_len)
        target_lens.append(target_len)
        processed.append(
            {
                "input_ids": encoded["input_ids"][0],
                "attention_mask": encoded["attention_mask"][0],
                "labels": encoded["input_ids"][0].clone(),
                "token_count": non_pad_len,
            }
        )
    token_t1 = time.perf_counter()
    preprocessing_s = token_t1 - token_t0
    print(f"[DATA] Tokenized/preprocessed in {format_seconds(preprocessing_s)}")
    input_stats = summarize_lengths("input_tokens", input_lens)
    target_stats = summarize_lengths("target_tokens", target_lens)

    log_section("Model Load")
    model_t0 = time.perf_counter()
    student = AutoModelForCausalLM.from_pretrained(args.model_name)
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    model_t1 = time.perf_counter()
    print(f"[MODEL] Loaded student+teacher in {format_seconds(model_t1 - model_t0)}")
    memory_snaps.append(print_memory_snapshot("after_model_load"))

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn"],
    )
    student = get_peft_model(student, lora_cfg)
    student.train()
    student.print_trainable_parameters()
    memory_snaps.append(print_memory_snapshot("after_lora_wrapping"))

    loader = DataLoader(processed, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    effective_samples = args.batch_size * args.grad_accum
    print(
        "[SHAPE] "
        f"per_device_batch_size={args.batch_size} "
        f"gradient_accumulation_steps={args.grad_accum} "
        f"effective_samples_per_optimizer_step={effective_samples}"
    )

    log_section("Training")
    total_optimizer_steps = 0
    cumulative_samples = 0
    cumulative_tokens = 0
    step_durations: List[float] = []
    first_fb_logged = False
    optimizer.zero_grad()

    for step, batch in enumerate(loader, start=1):
        step_t0 = time.perf_counter()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        token_count = int(batch["token_count"].sum().item())

        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits

        student_out = student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        ce_loss = student_out.loss

        t = args.temperature
        kl_loss = F.kl_div(
            F.log_softmax(student_out.logits / t, dim=-1),
            F.softmax(teacher_logits / t, dim=-1),
            reduction="batchmean",
        ) * (t ** 2)

        loss = args.alpha_kl * kl_loss + (1.0 - args.alpha_kl) * ce_loss
        (loss / args.grad_accum).backward()

        if not first_fb_logged:
            memory_snaps.append(print_memory_snapshot("after_first_forward_backward"))
            first_fb_logged = True

        should_step = (step % args.grad_accum == 0) or (step == len(loader))
        if should_step:
            optimizer.step()
            optimizer.zero_grad()
            total_optimizer_steps += 1
            cumulative_samples += input_ids.size(0) * args.grad_accum
            cumulative_tokens += token_count
            step_t1 = time.perf_counter()
            duration = step_t1 - step_t0
            step_durations.append(duration)
            elapsed = run_timer.elapsed()
            print(
                "[STEP] "
                f"opt_step={total_optimizer_steps} "
                f"elapsed={elapsed:.3f}s "
                f"step_duration={duration:.3f}s "
                f"cumulative_samples={cumulative_samples} "
                f"tokens_this_step={token_count} "
                f"cumulative_tokens={cumulative_tokens} "
                f"loss={loss.item():.6f} ce={ce_loss.item():.6f} kl={kl_loss.item():.6f}"
            )
            memory_snaps.append(print_memory_snapshot(f"optimizer_step_{total_optimizer_steps}"))

    log_section("Checkpoint Save")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_t0 = time.perf_counter()
    print(f"[CKPT] Save start ts_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    memory_snaps.append(print_memory_snapshot("before_save"))
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_t1 = time.perf_counter()
    print(f"[CKPT] Save end ts_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    save_stats = directory_size_and_count(args.output_dir)
    print(
        f"[CKPT] duration={format_seconds(save_t1 - save_t0)} "
        f"size_mb={save_stats['megabytes']:.4f} file_count={int(save_stats['file_count'])}"
    )
    memory_snaps.append(print_memory_snapshot("after_save"))

    avg_step = sum(step_durations) / len(step_durations) if step_durations else 0.0
    total_time = run_timer.elapsed()

    bottleneck_candidates = {
        "preprocessing_s": preprocessing_s,
        "avg_optimizer_step_s": avg_step,
        "checkpoint_save_s": save_t1 - save_t0,
        "max_rss_mb": max(s["rss_mb"] for s in memory_snaps),
    }
    likely = max(
        [
            ("preprocessing_overhead", bottleneck_candidates["preprocessing_s"]),
            ("checkpoint_io", bottleneck_candidates["checkpoint_save_s"]),
            ("wall_clock_training", bottleneck_candidates["avg_optimizer_step_s"]),
            ("ram_pressure_mb", bottleneck_candidates["max_rss_mb"] / 1000.0),
        ],
        key=lambda x: x[1],
    )[0]

    summary = {
        "likely_first_limit": likely,
        "notes": [
            "Inspect eval metrics for evaluation overhead and adapter reload time.",
            "Inspect sequence stats for token-length bottlenecks.",
            "Inspect environment thread settings for thread behavior anomalies.",
        ],
    }

    metrics = {
        "run": {
            "seed": args.seed,
            "model_name": args.model_name,
            "total_wall_time_s": total_time,
            "optimizer_steps": total_optimizer_steps,
            "effective_samples_per_optimizer_step": effective_samples,
            "gradient_accumulation_steps": args.grad_accum,
            "batch_size": args.batch_size,
        },
        "timings": {
            "data_load_s": load_t1 - load_t0,
            "preprocessing_s": preprocessing_s,
            "avg_optimizer_step_s": avg_step,
            "checkpoint_save_s": save_t1 - save_t0,
        },
        "length_stats": {
            "input": input_stats,
            "target": target_stats,
        },
        "checkpoint": save_stats,
        "environment": env_info,
        "memory_snapshots": memory_snaps,
        "bottleneck_summary": summary,
    }

    write_metrics_json(args.metrics_path, metrics)

    log_section("Final Bottleneck Summary")
    print(json.dumps(summary, indent=2))
    print(f"[DONE] training_run_time={format_seconds(total_time)}")


if __name__ == "__main__":
    main()
