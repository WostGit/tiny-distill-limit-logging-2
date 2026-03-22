import argparse
import json
import time
from typing import Dict, List

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import RunTimer, format_seconds, log_section, print_environment_info, print_memory_snapshot, summarize_lengths
from metrics_utils import write_metrics_json


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_prompt(row: Dict[str, str]) -> str:
    return f"Instruction: {row['instruction']}\nInput: {row['input']}\nResponse:"


def collate(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tiny LoRA distilled adapter with verbose logs")
    parser.add_argument("--eval-file", default="data/tiny_eval.jsonl")
    parser.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--adapter-dir", default="outputs/checkpoint")
    parser.add_argument("--metrics-path", default="outputs/metrics/eval_metrics.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--gen-max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    run_timer = RunTimer.start()
    env = print_environment_info()
    memory_snaps = [print_memory_snapshot("eval_process_start")]

    rows = load_jsonl(args.eval_file)
    print(f"[EVAL] Loaded {len(rows)} rows from {args.eval_file}")

    tok_t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processed = []
    input_lens = []
    target_lens = []
    for row in rows:
        prompt = make_prompt(row)
        full_text = f"{prompt} {row['output']}"
        enc = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_lens.append(int(enc["attention_mask"].sum().item()))
        target_lens.append(len(tokenizer(row["output"], truncation=True, max_length=args.max_length)["input_ids"]))
        processed.append(
            {
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "labels": enc["input_ids"][0].clone(),
            }
        )
    tok_t1 = time.perf_counter()
    print(f"[EVAL] Preprocessing time={format_seconds(tok_t1 - tok_t0)}")
    input_stats = summarize_lengths("eval_input_tokens", input_lens)
    target_stats = summarize_lengths("eval_target_tokens", target_lens)

    loader = DataLoader(processed, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    load_t0 = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    memory_snaps.append(print_memory_snapshot("eval_after_base_model_load"))
    adapter_t0 = time.perf_counter()
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    adapter_t1 = time.perf_counter()
    model.eval()
    load_t1 = time.perf_counter()
    memory_snaps.append(print_memory_snapshot("before_eval"))

    print(
        f"[EVAL] adapter_reload_time={format_seconds(adapter_t1 - adapter_t0)} "
        f"total_model_prep_time={format_seconds(load_t1 - load_t0)}"
    )

    eval_t0 = time.perf_counter()
    losses = []
    for idx, batch in enumerate(loader, start=1):
        with torch.no_grad():
            out = model(**batch)
        losses.append(float(out.loss.item()))
        print(f"[EVAL-STEP] step={idx} loss={out.loss.item():.6f} elapsed={run_timer.elapsed():.3f}s")
    eval_t1 = time.perf_counter()

    gen_t0 = time.perf_counter()
    sample_prompt = make_prompt(rows[0])
    inputs = tokenizer(sample_prompt, return_tensors="pt")
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=args.gen_max_new_tokens)
    gen_t1 = time.perf_counter()
    generated = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    preview = generated[:180]
    print(f"[EVAL] generation_duration={format_seconds(gen_t1 - gen_t0)}")
    print(f"[EVAL] generated_preview={preview!r}")
    memory_snaps.append(print_memory_snapshot("after_eval"))

    mean_loss = sum(losses) / len(losses) if losses else 0.0
    total_runtime = run_timer.elapsed()

    summary = {
        "likely_first_limit": "evaluation_overhead" if (eval_t1 - eval_t0) > (tok_t1 - tok_t0) else "preprocessing_overhead",
        "notes": [
            "Compare evaluation duration against training step durations in train metrics.",
            "Use memory snapshots to spot RAM spikes around model reload and generation.",
        ],
    }

    payload = {
        "run": {
            "base_model": args.base_model,
            "adapter_dir": args.adapter_dir,
            "batch_size": args.batch_size,
            "total_wall_time_s": total_runtime,
        },
        "timings": {
            "preprocessing_s": tok_t1 - tok_t0,
            "adapter_reload_s": adapter_t1 - adapter_t0,
            "eval_duration_s": eval_t1 - eval_t0,
            "generation_duration_s": gen_t1 - gen_t0,
        },
        "loss": {
            "mean_eval_loss": mean_loss,
            "num_eval_steps": len(losses),
        },
        "length_stats": {
            "input": input_stats,
            "target": target_stats,
        },
        "environment": env,
        "memory_snapshots": memory_snaps,
        "bottleneck_summary": summary,
    }
    write_metrics_json(args.metrics_path, payload)

    log_section("Eval Bottleneck Summary")
    print(json.dumps(summary, indent=2))
    print(f"[DONE] eval_run_time={format_seconds(total_runtime)}")


if __name__ == "__main__":
    main()
