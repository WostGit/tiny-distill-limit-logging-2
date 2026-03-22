#!/usr/bin/env python3
import argparse
import json
import time
from statistics import mean
from typing import Dict, List

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import VerboseLogger
from metrics_utils import write_metrics


def read_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


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
    parser.add_argument("--eval-file", default="data/tiny_eval.jsonl")
    parser.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--adapter-dir", default="outputs/checkpoints/tiny_adapter")
    parser.add_argument("--metrics-path", default="outputs/metrics/eval_metrics.json")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    logger = VerboseLogger("EVAL")
    env = logger.env_report()
    logger.snapshot_memory("before_eval")

    records = read_jsonl(args.eval_file)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = [f"Prompt: {r['prompt']}\nResponse: {r['target']}" for r in records]
    enc = tokenizer(inputs, truncation=True, max_length=args.max_length, padding=False)
    examples = [{"input_ids": ids, "labels": ids.copy()} for ids in enc["input_ids"]]
    loader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
    )

    reload_t0 = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(args.base_model)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()
    adapter_reload_seconds = time.perf_counter() - reload_t0
    logger.log(f"ADAPTER RELOAD adapter_reload_seconds={adapter_reload_seconds:.4f}")
    logger.snapshot_memory("after_adapter_reload")

    eval_t0 = time.perf_counter()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader, start=1):
            step_t0 = time.perf_counter()
            out = model(**batch)
            losses.append(float(out.loss.item()))
            logger.log(f"EVAL STEP idx={i} step_seconds={time.perf_counter() - step_t0:.4f} loss={out.loss.item():.6f}")
    eval_seconds = time.perf_counter() - eval_t0

    generation_t0 = time.perf_counter()
    sample_prompt = "Prompt: Explain why verbose CI logs are useful.\nResponse:"
    sample_ids = tokenizer(sample_prompt, return_tensors="pt")
    with torch.no_grad():
        _ = model.generate(**sample_ids, max_new_tokens=16)
    generation_seconds = time.perf_counter() - generation_t0

    logger.log(
        f"EVAL SUMMARY mean_loss={mean(losses):.6f} eval_seconds={eval_seconds:.4f} "
        f"generation_seconds={generation_seconds:.4f}"
    )
    logger.snapshot_memory("after_eval")

    dominant = {
        "adapter_reload_seconds": adapter_reload_seconds,
        "eval_seconds": eval_seconds,
        "generation_seconds": generation_seconds,
    }
    bottleneck = max(dominant.items(), key=lambda x: x[1])[0]
    logger.log(f"FINAL BOTTLENECK SUMMARY candidate={bottleneck} details={dominant}")

    metrics = {
        "run_type": "eval",
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "adapter_reload_seconds": adapter_reload_seconds,
        "eval_seconds": eval_seconds,
        "generation_seconds": generation_seconds,
        "mean_loss": mean(losses) if losses else None,
        "losses": losses,
        "memory_snapshots": logger.memory_as_dicts(),
        "environment": env,
        "bottleneck_candidate": bottleneck,
    }
    write_metrics(args.metrics_path, metrics)
    logger.log(f"WROTE METRICS path={args.metrics_path}")


if __name__ == "__main__":
    main()
