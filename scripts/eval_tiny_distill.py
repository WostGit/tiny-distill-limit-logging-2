import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from logging_utils import EventLogger, environment_snapshot, memory_snapshot
from metrics_utils import write_json


def load_jsonl(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate_adapter(
    base_model: str,
    adapter_dir: str,
    eval_data: str,
    max_length: int,
    metrics_dir: str,
    prefix: str = "",
) -> Dict[str, float]:
    logger = EventLogger()
    rows = load_jsonl(eval_data)
    env = environment_snapshot()
    memories = [memory_snapshot(f"{prefix}eval_process_start")]

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model)
    reload_start = time.perf_counter()
    model = PeftModel.from_pretrained(base, adapter_dir)
    adapter_reload_time = time.perf_counter() - reload_start
    model.eval()
    reload_total = time.perf_counter() - t0
    logger.log(
        "adapter_reload",
        adapter_reload_time_s=round(adapter_reload_time, 6),
        total_model_init_s=round(reload_total, 6),
        eval_rows=len(rows),
    )
    memories.append(memory_snapshot(f"{prefix}eval_after_model_reload"))

    eval_start = time.perf_counter()
    losses = []
    generation_total = 0.0
    for idx, row in enumerate(rows):
        text = row["prompt"].strip() + "\n" + row["target"].strip()
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        losses.append(float(out.loss.detach().cpu().item()))

        gen_inputs = tokenizer(row["prompt"], return_tensors="pt", truncation=True, max_length=max_length)
        gen_start = time.perf_counter()
        _ = model.generate(**gen_inputs, max_new_tokens=8, do_sample=False)
        generation_total += time.perf_counter() - gen_start

        logger.log(
            "eval_sample",
            sample_idx=idx,
            loss=losses[-1],
            input_tokens=int(enc["input_ids"].shape[1]),
        )

    eval_duration = time.perf_counter() - eval_start
    memories.append(memory_snapshot(f"{prefix}eval_after_eval_loop"))

    metrics = {
        "mean_loss": sum(losses) / max(len(losses), 1),
        "num_samples": len(losses),
        "adapter_reload_time_s": adapter_reload_time,
        "model_init_total_s": reload_total,
        "eval_duration_s": eval_duration,
        "generation_duration_s": generation_total,
    }

    payload = {
        "environment": env,
        "memory": memories,
        "metrics": metrics,
    }
    out_path = str(Path(metrics_dir) / f"{prefix}eval_metrics.json")
    write_json(out_path, payload)
    logger.log("eval_summary", **{k: round(v, 6) if isinstance(v, float) else v for k, v in metrics.items()})
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tiny eval for LoRA distilled adapter")
    p.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    p.add_argument("--adapter-dir", default="outputs/checkpoints/tiny_adapter")
    p.add_argument("--eval-data", default="data/tiny_eval.jsonl")
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--metrics-dir", default="outputs/metrics")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_adapter(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        eval_data=args.eval_data,
        max_length=args.max_length,
        metrics_dir=args.metrics_dir,
    )
