import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from logging_utils import RunClock, VerboseLogger, get_environment_snapshot, get_memory_snapshot
from metrics_utils import MetricsWriter


def load_jsonl(path: str):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="sshleifer/tiny-gpt2")
    p.add_argument("--adapter-dir", default="outputs/checkpoints/tiny_lora_adapter")
    p.add_argument("--eval-data", default="data/tiny_eval.jsonl")
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--metrics-out", default="outputs/metrics/eval_metrics.json")
    return p.parse_args()


def run_eval(base_model: str, adapter_dir: str, eval_data: str, max_length: int, metrics_out: str):
    torch.manual_seed(123)
    clock = RunClock.start()
    logger = VerboseLogger("tiny-distill-eval", clock)
    metrics = MetricsWriter()

    logger.log("env", **get_environment_snapshot(torch))
    snap = get_memory_snapshot("process_start")
    logger.log("memory", **snap)
    metrics.add_snapshot(snap)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    reload_s = time.perf_counter() - t0
    logger.log("adapter_reload", reload_s=round(reload_s, 6), adapter_dir=adapter_dir)

    snap = get_memory_snapshot("before_eval")
    logger.log("memory", **snap)
    metrics.add_snapshot(snap)

    data = load_jsonl(eval_data)
    prep_start = time.perf_counter()
    pairs = [f"{x['input']}\n{x['target']}" for x in data]
    batch = tokenizer(pairs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    preprocess_s = time.perf_counter() - prep_start
    logger.log("eval_preprocess", records=len(data), preprocess_s=round(preprocess_s, 6))

    eval_start = time.perf_counter()
    with torch.no_grad():
        out = model(**batch, labels=batch["input_ids"])
        eval_loss = out.loss.item()
    eval_s = time.perf_counter() - eval_start
    logger.log("eval_done", eval_s=round(eval_s, 6), eval_loss=round(eval_loss, 6))

    gen_start = time.perf_counter()
    prompt = tokenizer(data[0]["input"], return_tensors="pt")
    with torch.no_grad():
        _ = model.generate(**prompt, max_new_tokens=16)
    gen_s = time.perf_counter() - gen_start
    logger.log("generation_done", generation_s=round(gen_s, 6))

    snap = get_memory_snapshot("after_eval")
    logger.log("memory", **snap)
    metrics.add_snapshot(snap)

    summary = {
        "reload_s": round(reload_s, 6),
        "eval_s": round(eval_s, 6),
        "generation_s": round(gen_s, 6),
        "preprocess_s": round(preprocess_s, 6),
        "eval_loss": round(eval_loss, 6),
    }
    logger.log("eval_summary", **summary)
    metrics.set_summary(summary)
    metrics.write_json(metrics_out)


def main():
    args = parse_args()
    run_eval(args.base_model, args.adapter_dir, args.eval_data, args.max_length, args.metrics_out)


if __name__ == "__main__":
    main()
