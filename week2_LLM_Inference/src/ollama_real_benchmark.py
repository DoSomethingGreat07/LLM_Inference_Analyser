import argparse
import csv
import json
import os
import statistics
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PROMPT_PATH = PROJECT_ROOT / "examples" / "full_prompt_example.txt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ARCHIVE_OUTPUTS_DIR = OUTPUTS_DIR / "benchmark_runs"
RUNS_CSV_PATH = OUTPUTS_DIR / "real_benchmark_runs.csv"
SUMMARY_CSV_PATH = OUTPUTS_DIR / "real_benchmark_summary.csv"
SUMMARY_JSON_PATH = OUTPUTS_DIR / "real_benchmark_summary.json"

DEFAULT_MODELS = [
    "qwen3:4b",
    "mistral:instruct",
    "gemma3:4b",
]
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")


def compute_percentile(values, percentile):
    """Compute a percentile with linear interpolation."""
    if not values:
        raise ValueError("Cannot compute percentile for an empty list.")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 4)

    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return round(lower_value + (upper_value - lower_value) * fraction, 4)


def load_prompt(prompt_text=None, prompt_file=None):
    """Load the benchmark prompt from text or file."""
    if prompt_text:
        return prompt_text
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    return EXAMPLE_PROMPT_PATH.read_text(encoding="utf-8")


def stream_generate(model_name, prompt, num_predict, temperature, api_url):
    """Send one streaming generation request to Ollama and measure timings."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    started_at = time.perf_counter()
    first_token_at = None
    finished_at = None
    final_chunk = None

    with requests.post(api_url, json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            chunk = json.loads(line.decode("utf-8"))
            if first_token_at is None and (chunk.get("response") or chunk.get("thinking")):
                first_token_at = time.perf_counter()

            if chunk.get("done"):
                final_chunk = chunk
                finished_at = time.perf_counter()
                break

    if first_token_at is None or final_chunk is None or finished_at is None:
        raise RuntimeError(f"Did not receive a complete streaming response for model {model_name}.")

    ttft_seconds = first_token_at - started_at
    total_latency_seconds = finished_at - started_at
    decode_wall_seconds = finished_at - first_token_at

    prompt_eval_count = final_chunk.get("prompt_eval_count", 0)
    eval_count = final_chunk.get("eval_count", 0)
    prompt_eval_duration_ns = final_chunk.get("prompt_eval_duration", 0)
    eval_duration_ns = final_chunk.get("eval_duration", 0)

    prefill_seconds = round(prompt_eval_duration_ns / 1_000_000_000, 4) if prompt_eval_duration_ns else None
    decode_model_seconds = round(eval_duration_ns / 1_000_000_000, 4) if eval_duration_ns else None
    output_tokens_per_second = None
    if eval_duration_ns and eval_count:
        output_tokens_per_second = round(eval_count / (eval_duration_ns / 1_000_000_000), 4)

    return {
        "prompt_token_count": prompt_eval_count,
        "generated_token_count": eval_count,
        "ttft_seconds": round(ttft_seconds, 4),
        "prefill_seconds": prefill_seconds,
        "decode_wall_seconds": round(decode_wall_seconds, 4),
        "decode_model_seconds": decode_model_seconds,
        "total_latency_seconds": round(total_latency_seconds, 4),
        "output_tokens_per_second": output_tokens_per_second,
    }


def benchmark_model(model_name, prompt, num_predict, num_runs, warmup_runs, temperature, api_url):
    """Benchmark one model across warmup and measured runs."""
    for _ in range(warmup_runs):
        stream_generate(model_name, prompt, num_predict, temperature, api_url)

    runs = []
    for run_index in range(1, num_runs + 1):
        result = stream_generate(model_name, prompt, num_predict, temperature, api_url)
        result["model_name"] = model_name
        result["run_index"] = run_index
        runs.append(result)

    return runs


def summarize_runs(model_name, runs):
    """Build aggregate benchmark statistics for one model."""
    ttft_values = [run["ttft_seconds"] for run in runs]
    total_values = [run["total_latency_seconds"] for run in runs]
    decode_values = [run["decode_wall_seconds"] for run in runs]
    throughput_values = [run["output_tokens_per_second"] for run in runs if run["output_tokens_per_second"]]

    prompt_token_count = runs[0]["prompt_token_count"]
    generated_token_count = runs[0]["generated_token_count"]
    prefill_seconds = [run["prefill_seconds"] for run in runs if run["prefill_seconds"]]
    decode_model_seconds = [run["decode_model_seconds"] for run in runs if run["decode_model_seconds"]]

    return {
        "model_name": model_name,
        "prompt_token_count": prompt_token_count,
        "generated_token_count": generated_token_count,
        "num_runs": len(runs),
        "avg_ttft_seconds": round(statistics.mean(ttft_values), 4),
        "p50_ttft_seconds": compute_percentile(ttft_values, 50),
        "p90_ttft_seconds": compute_percentile(ttft_values, 90),
        "p95_ttft_seconds": compute_percentile(ttft_values, 95),
        "p99_ttft_seconds": compute_percentile(ttft_values, 99),
        "avg_decode_wall_seconds": round(statistics.mean(decode_values), 4),
        "p50_decode_wall_seconds": compute_percentile(decode_values, 50),
        "p90_decode_wall_seconds": compute_percentile(decode_values, 90),
        "p95_decode_wall_seconds": compute_percentile(decode_values, 95),
        "p99_decode_wall_seconds": compute_percentile(decode_values, 99),
        "avg_total_latency_seconds": round(statistics.mean(total_values), 4),
        "p50_total_latency_seconds": compute_percentile(total_values, 50),
        "p90_total_latency_seconds": compute_percentile(total_values, 90),
        "p95_total_latency_seconds": compute_percentile(total_values, 95),
        "p99_total_latency_seconds": compute_percentile(total_values, 99),
        "avg_prefill_seconds": round(statistics.mean(prefill_seconds), 4) if prefill_seconds else None,
        "avg_decode_model_seconds": round(statistics.mean(decode_model_seconds), 4) if decode_model_seconds else None,
        "avg_output_tokens_per_second": round(statistics.mean(throughput_values), 4) if throughput_values else None,
    }


def save_runs_csv(runs, output_path):
    """Save per-run measurements."""
    fieldnames = [
        "model_name",
        "run_index",
        "prompt_token_count",
        "generated_token_count",
        "ttft_seconds",
        "prefill_seconds",
        "decode_wall_seconds",
        "decode_model_seconds",
        "total_latency_seconds",
        "output_tokens_per_second",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs)


def save_summary(summary_rows, csv_path, json_path):
    """Save aggregate benchmark summary."""
    df = (
        statistics_to_dataframe(summary_rows)
        .sort_values("p95_total_latency_seconds")
        .reset_index(drop=True)
    )
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    return df


def build_output_paths(output_tag=None):
    """Build stable and archival output paths for a benchmark run."""
    if output_tag:
        suffix = output_tag
    else:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    archive_dir = ARCHIVE_OUTPUTS_DIR / suffix
    archive_dir.mkdir(parents=True, exist_ok=True)

    return {
        "latest_runs_csv": RUNS_CSV_PATH,
        "latest_summary_csv": SUMMARY_CSV_PATH,
        "latest_summary_json": SUMMARY_JSON_PATH,
        "archive_runs_csv": archive_dir / "real_benchmark_runs.csv",
        "archive_summary_csv": archive_dir / "real_benchmark_summary.csv",
        "archive_summary_json": archive_dir / "real_benchmark_summary.json",
        "archive_dir": archive_dir,
    }


def statistics_to_dataframe(summary_rows):
    """Convert summary rows into a dataframe."""
    return pd.DataFrame(summary_rows)


def build_arg_parser():
    """CLI arguments for the real benchmark script."""
    parser = argparse.ArgumentParser(description="Real Ollama benchmark for TTFT, decode, percentiles, and throughput.")
    parser.add_argument("--prompt-file", type=str, default=str(EXAMPLE_PROMPT_PATH), help="Prompt file path.")
    parser.add_argument("--prompt-text", type=str, help="Direct prompt text.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Ollama model names to benchmark.")
    parser.add_argument("--num-predict", type=int, default=128, help="Maximum output tokens to generate.")
    parser.add_argument("--num-runs", type=int, default=10, help="Measured runs per model.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs per model.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--api-url", type=str, default=DEFAULT_OLLAMA_URL, help="Streaming generate endpoint URL.")
    parser.add_argument("--output-tag", type=str, help="Optional folder name for archived benchmark outputs.")
    return parser


def main():
    """Run the real local benchmark and save results."""
    parser = build_arg_parser()
    args = parser.parse_args()

    prompt = load_prompt(prompt_text=args.prompt_text, prompt_file=args.prompt_file)
    all_runs = []
    summaries = []
    output_paths = build_output_paths(args.output_tag)

    for model_name in args.models:
        runs = benchmark_model(
            model_name=model_name,
            prompt=prompt,
            num_predict=args.num_predict,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs,
            temperature=args.temperature,
            api_url=args.api_url,
        )
        all_runs.extend(runs)
        summaries.append(summarize_runs(model_name, runs))

    save_runs_csv(all_runs, output_paths["latest_runs_csv"])
    save_runs_csv(all_runs, output_paths["archive_runs_csv"])
    summary_df = save_summary(summaries, output_paths["latest_summary_csv"], output_paths["latest_summary_json"])
    save_summary(summaries, output_paths["archive_summary_csv"], output_paths["archive_summary_json"])

    print("Saved latest per-run results to:", output_paths["latest_runs_csv"])
    print("Saved latest summary CSV to:", output_paths["latest_summary_csv"])
    print("Saved latest summary JSON to:", output_paths["latest_summary_json"])
    print("Saved archived outputs to:", output_paths["archive_dir"])
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
