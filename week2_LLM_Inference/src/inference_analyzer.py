import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from context_analyzer import calculate_context_usage
from decode_estimator import classify_decode_load, estimate_decode_time_seconds
from prefill_estimator import (
    BASE_TTFT_MS,
    PREFILL_TIME_PER_1K_TOKENS_MS,
    classify_prefill_load,
    estimate_ttft_ms,
)
from tokenizer_utils import (
    EXAMPLES_DIR,
    FIGURES_DIR,
    OUTPUTS_DIR,
    REQUESTED_MODEL_NAMES,
    count_characters,
    count_words,
    load_all_tokenizers,
    load_tokenizer,
    tokenize_text,
)

FULL_PROMPT_EXAMPLE_PATH = EXAMPLES_DIR / "full_prompt_example.txt"
FULL_INFERENCE_JSON_PATH = OUTPUTS_DIR / "full_inference_analysis.json"
FULL_INFERENCE_CSV_PATH = OUTPUTS_DIR / "full_inference_analysis.csv"
FULL_INFERENCE_COMPARE_JSON_PATH = OUTPUTS_DIR / "full_inference_analysis_compare.json"
FULL_INFERENCE_COMPARE_CSV_PATH = OUTPUTS_DIR / "full_inference_analysis_compare.csv"
DEFAULT_SIMULATION_RUNS = 50
DEFAULT_JITTER_PCT = 0.1
DEFAULT_RANDOM_SEED = 42


def load_text_input(text=None, file_path=None):
    """Load raw text from a direct string or a file path."""
    if text:
        return text

    if file_path:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.read_text(encoding="utf-8")

    raise ValueError("Provide either text or file_path.")


def build_warnings(
    fits_context_window,
    reserved_output_tokens,
    remaining_tokens,
    estimated_total_latency_seconds,
    prompt_tokens,
    prefill_load,
    expected_output_tokens,
    decode_load,
):
    """Build beginner-friendly warning messages for risky scenarios."""
    warnings = []

    if not fits_context_window:
        warnings.append("Prompt does not fit inside the selected context window.")
    if reserved_output_tokens > 2048:
        warnings.append("Reserved output tokens are very high for this simulated setup.")
    if remaining_tokens < 512:
        warnings.append("Remaining context tokens are low; prompt headroom is tight.")
    if estimated_total_latency_seconds > 30:
        warnings.append("Estimated total latency is high in this simulation.")
    if prompt_tokens >= 4000 and prefill_load == "high":
        warnings.append("Prompt token count is high, so prefill work is heavy.")
    if expected_output_tokens >= 1024 and decode_load == "high":
        warnings.append("Expected output is long, so decode time may dominate latency.")

    return warnings


def build_interpretation(
    prompt_tokens,
    context_used_percent,
    estimated_ttft_seconds,
    expected_output_tokens,
    tokens_per_second,
    estimated_decode_time_seconds,
    estimated_total_latency_seconds,
):
    """Create a one-row explanation that summarizes the simulated inference tradeoffs."""
    return (
        f"This prompt uses {prompt_tokens} input tokens and about {context_used_percent}% of the context window. "
        f"Estimated TTFT is {estimated_ttft_seconds} s. With {expected_output_tokens} output tokens at "
        f"{tokens_per_second} tokens/sec, decode takes about {estimated_decode_time_seconds} s and total "
        f"latency becomes about {estimated_total_latency_seconds} s. This is only a simulation, not real inference."
    )


def compute_percentile(values, percentile):
    """Compute a percentile from a list of numeric values using linear interpolation."""
    if not values:
        raise ValueError("Cannot compute a percentile from an empty list.")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 4)

    rank = (percentile / 100) * (len(sorted_values) - 1)
    lower_index = int(rank)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = rank - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    interpolated = lower_value + (upper_value - lower_value) * fraction
    return round(interpolated, 4)


def simulate_latency_distribution(
    estimated_ttft_seconds,
    estimated_decode_time_seconds,
    simulation_runs=DEFAULT_SIMULATION_RUNS,
    jitter_pct=DEFAULT_JITTER_PCT,
    random_seed=DEFAULT_RANDOM_SEED,
):
    """Simulate multiple runs with small random jitter to derive p50/p95/p99 style metrics."""
    rng = random.Random(random_seed)
    total_latencies = []

    for _ in range(simulation_runs):
        ttft_scale = 1 + rng.uniform(-jitter_pct, jitter_pct)
        decode_scale = 1 + rng.uniform(-jitter_pct, jitter_pct)
        simulated_total = (estimated_ttft_seconds * ttft_scale) + (
            estimated_decode_time_seconds * decode_scale
        )
        total_latencies.append(round(simulated_total, 4))

    avg_latency_seconds = round(sum(total_latencies) / len(total_latencies), 4)
    return {
        "simulation_runs": simulation_runs,
        "avg_latency_seconds": avg_latency_seconds,
        "p50_latency_seconds": compute_percentile(total_latencies, 50),
        "p95_latency_seconds": compute_percentile(total_latencies, 95),
        "p99_latency_seconds": compute_percentile(total_latencies, 99),
    }


def analyze_single_input(
    text,
    tokenizer_name,
    tokenizer,
    context_window,
    reserved_output_tokens,
    expected_output_tokens,
    tokens_per_second,
    simulation_runs=DEFAULT_SIMULATION_RUNS,
    jitter_pct=DEFAULT_JITTER_PCT,
    random_seed=DEFAULT_RANDOM_SEED,
):
    """Run the full simulated inference analysis for one tokenizer."""
    token_ids, _token_pieces = tokenize_text(text, tokenizer)
    prompt_token_count = len(token_ids)
    context_usage = calculate_context_usage(
        prompt_tokens=prompt_token_count,
        context_window=context_window,
        reserved_output_tokens=reserved_output_tokens,
    )

    prefill_load = classify_prefill_load(prompt_token_count)
    estimated_ttft_ms = estimate_ttft_ms(prompt_token_count)
    estimated_ttft_seconds = round(estimated_ttft_ms / 1000, 4)

    decode_load = classify_decode_load(expected_output_tokens)
    estimated_decode_time_seconds = estimate_decode_time_seconds(
        expected_output_tokens=expected_output_tokens,
        tokens_per_second=tokens_per_second,
    )
    estimated_total_latency_seconds = round(
        estimated_ttft_seconds + estimated_decode_time_seconds,
        4,
    )
    latency_distribution = simulate_latency_distribution(
        estimated_ttft_seconds=estimated_ttft_seconds,
        estimated_decode_time_seconds=estimated_decode_time_seconds,
        simulation_runs=simulation_runs,
        jitter_pct=jitter_pct,
        random_seed=random_seed,
    )
    output_throughput_tokens_per_second = round(
        expected_output_tokens / latency_distribution["avg_latency_seconds"],
        4,
    )
    requests_per_second = round(1 / latency_distribution["avg_latency_seconds"], 6)

    warnings = build_warnings(
        fits_context_window=context_usage["fits_context_window"],
        reserved_output_tokens=reserved_output_tokens,
        remaining_tokens=context_usage["remaining_tokens"],
        estimated_total_latency_seconds=latency_distribution["p95_latency_seconds"],
        prompt_tokens=prompt_token_count,
        prefill_load=prefill_load,
        expected_output_tokens=expected_output_tokens,
        decode_load=decode_load,
    )

    return {
        "tokenizer_name": tokenizer_name,
        "character_count": count_characters(text),
        "word_count": count_words(text),
        "prompt_token_count": prompt_token_count,
        "context_window": context_window,
        "reserved_output_tokens": reserved_output_tokens,
        "total_required_tokens": context_usage["total_required_tokens"],
        "remaining_tokens": context_usage["remaining_tokens"],
        "context_used_percent": context_usage["context_used_percent"],
        "fits_context_window": context_usage["fits_context_window"],
        "prefill_load": prefill_load,
        "estimated_ttft_ms": estimated_ttft_ms,
        "estimated_ttft_seconds": estimated_ttft_seconds,
        "expected_output_tokens": expected_output_tokens,
        "tokens_per_second": tokens_per_second,
        "decode_load": decode_load,
        "estimated_decode_time_seconds": estimated_decode_time_seconds,
        "estimated_total_latency_seconds": estimated_total_latency_seconds,
        "simulation_runs": latency_distribution["simulation_runs"],
        "avg_latency_seconds": latency_distribution["avg_latency_seconds"],
        "p50_latency_seconds": latency_distribution["p50_latency_seconds"],
        "p95_latency_seconds": latency_distribution["p95_latency_seconds"],
        "p99_latency_seconds": latency_distribution["p99_latency_seconds"],
        "output_throughput_tokens_per_second": output_throughput_tokens_per_second,
        "requests_per_second": requests_per_second,
        "warnings": warnings,
        "interpretation": build_interpretation(
            prompt_tokens=prompt_token_count,
            context_used_percent=context_usage["context_used_percent"],
            estimated_ttft_seconds=estimated_ttft_seconds,
            expected_output_tokens=expected_output_tokens,
            tokens_per_second=tokens_per_second,
            estimated_decode_time_seconds=estimated_decode_time_seconds,
            estimated_total_latency_seconds=estimated_total_latency_seconds,
        ),
    }


def analyze_text_across_tokenizers(
    text,
    context_window,
    reserved_output_tokens,
    expected_output_tokens,
    tokens_per_second,
    simulation_runs=DEFAULT_SIMULATION_RUNS,
    jitter_pct=DEFAULT_JITTER_PCT,
    random_seed=DEFAULT_RANDOM_SEED,
):
    """Run the unified analysis across the default tokenizer set."""
    tokenizers = load_all_tokenizers(REQUESTED_MODEL_NAMES)
    return [
        analyze_single_input(
            text=text,
            tokenizer_name=tokenizer_name,
            tokenizer=tokenizer,
            context_window=context_window,
            reserved_output_tokens=reserved_output_tokens,
            expected_output_tokens=expected_output_tokens,
            tokens_per_second=tokens_per_second,
            simulation_runs=simulation_runs,
            jitter_pct=jitter_pct,
            random_seed=random_seed,
        )
        for tokenizer_name, tokenizer in tokenizers.items()
    ]


def save_full_analysis(results, json_path=FULL_INFERENCE_JSON_PATH, csv_path=FULL_INFERENCE_CSV_PATH):
    """Save the full Step 7 analysis to JSON and CSV."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    df = pd.DataFrame(results)
    df = df[
        [
            "tokenizer_name",
            "character_count",
            "word_count",
            "prompt_token_count",
            "context_window",
            "reserved_output_tokens",
            "total_required_tokens",
            "remaining_tokens",
            "context_used_percent",
            "fits_context_window",
            "prefill_load",
            "estimated_ttft_ms",
            "estimated_ttft_seconds",
            "expected_output_tokens",
            "tokens_per_second",
            "decode_load",
            "estimated_decode_time_seconds",
            "estimated_total_latency_seconds",
            "simulation_runs",
            "avg_latency_seconds",
            "p50_latency_seconds",
            "p95_latency_seconds",
            "p99_latency_seconds",
            "output_throughput_tokens_per_second",
            "requests_per_second",
            "warnings",
            "interpretation",
        ]
    ].sort_values("estimated_total_latency_seconds", ascending=False)
    df.to_csv(csv_path, index=False)
    return df


def create_full_analysis_graphs(df, figures_dir=FIGURES_DIR):
    """Create and save all Step 7 full-analysis figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    context_usage = df.set_index("tokenizer_name")["context_used_percent"]
    ax = context_usage.plot(kind="bar", figsize=(10, 6), color="#1f77b4")
    ax.set_title("Context Usage Percent by Tokenizer")
    ax.set_xlabel("Tokenizer")
    ax.set_ylabel("Context Used Percent")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "full_context_usage.png", dpi=200)
    plt.close()

    latency_breakdown = df.set_index("tokenizer_name")[
        ["estimated_ttft_seconds", "estimated_decode_time_seconds"]
    ]
    ax = latency_breakdown.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title("Latency Breakdown: TTFT vs Decode Time")
    ax.set_xlabel("Tokenizer")
    ax.set_ylabel("Seconds")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "full_latency_breakdown.png", dpi=200)
    plt.close()

    total_latency = df.set_index("tokenizer_name")["estimated_total_latency_seconds"]
    ax = total_latency.plot(kind="bar", figsize=(10, 6), color="#2ca02c")
    ax.set_title("Total Estimated Latency by Tokenizer")
    ax.set_xlabel("Tokenizer")
    ax.set_ylabel("Estimated Total Latency (seconds)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "tokenizer_latency_comparison.png", dpi=200)
    plt.close()

    summary_df = df.set_index("tokenizer_name")[
        ["prompt_token_count", "remaining_tokens", "expected_output_tokens"]
    ]
    ax = summary_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Inference Summary: Prompt Tokens, Remaining Tokens, Expected Output Tokens")
    ax.set_xlabel("Tokenizer")
    ax.set_ylabel("Token Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "inference_summary.png", dpi=200)
    plt.close()


def run_full_analysis(
    text=None,
    file_path=None,
    tokenizer_name=None,
    context_window=8192,
    reserved_output_tokens=512,
    expected_output_tokens=512,
    tokens_per_second=35,
    compare_tokenizers=False,
    simulation_runs=DEFAULT_SIMULATION_RUNS,
    jitter_pct=DEFAULT_JITTER_PCT,
    random_seed=DEFAULT_RANDOM_SEED,
):
    """Run the full Step 7 unified analyzer for one tokenizer or across all tokenizers."""
    raw_text = load_text_input(text=text, file_path=file_path)

    if compare_tokenizers:
        results = analyze_text_across_tokenizers(
            text=raw_text,
            context_window=context_window,
            reserved_output_tokens=reserved_output_tokens,
            expected_output_tokens=expected_output_tokens,
            tokens_per_second=tokens_per_second,
            simulation_runs=simulation_runs,
            jitter_pct=jitter_pct,
            random_seed=random_seed,
        )
        json_path = FULL_INFERENCE_COMPARE_JSON_PATH
        csv_path = FULL_INFERENCE_COMPARE_CSV_PATH
    else:
        if not tokenizer_name:
            raise ValueError("Provide --tokenizer for single-tokenizer analysis.")
        tokenizer = load_tokenizer(tokenizer_name)
        results = [
            analyze_single_input(
                text=raw_text,
                tokenizer_name=tokenizer_name,
                tokenizer=tokenizer,
                context_window=context_window,
                reserved_output_tokens=reserved_output_tokens,
                expected_output_tokens=expected_output_tokens,
                tokens_per_second=tokens_per_second,
                simulation_runs=simulation_runs,
                jitter_pct=jitter_pct,
                random_seed=random_seed,
            )
        ]
        json_path = FULL_INFERENCE_JSON_PATH
        csv_path = FULL_INFERENCE_CSV_PATH

    df = save_full_analysis(results, json_path=json_path, csv_path=csv_path)
    create_full_analysis_graphs(df)
    return results, df, json_path, csv_path


def build_arg_parser():
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Full LLM Inference Analyzer CLI (simulation only).")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Direct text to analyze.")
    input_group.add_argument("--file", type=str, help="Path to a text file to analyze.")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer/model name for single analysis.")
    parser.add_argument("--compare-tokenizers", action="store_true", help="Compare all default tokenizers.")
    parser.add_argument("--context-window", type=int, required=True, help="Context window size.")
    parser.add_argument(
        "--reserved-output-tokens",
        type=int,
        required=True,
        help="Reserved output token budget.",
    )
    parser.add_argument(
        "--expected-output-tokens",
        type=int,
        required=True,
        help="Expected generated output tokens.",
    )
    parser.add_argument(
        "--tokens-per-second",
        type=float,
        required=True,
        help="Simulated generation speed in tokens per second.",
    )
    parser.add_argument(
        "--simulation-runs",
        type=int,
        default=DEFAULT_SIMULATION_RUNS,
        help="Number of simulated latency runs for percentile metrics.",
    )
    parser.add_argument(
        "--jitter-pct",
        type=float,
        default=DEFAULT_JITTER_PCT,
        help="Random jitter percentage for simulated latency distributions.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for simulated latency distributions.",
    )
    return parser


def main():
    """Run the Step 7 full analyzer from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args()

    results, df, json_path, csv_path = run_full_analysis(
        text=args.text,
        file_path=args.file,
        tokenizer_name=args.tokenizer,
        context_window=args.context_window,
        reserved_output_tokens=args.reserved_output_tokens,
        expected_output_tokens=args.expected_output_tokens,
        tokens_per_second=args.tokens_per_second,
        compare_tokenizers=args.compare_tokenizers,
        simulation_runs=args.simulation_runs,
        jitter_pct=args.jitter_pct,
        random_seed=args.random_seed,
    )

    print("Saved JSON to:", json_path)
    print("Saved CSV to:", csv_path)
    print("Saved figures to:", FIGURES_DIR)
    print()
    print(df.to_string(index=False))
    print()
    print("Simulation note: this is a conceptual analyzer, not real model inference.")
    print()
    for result in results:
        if result["warnings"]:
            print(f"Warnings for {result['tokenizer_name']}:")
            for warning in result["warnings"]:
                print(f"- {warning}")


if __name__ == "__main__":
    main()
