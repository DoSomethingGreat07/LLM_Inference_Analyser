from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tokenizer_utils import FIGURES_DIR, OUTPUTS_DIR

PREFILL_REPORT_PATH = OUTPUTS_DIR / "prefill_estimation_report.csv"
DECODE_REPORT_PATH = OUTPUTS_DIR / "decode_estimation_report.csv"

OUTPUT_TOKEN_BUDGETS = [128, 256, 512, 1024, 2048]
TOKENS_PER_SECOND_ASSUMPTIONS = [10, 25, 50, 100]


def load_prefill_report(prefill_report_path=PREFILL_REPORT_PATH):
    """Load the Step 5 prefill report that provides prompt size and TTFT estimates."""
    report_path = Path(prefill_report_path)
    if not report_path.exists():
        raise FileNotFoundError(
            "Missing outputs/prefill_estimation_report.csv. Run Step 5 before Step 6."
        )
    return pd.read_csv(report_path)


def classify_decode_load(expected_output_tokens):
    """Map expected output token counts to a simple decode load label."""
    if expected_output_tokens < 256:
        return "low"
    if expected_output_tokens < 1024:
        return "medium"
    return "high"


def estimate_decode_time_seconds(expected_output_tokens, tokens_per_second):
    """Estimate decode time from output token count and generation speed."""
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be greater than 0.")
    return round(expected_output_tokens / tokens_per_second, 4)


def build_decode_interpretation(
    prompt_tokens,
    estimated_ttft_seconds,
    expected_output_tokens,
    tokens_per_second,
    estimated_decode_time_seconds,
    estimated_total_latency_seconds,
):
    """Create a beginner-friendly explanation for one decode scenario."""
    return (
        f"With prompt_tokens={prompt_tokens}, estimated TTFT is about {estimated_ttft_seconds} s. "
        f"If the model generates {expected_output_tokens} output tokens at {tokens_per_second} tokens/sec, "
        f"decode takes about {estimated_decode_time_seconds} s and total latency becomes about "
        f"{estimated_total_latency_seconds} s. This is only a simulation, not a real benchmark."
    )


def build_decode_report(
    prefill_df,
    output_token_budgets=OUTPUT_TOKEN_BUDGETS,
    tokens_per_second_values=TOKENS_PER_SECOND_ASSUMPTIONS,
):
    """Create the Step 6 decode estimation report from the Step 5 prefill report."""
    rows = []

    for _, row in prefill_df.iterrows():
        for expected_output_tokens in output_token_budgets:
            for tokens_per_second in tokens_per_second_values:
                estimated_decode_time_seconds = estimate_decode_time_seconds(
                    expected_output_tokens=expected_output_tokens,
                    tokens_per_second=tokens_per_second,
                )
                estimated_ttft_seconds = float(row["estimated_ttft_seconds"])
                estimated_total_latency_seconds = round(
                    estimated_ttft_seconds + estimated_decode_time_seconds,
                    4,
                )

                rows.append(
                    {
                        "scenario_name": row["scenario_name"],
                        "tokenizer_name": row["tokenizer_name"],
                        "context_window": int(row["context_window"]),
                        "prompt_tokens": int(row["prompt_tokens"]),
                        "estimated_ttft_seconds": estimated_ttft_seconds,
                        "expected_output_tokens": expected_output_tokens,
                        "tokens_per_second": tokens_per_second,
                        "estimated_decode_time_seconds": estimated_decode_time_seconds,
                        "estimated_total_latency_seconds": estimated_total_latency_seconds,
                        "decode_load": classify_decode_load(expected_output_tokens),
                        "interpretation": build_decode_interpretation(
                            prompt_tokens=int(row["prompt_tokens"]),
                            estimated_ttft_seconds=estimated_ttft_seconds,
                            expected_output_tokens=expected_output_tokens,
                            tokens_per_second=tokens_per_second,
                            estimated_decode_time_seconds=estimated_decode_time_seconds,
                            estimated_total_latency_seconds=estimated_total_latency_seconds,
                        ),
                    }
                )

    return pd.DataFrame(rows)


def save_decode_report(df, output_path):
    """Save the Step 6 decode simulation report."""
    df = df[
        [
            "scenario_name",
            "tokenizer_name",
            "context_window",
            "prompt_tokens",
            "estimated_ttft_seconds",
            "expected_output_tokens",
            "tokens_per_second",
            "estimated_decode_time_seconds",
            "estimated_total_latency_seconds",
            "decode_load",
            "interpretation",
        ]
    ].sort_values("estimated_total_latency_seconds", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def build_decode_summary(df):
    """Build beginner-friendly Step 6 summary insights."""
    highest_total_row = df.sort_values("estimated_total_latency_seconds", ascending=False).iloc[0]
    highest_output_row = df.sort_values("expected_output_tokens", ascending=False).iloc[0]
    avg_by_tps = (
        df.groupby("tokens_per_second")["estimated_total_latency_seconds"]
        .mean()
        .round(4)
        .to_dict()
    )

    return {
        "longest_decode_output_budget": int(highest_output_row["expected_output_tokens"]),
        "highest_total_latency_scenario": highest_total_row["scenario_name"],
        "highest_total_latency_tokenizer": highest_total_row["tokenizer_name"],
        "how_tokens_per_second_changes_latency": avg_by_tps,
        "why_long_outputs_are_expensive": "Long outputs require more token-by-token generation steps, so decode time keeps growing as output length increases.",
        "why_decode_dominates_for_long_outputs": "When output length is large, token-by-token decode time can grow much larger than the initial TTFT.",
        "why_prefill_dominates_for_short_outputs": "When output is short, the up-front cost of reading the input prompt can be a bigger share of total latency.",
    }


def create_decode_graphs(df, figures_dir):
    """Create and save all Step 6 decode estimation figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    decode_by_budget = (
        df.groupby("expected_output_tokens")["estimated_decode_time_seconds"]
        .mean()
        .sort_values(ascending=False)
    )
    ax = decode_by_budget.plot(kind="bar", figsize=(10, 6), color="#1f77b4")
    ax.set_title("Decode Time by Output Token Budget")
    ax.set_xlabel("Expected Output Tokens")
    ax.set_ylabel("Average Decode Time (seconds)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "decode_time_by_output_budget.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for tokens_per_second, group in df.groupby("tokens_per_second"):
        plt.scatter(
            group["expected_output_tokens"],
            group["estimated_decode_time_seconds"],
            label=f"{tokens_per_second} tok/s",
            s=80,
        )
    plt.title("Output Tokens vs Decode Time")
    plt.xlabel("Expected Output Tokens")
    plt.ylabel("Estimated Decode Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "output_tokens_vs_decode_time.png", dpi=200)
    plt.close()

    total_latency = (
        df.groupby(["expected_output_tokens", "tokens_per_second"])["estimated_total_latency_seconds"]
        .mean()
        .unstack()
    )
    ax = total_latency.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Total Latency: Prefill + Decode")
    ax.set_xlabel("Expected Output Tokens")
    ax.set_ylabel("Estimated Total Latency (seconds)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "total_latency_prefill_plus_decode.png", dpi=200)
    plt.close()

    decode_tps = (
        df.groupby(["tokens_per_second", "expected_output_tokens"])["estimated_decode_time_seconds"]
        .mean()
        .unstack()
        .T
    )
    ax = decode_tps.plot(kind="line", figsize=(10, 6), marker="o")
    ax.set_title("Decode Time Across Tokens-Per-Second Values")
    ax.set_xlabel("Expected Output Tokens")
    ax.set_ylabel("Estimated Decode Time (seconds)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "decode_load_by_tokens_per_second.png", dpi=200)
    plt.close()


def run_decode_estimation():
    """Run the full Step 6 decode simulation pipeline."""
    prefill_df = load_prefill_report()
    df = build_decode_report(prefill_df)
    df = save_decode_report(df, DECODE_REPORT_PATH)
    create_decode_graphs(df, FIGURES_DIR)
    summary = build_decode_summary(df)
    return df, summary


def main():
    """Run the decode estimation simulation from the terminal."""
    df, summary = run_decode_estimation()

    print("Saved report to:", DECODE_REPORT_PATH)
    print("Saved figures to:", FIGURES_DIR)
    print()
    print("Top rows sorted by estimated total latency:")
    print(df.head(10).to_string(index=False))
    print()
    print("Summary insights:")
    for label, value in summary.items():
        print(f"- {label}: {value}")


if __name__ == "__main__":
    main()
