from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tokenizer_utils import FIGURES_DIR, OUTPUTS_DIR

CONTEXT_REPORT_PATH = OUTPUTS_DIR / "context_window_report.csv"
PROMPT_PACKING_REPORT_PATH = OUTPUTS_DIR / "prompt_packing_report.csv"
PREFILL_REPORT_PATH = OUTPUTS_DIR / "prefill_estimation_report.csv"

BASE_TTFT_MS = 100
PREFILL_TIME_PER_1K_TOKENS_MS = 40


def load_previous_reports(
    packing_report_path=PROMPT_PACKING_REPORT_PATH,
    context_report_path=CONTEXT_REPORT_PATH,
):
    """Load the Step 4 packing report if available, otherwise fall back to Step 3."""
    packing_path = Path(packing_report_path)
    context_path = Path(context_report_path)

    if packing_path.exists():
        df = pd.read_csv(packing_path)
        df["scenario_name"] = df.apply(
            lambda row: f"packed_prompt_{int(row['context_window'])}",
            axis=1,
        )
        df["prompt_tokens"] = df["packed_prompt_tokens"]
        return df, "packing"

    if context_path.exists():
        df = pd.read_csv(context_path)
        df["scenario_name"] = df["prompt_name"]
        df["prompt_tokens"] = df["prompt_token_count"]
        return df, "context"

    raise FileNotFoundError(
        "Could not find either outputs/prompt_packing_report.csv or outputs/context_window_report.csv."
    )


def classify_prefill_load(prompt_tokens):
    """Map prompt token counts to a simple prefill load label."""
    if prompt_tokens < 1000:
        return "low"
    if prompt_tokens < 4000:
        return "medium"
    return "high"


def estimate_ttft_ms(
    prompt_tokens,
    base_ttft_ms=BASE_TTFT_MS,
    prefill_time_per_1k_tokens_ms=PREFILL_TIME_PER_1K_TOKENS_MS,
):
    """Estimate TTFT from prompt size using a simple simulation formula."""
    return round(
        base_ttft_ms + (prompt_tokens / 1000) * prefill_time_per_1k_tokens_ms,
        4,
    )


def build_interpretation(prefill_load, prompt_tokens, estimated_ttft_ms, source_name):
    """Create a beginner-friendly interpretation string."""
    if source_name == "packing":
        source_text = "packed prompt tokens"
    else:
        source_text = "prompt tokens"

    return (
        f"Using {source_text}={prompt_tokens}, this scenario has {prefill_load} prefill load "
        f"and an estimated TTFT of about {estimated_ttft_ms} ms. This is only a simulation, not a real benchmark."
    )


def build_prefill_report(source_df, source_name):
    """Create the Step 5 prefill estimation report from Step 3 or Step 4 outputs."""
    rows = []

    for _, row in source_df.iterrows():
        prompt_tokens = int(row["prompt_tokens"])
        tokenizer_name = row["tokenizer_name"]
        context_window = int(row["context_window"])
        reserved_output_tokens = int(row["reserved_output_tokens"])
        prefill_load = classify_prefill_load(prompt_tokens)
        estimated_ttft_ms = estimate_ttft_ms(prompt_tokens)

        rows.append(
            {
                "scenario_name": row["scenario_name"],
                "tokenizer_name": tokenizer_name,
                "context_window": context_window,
                "reserved_output_tokens": reserved_output_tokens,
                "prompt_tokens": prompt_tokens,
                "prefill_load": prefill_load,
                "base_ttft_ms": BASE_TTFT_MS,
                "prefill_time_per_1k_tokens_ms": PREFILL_TIME_PER_1K_TOKENS_MS,
                "estimated_ttft_ms": estimated_ttft_ms,
                "estimated_ttft_seconds": round(estimated_ttft_ms / 1000, 4),
                "interpretation": build_interpretation(
                    prefill_load=prefill_load,
                    prompt_tokens=prompt_tokens,
                    estimated_ttft_ms=estimated_ttft_ms,
                    source_name=source_name,
                ),
            }
        )

    return pd.DataFrame(rows)


def save_prefill_report(df, output_path):
    """Save the Step 5 prefill simulation report."""
    df = df[
        [
            "scenario_name",
            "tokenizer_name",
            "context_window",
            "reserved_output_tokens",
            "prompt_tokens",
            "prefill_load",
            "base_ttft_ms",
            "prefill_time_per_1k_tokens_ms",
            "estimated_ttft_ms",
            "estimated_ttft_seconds",
            "interpretation",
        ]
    ].sort_values("estimated_ttft_ms", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def build_prefill_summary(df):
    """Build beginner-friendly Step 5 summary insights."""
    highest_row = df.sort_values("estimated_ttft_ms", ascending=False).iloc[0]
    highest_window_row = df.sort_values("prompt_tokens", ascending=False).iloc[0]
    avg_by_context_window = (
        df.groupby("context_window")["estimated_ttft_ms"].mean().round(2).to_dict()
    )

    return {
        "highest_prefill_load_scenario": highest_row["scenario_name"],
        "highest_prefill_load_tokenizer": highest_row["tokenizer_name"],
        "highest_input_size_context_window": int(highest_window_row["context_window"]),
        "why_long_rag_increases_ttft": "Long RAG prompts add many input tokens, so the model must process more tokens before the first output token can arrive.",
        "why_long_chat_history_increases_ttft": "Long chat history increases prompt tokens, which increases prefill work and pushes TTFT upward.",
        "average_estimated_ttft_by_context_window": avg_by_context_window,
    }


def create_prefill_graphs(df, figures_dir):
    """Create and save all Step 5 prefill estimation figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    scenario_ttft = (
        df.groupby("scenario_name")["estimated_ttft_ms"].mean().sort_values(ascending=False)
    )
    ax = scenario_ttft.plot(kind="bar", figsize=(12, 6), color="#1f77b4")
    ax.set_title("Prefill Load by Prompt or Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Average Estimated TTFT (ms)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "prefill_load_by_prompt.png", dpi=200)
    plt.close()

    scenario_ttft = (
        df.groupby(["scenario_name", "tokenizer_name"])["estimated_ttft_ms"]
        .mean()
        .unstack()
    )
    ax = scenario_ttft.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Estimated TTFT by Prompt or Scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Estimated TTFT (ms)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "estimated_ttft_by_prompt.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    for tokenizer_name, group in df.groupby("tokenizer_name"):
        plt.scatter(group["prompt_tokens"], group["estimated_ttft_ms"], label=tokenizer_name, s=90)
    plt.title("Prompt Tokens vs Estimated TTFT")
    plt.xlabel("Prompt Tokens")
    plt.ylabel("Estimated TTFT (ms)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "prompt_tokens_vs_estimated_ttft.png", dpi=200)
    plt.close()

    context_window_ttft = (
        df.groupby("context_window")["estimated_ttft_ms"].mean().sort_values(ascending=False)
    )
    ax = context_window_ttft.plot(kind="bar", figsize=(8, 5), color="#2ca02c")
    ax.set_title("Average Estimated TTFT by Context Window")
    ax.set_xlabel("Context Window")
    ax.set_ylabel("Estimated TTFT (ms)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "prefill_load_by_context_window.png", dpi=200)
    plt.close()


def run_prefill_estimation():
    """Run the full Step 5 prefill simulation pipeline."""
    source_df, source_name = load_previous_reports()
    df = build_prefill_report(source_df, source_name)
    df = save_prefill_report(df, PREFILL_REPORT_PATH)
    create_prefill_graphs(df, FIGURES_DIR)
    summary = build_prefill_summary(df)
    return df, summary


def main():
    """Run the prefill estimation simulation from the terminal."""
    df, summary = run_prefill_estimation()

    print("Saved report to:", PREFILL_REPORT_PATH)
    print("Saved figures to:", FIGURES_DIR)
    print()
    print("Top rows sorted by estimated TTFT:")
    print(df.head(10).to_string(index=False))
    print()
    print("Summary insights:")
    for label, value in summary.items():
        print(f"- {label}: {value}")


if __name__ == "__main__":
    main()
