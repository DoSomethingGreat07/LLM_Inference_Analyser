import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tokenizer_utils import (
    EXAMPLES_DIR,
    FIGURES_DIR,
    OUTPUTS_DIR,
    REQUESTED_MODEL_NAMES,
    count_characters,
    count_words,
    load_all_tokenizers,
    tokenize_text,
)

CONTEXT_EXAMPLES_PATH = EXAMPLES_DIR / "context_examples.json"
CONTEXT_REPORT_PATH = OUTPUTS_DIR / "context_window_report.csv"
DEFAULT_CONTEXT_WINDOWS = [4096, 8192, 32768]
DEFAULT_RESERVED_OUTPUT_TOKENS = [256, 512, 1024]


def calculate_context_usage(prompt_tokens, context_window, reserved_output_tokens):
    """Calculate whether prompt tokens plus reserved output fit in the context window."""
    total_required_tokens = prompt_tokens + reserved_output_tokens
    remaining_tokens = context_window - total_required_tokens
    context_used_percent = round((total_required_tokens / context_window) * 100, 4)
    fits_context_window = total_required_tokens <= context_window

    return {
        "total_required_tokens": total_required_tokens,
        "remaining_tokens": remaining_tokens,
        "context_used_percent": context_used_percent,
        "fits_context_window": fits_context_window,
    }


def analyze_prompt_context(
    prompt_name,
    text,
    tokenizer_name,
    tokenizer,
    context_window,
    reserved_output_tokens,
):
    """Analyze one prompt for one tokenizer, one context window, and one output reservation."""
    token_ids, _token_pieces = tokenize_text(text, tokenizer)
    prompt_token_count = len(token_ids)
    usage = calculate_context_usage(
        prompt_tokens=prompt_token_count,
        context_window=context_window,
        reserved_output_tokens=reserved_output_tokens,
    )

    return {
        "prompt_name": prompt_name,
        "tokenizer_name": tokenizer_name,
        "character_count": count_characters(text),
        "word_count": count_words(text),
        "prompt_token_count": prompt_token_count,
        "reserved_output_tokens": reserved_output_tokens,
        "context_window": context_window,
        "total_required_tokens": usage["total_required_tokens"],
        "remaining_tokens": usage["remaining_tokens"],
        "context_used_percent": usage["context_used_percent"],
        "fits_context_window": usage["fits_context_window"],
    }


def analyze_multiple_contexts(prompts, tokenizers, context_windows, reserved_output_tokens_list):
    """Analyze all prompt scenarios across tokenizers, context windows, and reserved outputs."""
    results = []

    for prompt in prompts:
        for tokenizer_name, tokenizer in tokenizers.items():
            for context_window in context_windows:
                for reserved_output_tokens in reserved_output_tokens_list:
                    results.append(
                        analyze_prompt_context(
                            prompt_name=prompt["prompt_name"],
                            text=prompt["text"],
                            tokenizer_name=tokenizer_name,
                            tokenizer=tokenizer,
                            context_window=context_window,
                            reserved_output_tokens=reserved_output_tokens,
                        )
                    )

    return results


def save_context_report(results, output_path):
    """Save the Step 3 context report as a sorted CSV file."""
    df = pd.DataFrame(results)
    df = df[
        [
            "prompt_name",
            "tokenizer_name",
            "character_count",
            "word_count",
            "prompt_token_count",
            "reserved_output_tokens",
            "context_window",
            "total_required_tokens",
            "remaining_tokens",
            "context_used_percent",
            "fits_context_window",
        ]
    ].sort_values("context_used_percent", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_context_examples(file_path):
    """Load prompt scenarios from JSON."""
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data["prompts"]


def build_context_summary(df):
    """Build beginner-friendly context-window insights from the report."""
    highest_usage_row = df.sort_values("context_used_percent", ascending=False).iloc[0]
    tokenizer_usage = (
        df.groupby("tokenizer_name")["context_used_percent"].mean().sort_values(ascending=False)
    )

    fits_4096_without_room = df[
        (df["context_window"] == 4096)
        & (df["fits_context_window"])
        & (df["remaining_tokens"] < 512)
    ]["prompt_name"].drop_duplicates().tolist()

    fits_8192 = df[
        (df["context_window"] == 8192) & (df["fits_context_window"])
    ]["prompt_name"].drop_duplicates().tolist()

    large_window_prompts = df[
        (df["context_window"] == 4096) & (~df["fits_context_window"])
    ]["prompt_name"].drop_duplicates().tolist()

    return {
        "most_context_heavy_prompt": highest_usage_row["prompt_name"],
        "highest_average_context_tokenizer": tokenizer_usage.index[0],
        "lowest_average_context_tokenizer": tokenizer_usage.index[-1],
        "tight_4096_prompts": fits_4096_without_room,
        "prompts_that_fit_8192": fits_8192,
        "prompts_that_need_larger_context_windows": large_window_prompts,
    }


def create_context_graphs(df, figures_dir):
    """Create and save all Step 3 context-window figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    graph_df = df[
        (df["context_window"] == 4096) & (df["reserved_output_tokens"] == 512)
    ]

    usage_by_prompt = (
        graph_df.groupby("prompt_name")["context_used_percent"].mean().sort_values(ascending=False)
    )
    ax = usage_by_prompt.plot(kind="bar", figsize=(12, 6), color="#1f77b4")
    ax.set_title("Context Used Percent by Prompt")
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Context Used Percent")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "context_usage_by_prompt.png", dpi=200)
    plt.close()

    remaining_by_prompt = (
        graph_df.groupby("prompt_name")["remaining_tokens"].mean().sort_values()
    )
    ax = remaining_by_prompt.plot(kind="bar", figsize=(12, 6), color="#ff7f0e")
    ax.set_title("Remaining Tokens by Prompt")
    ax.set_xlabel("Prompt")
    ax.set_ylabel("Remaining Tokens")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "remaining_tokens_by_prompt.png", dpi=200)
    plt.close()

    tokenizer_usage = (
        graph_df.groupby("tokenizer_name")["context_used_percent"].mean().sort_values(ascending=False)
    )
    ax = tokenizer_usage.plot(kind="bar", figsize=(10, 6), color="#2ca02c")
    ax.set_title("Average Context Used Percent by Tokenizer")
    ax.set_xlabel("Tokenizer")
    ax.set_ylabel("Average Context Used Percent")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "context_used_percent_by_tokenizer.png", dpi=200)
    plt.close()

    fit_counts = graph_df["fits_context_window"].value_counts().rename(
        index={True: "Fits", False: "Exceeds"}
    )
    ax = fit_counts.plot(kind="bar", figsize=(8, 5), color=["#4caf50", "#d62728"])
    ax.set_title("Prompts That Fit vs Exceed Context Window")
    ax.set_xlabel("Result")
    ax.set_ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "fits_vs_exceeds_context.png", dpi=200)
    plt.close()


def run_context_analysis(
    prompts_path=CONTEXT_EXAMPLES_PATH,
    output_path=CONTEXT_REPORT_PATH,
    model_names=None,
    context_windows=None,
    reserved_output_tokens_list=None,
):
    """Run the full Step 3 context-window analysis pipeline."""
    if model_names is None:
        model_names = REQUESTED_MODEL_NAMES
    if context_windows is None:
        context_windows = DEFAULT_CONTEXT_WINDOWS
    if reserved_output_tokens_list is None:
        reserved_output_tokens_list = DEFAULT_RESERVED_OUTPUT_TOKENS

    prompts = load_context_examples(prompts_path)
    tokenizers = load_all_tokenizers(model_names)
    results = analyze_multiple_contexts(
        prompts=prompts,
        tokenizers=tokenizers,
        context_windows=context_windows,
        reserved_output_tokens_list=reserved_output_tokens_list,
    )
    df = save_context_report(results, output_path)
    create_context_graphs(df, output_path.parent / "figures")
    summary = build_context_summary(df)
    return df, summary


def main():
    """Run the context-window analysis from the terminal."""
    df, summary = run_context_analysis()

    print("Saved report to:", CONTEXT_REPORT_PATH)
    print("Saved figures to:", FIGURES_DIR)
    print()
    print("Top rows sorted by context used percent:")
    print(df.head(10).to_string(index=False))
    print()
    print("Summary insights:")
    for label, value in summary.items():
        print(f"- {label}: {value}")


if __name__ == "__main__":
    main()
