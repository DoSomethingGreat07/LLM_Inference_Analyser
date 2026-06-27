import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tokenizer_utils import (
    EXAMPLES_DIR,
    FIGURES_DIR,
    OUTPUTS_DIR,
    REQUESTED_MODEL_NAMES,
    load_all_tokenizers,
    tokenize_text,
)

PROMPT_SECTIONS_PATH = EXAMPLES_DIR / "prompt_sections.json"
PROMPT_PACKING_REPORT_PATH = OUTPUTS_DIR / "prompt_packing_report.csv"
DEFAULT_CONTEXT_WINDOWS = [4096, 8192, 32768]
DEFAULT_RESERVED_OUTPUT_TOKENS = 512

PRIORITY_ORDER = {
    "required": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}


def load_prompt_sections(file_path):
    """Load structured prompt sections from JSON."""
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data["sections"]


def count_section_tokens(section, tokenizer):
    """Count how many tokens a single prompt section uses."""
    token_ids, _token_pieces = tokenize_text(section["text"], tokenizer)
    return len(token_ids)


def sort_sections_by_priority(sections):
    """Sort sections so required items are considered before lower-priority items."""
    return sorted(
        sections,
        key=lambda section: (
            PRIORITY_ORDER[section["priority"]],
            section.get("order", 0),
        ),
    )


def pack_prompt_sections(sections, tokenizer, context_window, reserved_output_tokens):
    """Pack prompt sections into the available input budget by priority."""
    available_input_budget = context_window - reserved_output_tokens
    sorted_sections = sort_sections_by_priority(sections)

    included_sections = []
    dropped_sections = []
    packed_prompt_tokens = 0
    total_original_prompt_tokens = 0
    section_token_map = {}

    for section in sorted_sections:
        section_tokens = count_section_tokens(section, tokenizer)
        section_token_map[section["name"]] = section_tokens
        total_original_prompt_tokens += section_tokens

        if packed_prompt_tokens + section_tokens <= available_input_budget:
            included_sections.append(section["name"])
            packed_prompt_tokens += section_tokens
        else:
            dropped_sections.append(section["name"])

    required_sections = [section["name"] for section in sections if section["priority"] == "required"]
    required_sections_fit = all(section_name in included_sections for section_name in required_sections)
    remaining_input_tokens = available_input_budget - packed_prompt_tokens
    fits_after_packing = required_sections_fit and packed_prompt_tokens <= available_input_budget
    packed_prompt_usage_percent = round((packed_prompt_tokens / available_input_budget) * 100, 4)

    return {
        "available_input_budget": available_input_budget,
        "total_original_prompt_tokens": total_original_prompt_tokens,
        "packed_prompt_tokens": packed_prompt_tokens,
        "remaining_input_tokens": remaining_input_tokens,
        "included_sections": included_sections,
        "dropped_sections": dropped_sections,
        "number_of_included_sections": len(included_sections),
        "number_of_dropped_sections": len(dropped_sections),
        "fits_after_packing": fits_after_packing,
        "required_sections_fit": required_sections_fit,
        "packed_prompt_usage_percent": packed_prompt_usage_percent,
        "section_token_map": section_token_map,
    }


def analyze_packing_across_tokenizers(sections, tokenizers, context_windows, reserved_output_tokens):
    """Analyze prompt packing across tokenizers and context windows."""
    results = []

    for tokenizer_name, tokenizer in tokenizers.items():
        for context_window in context_windows:
            packing_result = pack_prompt_sections(
                sections=sections,
                tokenizer=tokenizer,
                context_window=context_window,
                reserved_output_tokens=reserved_output_tokens,
            )
            results.append(
                {
                    "tokenizer_name": tokenizer_name,
                    "context_window": context_window,
                    "reserved_output_tokens": reserved_output_tokens,
                    "available_input_budget": packing_result["available_input_budget"],
                    "total_original_prompt_tokens": packing_result["total_original_prompt_tokens"],
                    "packed_prompt_tokens": packing_result["packed_prompt_tokens"],
                    "remaining_input_tokens": packing_result["remaining_input_tokens"],
                    "included_sections": packing_result["included_sections"],
                    "dropped_sections": packing_result["dropped_sections"],
                    "number_of_included_sections": packing_result["number_of_included_sections"],
                    "number_of_dropped_sections": packing_result["number_of_dropped_sections"],
                    "fits_after_packing": packing_result["fits_after_packing"],
                    "required_sections_fit": packing_result["required_sections_fit"],
                    "packed_prompt_usage_percent": packing_result["packed_prompt_usage_percent"],
                    "section_token_map": packing_result["section_token_map"],
                }
            )

    return results


def save_packing_report(results, output_path):
    """Save the Step 4 prompt packing report as CSV."""
    df = pd.DataFrame(results)
    df = df[
        [
            "tokenizer_name",
            "context_window",
            "reserved_output_tokens",
            "available_input_budget",
            "total_original_prompt_tokens",
            "packed_prompt_tokens",
            "remaining_input_tokens",
            "included_sections",
            "dropped_sections",
            "number_of_included_sections",
            "number_of_dropped_sections",
            "fits_after_packing",
            "required_sections_fit",
            "packed_prompt_usage_percent",
            "section_token_map",
        ]
    ].sort_values("number_of_dropped_sections", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def build_packing_summary(df):
    """Build beginner-friendly summary insights for prompt packing."""
    dropped_counts = {}
    for sections in df["dropped_sections"]:
        for section_name in sections:
            dropped_counts[section_name] = dropped_counts.get(section_name, 0) + 1

    windows_that_keep_all = []
    for context_window, group in df.groupby("context_window"):
        if (group["number_of_dropped_sections"] == 0).all():
            windows_that_keep_all.append(context_window)
    tokenizer_totals = (
        df.groupby("tokenizer_name")["total_original_prompt_tokens"].mean().sort_values(ascending=False)
    )
    required_fit = df.groupby("context_window")["required_sections_fit"].all().to_dict()
    remaining_budget = (
        df.groupby("context_window")["remaining_input_tokens"].mean().round(2).to_dict()
    )

    most_dropped_section = None
    if dropped_counts:
        most_dropped_section = max(dropped_counts, key=dropped_counts.get)

    return {
        "most_often_dropped_section": most_dropped_section,
        "context_windows_that_keep_all_sections": windows_that_keep_all,
        "highest_tokenizer_for_same_prompt": tokenizer_totals.index[0],
        "lowest_tokenizer_for_same_prompt": tokenizer_totals.index[-1],
        "required_sections_fit_by_context_window": required_fit,
        "average_remaining_input_tokens_by_context_window": remaining_budget,
    }


def create_packing_graphs(df, figures_dir):
    """Create and save all Step 4 prompt packing figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    included_vs_dropped = df.groupby("context_window")[
        ["number_of_included_sections", "number_of_dropped_sections"]
    ].mean()
    ax = included_vs_dropped.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Included vs Dropped Sections")
    ax.set_xlabel("Context Window")
    ax.set_ylabel("Average Section Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "included_vs_dropped_sections.png", dpi=200)
    plt.close()

    section_token_totals = {}
    for section_map in df["section_token_map"]:
        for section_name, token_count in section_map.items():
            section_token_totals.setdefault(section_name, []).append(token_count)
    section_token_avg = pd.Series(
        {name: sum(values) / len(values) for name, values in section_token_totals.items()}
    ).sort_values(ascending=False)
    ax = section_token_avg.plot(kind="bar", figsize=(12, 6), color="#ff7f0e")
    ax.set_title("Token Usage by Section")
    ax.set_xlabel("Section")
    ax.set_ylabel("Average Token Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "token_budget_by_section.png", dpi=200)
    plt.close()

    usage_percent = (
        df.groupby(["context_window", "tokenizer_name"])["packed_prompt_usage_percent"]
        .mean()
        .unstack()
    )
    ax = usage_percent.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Packed Prompt Usage Percent")
    ax.set_xlabel("Context Window")
    ax.set_ylabel("Usage Percent of Available Input Budget")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "packed_prompt_usage.png", dpi=200)
    plt.close()

    strategy_comparison = (
        df.groupby("context_window")["number_of_dropped_sections"].mean().sort_values(ascending=False)
    )
    ax = strategy_comparison.plot(kind="bar", figsize=(8, 5), color="#2ca02c")
    ax.set_title("Average Dropped Sections by Context Window")
    ax.set_xlabel("Context Window")
    ax.set_ylabel("Average Dropped Sections")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figures_dir / "strategy_comparison.png", dpi=200)
    plt.close()


def run_packing_analysis(
    sections_path=PROMPT_SECTIONS_PATH,
    output_path=PROMPT_PACKING_REPORT_PATH,
    model_names=None,
    context_windows=None,
    reserved_output_tokens=DEFAULT_RESERVED_OUTPUT_TOKENS,
):
    """Run the full Step 4 prompt packing analysis pipeline."""
    if model_names is None:
        model_names = REQUESTED_MODEL_NAMES
    if context_windows is None:
        context_windows = DEFAULT_CONTEXT_WINDOWS

    sections = load_prompt_sections(sections_path)
    tokenizers = load_all_tokenizers(model_names)
    results = analyze_packing_across_tokenizers(
        sections=sections,
        tokenizers=tokenizers,
        context_windows=context_windows,
        reserved_output_tokens=reserved_output_tokens,
    )
    df = save_packing_report(results, output_path)
    create_packing_graphs(df, output_path.parent / "figures")
    summary = build_packing_summary(df)
    return df, summary


def main():
    """Run the prompt packing analysis from the terminal."""
    df, summary = run_packing_analysis()

    print("Saved report to:", PROMPT_PACKING_REPORT_PATH)
    print("Saved figures to:", FIGURES_DIR)
    print()
    print("Top rows sorted by number of dropped sections:")
    print(df.head(10).to_string(index=False))
    print()
    print("Summary insights:")
    for label, value in summary.items():
        print(f"- {label}: {value}")


if __name__ == "__main__":
    main()
