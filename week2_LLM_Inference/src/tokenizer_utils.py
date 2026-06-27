import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
CASES_PATH = EXAMPLES_DIR / "tokenization_cases.json"
REPORT_PATH = OUTPUTS_DIR / "tokenization_behavior_report.csv"

REQUESTED_MODEL_NAMES = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

TOKENIZER_FALLBACKS = {
    "mistralai/Mistral-7B-Instruct-v0.2": "gpt2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "gpt2",
}

LOCAL_TOKENIZER_DIRS = {
    "Qwen/Qwen2.5-1.5B-Instruct": MODELS_DIR / "qwen_tokenizer",
    "mistralai/Mistral-7B-Instruct-v0.2": MODELS_DIR / "mistral_tokenizer",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": MODELS_DIR / "tinyllama_tokenizer",
}


def load_tokenizer(model_name):
    """Load one tokenizer from a local folder if present, otherwise from Hugging Face."""
    local_dir = LOCAL_TOKENIZER_DIRS.get(model_name)
    if local_dir is not None and local_dir.exists():
        return AutoTokenizer.from_pretrained(local_dir)
    return AutoTokenizer.from_pretrained(model_name)


def get_tokenizer_label(model_name):
    """Show whether a tokenizer came from a local folder or Hugging Face."""
    local_dir = LOCAL_TOKENIZER_DIRS.get(model_name)
    if local_dir is not None and local_dir.exists():
        return f"{model_name} (local)"
    return model_name


def load_all_tokenizers(model_names):
    """Load all requested tokenizers and fall back to a backup if needed."""
    tokenizers = {}

    for model_name in model_names:
        tokenizer_name = get_tokenizer_label(model_name)
        try:
            tokenizer = load_tokenizer(model_name)
        except Exception as exc:
            fallback_name = TOKENIZER_FALLBACKS.get(model_name)
            if fallback_name is None:
                raise RuntimeError(f"Could not load tokenizer '{model_name}': {exc}") from exc

            print(
                f"Could not load '{model_name}' because of: {exc}\n"
                f"Falling back to '{fallback_name}' for this experiment."
            )
            tokenizer = load_tokenizer(fallback_name)
            tokenizer_name = f"{model_name} -> {get_tokenizer_label(fallback_name)}"

        tokenizers[tokenizer_name] = tokenizer

    return tokenizers


def count_characters(text):
    """Count every character in the input text, including spaces and newlines."""
    return len(text)


def count_words(text):
    """Count words by splitting on whitespace."""
    return len(text.split())


def tokenize_text(text, tokenizer):
    """Convert text into token IDs and token pieces using one tokenizer."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
    return token_ids, token_pieces


def analyze_text_with_tokenizer(text, category, tokenizer_name, tokenizer):
    """Measure how one tokenizer splits one text example."""
    token_ids, token_pieces = tokenize_text(text, tokenizer)
    character_count = count_characters(text)
    word_count = count_words(text)
    token_count = len(token_ids)

    return {
        "category": category,
        "tokenizer_name": tokenizer_name,
        "text": text,
        "character_count": character_count,
        "word_count": word_count,
        "token_count": token_count,
        "tokens_per_word": round(token_count / word_count, 4) if word_count else 0.0,
        "tokens_per_character": round(token_count / character_count, 4) if character_count else 0.0,
        "token_pieces": token_pieces,
    }


def analyze_text_across_tokenizers(text, category, tokenizers):
    """Analyze one text example with every tokenizer."""
    return [
        analyze_text_with_tokenizer(text, category, tokenizer_name, tokenizer)
        for tokenizer_name, tokenizer in tokenizers.items()
    ]


def analyze_multiple_texts(cases, tokenizers):
    """Analyze all text cases with all tokenizers."""
    results = []

    for case in cases:
        results.extend(
            analyze_text_across_tokenizers(
                text=case["text"],
                category=case["category"],
                tokenizers=tokenizers,
            )
        )

    return results


def save_report(results, output_path):
    """Save the full tokenization comparison report as a CSV file."""
    df = pd.DataFrame(results)
    df = df[
        [
            "category",
            "tokenizer_name",
            "text",
            "character_count",
            "word_count",
            "token_count",
            "tokens_per_word",
            "tokens_per_character",
            "token_pieces",
        ]
    ].sort_values("token_count", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_cases(cases_path):
    """Load tokenization experiment cases from JSON."""
    with cases_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data["cases"]


def build_summary(df):
    """Build beginner-friendly summary facts from the report."""
    category_totals = df.groupby("category")["token_count"].mean().sort_values(ascending=False)
    tokenizer_totals = (
        df.groupby("tokenizer_name")["token_count"].mean().sort_values(ascending=False)
    )
    category_spread = (
        df.groupby("category")["token_count"].agg(["min", "max"]).assign(spread=lambda x: x["max"] - x["min"])
    )

    return {
        "most_token_heavy_category": category_totals.index[0],
        "highest_average_tokenizer": tokenizer_totals.index[0],
        "lowest_average_tokenizer": tokenizer_totals.index[-1],
        "most_variable_category": category_spread["spread"].sort_values(ascending=False).index[0],
    }


def create_graphs(df, figures_dir):
    """Create and save all Step 2 matplotlib visualizations."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    token_count_pivot = df.pivot(
        index="category",
        columns="tokenizer_name",
        values="token_count",
    )
    ax = token_count_pivot.plot(kind="bar", figsize=(14, 7))
    ax.set_title("Token Count by Category and Tokenizer")
    ax.set_xlabel("Category")
    ax.set_ylabel("Token Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "token_count_by_category_and_tokenizer.png", dpi=200)
    plt.close()

    tokens_per_word_pivot = df.pivot(
        index="category",
        columns="tokenizer_name",
        values="tokens_per_word",
    )
    ax = tokens_per_word_pivot.plot(kind="bar", figsize=(14, 7))
    ax.set_title("Tokens per Word by Category and Tokenizer")
    ax.set_xlabel("Category")
    ax.set_ylabel("Tokens per Word")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "tokens_per_word_by_category_and_tokenizer.png", dpi=200)
    plt.close()

    plt.figure(figsize=(12, 7))
    for tokenizer_name, group in df.groupby("tokenizer_name"):
        plt.scatter(
            group["character_count"],
            group["token_count"],
            label=tokenizer_name,
            s=90,
        )
    plt.title("Character Count vs Token Count")
    plt.xlabel("Character Count")
    plt.ylabel("Token Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "character_count_vs_token_count.png", dpi=200)
    plt.close()

    average_token_count = df.groupby("tokenizer_name")["token_count"].mean().sort_values(ascending=False)
    ax = average_token_count.plot(kind="bar", figsize=(10, 6), color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Average Token Count per Tokenizer")
    ax.set_xlabel("Tokenizer")
    ax.set_ylabel("Average Token Count")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "tokenizer_comparison_summary.png", dpi=200)
    plt.close()


def run_analysis(cases_path=CASES_PATH, output_path=REPORT_PATH, model_names=None):
    """Run the full Step 2 tokenization analysis pipeline."""
    if model_names is None:
        model_names = REQUESTED_MODEL_NAMES

    cases = load_cases(cases_path)
    tokenizers = load_all_tokenizers(model_names)
    results = analyze_multiple_texts(cases, tokenizers)
    df = save_report(results, output_path)
    create_graphs(df, output_path.parent / "figures")
    summary = build_summary(df)
    return df, summary


def main():
    """Run the tokenization behavior experiment from the terminal."""
    df, summary = run_analysis()

    print("Saved report to:", REPORT_PATH)
    print("Saved figures to:", FIGURES_DIR)
    print()
    print("Top rows sorted by token count:")
    print(df.head(10).to_string(index=False))
    print()
    print("Summary insights:")
    for label, value in summary.items():
        print(f"- {label}: {value}")


if __name__ == "__main__":
    main()
