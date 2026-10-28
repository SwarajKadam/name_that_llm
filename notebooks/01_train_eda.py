"""Exploratory analysis for the Name That LLM training set.

Run as a script from the repository root:
    python notebooks/01_train_eda.py

Or open this file in VS Code and run its ``# %%`` cells interactively. Charts are
shown inline in an interactive editor and are also saved under ``artifacts/eda``.
Only pandas, NumPy, and Matplotlib are required.
"""

# %% Imports and display configuration
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train.csv"
OUTPUT_DIR = ROOT / "artifacts" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 140)
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def finish_figure(name: str) -> None:
    """Apply consistent layout, save a figure, and display it."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close()


# %% Load and validate
df = pd.read_csv(DATA_PATH)
required_columns = {"id", "prompt_id", "text", "model"}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Models: {df['model'].nunique():,}")
print(f"Prompts: {df['prompt_id'].nunique():,}")
print("\nMissing values:\n", df.isna().sum())
print("\nExact duplicate texts:", int(df["text"].duplicated().sum()))
print("Duplicate IDs:", int(df["id"].duplicated().sum()))
display(df.head()) if "display" in globals() else print(df.head())


# %% Dataset overview
model_counts = df["model"].value_counts().sort_values()
prompt_sizes = df.groupby("prompt_id").size()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].barh(model_counts.index, model_counts.values, color=COLORS[: len(model_counts)])
axes[0].set_title("Training examples per model")
axes[0].set_xlabel("Passages")
for idx, value in enumerate(model_counts.values):
    axes[0].text(value + 0.15, idx, str(value), va="center", fontsize=8)

size_counts = prompt_sizes.value_counts().sort_index()
axes[1].bar(size_counts.index.astype(str), size_counts.values, color="#4C78A8")
axes[1].set_title("Responses available per prompt")
axes[1].set_xlabel("Number of model responses")
axes[1].set_ylabel("Number of prompts")
for idx, value in enumerate(size_counts.values):
    axes[1].text(idx, value + 1, str(value), ha="center")
finish_figure("01_dataset_overview")


# %% Prompt-model coverage and missing combinations
coverage = pd.crosstab(df["model"], df["prompt_id"])
missing_pairs = (
    coverage.eq(0)
    .stack()
    .loc[lambda values: values]
    .index.to_frame(index=False, name=["model", "prompt_id"])
)
print("Missing model-prompt combinations:")
print(missing_pairs.to_string(index=False) if len(missing_pairs) else "None")

fig, ax = plt.subplots(figsize=(16, 6))
ax.imshow(coverage.values, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
ax.set_title("Prompt-model coverage (white indicates a missing response)")
ax.set_xlabel("Prompt index (sorted anonymised ID)")
ax.set_ylabel("Model")
ax.set_yticks(range(len(coverage.index)))
ax.set_yticklabels(coverage.index, fontsize=8)
finish_figure("02_prompt_model_coverage")


# %% Engineer interpretable style and formatting features
def count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.MULTILINE))


features = pd.DataFrame(index=df.index)
features["characters"] = df["text"].str.len()
features["words"] = df["text"].str.findall(r"\b\w+\b").str.len()
features["sentences"] = df["text"].str.count(r"[.!?](?:\s|$)").clip(lower=1)
features["paragraphs"] = df["text"].str.count(r"\n\s*\n") + 1
features["avg_word_length"] = features["characters"] / features["words"].clip(lower=1)
features["avg_sentence_words"] = features["words"] / features["sentences"]
features["unique_word_ratio"] = df["text"].map(
    lambda value: len(set(re.findall(r"\b\w+\b", value.lower())))
    / max(1, len(re.findall(r"\b\w+\b", value.lower())))
)
features["newlines"] = df["text"].str.count("\n")
features["commas"] = df["text"].str.count(",")
features["semicolons"] = df["text"].str.count(";")
features["colons"] = df["text"].str.count(":")
features["question_marks"] = df["text"].str.count(r"\?")
features["exclamation_marks"] = df["text"].str.count("!")
features["parentheses"] = df["text"].str.count(r"[()]" )
features["em_dashes"] = df["text"].str.count("—")
features["markdown_bold"] = df["text"].str.count(r"\*\*") // 2
features["headings"] = df["text"].map(lambda value: count_pattern(value, r"^#{1,6}\s"))
features["list_items"] = df["text"].map(lambda value: count_pattern(value, r"^\s*(?:[-*+] |\d+[.)] )"))
features["code_fences"] = df["text"].str.count("```") // 2
features["latex_markers"] = df["text"].str.count(r"\$|\\\(|\\\[|\\begin\{")
features["uppercase_ratio"] = df["text"].map(
    lambda value: sum(char.isupper() for char in value) / max(1, sum(char.isalpha() for char in value))
)

eda = pd.concat([df, features], axis=1)
print(features.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.round(2))


# %% Passage length fingerprints by model
ordered_models = eda.groupby("model")["words"].median().sort_values().index
length_data = [eda.loc[eda["model"].eq(model), "words"] for model in ordered_models]

fig, ax = plt.subplots(figsize=(15, 7))
box = ax.boxplot(length_data, tick_labels=ordered_models, patch_artist=True, showfliers=False)
for patch, color in zip(box["boxes"], COLORS):
    patch.set_facecolor(color)
ax.axhline(40, color="grey", linestyle="--", linewidth=1, label="Competition lower bound")
ax.axhline(250, color="grey", linestyle=":", linewidth=1, label="Competition upper bound")
ax.set_title("Passage word-count distribution by model")
ax.set_ylabel("Words")
ax.tick_params(axis="x", rotation=65, labelsize=8)
ax.legend()
finish_figure("03_word_count_by_model")


# %% Length distribution independent of model
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(eda["words"], bins=35, color="#4C78A8", edgecolor="white")
axes[0].set_title("Overall word-count distribution")
axes[0].set_xlabel("Words")
axes[0].set_ylabel("Passages")

axes[1].scatter(eda["words"], eda["characters"], alpha=0.25, s=12, color="#F58518")
axes[1].set_title("Characters versus words")
axes[1].set_xlabel("Words")
axes[1].set_ylabel("Characters")
finish_figure("04_length_distribution")


# %% Normalized style fingerprint heatmap
fingerprint_columns = [
    "avg_word_length", "avg_sentence_words", "unique_word_ratio", "paragraphs",
    "newlines", "commas", "semicolons", "colons", "question_marks",
    "exclamation_marks", "parentheses", "em_dashes", "list_items",
    "markdown_bold", "latex_markers", "uppercase_ratio",
]
model_fingerprints = eda.groupby("model")[fingerprint_columns].mean()
z_scores = (model_fingerprints - model_fingerprints.mean()) / model_fingerprints.std(ddof=0).replace(0, 1)

fig, ax = plt.subplots(figsize=(16, 9))
image = ax.imshow(z_scores.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
ax.set_xticks(range(len(z_scores.columns)))
ax.set_xticklabels(z_scores.columns, rotation=55, ha="right", fontsize=8)
ax.set_yticks(range(len(z_scores.index)))
ax.set_yticklabels(z_scores.index, fontsize=8)
ax.set_title("Average style fingerprint by model (column-wise z-scores)")
fig.colorbar(image, ax=ax, label="Standard deviations from all-model mean")
finish_figure("05_style_fingerprint_heatmap")


# %% Formatting behavior: percentage of responses using each convention
binary_format_features = ["paragraphs", "newlines", "markdown_bold", "headings", "list_items", "code_fences", "latex_markers"]
presence = eda[binary_format_features].gt(0).join(eda["model"]).groupby("model").mean().mul(100)

fig, ax = plt.subplots(figsize=(13, 8))
image = ax.imshow(presence.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(1, presence.to_numpy().max()))
ax.set_xticks(range(len(presence.columns)))
ax.set_xticklabels(presence.columns, rotation=40, ha="right")
ax.set_yticks(range(len(presence.index)))
ax.set_yticklabels(presence.index, fontsize=8)
ax.set_title("Responses using each formatting convention (%)")
fig.colorbar(image, ax=ax, label="Percent of passages")
finish_figure("06_formatting_usage")


# %% Common response openings by model
def opening_phrase(text: str, words: int = 3) -> str:
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
    return " ".join(tokens[:words]) if tokens else "<no alphabetic opening>"


eda["opening_3gram"] = eda["text"].map(opening_phrase)
opening_rows = []
for model, group in eda.groupby("model"):
    for phrase, count in group["opening_3gram"].value_counts().head(3).items():
        opening_rows.append({"model": model, "opening": phrase, "count": count})
top_openings = pd.DataFrame(opening_rows).sort_values(["model", "count"], ascending=[True, False])
print("\nThree most common three-word openings per model:\n")
print(top_openings.to_string(index=False))


# %% Distinctive tokens using smoothed one-vs-rest log odds
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
documents = [set(TOKEN_PATTERN.findall(text.lower())) for text in df["text"]]
global_document_frequency = Counter(token for document in documents for token in document)
vocabulary = {token for token, count in global_document_frequency.items() if count >= 5}


def distinctive_tokens(model: str, top_n: int = 12) -> list[tuple[str, float]]:
    in_model = [documents[idx] for idx in df.index[df["model"].eq(model)]]
    out_model = [documents[idx] for idx in df.index[~df["model"].eq(model)]]
    in_counts = Counter(token for document in in_model for token in document)
    out_counts = Counter(token for document in out_model for token in document)
    n_in, n_out = len(in_model), len(out_model)
    scores = {
        token: np.log((in_counts[token] + 0.5) / (n_in - in_counts[token] + 0.5))
        - np.log((out_counts[token] + 0.5) / (n_out - out_counts[token] + 0.5))
        for token in vocabulary
    }
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]


distinctive = {
    model: ", ".join(token for token, _ in distinctive_tokens(model))
    for model in sorted(df["model"].unique())
}
print("\nCandidate distinctive tokens (validate with grouped CV before modeling):\n")
for model, tokens in distinctive.items():
    print(f"{model:24s} {tokens}")


# %% Within-prompt rank analysis: separates model style from prompt-driven length
eda["word_count_prompt_percentile"] = eda.groupby("prompt_id")["words"].rank(pct=True)
rank_summary = eda.groupby("model")["word_count_prompt_percentile"].agg(["mean", "median", "std"]).sort_values("mean")
print("\nWithin-prompt word-count percentile by model:\n", rank_summary.round(3))

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(rank_summary.index, rank_summary["mean"], xerr=rank_summary["std"], color=COLORS[: len(rank_summary)], alpha=0.85)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_xlim(0, 1)
ax.set_xlabel("Mean word-count percentile among responses to the same prompt")
ax.set_title("Prompt-controlled relative response length")
finish_figure("07_within_prompt_length_rank")


# %% Feature correlations
correlation_columns = [
    "characters", "words", "sentences", "paragraphs", "avg_word_length",
    "avg_sentence_words", "unique_word_ratio", "newlines", "commas",
    "semicolons", "colons", "parentheses", "em_dashes", "list_items",
]
correlations = eda[correlation_columns].corr()

fig, ax = plt.subplots(figsize=(11, 9))
image = ax.imshow(correlations.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(correlations.columns)))
ax.set_xticklabels(correlations.columns, rotation=55, ha="right", fontsize=8)
ax.set_yticks(range(len(correlations.index)))
ax.set_yticklabels(correlations.index, fontsize=8)
ax.set_title("Engineered-feature correlation matrix")
fig.colorbar(image, ax=ax, label="Pearson correlation")
finish_figure("08_feature_correlations")


# %% Compact findings table for downstream modeling
summary = eda.groupby("model").agg(
    passages=("id", "size"),
    median_words=("words", "median"),
    mean_sentences=("sentences", "mean"),
    mean_paragraphs=("paragraphs", "mean"),
    mean_unique_word_ratio=("unique_word_ratio", "mean"),
    list_usage_pct=("list_items", lambda values: 100 * values.gt(0).mean()),
    latex_usage_pct=("latex_markers", lambda values: 100 * values.gt(0).mean()),
    em_dash_usage_pct=("em_dashes", lambda values: 100 * values.gt(0).mean()),
).round(2)
summary.to_csv(OUTPUT_DIR / "model_style_summary.csv")
print("\nModel-level summary:\n", summary.sort_values("median_words"))
print(f"\nSaved charts and summary to: {OUTPUT_DIR}")

