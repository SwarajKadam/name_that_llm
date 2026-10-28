# Project Guide

## Objective

Classify each passage as one of 18 source language models for the Kaggle Name That LLM competition.

## Repository layout

- `data/`: downloaded competition data; do not modify source files.
- `notebooks/`: exploratory and experimental analyses. `01_train_eda.py` is a VS Code/Jupyter-style percent-cell notebook.
- `scripts/`: operational utilities such as the Kaggle downloader.
- `src/`: reusable training and inference code.
- `artifacts/`: generated analyses and model outputs; EDA charts are written to `artifacts/eda/`.
- `requirements.txt`: Python dependencies for data download, Parquet support, EDA, and interactive execution.
- `.gitignore`: excludes competition data, credentials, environments, caches, generated outputs, and model binaries.

## Data facts

- Training data has 2,699 passages, 150 prompt groups, and 18 model classes.
- Seventeen classes have 150 examples; `claude-fable-5` has 149.
- Nearly every prompt has one passage from every model.
- There are no missing values, duplicate IDs, or exact duplicate texts in the current training file.

## Modeling guardrail

Use grouped validation by `prompt_id`; never use a random row-level split for reported model selection. Test prompts are unseen, and mixing responses from one prompt across train and validation causes content leakage.

## EDA workflow

Run `python notebooks/01_train_eda.py` from the repository root. It requires pandas, NumPy, and Matplotlib, displays figures in interactive environments, and saves reproducible outputs under `artifacts/eda/`.

## Environment and generated files

Install dependencies with `python -m pip install -r requirements.txt`. Keep downloaded files under `data/` and generated results under `artifacts/`; both are ignored by Git and should be reproducible from scripts. Never commit Kaggle credentials, `.env` files, trained model binaries, or local virtual environments.

