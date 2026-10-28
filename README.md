# Name That LLM

Workspace for the Kaggle authorship-attribution competition that identifies which of 18 language models generated a passage.

## Setup

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the competition data after configuring Kaggle credentials:

```powershell
python scripts/kaggle_download.py
```

## Data

Competition files live in `data/`:

- `train.csv`: labelled passages with `id`, `prompt_id`, `text`, and `model`.
- `test.csv`: unlabelled passages.
- `sample_submission.parquet`: required submission schema.

## Exploratory data analysis

Run the training-data EDA from the repository root:

```powershell
python notebooks/01_train_eda.py
```

The file uses `# %%` cells and can also be run interactively in VS Code. It displays charts inline and saves PNG charts plus `model_style_summary.csv` to `artifacts/eda/`.

The analysis covers class balance, prompt-model coverage, passage length, prompt-controlled relative length, formatting conventions, punctuation/style fingerprints, common openings, distinctive tokens, and feature correlations. Treat all apparent fingerprints as hypotheses until they improve validation grouped by `prompt_id`.

Downloaded data and generated artifacts are intentionally excluded from Git. Recreate EDA outputs by running the analysis script.

## Validation rule

All modeling splits must keep every response for a `prompt_id` in the same fold. Random row-level splits leak prompt content and overstate generalization to the unseen prompts in the test set.

