from pathlib import Path
import kagglehub


kagglehub.login()
# Project folder / data, regardless of where you run the script
data_dir = Path(__file__).resolve().parent.parent / "data"

path = kagglehub.competition_download(
    "name-that-llm",
    output_dir=str(data_dir),
)

print("Path to competition files:", path)