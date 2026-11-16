import json
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

DATASET_FOLDERS = [
    "Data_Llama",
    "Data_SMOL",
    "Data_qwen",
]

COMBINED_OUTPUT_PATH = SRC_DIR / "original_features_combined.csv"

DETECTION_FIELD_MAP = {
    "num_tokens_scored": "initial_num_tokens_scored",
    "num_green_tokens": "initial_num_green_tokens",
    "green_fraction": "initial_green_fraction",
    "z_score": "initial_z_score",
    "p_value": "initial_p_value",
    "prediction": "initial_detected",
    "detection_threshold": "initial_detection_threshold",
}


def load_baseline_lookup(json_path: Path) -> dict[str, dict]:
    records = json.loads(json_path.read_text(encoding="utf-8"))
    lookup = {}
    for record in records:
        sample_id = str(record["sample_id"]).strip()
        lookup[sample_id] = record
    return lookup


def update_dataset_csv(folder_path: Path) -> pd.DataFrame:
    csv_path = folder_path / "original_features.csv"
    json_path = folder_path / "baseline_evaluation.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON: {json_path}")

    df = pd.read_csv(csv_path)
    lookup = load_baseline_lookup(json_path)

    sample_ids = df["file_name"].astype(str).apply(lambda x: Path(x).stem)

    missing_ids = [sid for sid in sample_ids if sid not in lookup]
    if missing_ids:
        raise KeyError(
            f"Sample IDs in {csv_path} not found in {json_path}: {missing_ids[:10]}"
        )

    model_names = [lookup[sid].get("model_name", "") for sid in sample_ids]

    if "model_name" in df.columns:
        df["model_name"] = model_names
    else:
        insert_pos = 2 if "relative_path" in df.columns else 1
        df.insert(insert_pos, "model_name", model_names)

    for json_key, col_name in DETECTION_FIELD_MAP.items():
        df[col_name] = [lookup[sid].get(json_key) for sid in sample_ids]

    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"Updated {csv_path} -> shape: {df.shape}")
    return df


def main():
    combined_dfs = []

    for folder_name in DATASET_FOLDERS:
        folder_path = SRC_DIR / folder_name
        updated_df = update_dataset_csv(folder_path)
        combined_dfs.append(updated_df)

    combined_df = pd.concat(combined_dfs, ignore_index=True)
    combined_df.to_csv(COMBINED_OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved combined CSV to {COMBINED_OUTPUT_PATH} -> shape: {combined_df.shape}")


if __name__ == "__main__":
    main()
