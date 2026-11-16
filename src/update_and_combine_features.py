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
ATTACK_COMBINED_PATH = SRC_DIR / "attack_settings_combined.csv"
MERGED_DATASET_PATH = SRC_DIR / "merged_robustness_dataset.csv"

VALID_DOMAINS = {"academic", "conversational", "instructional", "creative"}

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


def extract_domain(doc_id: str) -> str:
    prefix = str(doc_id).split("_")[0].strip().lower()
    if prefix not in VALID_DOMAINS:
        raise ValueError(f"Unrecognized domain prefix {prefix!r} from doc_id {doc_id!r}")
    return prefix


def update_dataset_csv(folder_path: Path) -> pd.DataFrame:
    csv_path = folder_path / "original_features.csv"
    json_path = folder_path / "baseline_evaluation.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing JSON: {json_path}")

    df = pd.read_csv(csv_path)
    lookup = load_baseline_lookup(json_path)

    doc_ids = df["file_name"].astype(str).apply(lambda x: Path(x).stem)
    domains = [extract_domain(did) for did in doc_ids]

    missing_ids = [sid for sid in doc_ids if sid not in lookup]
    if missing_ids:
        raise KeyError(
            f"Sample IDs in {csv_path} not found in {json_path}: {missing_ids[:10]}"
        )

    model_names = [lookup[sid].get("model_name", "") for sid in doc_ids]

    # Remove existing metadata columns if re-running so column order stays clean
    for col in ["doc_id", "domain", "model_name"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df.insert(0, "doc_id", doc_ids)
    insert_pos = 3 if "relative_path" in df.columns else 2
    df.insert(insert_pos, "domain", domains)
    df.insert(insert_pos + 1, "model_name", model_names)

    for json_key, col_name in DETECTION_FIELD_MAP.items():
        df[col_name] = [lookup[sid].get(json_key) for sid in doc_ids]

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
    print(f"\nSaved combined original features CSV to {COMBINED_OUTPUT_PATH} -> shape: {combined_df.shape}")

    if ATTACK_COMBINED_PATH.exists():
        attacks_df = pd.read_csv(ATTACK_COMBINED_PATH, low_memory=False)
        merged_df = attacks_df.merge(
            combined_df,
            on="doc_id",
            how="inner",
            validate="many_to_one",
        )
        merged_df["z_score_drop"] = (
            merged_df["initial_z_score"] - merged_df["baseline_z_score"]
        )
        merged_df["relative_z_drop_pct"] = (
            100.0 * merged_df["z_score_drop"] / merged_df["initial_z_score"]
        )
        merged_df["z_score_retention"] = (
            merged_df["baseline_z_score"] / merged_df["initial_z_score"]
        )
        merged_df["watermark_survived"] = merged_df["baseline_detected"].astype(bool)
        merged_df["watermark_removed"] = (
            merged_df["initial_detected"].astype(bool)
            & (~merged_df["baseline_detected"].astype(bool))
        )
        merged_df.to_csv(MERGED_DATASET_PATH, index=False, encoding="utf-8")
        print(f"Saved merged dataset (attacks + features) to {MERGED_DATASET_PATH} -> shape: {merged_df.shape}")


if __name__ == "__main__":
    main()
