import argparse
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

DATASET_FOLDERS = [
    "Data_Llama",
    "Data_SMOL",
    "Data_qwen",
]

STANDARD_ATTACKS = {
    "copy_paste",
    "helm_lowercase",
    "helm_contraction",
    "helm_expansion",
    "helm_misspelling",
    "helm_typo",
    "synonym_wordnet",
    "humarin",
}

DEFAULT_OUTPUT = SRC_DIR / "attack_settings_combined.csv"


def load_folder_attacks(folder_path: Path, keep_all: bool = False) -> pd.DataFrame:
    attack_csv = folder_path / "attacked" / "attack_settings.csv"
    features_csv = folder_path / "original_features.csv"

    if not attack_csv.exists():
        raise FileNotFoundError(f"Missing attack_settings.csv: {attack_csv}")

    df = pd.read_csv(attack_csv, low_memory=False)
    raw_len = len(df)

    if not keep_all:
        df = df[df["status"] == "ok"].copy()
        df = df[df["attack_name"].isin(STANDARD_ATTACKS)].copy()

        if features_csv.exists():
            features_df = pd.read_csv(features_csv)
            valid_doc_ids = set(
                features_df["file_name"].astype(str).apply(lambda x: Path(x).stem)
            )
            df = df[df["doc_id"].astype(str).isin(valid_doc_ids)].copy()

        df = df.drop_duplicates(
            subset=["doc_id", "attack_name", "intensity"], keep="last"
        )

    print(
        f"{folder_path.name}: loaded {raw_len} rows -> kept {len(df)} rows"
    )
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Combine attack_settings.csv from Data_Llama, Data_SMOL, and Data_qwen."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the combined attack_settings CSV.",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Include error/experimental rows without filtering to the 3,600 valid rows per folder.",
    )
    args = parser.parse_args()

    dfs = []
    for folder_name in DATASET_FOLDERS:
        folder_path = SRC_DIR / folder_name
        dfs.append(load_folder_attacks(folder_path, keep_all=args.keep_all))

    combined_df = pd.concat(dfs, ignore_index=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nSaved combined attack settings to {output_path} -> shape: {combined_df.shape}")


if __name__ == "__main__":
    main()
