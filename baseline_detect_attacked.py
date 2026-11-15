
import os

# Force CPU execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import pandas as pd
import torch
import traceback

from pathlib import Path
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ATTACKED_DIR = (
    PROJECT_ROOT / "src" / "data_code" / "raw" / "attacked"
)

CSV_PATH = ATTACKED_DIR / "attack_settings.csv"

# Directory containing watermark_processor.py
WATERMARK_REPO = Path.home() / "kgw-test"/"lm-watermarking"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

GAMMA = 0.25
DELTA = 2.0
BASE_SEED = 42

MAX_NEW_TOKENS = 600

SEEDING_SCHEME = "selfhash"
SELECT_GREEN_TOKENS = True
Z_THRESHOLD = 4.0

TEMPERATURE = 0.7
TOP_P = 0.9

DEVICE = torch.device("cpu")


# ============================================================
# WATERMARK DETECTOR IMPORT
# ============================================================

sys.path.insert(0, str(WATERMARK_REPO))

from extended_watermark_processor import WatermarkDetector


# ============================================================
# DETECTION COLUMNS
# ============================================================

DETECTION_COLUMNS = [
    "baseline_num_tokens_scored",
    "baseline_num_green_tokens",
    "baseline_green_fraction",
    "baseline_z_score",
    "baseline_p_value",
    "baseline_detected",
    "baseline_detection_status",
    "baseline_detection_error",
]


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_output_path(output_path):

    path = Path(str(output_path).strip())

    if path.is_absolute():
        return path

    # Normal case: path relative to attacked directory
    candidate = ATTACKED_DIR / path

    if candidate.exists():
        return candidate

    # Handle paths that already contain attacked/
    parts = path.parts

    if "attacked" in parts:
        index = parts.index("attacked")

        candidate = ATTACKED_DIR.joinpath(
            *parts[index + 1:]
        )

        if candidate.exists():
            return candidate

    return candidate


# ============================================================
# INITIALIZE TOKENIZER AND DETECTOR
# ============================================================

def initialize_detector():

    print("\nLoading tokenizer:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    vocab = list(
        tokenizer.get_vocab().values()
    )

    print("Vocabulary size:", len(vocab))

    print("Initializing KGW detector on CPU...")

    detector = WatermarkDetector(
        vocab=vocab,
        gamma=GAMMA,
        delta=DELTA,
        seeding_scheme=SEEDING_SCHEME,
        select_green_tokens=SELECT_GREEN_TOKENS,
        device=str(DEVICE),
        tokenizer=tokenizer,
        z_threshold=Z_THRESHOLD,
        normalizers=[],
        ignore_repeated_ngrams=False,
    )

    print("Detector initialized successfully.")

    return detector


# ============================================================
# BASELINE DETECTION
# ============================================================

def detect_text(detector, text):

    result = detector.detect(text)

    return {
        "baseline_num_tokens_scored":
            result.get("num_tokens_scored"),

        "baseline_num_green_tokens":
            result.get("num_green_tokens"),

        "baseline_green_fraction":
            result.get("green_fraction"),

        "baseline_z_score":
            result.get("z_score"),

        "baseline_p_value":
            result.get("p_value"),

        "baseline_detected":
            result.get("prediction"),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not ATTACKED_DIR.exists():
        raise FileNotFoundError(
            f"Attacked directory not found: {ATTACKED_DIR}"
        )

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}"
        )

    if not (
        WATERMARK_REPO / "watermark_processor.py"
    ).exists():

        raise FileNotFoundError(
            "watermark_processor.py not found. "
            "Update WATERMARK_REPO to its actual location."
        )

    # --------------------------------------------------------
    # LOAD EXISTING CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV_PATH,
        keep_default_na=False,
        dtype={"output_path": str}
    )

    if "output_path" not in df.columns:
        raise ValueError(
            "CSV must contain an output_path column."
        )

    # --------------------------------------------------------
    # CREATE MISSING COLUMNS
    # --------------------------------------------------------

    for column in DETECTION_COLUMNS:

        if column not in df.columns:
            df[column] = ""

    # IMPORTANT:
    # Pandas 3.x can infer string dtype for columns containing
    # empty strings. Detector outputs include integers, floats
    # and booleans, so use object dtype for all result columns.

    for column in DETECTION_COLUMNS:

        df[column] = df[column].astype("object")

    # --------------------------------------------------------
    # INITIALIZE DETECTOR
    # --------------------------------------------------------

    detector = initialize_detector()

    total = len(df)

    completed = 0
    skipped = 0
    failed = 0

    print("\n" + "=" * 70)
    print("KGW BASELINE DETECTION")
    print("=" * 70)

    print("Total entries:", total)
    print("Device:", DEVICE)
    print("Model:", MODEL_NAME)
    print("Gamma:", GAMMA)
    print("Delta:", DELTA)
    print("Seeding scheme:", SEEDING_SCHEME)
    print("Z threshold:", Z_THRESHOLD)

    # --------------------------------------------------------
    # PROCESS FILES
    # --------------------------------------------------------

    for index, row in df.iterrows():

        output_path = resolve_output_path(
            row["output_path"]
        )

        # Resume support
        if (
            str(df.at[index, "baseline_detection_status"])
            == "completed"
        ):

            skipped += 1
            continue

        print("\n" + "-" * 70)
        print(f"[{index + 1}/{total}]")
        print("File:", output_path)

        try:

            if not output_path.exists():

                raise FileNotFoundError(
                    f"File not found: {output_path}"
                )

            text = output_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            if not text.strip():

                raise ValueError(
                    "Empty text file."
                )

            # Run detection
            stats = detect_text(
                detector,
                text
            )

            # Store results
            for column, value in stats.items():

                df.at[index, column] = value

            # Display results
            print(
                "Tokens scored:",
                stats["baseline_num_tokens_scored"]
            )

            print(
                "Green tokens:",
                stats["baseline_num_green_tokens"]
            )

            print(
                "Green fraction:",
                stats["baseline_green_fraction"]
            )

            print(
                "Z-score:",
                stats["baseline_z_score"]
            )

            print(
                "P-value:",
                stats["baseline_p_value"]
            )

            print(
                "Detected:",
                stats["baseline_detected"]
            )

            df.at[
                index,
                "baseline_detection_status"
            ] = "completed"

            df.at[
                index,
                "baseline_detection_error"
            ] = ""

            completed += 1

        except Exception as e:

            print("ERROR:", str(e))

            df.at[
                index,
                "baseline_detection_status"
            ] = "error"

            df.at[
                index,
                "baseline_detection_error"
            ] = str(e)[:1000]

            failed += 1

            traceback.print_exc()

        finally:

            # Save checkpoint after EVERY document
            df.to_csv(
                CSV_PATH,
                index=False,
                encoding="utf-8-sig"
            )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE DETECTION FINISHED")
    print("=" * 70)

    print("Total entries:", total)
    print("Completed this run:", completed)
    print("Skipped (already completed):", skipped)
    print("Failed:", failed)

    print("\nUpdated CSV:", CSV_PATH)


if __name__ == "__main__":
    main()
