import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Paths
PROJECT_ROOT = Path.home()/ "majorproject"/"RobustnessPrediction"/"src"
KGW_DIR = Path.home() / "kgw-test" / "lm-watermarking"

INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "watermarked"
MANIFEST_PATH = INPUT_DIR / "manifest.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "baseline_detection.jsonl"

sys.path.insert(0, str(KGW_DIR))
from extended_watermark_processor import WatermarkDetector

# Match generation configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
GAMMA = 0.25
DELTA = 2.0
SEEDING_SCHEME = "selfhash"
SELECT_GREEN_TOKENS = True
Z_THRESHOLD = 4.0


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f if line.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=GAMMA,
        delta=DELTA,
        seeding_scheme=SEEDING_SCHEME,
        select_green_tokens=SELECT_GREEN_TOKENS,
        device=device,
        tokenizer=tokenizer,
        z_threshold=Z_THRESHOLD,
        normalizers=[],
        ignore_repeated_ngrams=False,
    )

    results = []

    for record in manifest:
        filename = record["filename"]
        text_path = INPUT_DIR / filename

        if not text_path.exists():
            print(f"WARNING: Missing file: {text_path}")
            continue

        text = text_path.read_text(encoding="utf-8").strip()

        scores = detector.detect(
            text=text,
            return_prediction=True,
            return_scores=True,
            convert_to_float=True,
        )

        result = {
            "document_id": record["document_id"],
            "domain": record["domain"],
            "filename": filename,
            "character_count": len(text),
            "num_tokens_scored": scores["num_tokens_scored"],
            "num_green_tokens": scores["num_green_tokens"],
            "green_fraction": scores["green_fraction"],
            "baseline_z_score": scores["z_score"],
            "baseline_p_value": scores["p_value"],
            "baseline_detected": scores["prediction"],
            "detection_threshold": Z_THRESHOLD,
            "gamma": GAMMA,
            "delta": DELTA,
            "seeding_scheme": SEEDING_SCHEME,
            "normalizers": [],
            "ignore_repeated_ngrams": False,
        }

        results.append(result)

        print(
            f"{filename}: "
            f"z={scores['z_score']:.3f}, "
            f"green_fraction={scores['green_fraction']:.3f}, "
            f"detected={scores['prediction']}"
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\nEvaluated {len(results)} documents.")
    print(f"Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
