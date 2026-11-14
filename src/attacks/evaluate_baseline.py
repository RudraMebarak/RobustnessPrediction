import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Paths
PROJECT_ROOT = Path.home()/ "majorproject"/"RobustnessPrediction"/"src"
KGW_DIR = Path.home() / "kgw-test" / "lm-watermarking"

INPUT_DIR = PROJECT_ROOT / "data1"
OUTPUT_DIR = PROJECT_ROOT / "data1" 
OUTPUT_PATH = OUTPUT_DIR / "baseline_detection_data1.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data1" / "paired_dataset.jsonl"

sys.path.insert(0, str(KGW_DIR))
from extended_watermark_processor import WatermarkDetector

# Match generation configuration
GAMMA = 0.25
DELTA = 2.0
SEEDING_SCHEME = "selfhash"
SELECT_GREEN_TOKENS = True
Z_THRESHOLD = 4.0
MODEL_1 = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_2 = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_3 = "HuggingFaceTB/SmolLM3-3B"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading tokenizers")

    tokenizer1 = AutoTokenizer.from_pretrained(MODEL_1)
    tokenizer2 = AutoTokenizer.from_pretrained(MODEL_2)
    tokenizer3 = AutoTokenizer.from_pretrained(MODEL_3)

    detector1 = WatermarkDetector(
        vocab=list(tokenizer1.get_vocab().values()),
        gamma=GAMMA,
        delta=DELTA,
        seeding_scheme=SEEDING_SCHEME,
        select_green_tokens=SELECT_GREEN_TOKENS,
        device=device,
        tokenizer=tokenizer1,
        z_threshold=Z_THRESHOLD,
        normalizers=[],
        ignore_repeated_ngrams=False,
    )
    detector2 = WatermarkDetector(
            vocab=list(tokenizer2.get_vocab().values()),
            gamma=GAMMA,
            delta=DELTA,
            seeding_scheme=SEEDING_SCHEME,
            select_green_tokens=SELECT_GREEN_TOKENS,
            device=device,
            tokenizer=tokenizer2,
            z_threshold=Z_THRESHOLD,
            normalizers=[],
            ignore_repeated_ngrams=False,
        )
    detector3 = WatermarkDetector(
            vocab=list(tokenizer3.get_vocab().values()),
            gamma=GAMMA,
            delta=DELTA,
            seeding_scheme=SEEDING_SCHEME,
            select_green_tokens=SELECT_GREEN_TOKENS,
            device=device,
            tokenizer=tokenizer3,
            z_threshold=Z_THRESHOLD,
            normalizers=[],
            ignore_repeated_ngrams=False,
        )
    

    results = []

    for record in manifest:
        if (record["watermarked"] == False):
            continue
        fpath = record["text_file"]
        text_path = INPUT_DIR / fpath

        if not text_path.exists():
            print(f"WARNING: Missing file: {text_path}")
            continue

        text = text_path.read_text(encoding="utf-8").strip()
        if (record["model_id"]==MODEL_1):
            detector = detector1
        elif (record["model_id"]==MODEL_2):
            detector = detector2
        elif (record["model_id"]==MODEL_3):
            detector = detector3
        else:
            print(f"WARNING: Unknown model ID: {record['model_id']}")
            continue

        scores = detector.detect(
            text=text,
            return_prediction=True,
            return_scores=True,
            convert_to_float=True,
        )

        result = {
            "sample_id": record["sample_id"],
            "domain": record["category"],
            "file_path": fpath,
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
            f"{record['sample_id']}: "
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
