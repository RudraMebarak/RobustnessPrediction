import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from click import prompt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
KGW_DIR = Path.home() / "kgw-test" / "lm-watermarking"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(KGW_DIR))

from extended_watermark_processor import WatermarkDetector, WatermarkLogitsProcessor

# Generation configuration
PROMPT = "Write the C++ code to implement radix sort algorithm"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
FNAME = "code_002.txt"
DOMAIN = "code"
GAMMA = 0.25
DELTA = 2.0
SEED = 42
MAX_NEW_TOKENS = 600
SEEDING_SCHEME = "selfhash"
SELECT_GREEN_TOKENS = True
Z_THRESHOLD = 4.0

OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "raw" / "watermarked"
MANIFEST = OUTPUT_DIR / "manifest.jsonl"
BASELINE_OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "processed"
BASELINE_OUTPUT = BASELINE_OUTPUT_DIR / "baseline_detection.jsonl"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Device: {device}")
    print(f"Loading model and tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    vocab = list(tokenizer.get_vocab().values())

    watermark_processor = WatermarkLogitsProcessor(
        vocab=vocab,
        gamma=GAMMA,
        delta=DELTA,
        seeding_scheme=SEEDING_SCHEME,
    )

    # Baseline detector uses the same KGW configuration as generation.
    detector = WatermarkDetector(
        vocab=vocab,
        gamma=GAMMA,
        delta=DELTA,
        seeding_scheme=SEEDING_SCHEME,
        select_green_tokens=SELECT_GREEN_TOKENS,
        device=str(device),
        tokenizer=tokenizer,
        z_threshold=Z_THRESHOLD,
        normalizers=[],
        ignore_repeated_ngrams=False,
    )

    random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    messages = [{"role": "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            logits_processor=[watermark_processor],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Save only generated text, not the prompt.
    generated_ids = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    if not text:
        raise RuntimeError("Generation returned empty text.")

    # FNAME already includes its extension; don't append '.txt' again.
    filename = FNAME
    text_path = OUTPUT_DIR / filename
    text_path.write_text(text + "\n", encoding="utf-8")

    # Run baseline KGW detection immediately on the generated sample.
    scores = detector.detect(
        text=text,
        return_prediction=True,
        return_scores=True,
        convert_to_float=True,
    )

    generated_at = datetime.now(timezone.utc).isoformat()
    record = {
        "document_id": FNAME,
        "domain": DOMAIN,
        "prompt": PROMPT,
        "model": MODEL_NAME,
        "watermark": "KGW",
        "gamma": GAMMA,
        "delta": DELTA,
        "seeding_scheme": SEEDING_SCHEME,
        "generation_seed": SEED,
        "filename": filename,
        "character_count": len(text),
        "word_count": len(text.split()),
        "num_tokens_scored": scores["num_tokens_scored"],
        "num_green_tokens": scores["num_green_tokens"],
        "green_fraction": scores["green_fraction"],
        "baseline_z_score": scores["z_score"],
        "baseline_p_value": scores["p_value"],
        "baseline_detected": scores["prediction"],
        "detection_threshold": Z_THRESHOLD,
        "normalizers": [],
        "ignore_repeated_ngrams": False,
        "generated_at": generated_at,
    }

    # Keep the generation manifest as the source record, now with baseline scores.
    with MANIFEST.open("a", encoding="utf-8") as manifest:
        manifest.write(json.dumps(record) + "\n")

    # Keep the baseline-evaluation output compatible with evaluate_baseline.py.
    baseline_record = {
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
    with BASELINE_OUTPUT.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(baseline_record) + "\n")

    print(f"\nGenerated: {text_path}")
    print(f"Words: {len(text.split())}")
    print(f"Baseline KGW z-score: {scores['z_score']:.3f}")
    print(f"Green-token fraction: {scores['green_fraction']:.3f}")
    print(f"Baseline detected: {scores['prediction']} (threshold={Z_THRESHOLD})")
    print(f"Manifest updated: {MANIFEST}")
    print(f"Baseline results appended to: {BASELINE_OUTPUT}")


if __name__ == "__main__":
    main()
