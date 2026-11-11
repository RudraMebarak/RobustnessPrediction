import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
KGW_DIR = Path.home() / "kgw-test" / "lm-watermarking"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(KGW_DIR))


# Model and KGW configuration: kept consistent with the single-sample script.
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

# Preserve the same text-output directory as the single-sample script.
OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "raw" / "watermarked"

# Use a NEW JSONL file so this run does not append to the old manifest or
# baseline_detection.jsonl.
RESULTS_JSONL = PROJECT_ROOT / "src" / "data" / "processed" / "code_samples_extra_50.jsonl"

CODE_ONLY_INSTRUCTION = (
    "Output only the complete source code. Do not include Markdown code fences, "
    "explanations, introductions, or commentary."
)


def find_prompt_csv() -> Path:
    candidate= PROJECT_ROOT / "src" / "50_code_prompts_html_jsx_cpp.csv"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "Could not find 50_code_prompts_html_jsx_cpp.csv."
    )


def load_prompts(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"prompt_id", "category", "language_or_stack", "prompt"}
    if not rows:
        raise ValueError(f"No prompts found in {csv_path}")

    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Prompt CSV is missing required columns: {sorted(missing)}"
        )

    if len(rows) != 50:
        raise ValueError(
            f"Expected exactly 50 prompts, but found {len(rows)} in {csv_path}"
        )

    for row in rows:
        if not row["prompt"].strip():
            raise ValueError(f"Empty prompt for {row['prompt_id']}")

    return rows


def main():
    prompt_csv = find_prompt_csv()
    prompts = load_prompts(prompt_csv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Prompt CSV: {prompt_csv}")
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


    with RESULTS_JSONL.open("w", encoding="utf-8") as results_file:
        for index, prompt_row in enumerate(prompts, start=1):
            prompt_id = prompt_row["prompt_id"]
            prompt_text = prompt_row["prompt"].strip()
            prompt_language = prompt_row["language_or_stack"].strip()
            sample_seed = BASE_SEED + index - 1
            filename = f"code_extra_{index}.txt"
            text_path = OUTPUT_DIR / filename
            prompt_text= prompt_text + f"Use Language: {prompt_language}]"
            print(f"\n[{index}/50] {prompt_id}: {prompt_row['category']} / "
                  f"{prompt_row['language_or_stack']}")

            random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(sample_seed)

            messages = [
                {"role": "system", "content": CODE_ONLY_INSTRUCTION},
                {"role": "user", "content": prompt_text},
            ]
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
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Save only newly generated tokens, not the prompt/template.
            generated_ids = output_ids[0, input_ids.shape[1]:]
            text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            if not text:
                raise RuntimeError(
                    f"Generation returned empty text for {prompt_id}"
                )

            text_path.write_text(text + "\n", encoding="utf-8")

            scores = detector.detect(
                text=text,
                return_prediction=True,
                return_scores=True,
                convert_to_float=True,
            )

            record = {
                "document_id": filename,
                "sample_index": index,
                "prompt_id": prompt_id,
                "domain": "code",
                "category": prompt_row["category"],
                "language_or_stack": prompt_row["language_or_stack"],
                "prompt": prompt_text,
                "system_instruction": CODE_ONLY_INSTRUCTION,
                "model": MODEL_NAME,
                "watermark": "KGW",
                "gamma": GAMMA,
                "delta": DELTA,
                "seeding_scheme": SEEDING_SCHEME,
                "generation_seed": sample_seed,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": True,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "filename": filename,
                "text_path": str(text_path),
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
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            results_file.flush()

            print(f"  Saved: {text_path.name}")
            print(f"  Words: {len(text.split())}")
            print(f"  Baseline KGW z-score: {scores['z_score']:.3f}")
            print(f"  Green-token fraction: {scores['green_fraction']:.3f}")
            print(
                f"  Baseline detected: {scores['prediction']} "
                f"(threshold={Z_THRESHOLD})"
            )

    print("\nFinished generating 50 code samples.")
    print(f"Text files: {OUTPUT_DIR}")
    print(f"New results JSONL: {RESULTS_JSONL}")


if __name__ == "__main__":
    main()
