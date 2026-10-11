import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
KGW_DIR = Path.home() / "kgw-test"/"lm-watermarking"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(KGW_DIR))

from prompts import PROMPTS
from extended_watermark_processor import WatermarkLogitsProcessor


# Experiment configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

GAMMA = 0.25
DELTA = 2.0
SEED = 42
MAX_NEW_TOKENS = 250

OUTPUT_DIR = PROJECT_ROOT /"src"/ "data" / "raw" / "watermarked"
MANIFEST = OUTPUT_DIR / "manifest.jsonl"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Require exactly the five expected domains.
    expected_domains = {
        "creative",
        "news",
        "code",
        "scientific",
        "conversational",
    }

    if set(PROMPTS) != expected_domains:
        raise ValueError(
            f"Expected prompt keys {expected_domains}; "
            f"received {set(PROMPTS)}"
        )

    # Avoid silently overwriting previous experimental data.
    output_files = [
        OUTPUT_DIR / f"{domain}_001.txt"
        for domain in PROMPTS
    ]

    if MANIFEST.exists() or any(p.exists() for p in output_files):
        raise FileExistsError(
            f"Output data already exists in {OUTPUT_DIR}. "
            "Move or archive it before starting a new run."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    print(f"Device: {device}")
    print(f"Loading {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=GAMMA,
        delta=DELTA,
        seeding_scheme="selfhash",
    )

    # One output document per domain.
    with MANIFEST.open("w", encoding="utf-8") as manifest:
        for index, (domain, prompt) in enumerate(
            PROMPTS.items(), start=1
        ):
            document_id = f"{domain.upper()}_001"

            seed = SEED + index
            random.seed(seed)
            torch.manual_seed(seed)

            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            messages = [{"role": "user", "content": prompt}]

            inputs = tokenizer.apply_chat_template(
		    messages,
		    add_generation_prompt=True,
		    tokenize=True,
		    return_dict=True,
		    return_tensors="pt",
		)

            inputs = {key: value.to(device) for key, value in inputs.items()}
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

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

            # Exclude the prompt tokens from the saved document.
            generated_ids = output_ids[0, input_ids.shape[1]:]
            text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            if not text:
                raise RuntimeError(
                    f"Generation returned empty text for {domain}"
                )

            filename = f"{domain}_001.txt"
            (OUTPUT_DIR / filename).write_text(
                text + "\n", encoding="utf-8"
            )

            record = {
                "document_id": document_id,
                "domain": domain,
                "prompt": prompt,
                "model": MODEL_NAME,
                "watermark": "KGW",
                "gamma": GAMMA,
                "delta": DELTA,
                "seeding_scheme": "selfhash",
                "generation_seed": seed,
                "filename": filename,
                "character_count": len(text),
                "word_count": len(text.split()),
                "baseline_z_score": None,
                "generated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            manifest.write(json.dumps(record) + "\n")
            manifest.flush()

            print(
                f"[{index}/5] {document_id}: "
                f"{len(text.split())} words -> {filename}"
            )

    print(f"\nGeneration complete: {OUTPUT_DIR}")
    print("Baseline detector scores have not been measured yet.")


if __name__ == "__main__":
    main()
