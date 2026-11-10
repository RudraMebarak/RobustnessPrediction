
from pathlib import Path
import csv, json, random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[1]

CSV_FILES = [
    ROOT / "100_code_prompts_code_only.csv",
    ROOT / "50_code_prompts_html_jsx_cpp.csv",  # Change to actual filename
]

OUTPUT_DIR = ROOT / "data" / "raw" / "unwatermarked"
BASELINE_DIR = ROOT / "data" / "baseline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# ---------- LOAD PROMPTS ----------
prompts = []

for csv_file in CSV_FILES:
    with open(csv_file, newline="", encoding="utf-8") as f:
        prompts.extend(list(csv.DictReader(f)))

assert len(prompts) == 150, f"Expected 150 prompts, got {len(prompts)}"

# ---------- LOAD MODEL ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

model.eval()

# ---------- LOAD KGW DETECTOR ----------
KGW_PATH = Path.home() / "kgw-test" / "lm-watermarking"
sys.path.insert(0, str(KGW_PATH))

from watermark_processor import WatermarkDetector

detector = WatermarkDetector(
    vocab=list(tokenizer.get_vocab().values()),
    gamma=0.25,
    delta=2.0,
    device=model.device,
    tokenizer=tokenizer,
    z_threshold=4.0,
    normalizers=None,
    ignore_repeated_ngrams=False,
    seeding_scheme="selfhash"
)

# ---------- GENERATION + DETECTION ----------
baseline_results = []

for index, prompt_row in enumerate(prompts, start=1):

    prompt_id = prompt_row["prompt_id"]
    prompt_text = prompt_row["prompt"].strip()
    prompt_language = prompt_row["language_or_stack"].strip()

    sample_seed = BASE_SEED + index - 1
    filename = f"code_extra_{index}.txt"
    text_path = OUTPUT_DIR / filename

    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)

    generation_prompt = (
        prompt_text +
        f"\n\nUse Language: {prompt_language}\n"
        "Write the complete solution strictly in this language. "
        "Do not use any other programming language. "
        "Return code only."
    )

    messages = [
        {"role": "user", "content": generation_prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=600,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = output[0][input_ids.shape[-1]:]
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    text_path.write_text(generated_text, encoding="utf-8")

    # Baseline KGW detection
    detection = detector.detect(
        generated_text,
        return_dict=True
    )

    baseline_results.append({
        "prompt_id": prompt_id,
        "filename": filename,
        "language": prompt_language,
        "seed": sample_seed,
        "z_score": float(detection["z_score"]),
        "p_value": float(detection["p_value"]),
        "detected": bool(detection["prediction"]),
        "green_fraction": float(detection["green_fraction"]),
        "num_green_tokens": int(detection["num_green_tokens"]),
        "num_tokens_scored": int(detection["num_tokens_scored"])
    })

    print(
        f"[{index}/150] {filename} | {prompt_language} | "
        f"Z-score: {detection['z_score']:.3f}"
    )

# ---------- SAVE BASELINE JSON ----------
baseline_file = BASELINE_DIR / "unwatermarked_baseline.json"

with open(baseline_file, "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, indent=4)

print("\nGeneration complete!")
print(f"Samples: {OUTPUT_DIR}")
print(f"Baseline: {baseline_file}")
