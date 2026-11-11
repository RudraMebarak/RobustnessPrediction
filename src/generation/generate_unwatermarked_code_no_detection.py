
from pathlib import Path
import csv, random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[1]

CSV_FILES = [
    ROOT / "100_code_prompts_code_only.csv",
    ROOT / "50_code_prompts_html_jsx_cpp.csv",  # Change to actual filename
]

OUTPUT_DIR = ROOT / "data" / "raw" / "unwatermarked"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CODE_ONLY_INSTRUCTION = (
    "Output only the complete source code. Do not include Markdown code fences, "
    "explanations, introductions, or commentary."
)

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

# ---------- GENERATION ----------
for index, prompt_row in enumerate(prompts, start=1):

    prompt_id = prompt_row["prompt_id"]
    prompt_text = prompt_row["prompt"].strip()
    prompt_language = prompt_row["language_or_stack"].strip()

    sample_seed = BASE_SEED + index - 1
    filename = f"code_{index}_unwatermarked.txt"
    text_path = OUTPUT_DIR / filename
    if text_path.exists():
        print(f"[{index}/150] {filename} already exists, skipping...")
        continue

    random.seed(sample_seed)
    np.random.seed(sample_seed)
    torch.manual_seed(sample_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(sample_seed)

    generation_prompt = (
        prompt_text + f"\n\nUse Language: {prompt_language}\n"
    )

    messages = [
        {"role": "system", "content": CODE_ONLY_INSTRUCTION},
        {"role": "user", "content": generation_prompt}
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        output = model.generate(
            **inputs,
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

    print(f"[{index}/150] {filename} | {prompt_language}")


print("\nGeneration complete!")
print(f"Samples: {OUTPUT_DIR}")
