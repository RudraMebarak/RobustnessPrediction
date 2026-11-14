"""Paraphrase text using Hugging Face's tuner007/pegasus_paraphrase with intensity presets.

Requirements:
    pip install -U transformers torch sentencepiece accelerate
"""

import argparse
from typing import List, Literal, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

MODEL_ID = "tuner007/pegasus_paraphrase"

# tuner007/pegasus_paraphrase does not have a built-in intensity control token,
# so intensity is approximated through core transformers decoding parameters.
INTENSITY_PRESETS = {
    "low": {
        "description": "Conservative, close to original wording (deterministic beam search)",
        "do_sample": False,
        "num_beams": 5,
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    },
    "medium": {
        "description": "Balanced rewording (multinomial beam sampling, no remote code needed)",
        "do_sample": True,
        "num_beams": 5,
        "temperature": 0.95,
        "top_p": 0.92,
        "repetition_penalty": 1.1,
    },
    "high": {
        "description": "Creative/stronger rewording (sampling with higher temperature)",
        "do_sample": True,
        "num_beams": 1,
        "temperature": 1.5,
        "top_p": 0.92,
        "repetition_penalty": 1.2,
    },
}


def load_model(model_id: str = MODEL_ID):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model, device


def paraphrase(
    text: str,
    tokenizer,
    model,
    device: str,
    intensity: Literal["low", "medium", "high"] = "medium",
    num_return_sequences: int = 3,
    max_length: int = 60,
    seed: Optional[int] = None,
) -> List[str]:
    if intensity not in INTENSITY_PRESETS:
        raise ValueError(f"intensity must be one of {list(INTENSITY_PRESETS.keys())}")

    if seed is not None:
        set_seed(seed)

    preset = INTENSITY_PRESETS[intensity]
    num_beams = max(preset["num_beams"], num_return_sequences) if preset["num_beams"] > 1 else 1

    inputs = tokenizer(
        [text],
        truncation=True,
        padding="longest",
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    gen_kwargs = {
        "max_length": max_length,
        "num_return_sequences": num_return_sequences,
        "do_sample": preset["do_sample"],
        "num_beams": num_beams,
        "temperature": preset["temperature"],
        "top_p": preset["top_p"],
        "repetition_penalty": preset["repetition_penalty"],
    }

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Paraphrase text with tuner007/pegasus_paraphrase.")
    parser.add_argument(
        "text",
        nargs="?",
        default="The ultimate test of your knowledge is your capacity to convey it to another.",
        help="Input sentence to paraphrase.",
    )
    parser.add_argument(
        "--intensity",
        choices=["low", "medium", "high"],
        default="medium",
        help="Paraphrasing intensity preset (default: medium).",
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=3,
        help="Number of paraphrases to return (default: 3).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=60,
        help="Maximum token length for input/output (default: 60).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation (especially with medium/high intensity).",
    )
    args = parser.parse_args()

    tokenizer, model, device = load_model()
    results = paraphrase(
        text=args.text,
        tokenizer=tokenizer,
        model=model,
        device=device,
        intensity="high",
        num_return_sequences=1,
        max_length=args.max_length,
        seed=2,
    )

    print(f"Original: {args.text}\n")
    for idx, candidate in enumerate(results, start=1):
        print(f"{idx}. {candidate}")


if __name__ == "__main__":
    main()