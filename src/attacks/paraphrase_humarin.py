"""
Paraphrase text with humarin/chatgpt_paraphraser_on_T5_base.

Designed as a drop-in-style companion to paraphrase_pegasus.py:
    load_model()
    paraphrase(text, tokenizer, model, device, intensity, num_return_sequences,
               max_length, seed)

This checkpoint is based on T5-base. Its tokenizer/config advertise a 512-token
sequence capacity, while its paraphrasing fine-tuning used max_length=128.
Long documents are therefore split into token-bounded chunks so that 500-1000
token inputs are not silently truncated. Chunks are paraphrased independently
and reassembled in order; this can reduce cross-chunk coherence.

Install:
    pip install -U torch transformers sentencepiece accelerate

Example:
    python paraphrase_humarin.py "Your text here" --intensity medium
"""

from __future__ import annotations

import argparse
import re
from typing import List, Literal, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed


MODEL_ID = "humarin/chatgpt_paraphraser_on_T5_base"

# These are experiment presets built around the model's documented generation
# interface. They are not official intensity levels supplied by the model author.
INTENSITY_PRESETS = {
    "low": {
        "description": "More conservative beam-search paraphrasing",
        "do_sample": False,
        "num_beams": 5,
        "num_beam_groups": 1,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
    },
    "medium": {
        "description": "More diverse beam-search paraphrasing",
        "do_sample": False,
        "num_beams": 5,
        "num_beam_groups": 1,
        "diversity_penalty": 1.0,
        "repetition_penalty": 2.0,
        "no_repeat_ngram_size": 2,
    },
    "high": {
        "description": "Sampling-based paraphrasing with greater variation",
        "do_sample": True,
        "num_beams": 1,
        "num_beam_groups": 1,
        "temperature": 1.0,
        "top_p": 0.92,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3,
    },
}


def load_model(model_id: str = MODEL_ID):
    """Load tokenizer and model once; return tokenizer, model, device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
    model.eval()

    print(f"Loaded: {model_id}")
    print(f"Device: {device}")
    print(f"Tokenizer max length: {tokenizer.model_max_length}")
    print(f"Model config n_positions: {getattr(model.config, 'n_positions', 'not exposed')}")
    return tokenizer, model, device


def _effective_chunk_size(tokenizer, model, requested_chunk_size: int) -> int:
    """Keep each encoded prompt below the model/tokenizer's advertised limit."""
    limits = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 1_000_000:
        limits.append(tokenizer_limit)

    model_limit = getattr(model.config, "n_positions", None)
    if isinstance(model_limit, int) and model_limit > 0:
        limits.append(model_limit)

    # Reserve a few tokens for the "paraphrase:" prefix and special tokens.
    hard_limit = min(limits) if limits else 512
    safe_limit = max(32, hard_limit - 8)
    return min(requested_chunk_size, safe_limit)


def _split_into_chunks(
    text: str,
    tokenizer,
    chunk_size: int,
) -> List[str]:
    """
    Pack sentence-like segments into chunks without truncating source tokens.
    Very long individual sentences are split at tokenizer-token boundaries.
    """
    if not text.strip():
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    current_parts: List[str] = []
    current_count = 0

    def token_count(value: str) -> int:
        return len(tokenizer.encode(value, add_special_tokens=False))

    def flush():
        nonlocal current_parts, current_count
        if current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_count = 0

    for sentence in sentences:
        if not sentence:
            continue

        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        if len(sentence_tokens) > chunk_size:
            flush()
            for start in range(0, len(sentence_tokens), chunk_size):
                piece_ids = sentence_tokens[start:start + chunk_size]
                piece = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
                if piece:
                    chunks.append(piece)
            continue

        added_count = len(sentence_tokens)
        if current_parts and current_count + added_count > chunk_size:
            flush()

        current_parts.append(sentence)
        current_count += added_count

    flush()
    return chunks or [text]


def _generation_kwargs(
    intensity: str,
    num_return_sequences: int,
    max_length: int,
) -> dict:
    preset = dict(INTENSITY_PRESETS[intensity])
    preset.pop("description", None)

    if preset["do_sample"]:
        preset["num_beams"] = max(1, num_return_sequences)
    else:
        preset["num_beams"] = max(preset["num_beams"], num_return_sequences)
        if preset["num_beam_groups"] > preset["num_beams"]:
            preset["num_beam_groups"] = 1

    # Group beam search requires num_beams to be divisible by num_beam_groups.
    if preset["num_beams"] % preset["num_beam_groups"] != 0:
        preset["num_beam_groups"] = 1
        preset.pop("diversity_penalty", None)

    preset.update({
        "num_return_sequences": num_return_sequences,
        "max_new_tokens": max_length,
        "early_stopping": True,
    })
    return preset


def paraphrase(
    text: str,
    tokenizer,
    model,
    device: str,
    intensity: Literal["low", "medium", "high"] = "medium",
    num_return_sequences: int = 1,
    max_length: int = 512,
    seed: Optional[int] = None,
    chunk_size: int = 500,
) -> List[str]:
    """
    Generate paraphrases for a complete document.

    Args:
        text: Source document.
        tokenizer/model/device: Returned by load_model().
        intensity: low, medium, or high generation preset.
        num_return_sequences: Number of full-document paraphrases.
        max_length: Maximum NEW tokens generated per chunk.
        seed: Reproducibility seed. Each chunk receives a derived seed.
        chunk_size: Approximate source-token limit per chunk. Automatically
            capped below the model/tokenizer's advertised sequence limit.

    Returns:
        A list containing num_return_sequences reassembled documents.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if intensity not in INTENSITY_PRESETS:
        raise ValueError(f"intensity must be one of {list(INTENSITY_PRESETS)}")
    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be at least 1")
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if chunk_size < 32:
        raise ValueError("chunk_size must be at least 32 tokens")
    if not text.strip():
        return [text] * num_return_sequences

    effective_chunk_size = _effective_chunk_size(tokenizer, model, chunk_size)
    chunks = _split_into_chunks(text, tokenizer, effective_chunk_size)
    print(
        f"Input tokens: {len(tokenizer.encode(text, add_special_tokens=False))} | "
        f"Chunks: {len(chunks)} | Chunk size: {effective_chunk_size}"
    )

    all_variants: List[List[str]] = [[] for _ in range(num_return_sequences)]
    gen_kwargs = _generation_kwargs(intensity, num_return_sequences, max_length)

    for chunk_index, chunk in enumerate(chunks):
        prompt = f"paraphrase: {chunk}"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # Defensive check: never silently truncate or feed an oversized chunk.
        advertised_limit = _effective_chunk_size(tokenizer, model, chunk_size) + 8
        if input_ids.shape[1] > advertised_limit:
            raise ValueError(
                f"Chunk {chunk_index} encoded to {input_ids.shape[1]} tokens, "
                f"above safe limit {advertised_limit}. Reduce --chunk-size."
            )

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        if seed is not None:
            set_seed(seed + chunk_index)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if len(decoded) != num_return_sequences:
            raise RuntimeError(
                f"Expected {num_return_sequences} variants for chunk {chunk_index}, "
                f"received {len(decoded)}."
            )

        for variant_index, candidate in enumerate(decoded):
            all_variants[variant_index].append(candidate.strip())

        print(f"Processed chunk {chunk_index + 1}/{len(chunks)}")

    return [" ".join(parts).strip() for parts in all_variants]


def main():
    parser = argparse.ArgumentParser(
        description="Paraphrase long text with humarin/chatgpt_paraphraser_on_T5_base."
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="The ultimate test of your knowledge is your capacity to convey it to another.",
        help="Input text to paraphrase.",
    )
    parser.add_argument(
        "--intensity",
        choices=["low", "medium", "high"],
        default="medium",
    )
    parser.add_argument("--num-variants", type=int, default=1)
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum new tokens generated per chunk.",
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model_id)
    results = paraphrase(
        text=args.text,
        tokenizer=tokenizer,
        model=model,
        device=device,
        intensity=args.intensity,
        num_return_sequences=args.num_variants,
        max_length=args.max_length,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    print("\nOriginal:\n", args.text)
    for index, candidate in enumerate(results, start=1):
        print(f"\nParaphrase {index}:\n{candidate}")


if __name__ == "__main__":
    main()
