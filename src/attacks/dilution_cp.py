"""Copy-Paste text attack for the robustness_project.

This module has no dataset/file-loading logic. It replaces k non-overlapping
spans in a target text with spans sampled from a donor text.

Tokenization is whitespace-based, so insertion_length means whitespace-delimited
words/tokens. Adapt _token_spans() if the experiment requires model-token spans.

Example:
    from copy_paste_attack import copy_paste_attack

    attacked = copy_paste_attack(
        target_text=watermarked_text,
        donor_text=unwatermarked_text,
        num_insertions=3,
        insertion_length=10,
        seed=42,
    )
"""

from __future__ import annotations

import random
import re
from typing import List, Optional, Tuple

TokenSpan = Tuple[int, int]


def _token_spans(text: str) -> List[TokenSpan]:
    """Return character offsets for whitespace-delimited tokens."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _sample_non_overlapping_starts(
    token_count: int,
    num_insertions: int,
    insertion_length: int,
    rng: random.Random,
) -> List[int]:
    """Sample starts for equal-length non-overlapping target spans.

    Adjacent spans are allowed. A compressed-coordinate transformation
    guarantees that the sampled spans cannot overlap.
    """
    if num_insertions * insertion_length > token_count:
        raise ValueError(
            "Replacement spans exceed the target length: "
            f"{num_insertions} * {insertion_length} > {token_count} tokens."
        )

    compressed_range = token_count - num_insertions * (insertion_length - 1)
    compressed_starts = sorted(rng.sample(range(compressed_range), num_insertions))
    return [
        compressed_starts[i] + i * (insertion_length - 1)
        for i in range(num_insertions)
    ]


def copy_paste_attack(
    target_text: str,
    donor_text: str,
    num_insertions: int = 3,
    insertion_length: int = 10,
    seed: Optional[int] = None,
) -> str:
    """Replace spans in ``target_text`` with spans from ``donor_text``.

    Args:
        target_text: Text to modify (e.g. a watermarked response).
        donor_text: Text supplying replacement spans (e.g. an unwatermarked
            response).
        num_insertions: Number of replacement spans.
        insertion_length: Whitespace-delimited tokens per span.
        seed: Optional seed for reproducible span selection.

    Returns:
        The attacked text. The number of target token positions selected is
        exactly ``num_insertions * insertion_length``.

    Raises:
        TypeError: If either text argument or an integer parameter has the
            wrong type.
        ValueError: If parameters are invalid or either text is too short.
    """
    if not isinstance(target_text, str) or not isinstance(donor_text, str):
        raise TypeError("target_text and donor_text must both be strings.")
    if not isinstance(num_insertions, int) or isinstance(num_insertions, bool):
        raise TypeError("num_insertions must be an integer.")
    if not isinstance(insertion_length, int) or isinstance(insertion_length, bool):
        raise TypeError("insertion_length must be an integer.")
    if num_insertions < 0:
        raise ValueError("num_insertions cannot be negative.")
    if insertion_length <= 0:
        raise ValueError("insertion_length must be greater than zero.")
    if num_insertions == 0:
        return target_text

    target_spans = _token_spans(target_text)
    donor_spans = _token_spans(donor_text)
    if not target_spans:
        raise ValueError("target_text contains no whitespace-delimited tokens.")
    if len(donor_spans) < insertion_length:
        raise ValueError(
            f"donor_text has {len(donor_spans)} tokens; at least "
            f"{insertion_length} are required per replacement span."
        )

    rng = random.Random(seed)
    target_starts = _sample_non_overlapping_starts(
        len(target_spans), num_insertions, insertion_length, rng
    )
    # Donor spans are sampled independently and may overlap one another.
    donor_starts = [
        rng.randint(0, len(donor_spans) - insertion_length)
        for _ in range(num_insertions)
    ]

    replacements = []
    for target_start, donor_start in zip(target_starts, donor_starts):
        target_char_start = target_spans[target_start][0]
        target_char_end = target_spans[target_start + insertion_length - 1][1]
        donor_char_start = donor_spans[donor_start][0]
        donor_char_end = donor_spans[donor_start + insertion_length - 1][1]
        replacements.append(
            (
                target_char_start,
                target_char_end,
                donor_text[donor_char_start:donor_char_end],
            )
        )

    # Replace right-to-left so earlier character offsets remain valid.
    attacked_text = target_text
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        attacked_text = attacked_text[:start] + replacement + attacked_text[end:]

    return attacked_text


def copy_paste_attack_with_metadata(
    target_text: str,
    donor_text: str,
    num_insertions: int = 3,
    insertion_length: int = 10,
    seed: Optional[int] = None,
) -> dict:
    """Return the attacked text together with basic attack metadata."""
    attacked_text = copy_paste_attack(
        target_text=target_text,
        donor_text=donor_text,
        num_insertions=num_insertions,
        insertion_length=insertion_length,
        seed=seed,
    )
    target_token_count = len(_token_spans(target_text))
    modified_tokens = num_insertions * insertion_length
    return {
        "attacked_text": attacked_text,
        "num_insertions": num_insertions,
        "insertion_length": insertion_length,
        "modified_tokens": modified_tokens,
        "target_token_count": target_token_count,
        "attacked_token_count": len(_token_spans(attacked_text)),
        "modified_percentage": (
            100.0 * modified_tokens / target_token_count if target_token_count else 0.0
        ),
        "seed": seed,
    }


if __name__ == "__main__":
    target = (
        "The system generates a response using a language model and "
        "then evaluates its semantic consistency across several tasks."
    )
    donor = (
        "A separate model produces alternative content that can replace "
        "selected portions of the generated response."
    )
    result = copy_paste_attack_with_metadata(
        target, donor, num_insertions=2, insertion_length=3, seed=42
    )
    print("Original:", target)
    print("Donor:   ", donor)
    print("Attacked:", result["attacked_text"])
    print("Metadata:", {k: v for k, v in result.items() if k != "attacked_text"})
