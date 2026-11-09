"""
Lightweight HELM-style text perturbations for robustness_project.

This module implements five independent transformations inspired by the
behaviours exposed by the WaterPark HELM wrapper:
    - lowercase
    - contraction
    - expansion
    - misspelling
    - typo

These are behaviour-based implementations, not a reproduction of the
original WaterPark/TextAttack implementation.

No model, API, or external NLP package is required.

Example:
    from helm_attack import helm_attack

    attacked = helm_attack(
        text,
        attack_type="typo",
        probability=0.15,
        seed=42,
    )
"""

from __future__ import annotations

import random
import re
from typing import Callable, Dict, Optional


# Common English contraction mappings. Longer phrases are matched first.
_CONTRACTIONS = {
    "are not": "aren't",
    "cannot": "can't",
    "could not": "couldn't",
    "did not": "didn't",
    "does not": "doesn't",
    "do not": "don't",
    "had not": "hadn't",
    "has not": "hasn't",
    "have not": "haven't",
    "is not": "isn't",
    "it is": "it's",
    "it has": "it's",
    "might not": "mightn't",
    "must not": "mustn't",
    "should not": "shouldn't",
    "was not": "wasn't",
    "were not": "weren't",
    "will not": "won't",
    "would not": "wouldn't",
    "I am": "I'm",
    "I have": "I've",
    "I will": "I'll",
    "I would": "I'd",
    "you are": "you're",
    "you have": "you've",
    "you will": "you'll",
    "you would": "you'd",
    "he is": "he's",
    "he has": "he's",
    "she is": "she's",
    "she has": "she's",
    "we are": "we're",
    "we have": "we've",
    "we will": "we'll",
    "we would": "we'd",
    "they are": "they're",
    "they have": "they've",
    "they will": "they'll",
    "they would": "they'd",
    "that is": "that's",
    "there is": "there's",
    "what is": "what's",
    "who is": "who's",
    "where is": "where's",
    "when is": "when's",
    "why is": "why's",
    "how is": "how's",
}

# Expansion is deliberately conservative: ambiguous contractions such as
# "he's" and "she's" are not expanded because they can mean "is" or "has".
_EXPANSIONS = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "haven't": "have not",
    "hasn't": "has not",
    "isn't": "is not",
    "mightn't": "might not",
    "mustn't": "must not",
    "shouldn't": "should not",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "wouldn't": "would not",
    "i'm": "I am",
    "i've": "I have",
    "i'll": "I will",
    "i'd": "I would",
    "you're": "you are",
    "you've": "you have",
    "you'll": "you will",
    "you'd": "you would",
    "we're": "we are",
    "we've": "we have",
    "we'll": "we will",
    "we'd": "we would",
    "they're": "they are",
    "they've": "they have",
    "they'll": "they will",
    "they'd": "they would",
    "that's": "that is",
    "there's": "there is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
}

# A small, explicit dictionary of common misspellings. This is intentionally
# transparent and easy to extend; it is not intended to cover every word.
_MISSPELLINGS = {
    "about": "abuot",
    "across": "acros",
    "address": "adress",
    "again": "agian",
    "already": "allready",
    "answer": "anwser",
    "because": "becuase",
    "believe": "beleive",
    "business": "buisness",
    "calendar": "calender",
    "cannot": "canot",
    "different": "diffrent",
    "environment": "enviroment",
    "example": "exmaple",
    "february": "febuary",
    "friend": "freind",
    "government": "goverment",
    "knowledge": "knowlege",
    "language": "langauge",
    "library": "libary",
    "necessary": "neccessary",
    "occasion": "ocassion",
    "people": "peaple",
    "receive": "recieve",
    "separate": "seperate",
    "similar": "similiar",
    "their": "thier",
    "through": "throught",
    "tomorrow": "tommorow",
    "until": "untill",
    "which": "wich",
    "would": "woud",
}


def _match_case(source: str, replacement: str) -> str:
    """Apply the source token's broad casing pattern to a replacement."""
    if source.isupper():
        return replacement.upper()
    if source.istitle():
        return replacement.capitalize()
    if source.islower():
        return replacement.lower()
    return replacement


def _replace_phrases(text: str, mapping: Dict[str, str]) -> str:
    """Replace whole-word/phrase matches, longest keys first, case-insensitively."""
    result = text
    for source in sorted(mapping, key=len, reverse=True):
        replacement = mapping[source]
        pattern = re.compile(r"(?<!\w)" + re.escape(source) + r"(?!\w)", re.IGNORECASE)

        def substitute(match: re.Match) -> str:
            return _match_case(match.group(0), replacement)

        result = pattern.sub(substitute, result)
    return result


def lowercase_attack(text: str) -> str:
    """Convert the entire text to lowercase."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.lower()


def contraction_attack(text: str) -> str:
    """Replace supported multi-word forms with common English contractions."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _replace_phrases(text, _CONTRACTIONS)


def expansion_attack(text: str) -> str:
    """Expand supported, unambiguous English contractions."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _replace_phrases(text, _EXPANSIONS)


def misspelling_attack(
    text: str,
    probability: float = 1.0,
    seed: Optional[int] = None,
) -> str:
    """
    Replace selected dictionary words with predefined common misspellings.

    probability is the chance of changing each eligible occurrence.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _validate_probability(probability)
    rng = random.Random(seed)

    # Match alphabetic words while leaving punctuation and whitespace intact.
    def substitute(match: re.Match) -> str:
        word = match.group(0)
        replacement = _MISSPELLINGS.get(word.lower())
        if replacement is None or rng.random() > probability:
            return word
        return _match_case(word, replacement)

    return re.sub(r"[A-Za-z]+", substitute, text)


def _typo_word(word: str, rng: random.Random) -> str:
    """Apply one random character-level typing error to a word."""
    if len(word) < 4:
        return word

    operation = rng.choice(("delete", "insert", "replace", "transpose"))
    index = rng.randrange(1, len(word) - 1)  # preserve first and last letters
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    if operation == "delete":
        return word[:index] + word[index + 1 :]

    if operation == "insert":
        char = rng.choice(alphabet)
        return word[:index] + char + word[index:]

    if operation == "replace":
        choices = alphabet.replace(word[index].lower(), "")
        char = rng.choice(choices)
        if word[index].isupper():
            char = char.upper()
        return word[:index] + char + word[index + 1 :]

    # Transpose adjacent internal characters.
    if index == len(word) - 1:
        index -= 1
    return word[:index] + word[index + 1] + word[index] + word[index + 2 :]


def _validate_probability(probability: float) -> None:
    if not isinstance(probability, (int, float)) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0.0 and 1.0")


def typo_attack(
    text: str,
    probability: float = 0.1,
    seed: Optional[int] = None,
) -> str:
    """
    Introduce random character-level typos in eligible words.

    Each alphabetic word of at least four characters is independently
    selected with the given probability. A selected word receives one
    insertion, deletion, substitution, or adjacent-character transposition.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _validate_probability(probability)
    rng = random.Random(seed)

    def substitute(match: re.Match) -> str:
        word = match.group(0)
        if len(word) < 4 or rng.random() > probability:
            return word
        return _typo_word(word, rng)

    return re.sub(r"[A-Za-z]+", substitute, text)


_ATTACKS: Dict[str, Callable[..., str]] = {
    "lowercase": lowercase_attack,
    "contraction": contraction_attack,
    "expansion": expansion_attack,
    "misspelling": misspelling_attack,
    "typo": typo_attack,
}


def helm_attack(
    text: str,
    attack_type: str = "typo",
    probability: float = 0.1,
    seed: Optional[int] = None,
) -> str:
    """
    Dispatch one HELM-style transformation.

    attack_type must be one of:
        lowercase, contraction, expansion, misspelling, typo

    probability is used by misspelling and typo. For the dictionary-based
    misspelling attack, it controls the chance of replacing each known word.
    For typo, it controls the chance of perturbing each eligible word.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized_type = attack_type.strip().lower().replace("_", "-")
    aliases = {
        "lowercaseattack": "lowercase",
        "contractionattack": "contraction",
        "expansionattack": "expansion",
        "misspellingattack": "misspelling",
        "typoattack": "typo",
    }
    normalized_type = aliases.get(normalized_type, normalized_type)

    if normalized_type not in _ATTACKS:
        raise ValueError(
            f"Unknown attack_type {attack_type!r}. Choose from: "
            + ", ".join(_ATTACKS)
        )

    attack = _ATTACKS[normalized_type]
    if normalized_type in ("misspelling", "typo"):
        return attack(text, probability=probability, seed=seed)
    return attack(text)


if __name__ == "__main__":
    sample = (
        "I am certain that the system does not work the way it is "
        "supposed to. This is an example of a different approach."
    )

    for name in _ATTACKS:
        print(f"\n{name}:")
        print(helm_attack(sample, attack_type=name, probability=0.25, seed=7))
