import random
import re

def substitution(text, rate=0.05, seed=42):
    """Replace selected words with a neutral placeholder."""
    rng = random.Random(seed)
    words = text.split()
    candidates = [
        i for i, word in enumerate(words)
        if re.search(r"[A-Za-z]", word)
    ]
    count = round(len(candidates) * rate)

    for i in rng.sample(candidates, min(count, len(candidates))):
        words[i] = "[REDACTED]"

    return " ".join(words)


def typoing(text, rate=0.03, seed=42):
    """Introduce reproducible adjacent-character swaps."""
    rng = random.Random(seed)
    words = text.split()
    candidates = [
        i for i, word in enumerate(words)
        if len(word) >= 4 and word.isalpha()
    ]
    count = round(len(candidates) * rate)

    for i in rng.sample(candidates, min(count, len(candidates))):
        word = words[i]
        j = rng.randrange(len(word) - 1)
        words[i] = (
            word[:j] + word[j + 1] + word[j]
            + word[j + 2:]
        )

    return " ".join(words)


def synonymization(text, replacements=None):
    """Apply a controlled, manually specified synonym mapping."""
    replacements = replacements or {
        "important": "significant",
        "rapid": "swift",
        "large": "substantial",
        "small": "little",
        "difficult": "challenging",
    }

    pattern = re.compile(
        r"\b(" + "|".join(
            re.escape(word) for word in replacements
        ) + r")\b",
        flags=re.IGNORECASE,
    )

    def replace(match):
        word = match.group(0)
        replacement = replacements[word.lower()]

        if word.isupper():
            return replacement.upper()
        if word.istitle():
            return replacement.title()
        return replacement

    return pattern.sub(replace, text)
