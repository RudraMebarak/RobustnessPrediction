"""
Synonym substitution attacks for LLM watermark robustness experiments.

Two independent candidate-generation methods are supported:
  - "wordnet": candidates from NLTK WordNet.
  - "llm": context-aware candidates from Qwen2.5-1.5B-Instruct.

Example:
    attack = SynonymAttack(method="wordnet", p=0.5, seed=42)
    attacked = attack.warp(text)

    attack = SynonymAttack(method="llm", p=0.5, seed=42)
    attacked = attack.warp(text)

For reproducible experiments, pass a stable per-document seed to warp():
    attacked = attack.warp(text, seed=doc_id * 1000 + trial_id)

The Qwen model is loaded lazily and only when method="llm".
"""

from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from typing import Iterable, Optional

import nltk
from nltk.corpus import wordnet
from nltk.tokenize import TreebankWordTokenizer

try:
    from pyinflect import getInflection
except ImportError:
    getInflection = None


class SynonymAttack:
    """Replace eligible words with synonyms from WordNet or Qwen."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    # Penn Treebank POS tags accepted by this attack.
    ALLOWED_TAGS = {
        "JJ", "JJR", "JJS",
        "RB", "RBR", "RBS",
        "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
        "NN", "NNS",
    }

    WORDNET_POS = {
        "JJ": wordnet.ADJ, "JJR": wordnet.ADJ, "JJS": wordnet.ADJ,
        "RB": wordnet.ADV, "RBR": wordnet.ADV, "RBS": wordnet.ADV,
        "VB": wordnet.VERB, "VBD": wordnet.VERB, "VBG": wordnet.VERB,
        "VBN": wordnet.VERB, "VBP": wordnet.VERB, "VBZ": wordnet.VERB,
        "NN": wordnet.NOUN, "NNS": wordnet.NOUN,
    }

    def __init__(
        self,
        method: str = "wordnet",
        p: float = 0.5,
        seed: int = 42,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 8,
        max_new_tokens: int = 80,
        context_window: int = 8,
        min_word_length: int = 4,
        no_replace_words: Optional[Iterable[str]] = None,
    ):
        method = method.lower().strip()
        if method not in {"wordnet", "llm"}:
            raise ValueError("method must be either 'wordnet' or 'llm'")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be between 0.0 and 1.0")

        self.method = method
        self.p = p
        self.seed = seed
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.context_window = context_window
        self.min_word_length = min_word_length
        self.no_replace_words = {w.lower() for w in (no_replace_words or {"as"})}

        self._tokenizer = TreebankWordTokenizer()
        self._candidate_cache = {}
        self._model = None
        self._hf_tokenizer = None
        self._torch = None

        self._ensure_nltk_resources()

    @staticmethod
    def _ensure_nltk_resources():
        """Check required NLTK resources and give actionable download errors."""
        resources = [
            ("tokenizers/punkt", "punkt"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        ]
        # Newer NLTK versions use these *_eng resources.
        for path, package in resources:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                except Exception:
                    pass

        # Probe the POS tagger; newer NLTK may require averaged_perceptron_tagger_eng.
        try:
            nltk.pos_tag(["test"])
        except LookupError:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.pos_tag(["test"])

    @staticmethod
    def _is_single_word(candidate: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", candidate.strip()))

    @staticmethod
    def _wordnet_pos(tag: str):
        return SynonymAttack.WORDNET_POS.get(tag)

    def _inflect(self, candidate: str, original_tag: str) -> str:
        """Try to restore the original word's inflection (plural/tense/etc.)."""
        if getInflection is None:
            return candidate

        try:
            forms = getInflection(candidate, tag=original_tag)
            if forms:
                return forms[0]
        except Exception:
            pass
        return candidate

    def _eligible(self, word: str, tag: str) -> bool:
        if tag not in self.ALLOWED_TAGS:
            return False
        if len(word) < self.min_word_length:
            return False
        if word.lower() in self.no_replace_words:
            return False
        if not word.isalpha():
            return False
        if tag.startswith("VB") and word.lower() in {"be", "have", "do"}:
            return False
        return True

    @staticmethod
    def _normalise(candidate: str) -> str:
        candidate = candidate.strip().strip("\"'`“”‘’.,;:!?()[]{}")
        candidate = candidate.replace("_", " ")
        return candidate

    def _wordnet_candidates(self, word: str, tag: str) -> list[str]:
        pos = self._wordnet_pos(tag)
        if pos is None:
            return []

        candidates = set()
        for synset in wordnet.synsets(word, pos=pos):
            for lemma in synset.lemmas():
                candidate = self._normalise(lemma.name())
                if not self._is_single_word(candidate):
                    continue
                if candidate.lower() == word.lower():
                    continue
                candidate = self._inflect(candidate, tag)
                if candidate.lower() != word.lower():
                    candidates.add(candidate)

        return sorted(candidates, key=str.lower)

    def _load_llm(self):
        """Load Qwen once, on demand."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._hf_tokenizer.pad_token is None:
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
        self._hf_tokenizer.padding_side = "left"

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
            ).to(self.device)

        self._model.eval()

    def _llm_prompt(self, word: str, tag: str, context: str) -> str:
        tag_description = {
            "JJ": "adjective", "JJR": "comparative adjective", "JJS": "superlative adjective",
            "RB": "adverb", "RBR": "comparative adverb", "RBS": "superlative adverb",
            "VB": "base-form verb", "VBD": "past-tense verb", "VBG": "gerund/present participle",
            "VBN": "past participle", "VBP": "present-tense verb", "VBZ": "third-person singular verb",
            "NN": "singular noun", "NNS": "plural noun",
        }.get(tag, "word")

        return (
            "You are helping with a controlled lexical substitution experiment. "
            "Suggest up to 5 one-word synonyms for the target word as it is used "
            "in the sentence. Preserve its part of speech and grammatical form. "
            "Do not change the sentence meaning. Do not return explanations. "
            "Return ONLY a JSON array of strings. If no safe synonym exists, return [].\n\n"
            f"Target word: {word}\n"
            f"Part of speech: {tag_description}\n"
            f"Sentence: {context}\n"
            'JSON array:'
        )

    def _parse_llm_candidates(self, generated: str) -> list[str]:
        generated = generated.strip()
        # Prefer a JSON array, including when the model wraps it in other text.
        match = re.search(r"\[[\s\S]*?\]", generated)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if isinstance(x, str)]
            except json.JSONDecodeError:
                pass

        # Conservative fallback for comma/newline-separated answers.
        generated = re.sub(r"^\s*(synonyms?|answers?)\s*:\s*", "", generated, flags=re.I)
        return re.split(r"[,;\n]", generated)

    def _llm_candidates_batch(self, requests: list[tuple[str, str, str]]) -> list[list[str]]:
        """Generate context-aware candidate lists for (word, POS, sentence) requests."""
        if not requests:
            return []

        self._load_llm()
        prompts = []
        for word, tag, context in requests:
            messages = [
                {"role": "system", "content": "You produce concise, accurate lexical substitutions."},
                {"role": "user", "content": self._llm_prompt(word, tag, context)},
            ]
            prompts.append(
                self._hf_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        outputs = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start:start + self.batch_size]
            encoded = self._hf_tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True
            )
            model_device = next(self._model.parameters()).device
            encoded = {k: v.to(model_device) for k, v in encoded.items()}

            with self._torch.inference_mode():
                generated = self._model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self._hf_tokenizer.pad_token_id,
                    eos_token_id=self._hf_tokenizer.eos_token_id,
                )

            input_width = encoded["input_ids"].shape[1]
            new_tokens = generated[:, input_width:]
            decoded = self._hf_tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            outputs.extend(decoded)

        result = []
        for (word, tag, _context), generated_text in zip(requests, outputs):
            candidates = []
            for raw in self._parse_llm_candidates(generated_text):
                candidate = self._normalise(raw)
                if not self._is_single_word(candidate):
                    continue
                if candidate.lower() == word.lower():
                    continue
                candidate = self._inflect(candidate, tag)
                if candidate.lower() != word.lower():
                    candidates.append(candidate)

            # Preserve order while removing duplicates.
            result.append(list(dict.fromkeys(candidates)))
        return result

    def warp(
        self,
        text: str,
        p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Return attacked text. `p` is the per-eligible-word replacement probability."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return text

        probability = self.p if p is None else p
        if not 0.0 <= probability <= 1.0:
            raise ValueError("p must be between 0.0 and 1.0")

        rng = random.Random(self.seed if seed is None else seed)

        spans = list(self._tokenizer.span_tokenize(text))
        tokens = [text[start:end] for start, end in spans]
        tagged = nltk.pos_tag(tokens)

        # Select positions first, then generate missing LLM candidates in batches.
        selected = []
        for i, ((start, end), (word, tag)) in enumerate(zip(spans, tagged)):
            if not self._eligible(word, tag) or rng.random() > probability:
                continue

            left = max(0, i - self.context_window)
            right = min(len(tokens), i + self.context_window + 1)
            context = " ".join(tokens[left:right])
            cache_key = (self.method, word.lower(), tag, context.lower())
            selected.append((start, end, word, tag, context, cache_key))

        missing = []
        seen_keys = set()
        for _start, _end, word, tag, context, cache_key in selected:
            if cache_key not in self._candidate_cache and cache_key not in seen_keys:
                missing.append((word, tag, context, cache_key))
                seen_keys.add(cache_key)

        if self.method == "wordnet":
            for word, tag, _context, cache_key in missing:
                self._candidate_cache[cache_key] = self._wordnet_candidates(word, tag)
        elif missing:
            requests = [(word, tag, context) for word, tag, context, _key in missing]
            candidate_lists = self._llm_candidates_batch(requests)
            for (_word, _tag, _context, cache_key), candidates in zip(missing, candidate_lists):
                self._candidate_cache[cache_key] = candidates

        replacements = []
        for start, end, word, _tag, _context, cache_key in selected:
            candidates = self._candidate_cache.get(cache_key, [])
            if candidates:
                replacement = rng.choice(candidates)
                # Preserve initial capitalization.
                if word[0].isupper():
                    replacement = replacement[0].upper() + replacement[1:]
                replacements.append((start, end, replacement))

        # Apply from right to left so character offsets remain valid.
        attacked = text
        for start, end, replacement in reversed(replacements):
            attacked = attacked[:start] + replacement + attacked[end:]

        return attacked

    # Compatibility alias with the WaterPark implementation's naming.
    def attack(self, text: str, p: Optional[float] = None, seed: Optional[int] = None) -> str:
        return self.warp(text, p=p, seed=seed)

    def clear_cache(self):
        """Clear generated synonym candidates (useful when changing experiment settings)."""
        self._candidate_cache.clear()
