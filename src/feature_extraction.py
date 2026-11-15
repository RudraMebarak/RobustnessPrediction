
import argparse
import logging
import re
import string
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import spacy
import textstat

from wordfreq import zipf_frequency
from spacy.lang.en.stop_words import STOP_WORDS


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {".txt"}

MATTR_WINDOW = 50

RARE_WORD_THRESHOLD = 3.5

TECHNICAL_WORD_MIN_LENGTH = 8
TECHNICAL_WORD_MAX_ZIPF = 4.0


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ============================================================
# TEXT UTILITIES
# ============================================================

def tokenize_words(doc):

    return [
        token.text.lower()
        for token in doc
        if token.is_alpha
    ]


def calculate_entropy(sequence):

    if not sequence:
        return 0.0

    counts = Counter(sequence)
    total = len(sequence)

    probabilities = np.array(
        list(counts.values()),
        dtype=float
    ) / total

    return float(
        -np.sum(probabilities * np.log2(probabilities))
    )


def calculate_ngram_repetition(words, n):

    if len(words) < n:
        return 0.0

    ngrams = [
        tuple(words[i:i+n])
        for i in range(len(words) - n + 1)
    ]

    counts = Counter(ngrams)

    repeated = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated / len(ngrams)


def calculate_mattr(words, window=MATTR_WINDOW):

    if not words:
        return 0.0

    if len(words) <= window:
        return len(set(words)) / len(words)

    scores = []

    for i in range(len(words) - window + 1):

        current = words[i:i + window]

        scores.append(
            len(set(current)) / window
        )

    return float(np.mean(scores))


def calculate_yules_k(words):

    """
    Yule's K lexical richness measure.
    Lower values generally indicate greater lexical diversity.
    """

    if not words:
        return 0.0

    frequencies = Counter(words)

    N = len(words)

    frequency_of_frequencies = Counter(
        frequencies.values()
    )

    summation = sum(
        count * (frequency ** 2)
        for frequency, count in frequency_of_frequencies.items()
    )

    return float(
        10000 * (summation - N) / (N ** 2)
    )


def token_depth(token):

    depth = 0
    current = token

    while current.head != current:
        depth += 1
        current = current.head

    return depth


def calculate_clause_count(doc):

    clause_dependencies = {
        "advcl",
        "ccomp",
        "xcomp",
        "relcl",
        "acl",
        "conj"
    }

    count = 0

    for token in doc:

        if token.dep_ == "ROOT":
            count += 1

        elif token.dep_ in clause_dependencies:
            count += 1

    return max(count, 1)


def calculate_passive_sentence_ratio(doc):

    sentences = list(doc.sents)

    if not sentences:
        return 0.0

    passive_count = 0

    for sentence in sentences:

        is_passive = any(
            token.dep_ in {"nsubjpass", "auxpass"}
            or token.dep_ == "nsubj:pass"
            for token in sentence
        )

        if is_passive:
            passive_count += 1

    return passive_count / len(sentences)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(text, nlp):

    doc = nlp(text)

    words = tokenize_words(doc)

    word_count = len(words)

    sentences = list(doc.sents)

    sentence_count = len(sentences)

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    paragraph_lengths = [
        len(re.findall(r"\b\w+\b", p))
        for p in paragraphs
    ]

    features = {}

    # --------------------------------------------------------
    # A. BASIC STATISTICS (8)
    # --------------------------------------------------------

    features["character_count"] = len(text)

    features["word_count"] = word_count

    features["sentence_count"] = sentence_count

    features["average_word_length"] = (
        float(np.mean([len(w) for w in words]))
        if words else 0.0
    )

    sentence_lengths = [
        len(tokenize_words(sentence))
        for sentence in sentences
    ]

    features["average_sentence_length"] = (
        float(np.mean(sentence_lengths))
        if sentence_lengths else 0.0
    )

    features["max_sentence_length"] = (
        max(sentence_lengths)
        if sentence_lengths else 0
    )

    features["vocabulary_size"] = len(set(words))

    features["type_token_ratio"] = (
        len(set(words)) / word_count
        if word_count else 0.0
    )

    # --------------------------------------------------------
    # B. LEXICAL DIVERSITY (8)
    # --------------------------------------------------------

    frequencies = Counter(words)

    features["mattr"] = calculate_mattr(words)

    features["hapax_legomena_ratio"] = (
        sum(1 for x in frequencies.values() if x == 1)
        / word_count if word_count else 0.0
    )

    features["dislegomena_ratio"] = (
        sum(1 for x in frequencies.values() if x == 2)
        / word_count if word_count else 0.0
    )

    content_tags = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

    content_words = [
        token for token in doc
        if token.is_alpha and token.pos_ in content_tags
    ]

    features["lexical_density"] = (
        len(content_words) / word_count
        if word_count else 0.0
    )

    stopword_count = sum(
        1 for word in words
        if word in STOP_WORDS
    )

    features["stopword_ratio"] = (
        stopword_count / word_count
        if word_count else 0.0
    )

    zipf_scores = [
        zipf_frequency(word, "en")
        for word in words
    ]

    features["rare_word_ratio"] = (
        sum(
            score < RARE_WORD_THRESHOLD
            for score in zipf_scores
        ) / word_count
        if word_count else 0.0
    )

    features["average_word_frequency"] = (
        float(np.mean(zipf_scores))
        if zipf_scores else 0.0
    )

    features["vocabulary_richness"] = calculate_yules_k(words)

    # --------------------------------------------------------
    # C. SYNTACTIC FEATURES (8)
    # --------------------------------------------------------

    pos_counts = Counter(
        token.pos_
        for token in doc
        if token.is_alpha
    )

    denominator = word_count or 1

    features["noun_ratio"] = (
        (pos_counts["NOUN"] + pos_counts["PROPN"])
        / denominator
    )

    features["verb_ratio"] = (
        pos_counts["VERB"] / denominator
    )

    features["adjective_ratio"] = (
        pos_counts["ADJ"] / denominator
    )

    features["adverb_ratio"] = (
        pos_counts["ADV"] / denominator
    )

    features["pronoun_ratio"] = (
        pos_counts["PRON"] / denominator
    )

    depths = [
        token_depth(token)
        for token in doc
        if token.is_alpha
    ]

    features["dependency_tree_depth"] = (
        float(np.mean(depths))
        if depths else 0.0
    )

    clause_count = calculate_clause_count(doc)

    features["average_clause_length"] = (
        word_count / clause_count
        if word_count else 0.0
    )

    features["passive_voice_ratio"] = (
        calculate_passive_sentence_ratio(doc)
    )

    # --------------------------------------------------------
    # D. READABILITY (5)
    # --------------------------------------------------------

    features["flesch_reading_ease"] = (
        textstat.flesch_reading_ease(text)
    )

    features["flesch_kincaid_grade"] = (
        textstat.flesch_kincaid_grade(text)
    )

    features["gunning_fog_index"] = (
        textstat.gunning_fog(text)
    )

    features["smog_index"] = (
        textstat.smog_index(text)
    )

    features["average_syllables_per_word"] = (
        textstat.avg_syllables_per_word(text)
    )

    # --------------------------------------------------------
    # E. LINGUISTIC / STRUCTURAL FEATURES (7)
    # --------------------------------------------------------

    bigrams = [
        tuple(words[i:i+2])
        for i in range(len(words)-1)
    ]

    trigrams = [
        tuple(words[i:i+3])
        for i in range(len(words)-2)
    ]

    features["bigram_entropy"] = calculate_entropy(bigrams)

    features["trigram_entropy"] = calculate_entropy(trigrams)

    features["repetition_ratio"] = (
        calculate_ngram_repetition(words, 2)
    )

    punctuation_count = sum(
        1 for char in text
        if char in string.punctuation
    )

    features["punctuation_density"] = (
        punctuation_count / len(text)
        if text else 0.0
    )

    comma_count = text.count(",")

    features["comma_ratio"] = (
        comma_count / sentence_count
        if sentence_count else 0.0
    )

    features["paragraph_count"] = len(paragraphs)

    features["paragraph_length_variance"] = (
        float(np.var(paragraph_lengths))
        if paragraph_lengths else 0.0
    )

    # --------------------------------------------------------
    # F. DOMAIN / STYLE FEATURES (6)
    # --------------------------------------------------------

    pronouns_first_person = {
        "i", "me", "my", "mine",
        "we", "us", "our", "ours"
    }

    pronouns_second_person = {
        "you", "your", "yours",
        "yourself", "yourselves"
    }

    features["first_person_pronoun_ratio"] = (
        sum(w in pronouns_first_person for w in words)
        / denominator
    )

    features["second_person_pronoun_ratio"] = (
        sum(w in pronouns_second_person for w in words)
        / denominator
    )

    technical_words = [
        word for word, score in zip(words, zipf_scores)
        if len(word) >= TECHNICAL_WORD_MIN_LENGTH
        and word not in STOP_WORDS
        and score <= TECHNICAL_WORD_MAX_ZIPF
    ]

    features["technical_vocabulary_ratio"] = (
        len(technical_words) / denominator
    )

    named_entity_count = sum(
        1 for entity in doc.ents
        if entity.label_ not in {
            "CARDINAL",
            "ORDINAL",
            "QUANTITY"
        }
    )

    features["named_entity_density"] = (
        named_entity_count / denominator
    )

    numeric_count = sum(
        1 for token in doc
        if token.like_num
    )

    features["numeric_expression_ratio"] = (
        numeric_count / denominator
    )

    conjunction_count = sum(
        1 for token in doc
        if token.pos_ in {"CCONJ", "SCONJ"}
    )

    features["conjunction_ratio"] = (
        conjunction_count / denominator
    )

    return features


# ============================================================
# FILE PROCESSING
# ============================================================

def read_text_file(path):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1"
    ]

    for encoding in encodings:

        try:
            return path.read_text(encoding=encoding)

        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to decode file.")


def process_folder(input_folder, output_csv, nlp):

    input_folder = Path(input_folder)
    output_csv = Path(output_csv)

    if not input_folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_folder}"
        )

    files = sorted(
        path for path in input_folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    logging.info("Found %d text files.", len(files))

    results = []
    failed_files = []

    for index, path in enumerate(files, start=1):

        relative_path = str(
            path.relative_to(input_folder)
        )

        logging.info(
            "[%d/%d] Processing: %s",
            index,
            len(files),
            relative_path
        )

        try:

            text = read_text_file(path)

            if not text.strip():
                logging.warning(
                    "Empty file skipped: %s",
                    relative_path
                )
                continue

            features = extract_features(text, nlp)

            row = {
                "file_name": path.name,
                "relative_path": relative_path,
                **features
            }

            results.append(row)

        except Exception as error:

            failed_files.append(relative_path)

            logging.exception(
                "Failed processing %s: %s",
                relative_path,
                error
            )

    if not results:
        logging.warning(
            "No files were successfully processed."
        )
        return

    dataframe = pd.DataFrame(results)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dataframe.to_csv(
        output_csv,
        index=False,
        encoding="utf-8"
    )

    logging.info("----------------------------------")
    logging.info("EXTRACTION COMPLETE")
    logging.info("Files found: %d", len(files))
    logging.info("Files processed: %d", len(results))
    logging.info("Failed files: %d", len(failed_files))
    logging.info("Features extracted: %d", len(results[0]) - 2)
    logging.info("Output: %s", output_csv.resolve())
    logging.info("----------------------------------")


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Extract 42 features from original text files."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing original text files."
    )

    parser.add_argument(
        "--output",
        default="text_features.csv",
        help="Output CSV path."
    )

    args = parser.parse_args()

    logging.info("Loading spaCy model...")

    nlp = spacy.load("en_core_web_sm")

    nlp.max_length = 5_000_000

    logging.info("spaCy model loaded successfully.")

    process_folder(
        args.input,
        args.output,
        nlp
    )


if __name__ == "__main__":
    main()
