#!/usr/bin/env python3
"""
Run the project's text attacks over a dataset, with intensity ladders,
per-document deterministic seeds, output files, and a resumable CSV manifest.

Place this file beside:
    dilution_cp.py
    helm.py
    paraphrase_humarin.py
    synonym_attack(1).py   (or synonym_attack.py)

Supported input formats: .jsonl, .json, .csv, .txt, or a directory containing them.

Expected record fields (auto-detected unless CLI overrides):
    doc_id / id / sample_id
    text / generated_text / watermarked_text / attacked_text
Optional copy-paste donor field:
    donor_text / donor / unwatermarked_text / clean_text

Example:
    python attackmode.py \
      --input data/watermarked \
      --output data/attacked \
      --id-field doc_id \
      --text-field text \
      --donor-data data/unwatermarked \
      --donor-id-field doc_id \
      --donor-text-field text

The default probability ladder (0.10, 0.30, 0.50) is a configurable
low/medium/high experimental ladder. It is NOT asserted to be WaterPark's
exact published schedule. HELM's lowercase/contraction/expansion functions
are whole-text deterministic transforms in the supplied implementation,
so those are run once at "full" intensity. HELM typo/misspelling and
synonym attacks use the probability ladder. Humarin uses chunked document-level paraphrasing.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import zlib
import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ID_CANDIDATES = ("doc_id", "id", "sample_id", "document_id", "original_id")
TEXT_CANDIDATES = ("text", "generated_text", "watermarked_text", "content", "response")
DONOR_CANDIDATES = ("donor_text", "donor", "unwatermarked_text", "clean_text", "non_watermarked_text")
PROBABILITY_LEVELS = {"low": 0.10, "medium": 0.30, "high": 0.50}
HELM_TYPES = ("lowercase", "contraction", "expansion", "misspelling", "typo")


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required attack module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_name(value: Any) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return (value[:120] or "document")


def detect_field(record: Dict[str, Any], explicit: Optional[str], candidates: Iterable[str], kind: str) -> str:
    if explicit:
        if explicit not in record:
            raise KeyError(f"Requested {kind} field {explicit!r} not found. Available: {list(record)}")
        return explicit
    for key in candidates:
        if key in record and record[key] is not None:
            return key
    raise KeyError(f"Could not detect {kind} field. Available fields: {list(record)}")


def read_records(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL/JSON/CSV, or treat each .txt file as one document."""
    if path.is_dir():
        records: List[Dict[str, Any]] = []
        files = sorted(
            f for f in path.rglob("*")
            if f.is_file() and f.suffix.lower() in {".jsonl", ".json", ".csv", ".txt"}
        )
        for file in files:
            records.extend(read_records(file))
        return records

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        out = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError(f"{path}:{line_no}: each JSONL line must be an object")
                    out.append(item)
        return out
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            for key in ("records", "samples", "data", "items"):
                if isinstance(obj.get(key), list):
                    return [x for x in obj[key] if isinstance(x, dict)]
            # A single record is also accepted.
            return [obj]
        raise ValueError(f"Unsupported JSON structure in {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".txt":
        return [{"doc_id": path.stem, "text": path.read_text(encoding="utf-8")}]
    raise ValueError(f"Unsupported input type: {path}")


def index_donors(records: List[Dict[str, Any]], id_field: Optional[str], text_field: Optional[str]) -> Dict[str, str]:
    """Index donor documents by ID/filename stem."""
    donors: Dict[str, str] = {}
    for i, record in enumerate(records):
        rid = str(record.get(id_field, record.get("doc_id", record.get("id", i)))) if id_field else str(record.get("doc_id", record.get("id", i)))
        field = detect_field(record, text_field, TEXT_CANDIDATES, "donor text")
        if rid in donors:
            raise ValueError(f"Duplicate donor ID/filename stem: {rid!r}")
        donors[rid] = str(record[field])
    return donors


def find_donor_text(doc_id: str, donors: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve the donor for the special data_code naming convention.

    code_sample_{i} -> code_{i}_*unwatermarked*
    code_extra_{i}  -> code*_{i+100}_unwatermarked*

    For all other IDs, use exact ID matching. A missing or ambiguous special
    match is returned as unresolved rather than silently choosing a donor.
    """
    match = re.fullmatch(r"code_sample_(\\d+)", doc_id)
    if match:
        number = int(match.group(1))
        pattern = f"code_{number}_*unwatermarked*"
    else:
        match = re.fullmatch(r"code_extra_(\\d+)", doc_id)
        if match:
            number = int(match.group(1)) + 100
            pattern = f"code*_{number}_unwatermarked*"
        else:
            return donors.get(doc_id), doc_id if doc_id in donors else None

    matches = sorted(name for name in donors if fnmatch.fnmatchcase(name, pattern))
    if len(matches) == 1:
        return donors[matches[0]], matches[0]
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No donor matched special mapping for {doc_id!r}; expected filename pattern {pattern!r}"
        )
    raise ValueError(
        f"Multiple donors matched {doc_id!r} using {pattern!r}: {matches}. "
        "Make the donor filename pattern unambiguous."
    )


def stable_seed(base_seed: int, doc_id: str, attack: str, intensity: str) -> int:
    payload = f"{base_seed}|{doc_id}|{attack}|{intensity}".encode("utf-8")
    return (base_seed + zlib.crc32(payload)) % (2**31 - 1)


def append_manifest(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "doc_id", "attack_name", "intensity", "seed", "probability",
        "output_path", "status", "error", "parameters_json",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})
        f.flush()
        os.fsync(f.fileno())


def output_exists_in_manifest(manifest: Path, output_path: str) -> bool:
    if not manifest.exists():
        return False
    with manifest.open("r", encoding="utf-8", newline="") as f:
        return any(row.get("output_path") == output_path and row.get("status") == "ok"
                   for row in csv.DictReader(f))


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def build_attacks(args):
    root = Path(__file__).resolve().parent
    cp = load_module("attackmode_copy_paste", root / "dilution_cp.py")
    helm = load_module("attackmode_helm", root / "helm.py")
    # Optional/heavy attack modules are imported only if selected.
    synonym_mod = None
    humarin_module = None
    humarin_objects = None
    synonym_objects = {}

    def get_synonym_module():
        nonlocal synonym_mod
        if synonym_mod is None:
            synonym_path = root / "synonym_attack.py"
            if not synonym_path.exists():
                synonym_path = root / "synonym_attack(1).py"
            synonym_mod = load_module("attackmode_synonym", synonym_path)
        return synonym_mod

    def run_attack(attack_name: str, text: str, record: Dict[str, Any],
                   donor_text: Optional[str], seed: int, intensity: str):
        probability = None
        params: Dict[str, Any] = {"seed": seed, "intensity": intensity}

        if attack_name == "copy_paste":
            if not donor_text:
                raise ValueError("copy_paste requires donor text; supply a donor field or --donor-data")
            fraction = PROBABILITY_LEVELS[intensity]
            target_tokens = len(cp._token_spans(text))
            donor_tokens = len(cp._token_spans(donor_text))
            if target_tokens == 0 or donor_tokens == 0:
                raise ValueError("copy_paste target/donor has no whitespace-delimited tokens")
            insertion_length = max(1, round(target_tokens * fraction))
            insertion_length = min(insertion_length, target_tokens, donor_tokens)
            num_insertions = 1
            probability = fraction
            params.update({
                "modified_fraction_target": fraction,
                "num_insertions": num_insertions,
                "insertion_length": insertion_length,
                "tokenization": "whitespace",
                "donor_id": record.get("_resolved_donor_id", ""),
            })
            result = cp.copy_paste_attack(
                target_text=text, donor_text=donor_text,
                num_insertions=num_insertions, insertion_length=insertion_length, seed=seed,
            )
            # This is a target-token replacement fraction, not a per-token probability.
            return result, None, params

        if attack_name.startswith("helm_"):
            kind = attack_name[len("helm_"):]
            if kind in ("typo", "misspelling"):
                probability = PROBABILITY_LEVELS[intensity]
                params["probability"] = probability
            else:
                params["intensity_note"] = "whole-text transform; supplied implementation has no partial-intensity parameter"
            params["attack_type"] = kind
            return helm.helm_attack(text, attack_type=kind,
                                    probability=probability if probability is not None else 1.0,
                                    seed=seed), probability, params

        if attack_name.startswith("synonym_"):
            method = attack_name[len("synonym_"):]
            probability = PROBABILITY_LEVELS[intensity]
            synonym_module = get_synonym_module()
            if method not in synonym_objects:
                synonym_objects[method] = synonym_module.SynonymAttack(
                    method=method,
                    p=probability,
                    seed=seed,
                    model_name=args.synonym_model,
                    device=args.device,
                    max_new_tokens=args.synonym_max_new_tokens,
                )
            attack_obj = synonym_objects[method]
            params.update({
                "method": method,
                "probability": probability,
                "model_name": args.synonym_model if method == "llm" else "",
                "max_new_tokens": args.synonym_max_new_tokens if method == "llm" else "",
            })
            return attack_obj.warp(text, p=probability, seed=seed), probability, params

        if attack_name == "humarin":
            nonlocal humarin_module, humarin_objects
            if humarin_module is None:
                humarin_module = load_module("attackmode_humarin", root / "paraphrase_humarin.py")
                humarin_objects = humarin_module.load_model(args.humarin_model)
            tokenizer, model, device = humarin_objects
            max_length = args.humarin_max_length
            chunk_size = args.humarin_chunk_size
            result_list = humarin_module.paraphrase(
                text=text, tokenizer=tokenizer, model=model, device=device,
                intensity=intensity, num_return_sequences=1,
                max_length=max_length, seed=seed, chunk_size=chunk_size,
            )
            params.update({
                "model_id": args.humarin_model,
                "num_return_sequences": 1,
                "max_length": max_length,
                "chunk_size": chunk_size,
                "decoding_preset": humarin_module.INTENSITY_PRESETS[intensity],
                "chunking": "sentence-aware token chunks, reassembled in original order",
            })
            return result_list[0] if result_list else text, None, params

        raise ValueError(f"Unknown attack: {attack_name}")

    return run_attack


def main():
    parser = argparse.ArgumentParser(description="Apply all configured text attacks to a dataset.")
    parser.add_argument("--input", required=True, help="Input file or directory (.jsonl/.json/.csv/.txt).")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--id-field", default=None, help="Document ID field; auto-detected by default.")
    parser.add_argument("--text-field", default=None, help="Text field; auto-detected by default.")
    parser.add_argument("--donor-field", default=None, help="Donor text field in input records.")
    parser.add_argument("--donor-data", default=None, help="Optional separate donor dataset, matched by document ID.")
    parser.add_argument("--donor-id-field", default=None, help="ID field in donor dataset.")
    parser.add_argument("--donor-text-field", default=None, help="Text field in donor dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--intensities", nargs="+", choices=("low", "medium", "high"),
                        default=("low", "medium", "high"))
    parser.add_argument("--attacks", nargs="+", default=[
        "copy_paste",
        "helm_lowercase", "helm_contraction", "helm_expansion", "helm_misspelling", "helm_typo",
        "synonym_wordnet", "humarin",
    ], help="Attack names. Default: all implemented attacks.")
    parser.add_argument("--synonym-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--synonym-max-new-tokens", type=int, default=80)
    parser.add_argument("--humarin-model", default="humarin/chatgpt_paraphraser_on_T5_base")
    parser.add_argument("--humarin-max-length", type=int, default=512,
                        help="Maximum new tokens generated per Humarin chunk.")
    parser.add_argument("--humarin-chunk-size", type=int, default=500,
                        help="Requested source-token chunk size for Humarin.")
    parser.add_argument("--device", default=None, help="Optional device for synonym LLM (e.g. cuda or cpu).")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing output files.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    records = read_records(input_path)
    if not records:
        parser.error("No records found in input path.")

    donors = {}
    if args.donor_data:
        donor_records = read_records(Path(args.donor_data).expanduser().resolve())
        donors = index_donors(donor_records, args.donor_id_field, args.donor_text_field)

    manifest = output_root / "attack_settings.csv"
    if "synonym_llm" in args.attacks:
        parser.error("synonym_llm has been removed. Use synonym_wordnet instead.")

    run_attack = build_attacks(args)

    completed = 0
    skipped = 0
    failed = 0

    for index, record in enumerate(records):
        try:
            id_field = detect_field(record, args.id_field, ID_CANDIDATES, "document ID") if args.id_field else next(
                (k for k in ID_CANDIDATES if k in record and record[k] not in (None, "")), None
            )
            doc_id = str(record[id_field]) if id_field else f"doc_{index:06d}"
            text_field = detect_field(record, args.text_field, TEXT_CANDIDATES, "text")
            original_text = record[text_field]
            if not isinstance(original_text, str):
                original_text = str(original_text)
            donor_text = None
            if args.donor_field and record.get(args.donor_field) is not None:
                donor_text = str(record[args.donor_field])
            elif not args.donor_data:
                for field in DONOR_CANDIDATES:
                    if record.get(field) is not None:
                        donor_text = str(record[field])
                        break
            donor_id_used = None
            if donor_text is None and donors:
                donor_text, donor_id_used = find_donor_text(doc_id, donors)

            record_for_attack = dict(record)
            record_for_attack["_resolved_donor_id"] = donor_id_used or ""

            for attack_name in args.attacks:
                # Full-text HELM transforms have no probability/intensity control in the supplied code.
                levels = ("full",) if attack_name in {
                    "helm_lowercase", "helm_contraction", "helm_expansion"
                } else args.intensities
                for intensity in levels:
                    effective_intensity = "medium" if intensity == "full" else intensity
                    seed = stable_seed(args.seed, doc_id, attack_name, intensity)
                    relative = Path(attack_name) / intensity / f"{safe_name(doc_id)}.txt"
                    destination = output_root / relative
                    output_key = str(relative.as_posix())

                    if destination.exists() and not args.overwrite:
                        if not output_exists_in_manifest(manifest, output_key):
                            append_manifest(manifest, {
                                "doc_id": doc_id, "attack_name": attack_name, "intensity": intensity,
                                "seed": seed, "probability": "",
                                "output_path": output_key, "status": "ok",
                                "parameters_json": json.dumps({"recovered_existing_output": True}),
                            })
                        skipped += 1
                        continue

                    try:
                        attacked, probability, params = run_attack(
                            attack_name, original_text, record_for_attack, donor_text, seed, effective_intensity
                        )
                        save_text(destination, attacked)
                        append_manifest(manifest, {
                            "doc_id": doc_id,
                            "attack_name": attack_name,
                            "intensity": intensity,
                            "seed": seed,
                            "probability": "" if probability is None else probability,
                            "output_path": output_key,
                            "status": "ok",
                            "parameters_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                        })
                        completed += 1
                        print(f"[OK] {doc_id} | {attack_name} | {intensity}")
                    except Exception as exc:
                        append_manifest(manifest, {
                            "doc_id": doc_id, "attack_name": attack_name, "intensity": intensity,
                            "seed": seed, "probability": "",
                            "output_path": output_key, "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "parameters_json": json.dumps({"seed": seed, "intensity": intensity}),
                        })
                        failed += 1
                        print(f"[ERROR] {doc_id} | {attack_name} | {intensity}: {exc}", file=sys.stderr)
        except Exception as exc:
            failed += 1
            print(f"[ERROR] record {index}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print("\nFinished.")
    print(f"Successful outputs: {completed}")
    print(f"Skipped existing outputs: {skipped}")
    print(f"Failures: {failed}")
    print(f"Output directory: {output_root}")
    print(f"Settings/checkpoint manifest: {manifest}")


if __name__ == "__main__":
    main()
