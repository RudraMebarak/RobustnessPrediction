"""
Minimal web UI for the six watermark-robustness models.

Run:
    venv/bin/python ui/app.py
then open http://127.0.0.1:8000
"""

import json
import math
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import pandas as pd
import spacy

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
MODEL_DIR = SRC / "trained_robustness_models" / "models"
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

sys.path.insert(0, str(SRC))
from feature_extraction import extract_features  # noqa: E402

# Display order: (experiment name, short label)
EXPERIMENTS = [
    ("absolute_z_initial_only", "Initial Z only"),
    ("absolute_z_features_only", "Features only"),
    ("absolute_z_initial_plus_features", "Initial Z + features"),
    ("relative_z_initial_only", "Initial Z only"),
    ("relative_z_features_only", "Features only"),
    ("relative_z_initial_plus_features", "Initial Z + features"),
]


ATTACK_MODEL_DIR = SRC / "attack_conditioned_models" / "models"

ATTACK_EXPERIMENTS = [
    ("absolute_initial_plus_attack", "Initial Z only"),
    ("absolute_features_attack", "Features only"),
    ("absolute_initial_features_attack", "Initial Z + features"),
    ("relative_initial_plus_attack", "Initial Z only"),
    ("relative_features_attack", "Features only"),
    ("relative_initial_features_attack", "Initial Z + features"),
]

# Attack families and the intensities seen in training.
ATTACKS = {
    "copy_paste": ["low", "medium", "high"],
    "helm_contraction": ["full"],
    "helm_expansion": ["full"],
    "helm_lowercase": ["full"],
    "helm_misspelling": ["low", "medium", "high"],
    "helm_typo": ["low", "medium", "high"],
    "humarin": ["low", "medium", "high"],
    "synonym_wordnet": ["low", "medium", "high"],
}


def load_models(model_dir, experiments, group):
    models = {}
    for name, label in experiments:
        meta = json.loads((model_dir / f"{name}.json").read_text())
        pipeline = joblib.load(model_dir / f"{name}.joblib")
        names_in = getattr(pipeline, "feature_names_in_", None)
        features = (list(names_in) if names_in is not None
                    else meta["input_features"])
        models[name] = {
            "label": label,
            "group": group,
            "pipeline": pipeline,
            "features": features,
            "algorithm": meta["selected_model"],
            "kind": "absolute" if name.startswith("absolute") else "relative",
            "test": meta.get("test_metrics", {}),
        }
    return models


print("Loading spaCy + models...")
NLP = spacy.load("en_core_web_sm")
NLP.max_length = 5_000_000
MODELS = {
    **load_models(MODEL_DIR, EXPERIMENTS, "general"),
    **load_models(ATTACK_MODEL_DIR, ATTACK_EXPERIMENTS, "attack"),
}
print(f"Loaded {len(MODELS)} models.")

CATEGORICAL = {"attack_name", "intensity"}


def clean(value):
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else value


def predict(text, initial_z, attack_name=None, intensity=None):
    features = extract_features(text, NLP)
    row = dict(features)
    if initial_z is not None:
        row["initial_z_score"] = initial_z
    if attack_name:
        if attack_name not in ATTACKS:
            raise ValueError(f"Unknown attack: {attack_name}")
        if intensity not in ATTACKS[attack_name]:
            intensity = ATTACKS[attack_name][0]
        row["attack_name"] = attack_name
        row["intensity"] = intensity

    results = []
    for name, m in MODELS.items():
        needs_z = "initial_z_score" in m["features"]
        entry = {
            "name": name,
            "label": m["label"],
            "group": m["group"],
            "kind": m["kind"],
            "algorithm": m["algorithm"],
            "test_mae": clean(m["test"].get("MAE")),
            "test_r2": clean(m["test"].get("R2")),
            "prediction": None,
        }
        if m["group"] == "attack" and not attack_name:
            entry["skipped"] = "select an attack"
        elif needs_z and initial_z is None:
            entry["skipped"] = "needs initial Z-score"
        else:
            X = pd.DataFrame([[row.get(f) for f in m["features"]]],
                             columns=m["features"])
            numeric = [f for f in m["features"] if f not in CATEGORICAL]
            X[numeric] = X[numeric].astype(float)
            entry["prediction"] = clean(m["pipeline"].predict(X)[0])
        results.append(entry)

    return {
        "features": {k: clean(v) for k, v in features.items()},
        "results": results,
        "initial_z": initial_z,
        "attack": {"name": attack_name, "intensity": intensity} if attack_name else None,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.read_text(), "text/html; charset=utf-8")
        elif self.path == "/attacks":
            self._send(200, json.dumps(ATTACKS), "application/json")
        else:
            self._send(404, "Not found", "text/plain")

    def do_POST(self):
        if self.path != "/predict":
            return self._send(404, "Not found", "text/plain")
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = (payload.get("text") or "").strip()
            if not text:
                raise ValueError("Please enter some text.")
            z_raw = payload.get("initial_z")
            initial_z = None if z_raw in (None, "") else float(z_raw)
            out = predict(text, initial_z,
                          payload.get("attack_name") or None,
                          payload.get("intensity") or None)
            self._send(200, json.dumps(out), "application/json")
        except Exception as e:  # report errors to the UI
            self._send(400, json.dumps({"error": str(e)}), "application/json")

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Open http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
