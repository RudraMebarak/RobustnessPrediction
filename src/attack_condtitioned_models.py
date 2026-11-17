
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

PROJECT_ROOT = Path(__file__).resolve().parent

candidate_dirs = [
    PROJECT_ROOT,
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "src" / "data",
]

DATA_DIR = next(
    (
        p for p in candidate_dirs
        if (p / "original_features_combined.csv").exists()
        and (p / "merged_robustness_dataset.csv").exists()
    ),
    None
)

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find original_features_combined.csv and "
        "merged_robustness_dataset.csv. Set DATA_DIR manually."
    )

ORIGINAL_PATH = DATA_DIR / "original_features_combined.csv"
MERGED_PATH = DATA_DIR / "merged_robustness_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "attack_conditioned_models"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ATTACK-CONDITIONED KGW PREDICTION — SIX EXPERIMENTS")
print("=" * 70)

orig_df = pd.read_csv(ORIGINAL_PATH, low_memory=False)
merged_df = pd.read_csv(MERGED_PATH, low_memory=False)

print("Original features:", orig_df.shape)
print("Attack dataset:", merged_df.shape)

if orig_df["doc_id"].duplicated().any():
    raise ValueError("Duplicate doc_id values in original feature table.")

required_attack = {
    "doc_id",
    "attack_name",
    "intensity",
    "initial_z_score",
    "z_score_drop",
    "relative_z_drop_pct",
}

missing = required_attack - set(merged_df.columns)

if missing:
    raise ValueError(f"Missing attack columns: {sorted(missing)}")


# ============================================================
# ORIGINAL-TEXT FEATURE SELECTION
# ============================================================

metadata_exclude = {
    "doc_id",
    "file_name",
    "relative_path",
    "domain",
    "model_name",
    "attack_name",
    "intensity",
    "output_path",
    "prompt_id",

    "initial_num_tokens_scored",
    "initial_num_green_tokens",
    "initial_green_fraction",
    "initial_z_score",
    "initial_p_value",
    "initial_detected",
    "initial_detection_threshold",

    "baseline_z_score",
    "baseline_p_value",
    "baseline_detected",

    "z_score_drop",
    "relative_z_drop_pct",
    "watermark_survived",
    "watermark_removed",
    "semantic_similarity_cosine",
}

feature_cols = [
    col for col in orig_df.columns
    if col not in metadata_exclude
    and pd.api.types.is_numeric_dtype(orig_df[col])
]

if not feature_cols:
    raise ValueError("No numeric original-text features found.")

print(f"\nOriginal-text features selected: {len(feature_cols)}")

if len(feature_cols) != 42:
    warnings.warn(
        f"Expected 42 original-text features, found {len(feature_cols)}."
    )

print(feature_cols)

feature_source = orig_df[["doc_id"] + feature_cols].copy()

for col in feature_cols:
    feature_source[col] = pd.to_numeric(
        feature_source[col], errors="coerce"
    )

feature_source[feature_cols] = feature_source[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)


# ============================================================
# MERGE ORIGINAL FEATURES INTO ATTACK DATA
# ============================================================

# Remove any existing copies to avoid duplicate/suffixed columns.
attack_data = merged_df.drop(
    columns=[c for c in feature_cols if c in merged_df.columns],
    errors="ignore",
).merge(
    feature_source,
    on="doc_id",
    how="left",
    validate="many_to_one",
)

numeric_cols = ["initial_z_score"] + feature_cols
categorical_cols = ["attack_name", "intensity"]

for col in numeric_cols:
    attack_data[col] = pd.to_numeric(
        attack_data[col], errors="coerce"
    )

attack_data[numeric_cols] = attack_data[numeric_cols].replace(
    [np.inf, -np.inf], np.nan
)

for col in categorical_cols:
    attack_data[col] = (
        attack_data[col].astype("string").str.strip()
    )

targets = [
    "z_score_drop",
    "relative_z_drop_pct",
]

for col in targets:
    attack_data[col] = pd.to_numeric(
        attack_data[col], errors="coerce"
    )

attack_data[targets] = attack_data[targets].replace(
    [np.inf, -np.inf], np.nan
)

# Require valid identifiers and attack metadata.
attack_data = attack_data.dropna(
    subset=["doc_id", "attack_name", "intensity"]
).copy()

print("\nFinal attack-level dataset:", attack_data.shape)
print("Unique documents:", attack_data["doc_id"].nunique())

print("\nAttack configurations:")
print(
    attack_data.groupby(["attack_name", "intensity"])
    .size()
    .to_string()
)


# ============================================================
# DEFINE THE SIX EXPERIMENTS
# ============================================================

experiments = {
    "absolute_initial_plus_attack": {
        "target": "z_score_drop",
        "numeric": ["initial_z_score"],
    },

    "relative_initial_plus_attack": {
        "target": "relative_z_drop_pct",
        "numeric": ["initial_z_score"],
    },

    "absolute_initial_features_attack": {
        "target": "z_score_drop",
        "numeric": ["initial_z_score"] + feature_cols,
    },

    "relative_initial_features_attack": {
        "target": "relative_z_drop_pct",
        "numeric": ["initial_z_score"] + feature_cols,
    },

    "absolute_features_attack": {
        "target": "z_score_drop",
        "numeric": feature_cols,
    },

    "relative_features_attack": {
        "target": "relative_z_drop_pct",
        "numeric": feature_cols,
    },
}


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(numeric_features, scale_numeric=False):

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )


# ============================================================
# CANDIDATE MODELS
# ============================================================

def build_candidates(numeric_features):

    return {
        "XGBoost": Pipeline([
            ("preprocessor", build_preprocessor(
                numeric_features, scale_numeric=False
            )),
            ("model", XGBRegressor(
                objective="reg:squarederror",
                n_estimators=400,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=3,
                reg_lambda=2.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),

        "RandomForest": Pipeline([
            ("preprocessor", build_preprocessor(
                numeric_features, scale_numeric=False
            )),
            ("model", RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "NeuralNetwork": Pipeline([
            ("preprocessor", build_preprocessor(
                numeric_features, scale_numeric=True
            )),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.01,
                learning_rate_init=0.001,
                batch_size=128,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=35,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    rho = spearmanr(y_true, y_pred).statistic

    if not np.isfinite(rho):
        rho = np.nan

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Spearman": rho,
    }


# ============================================================
# COMMON GROUPED HOLDOUT SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

# Split by document, never by individual attack row.
train_idx, test_idx = next(
    splitter.split(
        attack_data,
        groups=attack_data["doc_id"],
    )
)

train_data = attack_data.iloc[train_idx].copy()
test_data = attack_data.iloc[test_idx].copy()

train_docs = set(train_data["doc_id"].unique())
test_docs = set(test_data["doc_id"].unique())

assert train_docs.isdisjoint(test_docs)

print("\nGrouped holdout:")
print("Training documents:", len(train_docs))
print("Testing documents:", len(test_docs))
print("Training rows:", len(train_data))
print("Testing rows:", len(test_data))


# ============================================================
# TRAIN ALL SIX EXPERIMENTS
# ============================================================

all_cv_results = []
final_results = []
all_predictions = []

for experiment_name, config in experiments.items():

    print("\n" + "=" * 70)
    print("EXPERIMENT:", experiment_name)
    print("=" * 70)

    target = config["target"]
    numeric_features = config["numeric"]

    input_features = numeric_features + categorical_cols

    train_subset = train_data.dropna(subset=[target]).copy()
    test_subset = test_data.dropna(subset=[target]).copy()

    X_train = train_subset[input_features]
    y_train = train_subset[target]

    X_test = test_subset[input_features]
    y_test = test_subset[target]

    groups_train = train_subset["doc_id"]

    cv = GroupKFold(
        n_splits=min(N_SPLITS, groups_train.nunique())
    )

    candidate_results = []

    for model_name, pipeline in build_candidates(
        numeric_features
    ).items():

        print(f"\nEvaluating {model_name}...")

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )

        cv_mae = -scores.mean()
        cv_std = scores.std()

        print(f"CV MAE: {cv_mae:.5f} +/- {cv_std:.5f}")

        candidate_results.append({
            "experiment": experiment_name,
            "target": target,
            "model": model_name,
            "cv_mae_mean": cv_mae,
            "cv_mae_std": cv_std,
        })

    candidate_df = pd.DataFrame(candidate_results)
    candidate_df = candidate_df.sort_values("cv_mae_mean")

    all_cv_results.extend(candidate_df.to_dict("records"))

    best_model_name = candidate_df.iloc[0]["model"]

    print("\nSelected model:", best_model_name)

    # Refit on the complete training split.
    best_pipeline = build_candidates(
        numeric_features
    )[best_model_name]

    best_pipeline.fit(X_train, y_train)

    # Evaluate once on held-out documents.
    predictions = best_pipeline.predict(X_test)

    metrics = calculate_metrics(y_test, predictions)

    print("\nHeld-out test results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.5f}")

    # Export model.
    model_path = (
        MODEL_DIR / f"{experiment_name}.joblib"
    )

    joblib.dump(best_pipeline, model_path)

    # Export metadata for future UI integration.
    metadata = {
        "experiment": experiment_name,
        "target": target,
        "selected_model": best_model_name,
        "numeric_features": numeric_features,
        "categorical_features": categorical_cols,
        "initial_z_score_used": "initial_z_score" in numeric_features,
        "attack_family_used": True,
        "attack_intensity_used": True,
        "selection_metric": "5-fold GroupKFold MAE",
        "cv_mae": float(candidate_df.iloc[0]["cv_mae_mean"]),
        "cv_mae_std": float(candidate_df.iloc[0]["cv_mae_std"]),
        "test_metrics": {
            k: float(v) if np.isfinite(v) else None
            for k, v in metrics.items()
        },
        "random_state": RANDOM_STATE,
    }

    metadata_path = (
        MODEL_DIR / f"{experiment_name}.json"
    )

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    final_results.append({
        "experiment": experiment_name,
        "target": target,
        "selected_model": best_model_name,
        "number_of_numeric_features": len(numeric_features),
        **metrics,
        "model_path": str(model_path),
    })

    # Save held-out predictions with attack metadata.
    prediction_df = test_subset[
        ["doc_id", "attack_name", "intensity"]
    ].copy()

    prediction_df["experiment"] = experiment_name
    prediction_df["actual"] = y_test.values
    prediction_df["predicted"] = predictions

    all_predictions.append(prediction_df)

    print("Exported:", model_path)


# ============================================================
# EXPORT COMPARISON TABLES
# ============================================================

cv_results_df = pd.DataFrame(all_cv_results)
final_results_df = pd.DataFrame(final_results)
predictions_df = pd.concat(all_predictions, ignore_index=True)

cv_results_df.to_csv(
    RESULT_DIR / "candidate_cv_comparison.csv",
    index=False,
)

final_results_df.to_csv(
    RESULT_DIR / "final_model_results.csv",
    index=False,
)

predictions_df.to_csv(
    RESULT_DIR / "heldout_test_predictions.csv",
    index=False,
)

with open(OUTPUT_DIR / "feature_schema.json", "w") as f:
    json.dump({
        "original_text_features": feature_cols,
        "attack_inputs": categorical_cols,
    }, f, indent=4)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ALL SIX ATTACK-CONDITIONED EXPERIMENTS COMPLETE")
print("=" * 70)

print(
    final_results_df[
        [
            "experiment",
            "selected_model",
            "MAE",
            "RMSE",
            "R2",
            "Spearman",
        ]
    ].to_string(index=False)
)

print("\nOutputs saved to:")
print(OUTPUT_DIR.resolve())
