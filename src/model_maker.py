
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

# Automatically locate the two source CSV files.
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
        "Could not locate original_features_combined.csv and "
        "merged_robustness_dataset.csv. Set DATA_DIR manually."
    )

ORIGINAL_PATH = DATA_DIR / "original_features_combined.csv"
MERGED_PATH = DATA_DIR / "merged_robustness_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "trained_robustness_models"
MODEL_DIR = OUTPUT_DIR / "models"
RESULT_DIR = OUTPUT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("KGW ROBUSTNESS PREDICTION — MODEL TRAINING")
print("=" * 65)

orig_df = pd.read_csv(ORIGINAL_PATH, low_memory=False)
merged_df = pd.read_csv(MERGED_PATH, low_memory=False)

print("\nOriginal feature table:", orig_df.shape)
print("Merged attack table:", merged_df.shape)

required_original = {"doc_id"}
required_attack = {
    "doc_id",
    "initial_z_score",
    "z_score_drop",
    "relative_z_drop_pct",
    "attack_name",
}

if not required_original.issubset(orig_df.columns):
    raise ValueError(
        f"Missing original columns: {required_original - set(orig_df.columns)}"
    )

if not required_attack.issubset(merged_df.columns):
    raise ValueError(
        f"Missing attack columns: {required_attack - set(merged_df.columns)}"
    )

if orig_df["doc_id"].duplicated().any():
    raise ValueError("original_features_combined.csv has duplicate doc_id values.")


# ============================================================
# SELECT ORIGINAL-TEXT FEATURES
# ============================================================

# These columns must never become original-text predictors.
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
    col
    for col in orig_df.columns
    if col not in metadata_exclude
    and pd.api.types.is_numeric_dtype(orig_df[col])
]

if not feature_cols:
    raise ValueError("No numeric original-text features found.")

print(f"\nSelected original-text features: {len(feature_cols)}")

if len(feature_cols) != 42:
    warnings.warn(
        f"Expected 42 original-text features based on the EDA, "
        f"but found {len(feature_cols)}. Check the feature table."
    )

print(feature_cols)

# Convert features to numeric and replace infinite values.
for col in feature_cols:
    orig_df[col] = pd.to_numeric(orig_df[col], errors="coerce")

orig_df[feature_cols] = orig_df[feature_cols].replace(
    [np.inf, -np.inf], np.nan
)


# ============================================================
# BUILD DOCUMENT-LEVEL TARGETS
# ============================================================

# First average attack configurations within each attack family.
family_doc = (
    merged_df
    .groupby(["doc_id", "attack_name"], as_index=False)
    .agg(
        family_absolute_drop=("z_score_drop", "mean"),
        family_relative_drop=("relative_z_drop_pct", "mean"),
    )
)

# Then give each attack family equal weight.
equal_family = (
    family_doc
    .groupby("doc_id", as_index=False)
    .agg(
        equal_family_absolute_drop=("family_absolute_drop", "mean"),
        equal_family_relative_drop=("family_relative_drop", "mean"),
    )
)

# Initial Z-score is constant per document in the experiment.
initial_scores = (
    merged_df
    .groupby("doc_id", as_index=False)
    .agg(initial_z_score=("initial_z_score", "mean"))
)

initial_std = (
    merged_df
    .groupby("doc_id")["initial_z_score"]
    .std()
    .fillna(0)
)

if initial_std.max() > 1e-5:
    warnings.warn(
        "Initial Z-score varies within some documents. "
        "The document-level mean is being used."
    )

# One row per original document.
doc_model = (
    orig_df[["doc_id"] + feature_cols]
    .merge(initial_scores, on="doc_id", how="inner", validate="one_to_one")
    .merge(equal_family, on="doc_id", how="inner", validate="one_to_one")
)

print("\nDocument-level modelling dataset:", doc_model.shape)
print("Unique documents:", doc_model["doc_id"].nunique())

targets = [
    "equal_family_absolute_drop",
    "equal_family_relative_drop",
]

print("\nTarget statistics:")
print(doc_model[targets].describe().round(4))


# ============================================================
# DEFINE THE SIX EXPERIMENTS
# ============================================================

experiments = {
    "absolute_z_initial_only": {
        "target": "equal_family_absolute_drop",
        "features": ["initial_z_score"],
    },

    "relative_z_initial_only": {
        "target": "equal_family_relative_drop",
        "features": ["initial_z_score"],
    },

    "absolute_z_initial_plus_features": {
        "target": "equal_family_absolute_drop",
        "features": ["initial_z_score"] + feature_cols,
    },

    "relative_z_initial_plus_features": {
        "target": "equal_family_relative_drop",
        "features": ["initial_z_score"] + feature_cols,
    },

    "absolute_z_features_only": {
        "target": "equal_family_absolute_drop",
        "features": feature_cols,
    },

    "relative_z_features_only": {
        "target": "equal_family_relative_drop",
        "features": feature_cols,
    },
}


# ============================================================
# DEFINE CANDIDATE MODELS
# ============================================================

def build_candidates():
    """
    Each candidate is a complete preprocessing + regression pipeline.
    """

    return {
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                objective="reg:squarederror",
                n_estimators=350,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=3,
                reg_lambda=2.0,
                reg_alpha=0.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),

        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
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
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.01,
                learning_rate_init=0.001,
                batch_size=32,
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
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
# TRAIN, SELECT AND EXPORT
# ============================================================

all_cv_results = []
final_results = []
all_test_predictions = []

for experiment_name, config in experiments.items():

    print("\n" + "=" * 65)
    print("EXPERIMENT:", experiment_name)
    print("=" * 65)

    target = config["target"]
    selected_features = config["features"]

    # Remove rows without a valid target.
    data = doc_model.dropna(subset=[target]).copy()

    X = data[selected_features].copy()
    y = pd.to_numeric(data[target], errors="coerce")
    groups = data["doc_id"]

    valid = y.notna() & np.isfinite(y)

    X = X.loc[valid]
    y = y.loc[valid]
    groups = groups.loc[valid]

    # One grouped holdout split. Test data is never used for selection.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    assert set(groups_train).isdisjoint(set(groups.iloc[test_idx]))

    print("Features:", len(selected_features))
    print("Training documents:", len(X_train))
    print("Testing documents:", len(X_test))

    # Grouped CV is used only on the training documents.
    n_splits = min(N_SPLITS, groups_train.nunique())

    if n_splits < 2:
        raise ValueError("Not enough training documents for grouped CV.")

    cv = GroupKFold(n_splits=n_splits)

    candidate_results = []

    candidates = build_candidates()

    for model_name, pipeline in candidates.items():

        print(f"\nEvaluating {model_name}...")

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=1,
        )

        mean_cv_mae = -cv_scores.mean()
        std_cv_mae = cv_scores.std()

        candidate_results.append({
            "experiment": experiment_name,
            "target": target,
            "model": model_name,
            "cv_mae_mean": mean_cv_mae,
            "cv_mae_std": std_cv_mae,
        })

        print(
            f"CV MAE: {mean_cv_mae:.5f} "
            f"+/- {std_cv_mae:.5f}"
        )

    # Choose the model with the lowest mean grouped-CV MAE.
    candidate_df = pd.DataFrame(candidate_results)
    candidate_df = candidate_df.sort_values("cv_mae_mean")

    all_cv_results.extend(candidate_df.to_dict("records"))

    best_model_name = candidate_df.iloc[0]["model"]

    print("\nSelected model:", best_model_name)

    # Refit selected model on all training documents.
    best_pipeline = build_candidates()[best_model_name]

    best_pipeline.fit(X_train, y_train)

    # Final held-out evaluation.
    test_predictions = best_pipeline.predict(X_test)

    test_metrics = calculate_metrics(y_test, test_predictions)

    print("\nHeld-out test results:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.5f}")

    # Export selected pipeline.
    model_path = MODEL_DIR / f"{experiment_name}.joblib"

    joblib.dump(best_pipeline, model_path)

    # Save metadata for later UI integration.
    metadata = {
        "experiment": experiment_name,
        "target": target,
        "selected_model": best_model_name,
        "input_features": selected_features,
        "number_of_features": len(selected_features),
        "selection_metric": "5-fold GroupKFold MAE",
        "cv_mae": float(candidate_df.iloc[0]["cv_mae_mean"]),
        "cv_mae_std": float(candidate_df.iloc[0]["cv_mae_std"]),
        "test_metrics": {
            k: float(v) if np.isfinite(v) else None
            for k, v in test_metrics.items()
        },
        "random_state": RANDOM_STATE,
    }

    metadata_path = MODEL_DIR / f"{experiment_name}.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    final_results.append({
        "experiment": experiment_name,
        "target": target,
        "selected_model": best_model_name,
        "number_of_features": len(selected_features),
        **test_metrics,
        "model_path": str(model_path),
    })

    prediction_df = pd.DataFrame({
        "doc_id": data.iloc[test_idx]["doc_id"].values,
        "actual": y_test.values,
        "predicted": test_predictions,
        "experiment": experiment_name,
    })

    all_test_predictions.append(prediction_df)

    print("Exported:", model_path)


# ============================================================
# SAVE ALL RESULTS
# ============================================================

cv_results_df = pd.DataFrame(all_cv_results)
final_results_df = pd.DataFrame(final_results)
predictions_df = pd.concat(all_test_predictions, ignore_index=True)

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

# Save exact feature schema.
with open(OUTPUT_DIR / "feature_schema.json", "w") as f:
    json.dump(feature_cols, f, indent=4)

print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)

print("\nFinal selected models:")
print(
    final_results_df[
        ["experiment", "selected_model", "MAE", "RMSE", "R2", "Spearman"]
    ].to_string(index=False)
)

print("\nAll outputs saved to:")
print(OUTPUT_DIR.resolve())
