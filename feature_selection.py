"""
XGBoost-SHAP feature selection with repeated stratified K-fold CV.

Workflow
--------
1. Load a train-only TPM feature matrix and apply log2(TPM + 1).
2. Undersample the majority class within each training fold.
3. Fit MinMaxScaler on the training fold only.
4. Train XGBoost and evaluate the held-out validation fold.
5. Compute SHAP values on the validation fold.
6. Aggregate feature importance across repeats and folds.
7. Select candidates up to the cumulative SHAP cutoff.
8. Retain candidates with stability frequency >= 0.40.

The independent test set is not used during feature selection.
The classification column must be binary: 1 = target stress, 0 = non-target class.
"""

import argparse
import json
import os
from typing import List, Sequence, Tuple
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Configuration

SEED = 42
DEFAULT_STABILITY_THRESHOLD = 0.40

np.random.seed(SEED)
META_COLS = ["sample", "treatment", "classification"]


XGB_PARAMS = {
    "max_depth": 3,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "reg_lambda": 3.0,
    "reg_alpha": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
}


# Data loading


def load_train_matrix(
    data_path: str,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:

    df_train = pd.read_excel(data_path)

    missing_meta_cols = [c for c in META_COLS if c not in df_train.columns]
    if missing_meta_cols:
        raise ValueError(f"Missing required metadata columns: {missing_meta_cols}")

    feature_cols = [c for c in df_train.columns if c not in META_COLS]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns were found.")

    df_train["classification"] = df_train["classification"].astype(int)

    X_full = np.log2(df_train[feature_cols] + 1)
    y_full = df_train["classification"].copy()

    print("===================================================")
    print("Train-only matrix loaded for SHAP")
    print("===================================================")
    print("File:", data_path)
    print("Samples:", df_train.shape[0])
    print("Features:", len(feature_cols))
    print("Class distribution:")
    print(y_full.value_counts().sort_index())
    print()

    return (
        X_full.reset_index(drop=True),
        y_full.reset_index(drop=True),
        list(feature_cols),
    )


# Fold preprocessing


def undersample_training_fold(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    rng_seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:

    stress_idx = y_train_full[y_train_full == 1].index
    control_idx = y_train_full[y_train_full == 0].index

    n_sample = min(len(stress_idx), len(control_idx))
    if n_sample == 0:
        raise ValueError("One class has zero samples in the training fold.")

    sampled_stress_idx = (
        pd.Series(stress_idx).sample(n=n_sample, random_state=rng_seed).values
    )
    sampled_control_idx = (
        pd.Series(control_idx).sample(n=n_sample, random_state=rng_seed).values
    )

    sampled_idx = np.concatenate([sampled_stress_idx, sampled_control_idx])

    rng = np.random.RandomState(rng_seed)
    sampled_idx = rng.permutation(sampled_idx)

    X_train = X_train_full.loc[sampled_idx].copy()
    y_train = y_train_full.loc[sampled_idx].copy()

    return X_train, y_train


def scale_training_and_validation_fold(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    scaler = MinMaxScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index,
        columns=X_val.columns,
    )

    return X_train_scaled, X_val_scaled


# SHAP utilities


def extract_binary_shap_values(
    explainer: shap.TreeExplainer,
    X_scaled: pd.DataFrame,
) -> np.ndarray:

    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]
        elif shap_values.shape[0] == 2:
            shap_values = shap_values[1]
        else:
            raise ValueError(f"Unexpected 3D SHAP shape: {shap_values.shape}")

    if shap_values.ndim != 2 or shap_values.shape[1] != X_scaled.shape[1]:
        raise ValueError(
            "Invalid SHAP value shape. "
            f"shape={shap_values.shape}, expected_features={X_scaled.shape[1]}"
        )

    return shap_values


# Fold training and SHAP


def fit_fold_and_compute_shap(
    X_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    X_val_scaled: pd.DataFrame,
    y_val: pd.Series,
    rng_seed: int,
) -> Tuple[np.ndarray, float, float, float]:

    model = XGBClassifier(**XGB_PARAMS, random_state=rng_seed)
    model.fit(X_train_scaled, y_train, verbose=False)

    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_prob)

    explainer = shap.TreeExplainer(model)
    shap_values = extract_binary_shap_values(explainer, X_val_scaled)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    return mean_abs_shap, acc, f1, auc


# Feature importance aggregation


def build_importance_table(
    feature_names: Sequence[str],
    shap_sum: np.ndarray,
    shap_matrix: np.ndarray,
    total_models: int,
) -> pd.DataFrame:

    n_features = len(feature_names)

    mean_shap = shap_sum / total_models
    if mean_shap.sum() == 0:
        raise ValueError(
            "The sum of mean SHAP values is zero. Check the input features and model fit."
        )

    # Frequency of appearing among the Top 10% features across folds/repeats
    top_k = max(1, int(np.ceil(n_features * 0.1)))
    stability_count = np.zeros(n_features)
    for row in shap_matrix:
        top_indices = np.argsort(row)[::-1][:top_k]
        stability_count[top_indices] += 1
    stability_freq = stability_count / total_models

    df_importance = (
        pd.DataFrame(
            {
                "Feature": list(feature_names),
                "MeanAbsSHAP": mean_shap,
                "StabilityFreq_Top10pct": stability_freq,
            }
        )
        .sort_values("MeanAbsSHAP", ascending=False)
        .reset_index(drop=True)
    )

    df_importance["Rank"] = np.arange(1, len(df_importance) + 1)
    df_importance["Cumulative_ratio"] = (
        df_importance["MeanAbsSHAP"].cumsum() / df_importance["MeanAbsSHAP"].sum()
    )

    df_importance = df_importance[
        ["Rank", "Feature", "MeanAbsSHAP", "StabilityFreq_Top10pct", "Cumulative_ratio"]
    ]

    return df_importance


def select_cumulative_cutoff_genes(
    df_importance: pd.DataFrame,
    cutoff_ratio: float,
) -> pd.DataFrame:

    matches = df_importance[df_importance["Cumulative_ratio"] >= cutoff_ratio].index
    if len(matches) == 0:
        raise ValueError(f"Cumulative SHAP did not reach cutoff_ratio={cutoff_ratio}.")

    cutoff_idx = int(matches[0])
    return df_importance.iloc[: cutoff_idx + 1].copy()


def select_stable_candidates(
    cutoff_genes: pd.DataFrame,
    stability_threshold: float,
) -> pd.DataFrame:

    stable_genes = cutoff_genes[
        cutoff_genes["StabilityFreq_Top10pct"] >= stability_threshold
    ].copy()

    return stable_genes.sort_values("Rank").reset_index(drop=True)


# Output utilities


def save_cumulative_shap_plot(
    df_importance: pd.DataFrame,
    cutoff_ratio: float,
    cutoff_rank: int,
    output_path: str,
) -> None:

    plt.figure(figsize=(8, 5))
    plt.plot(df_importance["Rank"], df_importance["Cumulative_ratio"], linewidth=2)
    plt.axhline(cutoff_ratio, linestyle="--")
    plt.axvline(cutoff_rank, linestyle="--")
    plt.xlabel("Feature rank")
    plt.ylabel("Cumulative SHAP contribution")
    plt.title("Cumulative SHAP Contribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_parameter_record(
    output_path: str,
    n_splits: int,
    n_repeats: int,
    cutoff_ratio: float,
    top_k: int,
    stability_threshold: float,
) -> None:

    parameter_record = {
        "seed": SEED,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "total_models": n_splits * n_repeats,
        "cutoff_ratio": cutoff_ratio,
        "stability_top_k": top_k,
        "stability_top_fraction": 0.10,
        "stability_frequency_threshold": stability_threshold,
        "xgboost_parameters": XGB_PARAMS,
        "early_stopping": False,
        "feature_ranking_input": "train-only matrix",
        "input_transformation": "log2(TPM + 1)",
        "scaling": (
            "MinMaxScaler fitted on the undersampled training partition of each "
            "CV fold; validation partition transformed only."
        ),
        "shap_calculation": "scaled outer validation folds",
        "undersampling": "within each training fold",
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parameter_record, f, indent=4, ensure_ascii=False)


# Repeated CV feature selection


def repeated_kfold_shap(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: Sequence[str],
    out_dir: str,
    n_splits: int = 5,
    n_repeats: int = 5,
    cutoff_ratio: float = 0.90,
    stability_threshold: float = DEFAULT_STABILITY_THRESHOLD,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    os.makedirs(out_dir, exist_ok=True)

    feature_names = list(feature_names)
    n_features = len(feature_names)
    total_models = n_splits * n_repeats

    min_class_count = int(y.value_counts().min())
    if n_splits > min_class_count:
        raise ValueError(
            f"n_splits={n_splits} exceeds the minimum class size ({min_class_count})."
        )

    shap_sum = np.zeros(n_features)
    shap_matrix = []
    performance_records = []

    for r in range(n_repeats):

        print(f"\nRepeat {r + 1}/{n_repeats}")

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=SEED + r,
        )

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):

            print(f"  Fold {fold_idx + 1}/{n_splits}")

            X_train_full = X.iloc[train_idx].copy()
            y_train_full = y.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_val = y.iloc[val_idx].copy()

            rng_seed = SEED + (r * 100) + fold_idx

            # 1) Undersample the training fold
            X_train, y_train = undersample_training_fold(
                X_train_full=X_train_full,
                y_train_full=y_train_full,
                rng_seed=rng_seed,
            )

            # 2) Fit MinMaxScaler on the training fold only
            X_train_scaled, X_val_scaled = scale_training_and_validation_fold(
                X_train=X_train,
                X_val=X_val,
            )

            # 3) Train XGBoost, evaluate validation data, and compute SHAP
            mean_abs_shap, acc, f1, auc = fit_fold_and_compute_shap(
                X_train_scaled=X_train_scaled,
                y_train=y_train,
                X_val_scaled=X_val_scaled,
                y_val=y_val,
                rng_seed=rng_seed,
            )

            performance_records.append(
                {
                    "Repeat": r + 1,
                    "Fold": fold_idx + 1,
                    "Train_positive_after_undersampling": int((y_train == 1).sum()),
                    "Train_negative_after_undersampling": int((y_train == 0).sum()),
                    "Validation_positive": int((y_val == 1).sum()),
                    "Validation_negative": int((y_val == 0).sum()),
                    "Accuracy": acc,
                    "F1": f1,
                    "AUC": auc,
                }
            )

            shap_sum += mean_abs_shap
            shap_matrix.append(mean_abs_shap)

    shap_matrix = np.array(shap_matrix)

    # Build importance table and select cumulative-SHAP candidates

    df_importance = build_importance_table(
        feature_names=feature_names,
        shap_sum=shap_sum,
        shap_matrix=shap_matrix,
        total_models=total_models,
    )

    df_perf = pd.DataFrame(performance_records)
    cutoff_genes = select_cumulative_cutoff_genes(df_importance, cutoff_ratio)
    stable_genes = select_stable_candidates(
        cutoff_genes=cutoff_genes,
        stability_threshold=stability_threshold,
    )
    cutoff_rank = int(cutoff_genes["Rank"].iloc[-1])
    top_k = max(1, int(np.ceil(n_features * 0.1)))

    # Save outputs

    importance_path = os.path.join(out_dir, "Repeated5fold_SHAP_importance.xlsx")
    perf_path = os.path.join(out_dir, "Repeated5fold_performance.xlsx")
    cutoff_path = os.path.join(out_dir, "SHAP_90pct_candidate_genes.xlsx")
    stable_path = os.path.join(out_dir, "SHAP_stable_candidate_genes.xlsx")
    params_path = os.path.join(out_dir, "SHAP_feature_selection_parameters.json")
    curve_path = os.path.join(out_dir, "Cumulative_SHAP_curve.png")

    df_importance.to_excel(importance_path, index=False)
    df_perf.to_excel(perf_path, index=False)
    cutoff_genes.to_excel(cutoff_path, index=False)
    stable_genes.to_excel(stable_path, index=False)
    save_parameter_record(
        params_path,
        n_splits,
        n_repeats,
        cutoff_ratio,
        top_k,
        stability_threshold,
    )
    save_cumulative_shap_plot(df_importance, cutoff_ratio, cutoff_rank, curve_path)

    print("\nPerformance summary:")
    print(df_perf[["Accuracy", "F1", "AUC"]].describe())

    print(f"\n90% cumulative SHAP candidates: {len(cutoff_genes)}")
    print(
        f"Stable candidates (frequency >= {stability_threshold:.2f}): "
        f"{len(stable_genes)}"
    )

    print("\nSaved files:")
    for path in (
        importance_path,
        perf_path,
        cutoff_path,
        stable_path,
        params_path,
        curve_path,
    ):
        print(path)

    return df_importance, df_perf, cutoff_genes, stable_genes


# Command-line interface


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated stratified K-fold XGBoost-SHAP feature selection "
            "on a train-only TPM feature matrix."
        )
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help=(
            "Path to the input Excel file. Required metadata columns: "
            "sample, treatment, classification."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where feature-selection results will be saved.",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=DEFAULT_STABILITY_THRESHOLD,
        help="Minimum Top-10%% stability frequency (default: 0.40).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X_train, y_train, feature_names = load_train_matrix(args.data_path)

    _, _, cutoff_genes, stable_genes = repeated_kfold_shap(
        X=X_train,
        y=y_train,
        feature_names=feature_names,
        out_dir=args.out_dir,
        n_splits=5,
        n_repeats=5,
        cutoff_ratio=0.90,
        stability_threshold=args.stability_threshold,
    )

    print(f"\n90% cumulative SHAP cutoff: {len(cutoff_genes)} features")
    print(
        f"Stability-filtered candidates: {len(stable_genes)} features "
        f"(threshold >= {args.stability_threshold:.2f})"
    )
    print("Feature selection completed.")


if __name__ == "__main__":
    main()
