"""
Top-N logistic regression evaluation using a curated final feature ranking.

Workflow
--------
1. Load the finalized ranked feature table.
2. Use the existing ranking order without reapplying SHAP cutoff or stability filtering.
3. Validate ranked features against the expression matrix.
4. Evaluate requested Top-N feature sets with stratified K-fold cross-validation.
5. Fit each Top-N logistic regression model on the full training set.
6. Evaluate each model on the fixed independent test set.
7. Save metrics, predictions, coefficients, fitted models, scalers, and parameters.

"""

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Configuration

SEED = 42
np.random.seed(SEED)

# Top-N checkpoints for marker-panel evaluation

BASE_TOPN_COUNTS = [3, 5, 7, 9, 10, 15, 20]


# Curated final feature ranking


def load_final_features(
    ranking_path,
):

    ranking_df = pd.read_excel(ranking_path)

    ranking_df.columns = ranking_df.columns.astype(str).str.strip()

    if "Feature" not in ranking_df.columns:
        raise ValueError(
            "The ranking file must contain a 'Feature' column.\n"
            f"Available columns: {ranking_df.columns.tolist()}"
        )

    ranking_df["Feature"] = ranking_df["Feature"].astype(str).str.strip()

    ranking_df = ranking_df[
        ~ranking_df["Feature"].isin(["", "nan", "None", "<NA>"])
    ].copy()

    numeric_cols = [
        "Final_rank",
        "Rank",
        "MeanAbsSHAP",
        "StabilityFreq_Top10pct",
        "Cumulative_ratio",
    ]

    for col in numeric_cols:
        if col in ranking_df.columns:
            ranking_df[col] = pd.to_numeric(
                ranking_df[col],
                errors="coerce",
            )

    ranking_columns = [
        "Final_rank",
        "Rank",
    ]

    ranking_col = next(
        (col for col in ranking_columns if col in ranking_df.columns),
        None,
    )

    if ranking_col is not None:

        if ranking_df[ranking_col].isna().any():
            raise ValueError(
                f"The ranking column '{ranking_col}' contains missing values."
            )

        ranking_df = ranking_df.sort_values(ranking_col).reset_index(drop=True).copy()

    elif "MeanAbsSHAP" in ranking_df.columns:

        if ranking_df["MeanAbsSHAP"].isna().any():
            raise ValueError("The MeanAbsSHAP column contains missing values.")

        ranking_df = (
            ranking_df.sort_values(
                "MeanAbsSHAP",
                ascending=False,
            )
            .reset_index(drop=True)
            .copy()
        )

        ranking_col = "MeanAbsSHAP"

    else:
        raise ValueError(
            "The ranking file must contain one of: " "Final_rank, Rank, or MeanAbsSHAP."
        )

    duplicated_features = ranking_df.loc[
        ranking_df["Feature"].duplicated(keep=False),
        "Feature",
    ].tolist()

    if duplicated_features:
        raise ValueError(
            "Duplicate Feature IDs were found in the ranking file.\n"
            f"Examples: {duplicated_features[:20]}"
        )

    ranking_df["FinalRank"] = np.arange(
        1,
        len(ranking_df) + 1,
    )

    print("===================================================")
    print("Final feature ranking")
    print("===================================================")
    print("Ranking file:", ranking_path)
    print("Ranking column:", ranking_col)
    print("Number of candidate features:", len(ranking_df))
    print()

    return ranking_df


# Top-N logistic regression evaluation


def evaluate_logistic_with_fixed_test(
    data_path,
    ranking_path,
    fixed_test_path,
    out_dir,
    feature_counts=None,
    n_splits=5,
):

    os.makedirs(out_dir, exist_ok=True)

    # Data loading

    df = pd.read_excel(data_path)
    fixed_test = pd.read_excel(fixed_test_path)

    meta_cols = ["sample", "treatment", "classification"]

    missing_meta_cols = [c for c in meta_cols if c not in df.columns]
    if missing_meta_cols:
        raise ValueError(f"Missing required metadata columns: {missing_meta_cols}")

    if "sample" not in fixed_test.columns:
        raise ValueError("The fixed-test file must contain a 'sample' column.")

    # Normalize sample identifiers
    df["sample"] = df["sample"].astype(str).str.strip()
    fixed_test["sample"] = fixed_test["sample"].astype(str).str.strip()

    df["classification"] = df["classification"].astype(int)
    df["treatment"] = df["treatment"].astype(str).str.strip()

    feature_cols = [c for c in df.columns if c not in meta_cols]

    if len(feature_cols) == 0:
        raise ValueError(
            "No feature columns were found. Check the metadata-column specification."
        )

    # Validate fixed test-sample matching
    fixed_test_samples = set(fixed_test["sample"])
    data_samples = set(df["sample"])

    missing_test_samples = sorted(fixed_test_samples - data_samples)

    if missing_test_samples:
        raise ValueError(
            "Some fixed-test samples are missing from the input data.\n"
            f"Number of missing samples: {len(missing_test_samples)}\n"
            f"Examples: {missing_test_samples[:20]}"
        )

    # Split fixed test samples from the training data
    df_train = df[~df["sample"].isin(fixed_test_samples)].copy()
    df_test = df[df["sample"].isin(fixed_test_samples)].copy()

    if df_train.shape[0] == 0:
        raise ValueError("No training samples remain after the fixed test split.")

    if df_test.shape[0] == 0:
        raise ValueError("No samples from the fixed-test file matched the input data.")

    # Save the train/test split record
    split_record = df[["sample", "treatment", "classification"]].copy()
    split_record["split"] = np.where(
        split_record["sample"].isin(fixed_test_samples),
        "test",
        "train",
    )

    split_record.to_csv(
        os.path.join(out_dir, "train_test_split_used.csv"),
        index=False,
    )

    print("===================================================")
    print("Train/test split")
    print("===================================================")
    print("Data file:", data_path)
    print("Fixed test file:", fixed_test_path)
    print("Training samples:", df_train.shape[0])
    print("Test samples:", df_test.shape[0])
    print("Train class distribution:")
    print(df_train["classification"].value_counts())
    print("Test class distribution:")
    print(df_test["classification"].value_counts())
    print()

    # Feature matrix

    X_train_full = np.log2(df_train[feature_cols] + 1)
    y_train_full = df_train["classification"].copy().reset_index(drop=True)

    X_test = np.log2(df_test[feature_cols] + 1)
    y_test = df_test["classification"].copy().reset_index(drop=True)

    X_train_full = X_train_full.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    train_meta = df_train[["sample", "classification", "treatment"]].copy()
    test_meta = df_test[["sample", "classification", "treatment"]].copy()

    train_meta = train_meta.reset_index(drop=True)
    test_meta = test_meta.reset_index(drop=True)

    # Load final ranked features

    final_ranking_df = load_final_features(
        ranking_path=ranking_path,
    )

    final_ranking_df.to_excel(
        os.path.join(
            out_dir,
            "Final_feature_ranking_used.xlsx",
        ),
        index=False,
    )

    final_features = final_ranking_df["Feature"].tolist()

    if len(final_features) == 0:
        raise ValueError("No candidate features were found in the ranking file.")

    # Determine Top-N feature counts

    if feature_counts is None:
        max_features = len(final_features)

        feature_counts = [n for n in BASE_TOPN_COUNTS if n <= max_features]

        if max_features > 20:
            feature_counts.extend(range(30, max_features + 1, 10))

        feature_counts.append(max_features)
        feature_counts = sorted(set(feature_counts))

    else:

        feature_counts = sorted({int(n) for n in feature_counts if int(n) > 0})

    print("Top-N feature counts to evaluate:", feature_counts)
    print()

    missing_final_features = [
        gene for gene in final_features if gene not in X_train_full.columns
    ]

    if missing_final_features:
        raise ValueError(
            "Some ranked features are missing from the feature matrix.\n"
            f"Number of missing features: {len(missing_final_features)}\n"
            f"Examples: {missing_final_features[:20]}"
        )

    # Ignore matrix features that are absent from the finalized ranking
    extra_matrix_features = [
        gene for gene in feature_cols if gene not in final_features
    ]

    if extra_matrix_features:
        print(
            "Note: matrix features not present in the final ranking: "
            f"{len(extra_matrix_features)}"
        )

    valid_feature_counts = [n for n in feature_counts if n <= len(final_features)]

    skipped_feature_counts = [n for n in feature_counts if n > len(final_features)]

    if skipped_feature_counts:
        print(
            "Skipping Top-N values larger than the number of ranked features:",
            skipped_feature_counts,
        )

    if len(valid_feature_counts) == 0:
        raise ValueError(
            "No requested Top-N value can be evaluated. "
            f"Number of ranked features: {len(final_features)}"
        )

    min_class_count = y_train_full.value_counts().min()

    if n_splits > min_class_count:
        raise ValueError(
            f"n_splits={n_splits} exceeds the minimum training-class size ({min_class_count})."
        )

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED,
    )

    overall_summary = []

    parameter_record = {
        "seed": SEED,
        "data_path": data_path,
        "ranking_path": ranking_path,
        "fixed_test_path": fixed_test_path,
        "n_splits": n_splits,
        "feature_counts_requested": feature_counts,
        "feature_counts_used": valid_feature_counts,
        "feature_selection_input": "curated final ranked feature table after stability filtering and gene-level redundancy removal",
        "logistic_model": {
            "solver": "liblinear",
            "max_iter": 500,
            "class_weight": "balanced",
            "scaling": (
                "MinMaxScaler fitted within each CV training fold; "
                "final scaler fitted on the full training set"
            ),
        },
        "undersampling": False,
        "test_set_usage": "held-out test set used only for final evaluation",
    }

    with open(
        os.path.join(out_dir, "logistic_TopN_parameters.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(parameter_record, f, indent=4, ensure_ascii=False)

    # Top-N evaluation loop

    for n in valid_feature_counts:

        print(f"\nProcessing Top {n}")

        top_dir = os.path.join(out_dir, f"Top_{n}")
        os.makedirs(top_dir, exist_ok=True)

        selected_genes = final_features[:n]

        selected_df = final_ranking_df[
            final_ranking_df["Feature"].isin(selected_genes)
        ].copy()

        selected_df["Feature"] = pd.Categorical(
            selected_df["Feature"],
            categories=selected_genes,
            ordered=True,
        )

        selected_df = selected_df.sort_values("Feature").reset_index(drop=True)
        selected_df["TopN_Rank"] = np.arange(1, len(selected_df) + 1)

        selected_df.to_csv(
            os.path.join(top_dir, "selected_genes.csv"),
            index=False,
        )

        # Cross-validation on training set

        fold_results = []
        oof_rows = []

        for fold, (train_idx, val_idx) in enumerate(
            skf.split(X_train_full, y_train_full),
            start=1,
        ):

            X_train = X_train_full.iloc[train_idx][selected_genes].copy()
            X_val = X_train_full.iloc[val_idx][selected_genes].copy()

            y_train = y_train_full.iloc[train_idx].copy()
            y_val = y_train_full.iloc[val_idx].copy()

            val_meta = train_meta.iloc[val_idx].copy()

            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = LogisticRegression(
                max_iter=500,
                solver="liblinear",
                class_weight="balanced",
                random_state=SEED,
            )

            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_val_scaled)
            y_prob = model.predict_proba(X_val_scaled)[:, 1]

            fold_results.append(
                {
                    "Top_N": n,
                    "Fold": fold,
                    "Train_positive": int((y_train == 1).sum()),
                    "Train_negative": int((y_train == 0).sum()),
                    "Validation_positive": int((y_val == 1).sum()),
                    "Validation_negative": int((y_val == 0).sum()),
                    "ACC": accuracy_score(y_val, y_pred),
                    "Precision": precision_score(
                        y_val,
                        y_pred,
                        zero_division=0,
                    ),
                    "Recall": recall_score(
                        y_val,
                        y_pred,
                        zero_division=0,
                    ),
                    "F1": f1_score(
                        y_val,
                        y_pred,
                        zero_division=0,
                    ),
                    "AUC": roc_auc_score(y_val, y_prob),
                }
            )

            fold_oof = pd.DataFrame(
                {
                    "Top_N": n,
                    "Fold": fold,
                    "Sample": val_meta["sample"].values,
                    "Treatment": val_meta["treatment"].values,
                    "TrueLabel": y_val.values,
                    "PredLabel": y_pred,
                    "PredProb_1": y_prob,
                }
            )

            oof_rows.append(fold_oof)

        df_cv = pd.DataFrame(fold_results)
        df_cv.to_csv(
            os.path.join(top_dir, "CV_fold_results.csv"),
            index=False,
        )

        df_oof = pd.concat(oof_rows, axis=0).reset_index(drop=True)
        df_oof.to_csv(
            os.path.join(top_dir, "CV_oof_predictions.csv"),
            index=False,
        )

        mean_metrics = df_cv.mean(numeric_only=True)
        std_metrics = df_cv.std(numeric_only=True)

        df_cv_summary = pd.DataFrame(
            [
                {
                    "Top_N": n,
                    "ACC_mean": mean_metrics["ACC"],
                    "ACC_std": std_metrics["ACC"],
                    "Precision_mean": mean_metrics["Precision"],
                    "Precision_std": std_metrics["Precision"],
                    "Recall_mean": mean_metrics["Recall"],
                    "Recall_std": std_metrics["Recall"],
                    "F1_mean": mean_metrics["F1"],
                    "F1_std": std_metrics["F1"],
                    "AUC_mean": mean_metrics["AUC"],
                    "AUC_std": std_metrics["AUC"],
                }
            ]
        )

        df_cv_summary.to_csv(
            os.path.join(top_dir, "CV_summary.csv"),
            index=False,
        )

        # Final independent test

        scaler_final = MinMaxScaler()

        X_train_scaled_final = scaler_final.fit_transform(X_train_full[selected_genes])

        X_test_scaled = scaler_final.transform(X_test[selected_genes])

        model_final = LogisticRegression(
            max_iter=500,
            solver="liblinear",
            class_weight="balanced",
            random_state=SEED,
        )

        model_final.fit(X_train_scaled_final, y_train_full)

        y_pred_test = model_final.predict(X_test_scaled)
        y_prob_test = model_final.predict_proba(X_test_scaled)[:, 1]

        final_metrics = {
            "Top_N": n,
            "ACC_test": accuracy_score(y_test, y_pred_test),
            "Precision_test": precision_score(
                y_test,
                y_pred_test,
                zero_division=0,
            ),
            "Recall_test": recall_score(
                y_test,
                y_pred_test,
                zero_division=0,
            ),
            "F1_test": f1_score(
                y_test,
                y_pred_test,
                zero_division=0,
            ),
            "AUC_test": roc_auc_score(y_test, y_prob_test),
        }

        pd.DataFrame([final_metrics]).to_csv(
            os.path.join(top_dir, "Test_summary.csv"),
            index=False,
        )

        df_test_pred = pd.DataFrame(
            {
                "Top_N": n,
                "Sample": test_meta["sample"].values,
                "Treatment": test_meta["treatment"].values,
                "TrueLabel": y_test.values,
                "PredLabel": y_pred_test,
                "PredProb_1": y_prob_test,
            }
        )

        df_test_pred.to_csv(
            os.path.join(top_dir, "Test_predictions.csv"),
            index=False,
        )

        cm = confusion_matrix(y_test, y_pred_test)

        pd.DataFrame(
            cm,
            index=["True_0", "True_1"],
            columns=["Pred_0", "Pred_1"],
        ).to_csv(os.path.join(top_dir, "Test_confusion_matrix.csv"))

        joblib.dump(
            model_final,
            os.path.join(top_dir, "logistic_model.pkl"),
        )

        joblib.dump(
            scaler_final,
            os.path.join(top_dir, "scaler.pkl"),
        )

        coef_df = pd.DataFrame(
            {
                "Gene": selected_genes,
                "Coefficient": model_final.coef_[0],
                "AbsCoefficient": np.abs(model_final.coef_[0]),
            }
        )

        coef_df = coef_df.sort_values(
            "AbsCoefficient",
            ascending=False,
        )

        coef_df.to_csv(
            os.path.join(top_dir, "coefficients.csv"),
            index=False,
        )

        overall_summary.append(
            {
                "Top_N": n,
                "Num_candidate_features": len(final_features),
                "CV_ACC_mean": mean_metrics["ACC"],
                "CV_Precision_mean": mean_metrics["Precision"],
                "CV_Recall_mean": mean_metrics["Recall"],
                "CV_F1_mean": mean_metrics["F1"],
                "CV_AUC_mean": mean_metrics["AUC"],
                "CV_ACC_std": std_metrics["ACC"],
                "CV_Precision_std": std_metrics["Precision"],
                "CV_Recall_std": std_metrics["Recall"],
                "CV_F1_std": std_metrics["F1"],
                "CV_AUC_std": std_metrics["AUC"],
                "Test_ACC": final_metrics["ACC_test"],
                "Test_Precision": final_metrics["Precision_test"],
                "Test_Recall": final_metrics["Recall_test"],
                "Test_F1": final_metrics["F1_test"],
                "Test_AUC": final_metrics["AUC_test"],
            }
        )

        print(
            f"Top {n} -> "
            f"CV F1: {mean_metrics['F1']:.4f} | "
            f"Test F1: {final_metrics['F1_test']:.4f}"
        )

    df_overall = pd.DataFrame(overall_summary)

    df_overall.to_csv(
        os.path.join(out_dir, "Overall_Summary.csv"),
        index=False,
    )

    print("\nAll Top-N evaluations completed.")
    print("Overall summary:")
    print(df_overall)

    return df_overall


# Command-line interface


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Top-N logistic regression models using a curated final "
            "feature ranking and a fixed independent test set."
        )
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help=(
            "Path to the train+test feature matrix (.xlsx). Required metadata "
            "columns: sample, treatment, classification."
        ),
    )
    parser.add_argument(
        "--ranking-path",
        required=True,
        help=(
            "Path to the finalized ranked feature table (.xlsx). "
            "No SHAP cutoff or stability filtering is reapplied."
        ),
    )
    parser.add_argument(
        "--fixed-test-path",
        required=True,
        help="Path to the fixed-test sample file (.xlsx) containing a 'sample' column.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where Top-N evaluation results will be saved.",
    )
    parser.add_argument(
        "--feature-counts",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional custom Top-N feature counts. If omitted, the script "
            "evaluates predefined checkpoints (3, 5, 7, 9, 10, 15, and 20), "
            "then adds 10-feature intervals above 20 (30, 40, 50, ...), "
            "where available. The complete candidate-feature set is always "
            "included. Values larger than the available ranked feature count "
            "are skipped automatically."
        ),
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of stratified CV folds. Default: 5.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    evaluate_logistic_with_fixed_test(
        data_path=args.data_path,
        ranking_path=args.ranking_path,
        fixed_test_path=args.fixed_test_path,
        out_dir=args.out_dir,
        feature_counts=args.feature_counts,
        n_splits=args.n_splits,
    )


if __name__ == "__main__":
    main()
