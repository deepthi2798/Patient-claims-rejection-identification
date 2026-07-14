"""
train.py
Trains a baseline Logistic Regression and an XGBoost classifier on
claims_history.csv (using the provided train/validation split), compares
them on the metric that matters for this problem -- denial capture rate
within the top 25% of claims by risk score -- and persists the winner.

Usage
-----
python src/train.py --data_path data/claims_history.csv --seed 42
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data_prep import engineer_features, get_feature_columns, TARGET_COL


def capture_rate_at_k(y_true: np.ndarray, y_score: np.ndarray, top_frac: float = 0.25) -> dict:
    """Of the claims flagged in the top `top_frac` by score, what fraction of
    ALL actual denials did we catch (recall/capture rate), and what fraction
    of the flagged claims were actually denied (precision)?"""
    n = len(y_true)
    k = max(1, int(np.ceil(n * top_frac)))
    order = np.argsort(-y_score)
    top_idx = order[:k]
    flagged_denials = y_true[top_idx].sum()
    total_denials = y_true.sum()
    capture_rate = flagged_denials / total_denials if total_denials > 0 else np.nan
    precision_at_k = flagged_denials / k
    return {
        "k": k,
        "n": n,
        "capture_rate_at_top25": float(capture_rate),
        "precision_at_top25": float(precision_at_k),
    }


def build_preprocessor() -> ColumnTransformer:
    cols = get_feature_columns()
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cols["categorical"]),
            ("num", StandardScaler(), cols["numeric"]),
            ("bin", "passthrough", cols["binary"]),
        ]
    )


def get_feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/claims_history.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    df = engineer_features(df)

    cols = get_feature_columns()
    feature_cols = cols["categorical"] + cols["numeric"] + cols["binary"]

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL].values
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COL].values

    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    feature_names = get_feature_names_out(preprocessor)

    candidates = {}

    # --- Baseline: Logistic Regression ---
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    logreg.fit(X_train_t, y_train)
    val_scores_lr = logreg.predict_proba(X_val_t)[:, 1]
    candidates["logistic_regression"] = {
        "model": logreg,
        "val_roc_auc": roc_auc_score(y_val, val_scores_lr),
        "val_pr_auc": average_precision_score(y_val, val_scores_lr),
        **capture_rate_at_k(y_val, val_scores_lr),
    }

    # --- Candidate: XGBoost ---
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos  # correct for the ~21% denial base rate

    xgb = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=args.seed,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    xgb.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)
    val_scores_xgb = xgb.predict_proba(X_val_t)[:, 1]
    candidates["xgboost"] = {
        "model": xgb,
        "val_roc_auc": roc_auc_score(y_val, val_scores_xgb),
        "val_pr_auc": average_precision_score(y_val, val_scores_xgb),
        **capture_rate_at_k(y_val, val_scores_xgb),
    }

    print("=== Validation comparison ===")
    for name, res in candidates.items():
        print(
            f"{name:20s} ROC-AUC={res['val_roc_auc']:.3f}  PR-AUC={res['val_pr_auc']:.3f}  "
            f"capture@top25%={res['capture_rate_at_top25']:.3f}  precision@top25%={res['precision_at_top25']:.3f}"
        )

    # Selection criterion: capture rate at the operational top-25% threshold is
    # what actually matters for this workflow (review team can only see 25% of
    # volume), so we pick the model that maximizes it. PR-AUC is used as a tie-breaker.
    best_name = max(candidates, key=lambda n: (candidates[n]["capture_rate_at_top25"], candidates[n]["val_pr_auc"]))
    best_model = candidates[best_name]["model"]
    print(f"\nSelected model: {best_name}")

    # Risk-tier thresholds are derived from the VALIDATION score distribution
    # (not train, to avoid overfit thresholds; not test, to avoid leakage).
    # High = the score cutoff that corresponds to the top 25% review capacity.
    # Medium = next band down to the median score. Low = below median.
    best_val_scores = candidates[best_name]["model"].predict_proba(X_val_t)[:, 1]
    threshold_high = float(np.quantile(best_val_scores, 0.75))
    threshold_medium = float(np.quantile(best_val_scores, 0.50))
    print(f"Risk tier thresholds -> High: >={threshold_high:.3f}, Medium: >={threshold_medium:.3f}, Low: below")

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "feature_names_out": feature_names,
        "seed": args.seed,
        "threshold_high": threshold_high,
        "threshold_medium": threshold_medium,
        "X_train_t_sample": X_train_t[:200],  # small background sample for SHAP
    }
    with open(out_dir / "model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved model artifact to {out_dir / 'model.pkl'}")


if __name__ == "__main__":
    main()
