"""
evaluate.py
Evaluates the persisted model on the held-out test split of claims_history.csv.
Produces:
  - printed metrics (ROC-AUC, PR-AUC, capture rate & precision at top 25%)
  - outputs/plots/roc_curve.png
  - outputs/plots/pr_curve.png
  - outputs/plots/capture_rate_curve.png   (capture rate vs. % of claims reviewed)
  - outputs/plots/feature_importance.png
  - outputs/plots/denial_reason_breakdown.png (error analysis using denial_reason,
    which is available post-hoc on the test set only, never as a model input)

Usage
-----
python src/evaluate.py --model_path outputs/model.pkl --data_path data/claims_history.csv
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from data_prep import engineer_features, TARGET_COL
from train import capture_rate_at_k


def load_artifact(model_path: str) -> dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def evaluate_test_set(artifact: dict, data_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    X_test = test_df[artifact["feature_cols"]]
    y_test = test_df[TARGET_COL].values

    X_test_t = artifact["preprocessor"].transform(X_test)
    scores = artifact["model"].predict_proba(X_test_t)[:, 1]

    return test_df, y_test, scores


def plot_roc(y_test, scores, out_path: Path):
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr(y_test, scores, out_path: Path):
    precision, recall, _ = precision_recall_curve(y_test, scores)
    ap = average_precision_score(y_test, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {ap:.3f}")
    plt.axhline(y_test.mean(), linestyle="--", color="gray", label="Base rate")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_capture_rate_curve(y_test, scores, out_path: Path):
    """Shows capture rate (% of all denials caught) as a function of the % of
    claims the review team is willing to inspect. Makes the 25% constraint concrete."""
    fracs = np.linspace(0.01, 1.0, 100)
    capture_rates = []
    for frac in fracs:
        res = capture_rate_at_k(y_test, scores, top_frac=frac)
        capture_rates.append(res["capture_rate_at_top25"])
    plt.figure(figsize=(6, 5))
    plt.plot(fracs * 100, capture_rates)
    plt.axvline(25, linestyle="--", color="red", label="Review capacity (25%)")
    frac25 = capture_rate_at_k(y_test, scores, top_frac=0.25)["capture_rate_at_top25"]
    plt.scatter([25], [frac25], color="red", zorder=5)
    plt.annotate(f"{frac25:.0%} of denials caught", (25, frac25), textcoords="offset points", xytext=(10, -10))
    plt.xlabel("% of claims reviewed (sorted by risk score)")
    plt.ylabel("% of all denials captured")
    plt.title("Denial capture rate vs. review capacity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_feature_importance(artifact: dict, out_path: Path, top_n: int = 15):
    model = artifact["model"]
    names = artifact["feature_names_out"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return
    order = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(7, 6))
    plt.barh([names[i] for i in order][::-1], importances[order][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} feature importances ({artifact['model_name']})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_denial_reason_breakdown(test_df, y_test, scores, out_path: Path, top_frac: float = 0.25):
    """Error analysis: among ACTUAL denials, which denial_reason categories
    does the model catch vs. miss within the top-25% review window?
    denial_reason is used here only for analysis, never as a model input."""
    n = len(y_test)
    k = int(np.ceil(n * top_frac))
    order = np.argsort(-scores)
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:k]] = True

    denied_mask = y_test == 1
    reasons = test_df.loc[denied_mask, "denial_reason"]
    caught = flagged[denied_mask]

    breakdown = pd.DataFrame({"reason": reasons.values, "caught": caught})
    summary = breakdown.groupby("reason")["caught"].agg(["mean", "count"]).sort_values("mean")

    plt.figure(figsize=(7, 5))
    plt.barh(summary.index, summary["mean"])
    plt.xlabel("Fraction caught within top 25% reviewed")
    plt.title("Denial capture rate by denial reason (test set)")
    for i, (idx, row) in enumerate(summary.iterrows()):
        plt.text(row["mean"] + 0.01, i, f"n={int(row['count'])}", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/model.pkl")
    parser.add_argument("--data_path", default="data/claims_history.csv")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact(args.model_path)
    test_df, y_test, scores = evaluate_test_set(artifact, args.data_path)

    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    capture = capture_rate_at_k(y_test, scores, top_frac=0.25)

    print("=== Test set metrics ===")
    print(f"Model: {artifact['model_name']}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")
    print(f"Base denial rate: {y_test.mean():.3f}")
    print(f"Capture rate @ top 25% reviewed: {capture['capture_rate_at_top25']:.3f}")
    print(f"Precision @ top 25% reviewed: {capture['precision_at_top25']:.3f}")
    print(f"(k = {capture['k']} of {capture['n']} test claims)")

    plot_roc(y_test, scores, plot_dir / "roc_curve.png")
    plot_pr(y_test, scores, plot_dir / "pr_curve.png")
    plot_capture_rate_curve(y_test, scores, plot_dir / "capture_rate_curve.png")
    plot_feature_importance(artifact, plot_dir / "feature_importance.png")
    reason_summary = plot_denial_reason_breakdown(test_df, y_test, scores, plot_dir / "denial_reason_breakdown.png")

    print("\n=== Capture rate by denial reason (top 25% reviewed) ===")
    print(reason_summary)

    print(f"\nPlots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
