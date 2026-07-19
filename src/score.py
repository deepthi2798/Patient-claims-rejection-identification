"""
score.py
Scores current_claims.csv with the trained model and produces
outputs/predictions_current_claims.csv with columns:
  claim_id, denial_probability, predicted_denial, risk_tier, top_risk_factors

Usage
-----
python src/score.py --model_path outputs/model.pkl --data_path data/current_claims.csv
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from data_prep import engineer_features

# Human-readable labels for engineered / binary features, with the direction
# that makes a claim RISKIER spelled out. Used to turn SHAP's top drivers into
# analyst-facing phrases instead of raw column names.
FEATURE_LABELS = {
    "prior_auth_gap": "prior authorization required but not on file",
    "referral_gap": "referral required but not present",
    "eligibility_gap": "patient eligibility not verified",
    "network_gap": "provider out of network for this payer",
    "missing_documentation_flag": "documentation appears incomplete or missing",
    "late_submission_flag": "claim submitted more than 30 days after service",
    "total_risk_gaps": "multiple risk gaps present on this claim",
    "days_to_submit": "submission timing",
    "payment_ratio": "expected payment vs. billed amount",
    "procedures_per_diagnosis": "procedure/diagnosis code volume",
    "num_procedures": "number of procedure codes",
    "num_diagnoses": "number of diagnosis codes",
    "total_billed": "total billed amount",
    "expected_payment": "expected payment amount",
    "month_sin": "seasonality",
    "month_cos": "seasonality",
    "prior_auth_required": "prior authorization requirement",
    "has_prior_auth": "prior authorization on file",
    "is_in_network": "network status",
    "eligibility_verified": "eligibility verification status",
    "referral_required": "referral requirement",
    "referral_present": "referral on file",
}


def humanize_feature(raw_name: str) -> str:
    """Map a preprocessor output feature name (e.g. 'cat__payer_type_BCBS',
    'num__prior_auth_gap') to a short human-readable label."""
    name = raw_name.split("__", 1)[-1] if "__" in raw_name else raw_name

    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]

    # One-hot encoded categorical, e.g. "payer_type_BCBS" or "visit_type_Emergency"
    for prefix, readable in [
        ("payer_type_", "payer type"),
        ("visit_type_", "visit type"),
        ("payer_id_", "payer"),
    ]:
        if name.startswith(prefix):
            value = name[len(prefix):]
            return f"{readable} ({value})"

    return name.replace("_", " ")


def load_artifact(model_path: str) -> dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


# Excluded from analyst-facing output: either redundant with the individual
# gap flags it aggregates (total_risk_gaps) or not actionable/meaningful on
# its own (month_sin/cos are a seasonality encoding, not a real-world driver).
_EXCLUDED_RAW_FEATURES = {"total_risk_gaps", "month_sin", "month_cos"}


def get_top_risk_factors(shap_values: np.ndarray, feature_names: list[str], top_n: int = 3) -> list[str]:
    """For a single claim's SHAP row, return the top_n features pushing the
    score UP (toward denial), as human-readable labels. Only positive
    contributions are surfaced -- factors that reduce risk aren't 'risk factors'."""
    order = np.argsort(shap_values)[::-1]
    labels = []
    seen = set()
    for idx in order:
        if shap_values[idx] <= 0:
            break
        raw_name = feature_names[idx].split("__", 1)[-1]
        if raw_name in _EXCLUDED_RAW_FEATURES:
            continue
        label = humanize_feature(feature_names[idx])
        if label not in seen:
            labels.append(label)
            seen.add(label)
        if len(labels) >= top_n:
            break
    if not labels:
        labels = ["no single dominant risk driver identified"]
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="outputs/model.pkl")
    parser.add_argument("--data_path", default="data/current_claims.csv")
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = load_artifact(args.model_path)
    model = artifact["model"]
    preprocessor = artifact["preprocessor"]
    feature_names = artifact["feature_names_out"]

    raw_df = pd.read_csv(args.data_path)
    df = engineer_features(raw_df)
    X = df[artifact["feature_cols"]]
    X_t = preprocessor.transform(X)
    # Densify: SHAP explainers expect dense arrays; OneHotEncoder output is sparse.
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    scores = model.predict_proba(X_t)[:, 1]

    threshold_high = artifact["threshold_high"]
    threshold_medium = artifact["threshold_medium"]

    risk_tier = np.where(
        scores >= threshold_high, "High", np.where(scores >= threshold_medium, "Medium", "Low")
    )
    predicted_denial = (scores >= threshold_high).astype(int)

    # --- SHAP explanations ---
    if hasattr(model, "feature_importances_"):  # tree-based (XGBoost)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_t)
    else:  # linear model
        background = artifact["X_train_t_sample"]
        if hasattr(background, "toarray"):
            background = background.toarray()
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_t)

    top_risk_factors = [
        "; ".join(get_top_risk_factors(shap_values[i], feature_names, top_n=3))
        for i in range(len(scores))
    ]

    result = pd.DataFrame(
        {
            "claim_id": raw_df["claim_id"],
            "denial_probability": np.round(scores, 4),
            "predicted_denial": predicted_denial,
            "risk_tier": risk_tier,
            "top_risk_factors": top_risk_factors,
        }
    ).sort_values("denial_probability", ascending=False).reset_index(drop=True)

    out_path = out_dir / "predictions_current_claims.csv"
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} scored claims to {out_path}")
    print(result["risk_tier"].value_counts())
    print("\nTop 10 highest-risk claims:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
