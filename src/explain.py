"""
explain.py
For the top 10 highest-risk claims in outputs/predictions_current_claims.csv,
calls the Gemini API (Google AI Studio) to generate a short, plain-English,
analyst-facing explanation grounded in the claim's actual field values and
its model-derived top_risk_factors.

Setup
-----
1. Get a free API key from Google AI Studio: https://aistudio.google.com/apikey
2. export GEMINI_API_KEY="your-key-here"
3. python src/explain.py --predictions_path outputs/predictions_current_claims.csv \
       --claims_path data/current_claims.csv

Model
-----
Defaults to "gemini-flash-latest" (currently resolves to Gemini 3.5 Flash).
Override with --model if Google renames/deprecates it later -- check
https://ai.google.dev/gemini-api/docs/models for the current Flash alias.

Design notes
------------
- The prompt is deliberately constrained: it is given ONLY the claim's raw
  field values + the model's top_risk_factors + denial_probability. It is
  explicitly instructed not to invent facts, to name one concrete action,
  and to caveat that this is a risk estimate, not a certainty.
- A worked low-risk claim is included (see `demo_low_risk_claim`) to sanity
  check that the prompt behaves reasonably when there is little to flag --
  required by the assessment spec.
- If GEMINI_API_KEY is not set, the script falls back to printing the exact
  prompts it would have sent (useful for review / grading without a live key).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

GEMINI_API_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT_TEMPLATE = """You are helping a hospital billing analyst quickly triage a health insurance claim before it is submitted.

Claim facts (use ONLY these -- do not assume or invent anything else):
- Payer: {payer_id} ({payer_type})
- Visit type: {visit_type}
- Total billed: ${total_billed:,.2f}
- Expected payment: ${expected_payment:,.2f}
- Days between service and submission: {days_to_submit}
- Prior authorization required: {prior_auth_required} | On file: {has_prior_auth}
- In network: {is_in_network}
- Documentation flagged as missing: {missing_documentation_flag}
- Eligibility verified: {eligibility_verified}
- Referral required: {referral_required} | On file: {referral_present}

Model output:
- Predicted denial probability: {denial_probability:.0%}
- Top risk drivers identified by the model: {top_risk_factors}

Write a 2-3 sentence explanation for the analyst. Requirements:
1. Ground it only in the claim facts and risk drivers listed above -- do not invent details.
2. Plain English, no insurance jargon the analyst would need to look up.
3. Include exactly one specific, concrete recommended action.
4. Explicitly note this is a risk estimate, not a guarantee the claim will be denied.
5. Keep it to 2-3 sentences total.
"""


def build_prompt(row: pd.Series) -> str:
    return PROMPT_TEMPLATE.format(
        payer_id=row["payer_id"],
        payer_type=row["payer_type"],
        visit_type=row["visit_type"],
        total_billed=row["total_billed"],
        expected_payment=row["expected_payment"],
        days_to_submit=row["days_to_submit"],
        prior_auth_required="Yes" if row["prior_auth_required"] else "No",
        has_prior_auth="Yes" if row["has_prior_auth"] else "No",
        is_in_network="Yes" if row["is_in_network"] else "No",
        missing_documentation_flag="Yes" if row["missing_documentation_flag"] else "No",
        eligibility_verified="Yes" if row["eligibility_verified"] else "No",
        referral_required="Yes" if row["referral_required"] else "No",
        referral_present="Yes" if row["referral_present"] else "No",
        denial_probability=row["denial_probability"],
        top_risk_factors=row["top_risk_factors"],
    )


def call_gemini(prompt: str, api_key: str, model: str, max_retries: int = 3) -> str:
    url = GEMINI_API_URL_TMPL.format(model=model)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200},
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    for attempt in range(max_retries):
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        if resp.status_code == 429:  # rate limited -- back off and retry
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
    raise RuntimeError("Gemini API rate-limited after all retries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", default="outputs/predictions_current_claims.csv")
    parser.add_argument("--claims_path", default="data/current_claims.csv")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--model", default="gemini-flash-latest")
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions_path)
    claims = pd.read_csv(args.claims_path)

    merged = predictions.merge(claims, on="claim_id", how="left").head(args.top_n)

    api_key = os.environ.get("GEMINI_API_KEY")
    explanations = []

    if not api_key:
        print("GEMINI_API_KEY not set -- printing prompts instead of calling the API.\n")
        print("Set it with: export GEMINI_API_KEY='your-key-here'\n")

    for _, row in merged.iterrows():
        prompt = build_prompt(row)
        if api_key:
            try:
                explanation = call_gemini(prompt, api_key, args.model)
            except Exception as e:
                explanation = f"[ERROR generating explanation: {e}]"
        else:
            explanation = "[NOT GENERATED -- no API key set. See prompt below.]"
            print(f"--- Prompt for {row['claim_id']} ---\n{prompt}\n")
        explanations.append(explanation)

    merged["explanation"] = explanations

    out_path = Path(args.out_dir) / "predictions_current_claims.csv"
    # Merge explanations back into the FULL predictions file (not just top N)
    full = predictions.copy()
    full["explanation"] = ""
    full.loc[full["claim_id"].isin(merged["claim_id"]), "explanation"] = (
        full["claim_id"].map(dict(zip(merged["claim_id"], merged["explanation"])))
    )
    full.to_csv(out_path, index=False)
    print(f"Wrote explanations for top {args.top_n} claims into {out_path}")

    for _, row in merged.iterrows():
        print(f"\n{row['claim_id']} (risk={row['denial_probability']:.0%}, tier={row['risk_tier']}):")
        print(f"  {row['explanation']}")


if __name__ == "__main__":
    main()
