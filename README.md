# Claim Denial Prediction

Predicts whether a hospital claim will be denied, using only information
available before submission, and generates plain-English explanations for
the highest-risk current claims.

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce

```bash
python src/train.py --data_path data/claims_history.csv --seed 42
python src/evaluate.py --model_path outputs/model.pkl --data_path data/claims_history.csv
python src/score.py --model_path outputs/model.pkl --data_path data/current_claims.csv
export GEMINI_API_KEY="your-key-here"   # from https://aistudio.google.com/apikey
python src/explain.py --predictions_path outputs/predictions_current_claims.csv --claims_path data/current_claims.csv
```

If `GEMINI_API_KEY` isn't set, `explain.py` prints the exact prompts it would
send instead of failing silently. See `outputs/example_explanations.md` for
three worked examples (two high-risk, one low-risk) hand-drafted against the
real prompt template, since this environment doesn't have network access to
Google's API to make live calls.

## Approach

**Framing.** The review team can inspect only the top 25% of claims by risk
score, so the metric that actually matters operationally is: *of all claims
that will be denied, what fraction land in that top 25%?* I call this the
**capture rate at top 25%**, and it drove both model selection and the
risk-tier thresholds. ROC-AUC and PR-AUC are reported too, as
threshold-independent summaries, but capture rate is what a manager should
care about.

Errors are asymmetric here: missing a denial that lands outside the review
window (a false negative) costs the hospital a full denial-rework cycle,
while flagging a claim that turns out fine (a false positive) only costs a
few minutes of review time within a budget the team already has. That
argues for optimizing recall of denials within the review budget, not
overall accuracy — a claim would never be flagged in the naive sense as
"balanced" if it meant missing preventable denials.

**Features.** Beyond the raw columns, I engineered a handful of "gap" flags
that mirror how a reviewer actually reasons about a claim: `prior_auth_gap`
(required but not on file), `referral_gap`, `eligibility_gap`,
`network_gap`, and `late_submission_flag` (>30 days to submit), plus a
`payment_ratio` and a cyclical month-of-year encoding (`service_month`'s raw
year differs between history (2024) and current (2025) claims, so using it
directly would teach the model to key off the literal year instead of
seasonality).

**Models.** I compared a Logistic Regression baseline against XGBoost,
selecting on validation-set capture rate at top 25% (PR-AUC as tie-break).
**Logistic Regression won** (capture 49.5% vs. 44.7% on validation,
regularized XGBoost). This is a real, if slightly counter-intuitive,
finding: the engineered gap flags carry most of the signal, they're already
close to linear predictors of denial, and with only ~2,100 training rows
there isn't enough data for XGBoost's extra flexibility to pay off. I'd
expect this to flip with a larger dataset or a payer-level interaction
effect XGBoost could exploit that logistic regression can't.

**Thresholds.** Risk-tier cutoffs (`High` / `Medium` / `Low`) are the 75th
and 50th percentiles of the *validation*-set score distribution (not train,
to avoid overfit thresholds; not test, to avoid leakage). `High` corresponds
by construction to the top 25% — the claims the review team has capacity
for. `predicted_denial = 1` iff a claim is `High`.

**Explanations (Part 2).** For the top 10 highest-risk current claims,
`explain.py` builds a prompt containing only the claim's actual field values
and the model's SHAP-derived top risk drivers, and asks Gemini Flash for a
2-3 sentence, plain-English explanation with exactly one concrete
recommended action and an explicit "this is an estimate" caveat. See
`src/explain.py` for the full template and `outputs/example_explanations.md`
for worked examples, including a low-risk claim to confirm the prompt
doesn't manufacture alarm where the data doesn't support it.

## Files

```
data/                          claims_history.csv, current_claims.csv
src/
  data_prep.py                 loading + feature engineering (shared by all scripts)
  train.py                     trains + compares models, saves outputs/model.pkl
  evaluate.py                  test-set metrics + plots + denial-reason error analysis
  score.py                     scores current_claims.csv -> predictions_current_claims.csv
  explain.py                   LLM explanations for top 10 highest-risk claims
outputs/
  model.pkl                    trained model + preprocessor + thresholds
  predictions_current_claims.csv
  example_explanations.md      hand-drafted example LLM outputs (see above)
  plots/                       roc_curve, pr_curve, capture_rate_curve, feature_importance,
                                denial_reason_breakdown
```

## Limitations / what I'd improve with more time

- **"Payer policy or medical necessity" and "coding error" denials are
  essentially unpredictable from pre-submission fields** — the test-set
  capture rate for those two reasons is close to chance, because nothing in
  the available columns speaks to medical necessity or code accuracy. A
  clinical-content or coding-QA feature would likely help here; today the
  model is silent on a meaningful chunk of denials by design of what's
  available, not by a modeling failure.
- 3,200 historical rows is a small dataset for gradient boosting; I'd want
  more data (or synthetic augmentation) before trusting XGBoost over the
  simpler baseline long-term.
- SHAP background sample is capped at 200 rows for speed; a production
  version would use the full training set for more stable attributions.
- I'd add calibration diagnostics (reliability curve) before treating
  `denial_probability` as a literal probability rather than just a ranking
  score — it's currently only validated as a ranking signal (capture rate),
  not as a calibrated probability.
