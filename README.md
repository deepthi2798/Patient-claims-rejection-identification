# Claim Denial Prediction

Predicts whether a hospital claim will get denied, using only the info
available before it's submitted, and writes plain-English explanations
for the riskiest current claims.

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

If `GEMINI_API_KEY` isn't set, `explain.py` just prints the prompts it
would have sent instead of failing. `outputs/example_explanations.md` has
three worked examples (two high-risk, one low-risk) I drafted by hand
against the real prompt, since I didn't have network access to Google's
API from this environment to make live calls.

`outputs/writeup.pdf` walks through the whole thing end to end — problem
framing, data findings, model comparison, test-set metrics, and the LLM
explanation step — if you want the short version instead of reading the
code.

## How I approached it

The review team can only look at the top 25% of claims by risk score.
So the number that actually matters isn't accuracy — it's: of all the
claims that will get denied, how many end up in that top 25%? I'm calling
that the capture rate at top 25%, and it's what drove both model
selection and the risk-tier cutoffs. I still report ROC-AUC and PR-AUC
since they're standard, but capture rate is the one a manager should
actually care about.

The errors aren't symmetric either. Missing a denial that falls outside
the review window costs a full rework cycle. Flagging a claim that turns
out fine just costs a few minutes of a reviewer's time they already have
budgeted. So it makes sense to optimize for catching denials within that
25% budget, not for "balanced" accuracy overall.

**Features.** On top of the raw columns, I added a few "gap" flags that
match how a reviewer would actually think about a claim: `prior_auth_gap`
(required but not on file), `referral_gap`, `eligibility_gap`,
`network_gap`, and `late_submission_flag` (over 30 days to submit). Also
added `payment_ratio` and a cyclical encoding for month of service —
`service_month` is 2024 in the history data and 2025 in the current
claims, so feeding the raw year in would just teach the model to key off
the year instead of any real seasonal pattern.

**Models.** I compared a logistic regression baseline against XGBoost,
picking whichever had the better validation capture rate at top 25%
(PR-AUC as the tiebreaker). Logistic regression won — 49.5% capture vs.
44.7% for XGBoost. That surprised me a little, but it holds up: the gap
flags carry most of the signal and they're already close to linear
predictors of denial, and with ~2,100 training rows there's not really
enough data for XGBoost's extra flexibility to help. I'd expect that to
flip with more data, or if there were payer-level interaction effects for
it to exploit.

**Thresholds.** The High/Medium/Low cutoffs are the 75th and 50th
percentiles of the validation set's score distribution — not train
(would overfit the thresholds) and not test (would leak). High lines up
with the top 25% by construction, since that's what the review team can
actually get through. `predicted_denial = 1` whenever a claim is High.

**Explanations (Part 2).** For the 10 riskiest current claims,
`explain.py` builds a prompt out of the claim's actual field values plus
the model's SHAP-based top risk drivers, and asks Gemini Flash for a
2-3 sentence explanation with one concrete action and a caveat that
it's an estimate, not a guarantee. Full prompt is in `src/explain.py`,
worked examples in `outputs/example_explanations.md` — including a
low-risk claim, to check the prompt doesn't manufacture alarm when the
data doesn't support it.

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

## What I'd do differently with more time

- Denials reasoned as "payer policy / medical necessity" or "coding
  error" are basically unpredictable from the fields we get — capture
  rate on those two reasons is close to chance, because nothing in the
  data speaks to medical necessity or code accuracy. That's a data gap,
  not really a modeling failure, but it means the model stays quiet on
  a real chunk of denials.
- 3,200 rows is on the small side for gradient boosting. I'd want more
  data before trusting XGBoost over the simpler baseline long-term.
- The SHAP background sample is capped at 200 rows for speed. With more
  time I'd use the full training set for steadier attributions.
- I'd also want to check calibration (a reliability curve) before
  treating `denial_probability` as an actual probability rather than
  just a ranking score. Right now it's only validated as a ranking
  signal, not a calibrated one.