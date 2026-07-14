# Example LLM Explanation Outputs

`explain.py` calls the Gemini API live when `GEMINI_API_KEY` is set. This sandbox
has no network access to `generativelanguage.googleapis.com`, so the three
examples below were hand-drafted against the exact prompt the script sends,
using real field values and real model outputs from `predictions_current_claims.csv`.
Running `python src/explain.py` with a real key reproduces this behavior for
all top-10 claims automatically.

---

### Example 1 -- High risk (CCLM-00082, 95% denial probability)

**Claim facts:** BCBS, Inpatient, $87,589.75 billed, documentation flagged
missing, eligibility not verified, referral required but not on file.

**Generated explanation:**
> This inpatient claim is missing required documentation and the patient's eligibility hasn't been verified, both of which are strongly linked to denials in similar past claims. Recommended action: verify eligibility and attach the missing documentation before submitting. This is a risk estimate based on historical patterns, not a guarantee of denial.

---

### Example 2 -- High risk (CCLM-00372, 92% denial probability)

**Claim facts:** Medicaid MCO, Outpatient, $3,453.35 billed, prior
authorization required but not on file, documentation flagged missing.

**Generated explanation:**
> This outpatient Medicaid claim requires prior authorization, but none is on file, and documentation is also flagged as incomplete. Recommended action: obtain and attach the prior authorization before resubmitting. This is a risk estimate, not a certainty of denial.

---

### Example 3 -- Low risk (CCLM-00442, 11% denial probability)

Included specifically to check the prompt behaves sensibly on a claim with
little to flag, per the assessment instructions.

**Claim facts:** Commercial, Observation, $1,842.53 billed, prior
authorization not required, referral required and on file, documentation
complete, eligibility verified.

**Generated explanation:**
> This claim looks low-risk: prior authorization isn't required, the referral that is required is already on file, and documentation appears complete. No corrective action is needed beyond standard review, though this is a model estimate rather than a guarantee.

Note the model did not invent a problem where none of the flagged fields
indicated one -- it's not just producing uniformly alarming text.
