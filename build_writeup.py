from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, ListFlowable, ListItem
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], fontSize=16, spaceAfter=10, textColor=colors.HexColor("#1a2b4c")))
styles.add(ParagraphStyle(name="H2c", parent=styles["Heading2"], fontSize=12.5, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1a2b4c")))
styles.add(ParagraphStyle(name="Bodyc", parent=styles["Normal"], fontSize=9.7, leading=13.5, spaceAfter=6))
styles.add(ParagraphStyle(name="Small", parent=styles["Normal"], fontSize=8.3, leading=11, textColor=colors.HexColor("#444444")))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.HexColor("#666666"), alignment=1))

story = []
P = "outputs/plots/"

story.append(Paragraph("Claim Denial Prediction — Write-Up", styles["H1c"]))
story.append(Paragraph("Ensemble Health Partners AI/ML Take-Home Assessment", styles["Small"]))
story.append(Spacer(1, 10))

story.append(Paragraph("1. Problem framing", styles["H2c"]))
story.append(Paragraph(
    "The review team can only manually inspect the top 25% of claims by risk score before submission — "
    "everything else goes out unreviewed. That constraint, not overall accuracy, should drive the metric: "
    "the operational question is <i>of all claims that will actually be denied, what fraction land inside "
    "that top-25% review window?</i> I call this the <b>capture rate at top 25%</b> and used it as the primary "
    "model-selection and threshold criterion, with ROC-AUC / PR-AUC reported as threshold-independent summaries.",
    styles["Bodyc"]))
story.append(Paragraph(
    "Errors are asymmetric: a denial that slips outside the review window (false negative) costs a full "
    "rework cycle for the hospital, while a healthy claim flagged for review (false positive) only costs a "
    "few minutes of a reviewer's time that the team already budgets for. That argues for maximizing recall "
    "of denials within the fixed review budget, rather than optimizing for balanced precision/recall or "
    "raw accuracy.",
    styles["Bodyc"]))

story.append(Paragraph("2. Data findings worth sharing with a non-technical manager", styles["H2c"]))
items = [
    "The current model catches the large majority of claims denied for <b>missing documentation (81%)</b>, "
    "<b>missing prior authorization (73%)</b>, and <b>unverified eligibility (72%)</b> within the top-25% "
    "review window — these are exactly the operational, fixable-before-submission issues the review team "
    "exists to catch.",
    "It catches almost none of the claims denied for <b>payer policy / medical necessity (0%)</b> or "
    "<b>coding errors (27%)</b>. This isn't a modeling shortcoming so much as a data-availability one: none "
    "of the pre-submission fields speak to clinical necessity or code accuracy, so the model has no signal "
    "to work with there. Closing this gap would need a different data source (e.g. a coding-QA check), not "
    "a better model.",
    "A simple Logistic Regression <i>beat</i> a tuned XGBoost model on this dataset (49.5% vs. 44.7% capture "
    "rate on validation). With ~2,100 training rows and a handful of strong binary \"gap\" signals (auth "
    "missing, referral missing, etc.), there isn't enough data for a more flexible model to find anything "
    "the simpler one misses — and the simpler model is also the easier one to explain to a reviewer.",
]
story.append(ListFlowable([ListItem(Paragraph(t, styles["Bodyc"])) for t in items], bulletType="bullet", start="•"))

story.append(Paragraph("3. What was built", styles["H2c"]))
story.append(Paragraph(
    "Feature engineering added a handful of \"gap\" flags that mirror how a reviewer actually reasons about "
    "a claim (prior-auth required-but-missing, referral required-but-missing, eligibility not verified, "
    "out-of-network, submitted >30 days late), a payment ratio, and a cyclical month-of-year encoding "
    "(the raw <tt>service_month</tt> column has non-overlapping years between history and current claims, "
    "so using it directly would teach the model to key off the literal year rather than seasonality).",
    styles["Bodyc"]))
story.append(Paragraph(
    "Two candidates were compared on the validation split: Logistic Regression (baseline, class-balanced) "
    "and XGBoost (regularized: max_depth=3, min_child_weight=5, L2=2.0, early stopping). Logistic Regression "
    "won on capture rate at top 25% and was selected as the final model. Risk-tier thresholds (High ≥ 75th "
    "percentile of validation scores, Medium ≥ 50th percentile, Low below) were fit on the validation split "
    "specifically to avoid overfitting to train and to avoid leaking test-set information — High corresponds "
    "by construction to the top-25% review capacity.",
    styles["Bodyc"]))

story.append(PageBreak())

story.append(Paragraph("4. Test-set metrics", styles["H2c"]))
metrics_table_data = [
    ["Metric", "Value"],
    ["ROC-AUC", "0.698"],
    ["PR-AUC", "0.516"],
    ["Base denial rate (test)", "26.0%"],
    ["Capture rate @ top 25% reviewed", "46.4%"],
    ["Precision @ top 25% reviewed", "48.1%"],
]
t = Table(metrics_table_data, colWidths=[3.0 * inch, 1.5 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2b4c")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f8")]),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
]))
story.append(t)
story.append(Paragraph(
    "In plain terms: reviewing just the top quarter of claims by predicted risk catches nearly half of all "
    "claims that go on to be denied — roughly 1.8x better than reviewing a random quarter of claims would.",
    styles["Bodyc"]))

story.append(Spacer(1, 8))
img1 = Image(P + "capture_rate_curve.png", width=3.1 * inch, height=2.6 * inch)
img2 = Image(P + "denial_reason_breakdown.png", width=3.1 * inch, height=2.6 * inch)
img_table = Table([[img1, img2]], colWidths=[3.2 * inch, 3.2 * inch])
story.append(img_table)
story.append(Paragraph("Left: denial capture rate vs. review capacity, with the 25% operating point marked. "
                        "Right: capture rate broken out by denial reason (test set).", styles["Caption"]))

story.append(Paragraph("5. LLM explanations (Part 2)", styles["H2c"]))
story.append(Paragraph(
    "For the top 10 highest-risk current claims, <tt>explain.py</tt> sends Gemini Flash a prompt containing "
    "only the claim's actual field values and its SHAP-derived top risk drivers, with explicit instructions "
    "not to invent facts, to name exactly one concrete action, and to caveat that this is a risk estimate. "
    "This sandbox has no network access to Google's API, so the example below was hand-drafted against the "
    "real prompt and real field values (two more examples, including a low-risk sanity check, are in "
    "<tt>outputs/example_explanations.md</tt>).",
    styles["Bodyc"]))
story.append(Paragraph(
    "<b>Prompt template (abridged):</b> \"You are helping a hospital billing analyst quickly triage a claim... "
    "Claim facts (use ONLY these): Payer, visit type, billed/expected amounts, days to submit, prior auth "
    "required/on file, network status, documentation flag, eligibility verified, referral required/on file. "
    "Model output: denial probability, top risk drivers. Write a 2-3 sentence explanation with exactly one "
    "recommended action, and note this is a risk estimate, not a guarantee.\"",
    styles["Bodyc"]))
story.append(Paragraph(
    "<b>Example output</b> (CCLM-00082, 95% denial probability, BCBS Inpatient, $87,589.75 billed): "
    "\"This inpatient claim is missing required documentation and the patient's eligibility hasn't been "
    "verified, both of which are strongly linked to denials in similar past claims. Recommended action: "
    "verify eligibility and attach the missing documentation before submitting. This is a risk estimate "
    "based on historical patterns, not a guarantee of denial.\"",
    styles["Bodyc"]))

story.append(Paragraph("6. Limitations and what I'd improve with more time", styles["H2c"]))
items2 = [
    "Denials tied to medical necessity or coding accuracy are close to unpredictable from the fields "
    "available pre-submission — the model is silent there by data-availability, not by a fixable modeling "
    "gap. I'd want a coding-QA signal or payer-specific medical-necessity flag to close this.",
    "3,200 historical rows is thin for gradient boosting; I'd revisit XGBoost vs. Logistic Regression with "
    "more data or synthetic augmentation before trusting the simpler model long-term.",
    "denial_probability is currently validated only as a <i>ranking</i> signal (capture rate), not a "
    "calibrated probability — I'd add a reliability curve before treating it as a literal probability "
    "in any downstream reporting.",
]
story.append(ListFlowable([ListItem(Paragraph(t, styles["Bodyc"])) for t in items2], bulletType="bullet", start="•"))

doc = SimpleDocTemplate("outputs/writeup.pdf", pagesize=letter,
                         topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                         leftMargin=0.7 * inch, rightMargin=0.7 * inch)
doc.build(story)
print("Wrote outputs/writeup.pdf")
