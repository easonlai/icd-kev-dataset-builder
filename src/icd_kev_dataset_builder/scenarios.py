from __future__ import annotations

# Curated complete, reportable, non-pilot codes. Each is verified against the selected
# release's codebook at run time; an absent code is an explicit error, never a substitute.
OUTSIDE_DIAGNOSIS_CODES = (
    "E66.9", "M54.50", "G43.909", "F41.1", "N40.0", "M17.9", "J02.9", "B34.9",
    "L20.9", "E78.5", "J30.9", "K58.9", "E03.9", "D64.9", "E55.9", "H66.90",
    "R51.9", "G47.00", "K76.0", "L70.0", "H10.9", "J01.90", "R05.9", "F32.9",
    "G40.909", "D50.9", "M10.9", "E04.9", "K80.20", "N20.0", "L40.9", "H40.9",
    "J32.9", "M81.0", "B35.1", "K64.9", "M75.100", "L03.90", "F90.9", "G47.33",
)
OUTSIDE_BRANCH_CODES = (
    "E11.319", "E11.51", "E11.621", "E10.10", "J45.50", "J45.901", "I50.9",
    "I48.91", "N40.1", "K29.60",
)
OUTSIDE_CODES_BY_TRAIT = {
    "nonpilot_diagnosis": OUTSIDE_DIAGNOSIS_CODES,
    "nonpilot_branch_of_pilot_condition": OUTSIDE_BRANCH_CODES,
}

# (topic, scenario hint) pairs for NEEDS_REVIEW notes. The hint describes a genuine
# contradiction inside the note; the application, not the model, labels the outcome.
NEEDS_REVIEW_SCENARIOS = (
    ("diabetes mellitus", "the assessment states both type 1 and type 2 diabetes for the current visit"),
    ("asthma", "severity is documented as both mild persistent and moderate persistent in the same assessment"),
    ("chronic kidney disease", "the stage is documented as stage 3 in one sentence and stage 4 in another"),
    ("atrial fibrillation", "the type is documented as both paroxysmal and permanent today"),
    ("chronic obstructive pulmonary disease", "the note says both with and without acute exacerbation at this visit"),
    ("heart failure", "the type is documented as both systolic and diastolic with no reconciliation"),
    ("gastroesophageal reflux disease", "the note states both with and without esophagitis for the same visit"),
    ("gastritis", "the note states both with and without bleeding for the same assessment"),
    ("hypertension", "the problem list says hypertension is current but the assessment says it was never diagnosed"),
    ("urinary tract infection", "the note states the infection is both current and fully resolved"),
    ("migraine", "the note states both with and without aura for the same headache pattern"),
    ("hypothyroidism", "the note states the patient is both on stable replacement and has never been treated"),
    ("osteoarthritis", "the affected knee is documented as both the left and the right in the same assessment"),
    ("gastric ulcer", "the note states both with and without hemorrhage in the same assessment"),
    ("coronary artery disease", "the note states both with and without angina at this visit"),
    ("chronic kidney disease", "the note documents both stage 2 and stage 5 for the same visit"),
    ("pneumonia", "the note says pneumonia is both active today and resolved on the same imaging review"),
    ("constipation", "the note says the patient has chronic constipation and also that bowel habits have always been normal"),
)

STYLE_TEXT = {
    "soap_brief": "a brief SOAP-style note (Subjective / Objective / Assessment / Plan content in running sentences)",
    "free_text_paragraph": "a single free-text clinic paragraph",
    "bulleted_problem_list": "a short problem-list style note where each line is one sentence",
    "telephone_followup": "a telephone follow-up note",
    "abbreviation_heavy": "a terse note using common clinical abbreviations",
}
TRAIT_TEXT = {
    "plain": "State the focused diagnosis plainly.",
    "negation_of_related_condition": "Include a clear negation of one related condition that is NOT present.",
    "history_vs_current": "Mention a relevant past/history item that is not the focus, while the current diagnosis is clear.",
    "irrelevant_distractor": "Include one irrelevant distractor detail unrelated to the focused diagnosis.",
    "sparse_detail": "Keep documentation sparse while still making the focused diagnosis clear.",
    "detailed_exam": "Include a few concrete exam or test details that support the focused diagnosis.",
    "nonpilot_diagnosis": "The focused diagnosis is the one described below.",
    "nonpilot_branch_of_pilot_condition": "The focused diagnosis is the one described below.",
    "documentation_conflict": "The documentation contains a genuine contradiction described below.",
    "unresolved_detail_required_for_code": (
        "The documentation contains a genuine contradiction described below. State it plainly "
        "and do not reconcile it; do not use the phrase \"clarification is needed\" and vary how, "
        "or whether, the plan mentions the discrepancy."
    ),
}
LENGTH_TEXT = {"short": "2-3 sentences", "medium": "4-6 sentences"}
