# Kev ICD-10-CM Fine-Tuning: Evaluation Report

**Model:** `jaredpalmer/kev-0.8b` on base `Qwen/Qwen3.5-0.8B-Base`
**Training settings:** learning rate 4e-5, 2 epochs, batch 1 x gradient accumulation 8, max state 2048 tokens
**Calibration:** one temperature value, T = 2.52, fitted on the development set and saved with the checkpoint

> **Scope.** This report evaluates Kev on held-out **synthetic** tasks from the dataset builder. It shows how well the model answers the supplied decision questions. It does not show clinical validity, performance on real notes, billing readiness, or a live run from the ICD-10-CM root to a final code. Every number is calculated from the saved evaluation files in this session.

---

## 1. Executive summary

On the held-out synthetic test questions, accuracy moved from 58.33% to 98.80%. The lowest-scoring slice after fine-tuning is Label: OUT_OF_SCOPE at 63.64% (21/33).

- **Question accuracy** on 3,247 held-out test questions rose from **58.33%** to **98.80%** (+40.47 percentage points). A case-level bootstrap gives a 95% interval of +39.08 to +41.79 points.
- **Wins and losses:** 1,323 questions were fixed and 9 regressed (about 147-to-1); net +1,314. Another 30 stayed wrong and 1,885 stayed right.
- **Case-level routing-task exact match** (332 cases): 80 (24.1%) before, 315 (94.9%) after. A case counts only if every supplied routing question was right. This is not a live ROOT-to-code run.
- **Probability quality** on test: expected calibration error 0.3818 to 0.0035 (99.1% lower) and Brier score 0.7077 to 0.0194 (97.3% lower).
- **Retrospective coverage:** when ranked by confidence, the share of test questions that stays within a 5% error budget rose from 0.03% to 100.00%, and within a 1% budget from 0.03% to 99.45%. This describes this test set only. It does not show that the model is safe to run unattended.
- **Weakest area:** `OUT_OF_SCOPE` was correct for 21 of 33 questions (63.6%), with 6 confident errors.

---

## 2. Data and split audit

| Split | Task records | Cases | Questions | Routing (choice) | Wrong-branch (choice) | Evidence records (noul) | Choice questions | Noul questions |
|---|---|---|---|---|---|---|---|---|
| Test | 2,414 | 332 | 3,247 | 1,598 | 544 | 272 | 2,142 | 1,105 |

A task record is one exported training example. One case can yield several records, and one record can hold several questions. Counts of records and questions therefore overstate the number of independent cases.

Case-ID overlap was not rechecked because not all three normalized split files are present in this session.

---

## 3. Progression across development and test

| Stage | Split (questions) | Accuracy | Brier | NLL | ECE | Confident errors (p>=0.90) | Coverage at <=5% error | Coverage at <=1% error | AURC |
|---|---|---|---|---|---|---|---|---|---|
| 1. Dev baseline (before fine-tuning) | Dev (1,001) | 58.64% | 0.7043 | 1.2722 | 0.3872 | 7.29% | 0.00% | 0.00% | 0.4674 |
| 2. Dev fine-tuned, T=1.00 | Dev (1,001) | 98.80% | 0.0199 | 0.0686 | 0.0104 | 0.90% | 100.00% | 99.70% | 0.0005 |
| 3. Dev fine-tuned, calibrated | Dev (1,001) | 98.80% | 0.0177 | 0.0388 | 0.0081 | 0.20% | 100.00% | 99.70% | 0.0005 |
| 4. Test baseline (before fine-tuning) | Test (3,247) | 58.33% | 0.7077 | 1.2905 | 0.3818 | 7.76% | 0.03% | 0.03% | 0.4826 |
| 5. Test fine-tuned, calibrated | Test (3,247) | 98.80% | 0.0194 | 0.0371 | 0.0035 | 0.52% | 100.00% | 99.45% | 0.0005 |

Lower is better for Brier, NLL, ECE, confident errors and AURC. Higher is better for accuracy and coverage. Confident errors are wrong answers made with confidence of at least 0.90. AURC (area under the risk-coverage curve) summarizes error when answers are kept in confidence order.

### Effect of temperature calibration (development set)

Calibration changed the probabilities but not the answers. Development accuracy was 98.80% before and 98.80% after. NLL fell from 0.0686 to 0.0388 (43.4% lower), and confident errors fell from 0.90% to 0.20%.

The fitted temperature was 2.52, with fold temperatures [2.64, 2.76, 2.41, 2.64, 2.19] across cross-validation folds. The folds agree reasonably well, which suggests the fit is stable on this development set.

ECE moved from 0.0104 to 0.0081 and Brier from 0.0199 to 0.0177. Check the saved `calibration.json` intervals before treating small changes in these measures as real.

The test set has no uncalibrated fine-tuned run, so the effect of fine-tuning and the effect of calibration cannot be separated on test.

Calibrated development accuracy was 98.80% and test accuracy was 98.80%. A small gap between them suggests no large difference between the two splits, but this is one run and does not rule out overfitting to the synthetic generator.

---

## 4. Performance by task and label (test set)

| Slice | Questions | Acc before | Acc after | Change (pp) | Correct after | Brier before | Brier after | ECE before | ECE after | Conf. errors before | Conf. errors after | Cov<=5% before | Cov<=5% after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Overall (all questions) | 3,247 | 58.33% | 98.80% | +40.47 | 3208/3247 | 0.7077 | 0.0194 | 0.3818 | 0.0035 | 7.76% | 0.52% | 0.0% | 100.0% |
| Task: routing | 1,598 | 75.53% | 98.94% | +23.40 | 1581/1598 | 0.5406 | 0.0174 | 0.3763 | 0.0037 | 0.00% | 0.56% | 33.4% | 100.0% |
| Task: wrong-branch (NO_MATCH) | 544 | 61.76% | 99.26% | +37.50 | 540/544 | 0.6717 | 0.0095 | 0.3044 | 0.0056 | 0.00% | 0.18% | 0.0% | 100.0% |
| Task: evidence sentences | 1,105 | 31.76% | 98.37% | +66.61 | 1087/1105 | 0.9670 | 0.0272 | 0.5215 | 0.0088 | 22.81% | 0.63% | 0.1% | 100.0% |
| Label: STOP | 272 | 82.72% | 100.00% | +17.28 | 272/272 | 0.4470 | 0.0004 | 0.3688 | 0.0023 | 0.00% | 0.00% | 75.0% | 100.0% |
| Label: NEEDS_REVIEW | 27 | 0.00% | 92.59% | +92.59 | 25/27 | 0.9473 | 0.1009 | 0.3270 | 0.0637 | 0.00% | 3.70% | 0.0% | 96.3% |
| Label: OUT_OF_SCOPE | 33 | 0.00% | 63.64% | +63.64 | 21/33 | 1.0696 | 0.5441 | 0.3682 | 0.2659 | 0.00% | 18.18% | 0.0% | 21.2% |

Label rows are subsets of the routing questions, so do not add them to the task rows. Accuracy is per question, so cases with more questions carry more weight.

### Case-level routing-task exact match

| Case group | Cases | Before: all routing questions correct | After: all routing questions correct | Cases still missed |
|---|---|---|---|---|
| All routing cases | 332 | 80 (24.1%) | 315 (94.9%) | 17 |
| MAPPED (reaches STOP) | 272 | 80 (29.4%) | 269 (98.9%) | 3 |
| OUT_OF_SCOPE | 33 | 0 (0.0%) | 21 (63.6%) | 12 |
| NEEDS_REVIEW | 27 | 0 (0.0%) | 25 (92.6%) | 2 |

Of the 17 cases that still had at least one wrong routing answer, 12 were `OUT_OF_SCOPE` cases. This is a case-level count, so it is not the same as the 12 missed `OUT_OF_SCOPE` questions.

---

## 5. Confidence behavior

### Reliability by confidence bin (fine-tuned model)

| Confidence bin | Questions | Share | Mean confidence | Observed accuracy |
|---|---|---|---|---|
| 0.4-0.5 | 1 | 0.0% | 0.497 | 100.00% |
| 0.5-0.6 | 8 | 0.2% | 0.535 | 62.50% |
| 0.6-0.7 | 8 | 0.2% | 0.668 | 62.50% |
| 0.7-0.8 | 31 | 1.0% | 0.748 | 58.06% |
| 0.8-0.9 | 44 | 1.4% | 0.865 | 93.18% |
| 0.9-1.0 | 3155 | 97.2% | 0.994 | 99.46% |

82.3% of test questions have confidence above 0.99, and 99.93% of those were correct.

### Reliability by confidence bin (baseline)

| Confidence bin | Questions | Share | Mean confidence | Observed accuracy |
|---|---|---|---|---|
| 0.1-0.2 | 169 | 5.2% | 0.174 | 35.50% |
| 0.2-0.3 | 544 | 16.8% | 0.255 | 56.80% |
| 0.3-0.4 | 712 | 21.9% | 0.345 | 73.60% |
| 0.4-0.5 | 378 | 11.6% | 0.444 | 88.36% |
| 0.5-0.6 | 267 | 8.2% | 0.548 | 79.03% |
| 0.6-0.7 | 193 | 5.9% | 0.648 | 55.96% |
| 0.7-0.8 | 184 | 5.7% | 0.755 | 28.80% |
| 0.8-0.9 | 319 | 9.8% | 0.860 | 20.69% |
| 0.9-1.0 | 481 | 14.8% | 0.929 | 47.61% |

### Error as more questions are kept, highest confidence first

| Highest-confidence questions kept | Error before | Error after |
|---|---|---|
| 1% | 40.62% | 0.00% |
| 10% | 48.62% | 0.00% |
| 50% | 48.95% | 0.00% |
| 90% | 39.66% | 0.10% |
| 100% | 41.67% | 1.20% |

The saved chart `test_calibration_and_coverage.png` shows the same information. It is a retrospective summary of one synthetic test set. It cannot be used to choose a safe automation threshold, because that threshold would need to be chosen on separate data and then tested.

---

## 6. Remaining errors

The fine-tuned model made **39** errors across 3,247 questions: 9 were regressions (right before, wrong now) and 30 were wrong in both runs.

| Task | Questions | Errors | Error rate |
|---|---|---|---|
| evidence_selection | 1105 | 18 | 1.63% |
| routing | 1598 | 17 | 1.06% |
| routing_no_match | 544 | 4 | 0.74% |

### Most common error patterns

| Task | Gold | Predicted | Errors |
|---|---|---|---|
| evidence_selection | false | true | 11 |
| evidence_selection | true | false | 7 |
| routing | OUT_OF_SCOPE | node-0011145 | 4 |
| routing | OUT_OF_SCOPE | node-0012963 | 4 |
| routing_no_match | NO_MATCH | STOP | 3 |
| routing | NEEDS_REVIEW | node-0013453 | 2 |
| routing | OUT_OF_SCOPE | node-0015798 | 2 |
| routing | OUT_OF_SCOPE | node-0004970 | 1 |
| routing | OUT_OF_SCOPE | node-0003949 | 1 |
| routing | node-0004215 | NO_MATCH | 1 |
| routing | node-0013192 | node-0013089 | 1 |
| routing | node-0013453 | NEEDS_REVIEW | 1 |

### OUT_OF_SCOPE misses

12 of 33 `OUT_OF_SCOPE` questions were missed. The first rows below show the note text so the cause can be judged from the evidence. Ordered by fine-tuned confidence.

| Record | Question | Gold | Before | After | Note excerpt |
|---|---|---|---|---|---|
| rec-51e6ec4f4ab4:routing:0 | route | OUT_OF_SCOPE | node-0011145 (0.26) | node-0011145 (0.96) | Assessment: HF, type and acuity unspecified, at outpatient f/u. Reports unchanged DOE when climbing stairs and no SOB at rest. Exam: lungs CTA, trace bilat ankle edema. No EF or prior echo report available for review. Plan: continue current regimen, monitor da... |
| rec-20ab30178e1e:routing:0 | route | OUT_OF_SCOPE | node-0004970 (0.41) | node-0004970 (0.95) | Active problem: Attention-deficit hyperactivity disorder, unspecified type, reviewed at an outpatient medication follow-up. The patient reports continued difficulty organizing work assignments, with some improvement using scheduled reminders. Plan: Continue th... |
| rec-42ac73c24c75:routing:0 | route | OUT_OF_SCOPE | node-0012963 (0.37) | node-0012963 (0.93) | Active problem: asthma with an acute flare evaluated in clinic. History: three days of worsening wheeze and cough, and rescue inhaler benefit does not last. Findings: room-air saturation is 96%, expiratory wheezing is diffuse, and accessory muscles are not bei... |
| rec-200603fb9bd5:routing:0 | route | OUT_OF_SCOPE | node-0012963 (0.43) | node-0012963 (0.92) | Asthma: acute exacerbation assessed at today's outpatient visit. Symptoms: cough and wheezing have increased over three days, with only brief relief from the rescue inhaler. Exam: diffuse expiratory wheezes, oxygen saturation 96% on room air, and no accessory... |
| rec-2faad7d61c9d:routing:0 | route | OUT_OF_SCOPE | node-0011145 (0.31) | node-0011145 (0.91) | Outpt f/u: Atrial fibrillation confirmed on ECG; subtype not documented. HR 84, irregular; pt denies palpitations, CP, or syncope. Continue current management and arrange cardiology f/u. |
| rec-1cafe58ee994:routing:0 | route | OUT_OF_SCOPE | node-0011145 (0.30) | node-0011145 (0.91) | Telephone follow-up addressed the patient's established heart failure. The available documentation does not specify the heart failure type or acuity. The patient reports unchanged breathing, stable home weight, and no new ankle swelling. The patient continues... |
| rec-1855abeb736c:routing:0 | route | OUT_OF_SCOPE | node-0012963 (0.43) | node-0012963 (0.90) | Subjective: The patient with established asthma reports increased wheezing and chest tightness after exposure to renovation dust. Objective: Diffuse expiratory wheezes are present, with comfortable speech in full sentences and no accessory muscle use. Assessme... |
| rec-03398d95fc79:routing:0 | route | OUT_OF_SCOPE | node-0015798 (0.44) | node-0015798 (0.81) | During a follow-up call, the patient describes stair-related knee pain and short-lived stiffness when rising after sitting, without any recent falls. Assessment remains osteoarthritis involving the knee. The home exercise program provides benefit and will be c... |
| rec-4bfea6dbc93d:routing:0 | route | OUT_OF_SCOPE | node-0012963 (0.31) | node-0012963 (0.79) | Telephone follow-up for an acute exacerbation of asthma; baseline asthma severity and persistence category remain unspecified. The patient reports increased wheezing and chest tightness after cleaning a dusty storage room two days ago. Rescue inhaler use provi... |
| rec-2b36baf825df:routing:0 | route | OUT_OF_SCOPE | node-0015798 (0.38) | node-0015798 (0.76) | Telephone follow-up addressed established knee osteoarthritis. The patient reports discomfort on stairs and brief stiffness after sitting, with no recent falls. Home exercises remain helpful, and the plan is to continue them with an office reassessment if symp... |

### Regressions

2 routing and 7 evidence regressions. In the baseline, these were often right at low confidence, so some may reflect guessing rather than skill.

| Record | Question | Gold | Before | After | Note excerpt |
|---|---|---|---|---|---|
| rec-0635ca186148:routing:3 | route | node-0004215 | node-0004215 (0.22) | NO_MATCH (0.99) | DM2 with diabetic CKD, stage 3b, at routine OP f/u. Home BG log unavailable; repeat BMP and bring log next visit. |
| rec-07bc27bc1215:evidence:0 | sentence_4 | true | true (0.70) | false (0.97) | OP f/u for chronic diastolic CHF, compensated. DOE after two blocks is unchanged. No PND; trace pedal edema remains at baseline. No acute exacerbation today. Cont current meds and low-sodium diet. |
| rec-0528f6d13eea:evidence:0 | sentence_4 | true | true (0.91) | false (0.95) | The patient returns to clinic for monitoring of chronic kidney disease, documented as stage 3 without a specified substage. There is a history of a ureteral stone that passed several years ago, with no recurrent flank pain. The patient reports normal appetite... |
| rec-1e07b2fb0c9b:routing:0 | route | node-0013453 | node-0013453 (0.28) | NEEDS_REVIEW (0.94) | During telephone follow-up for chronic gastritis with bleeding of unspecified type, the patient reports less epigastric burning but another dark stool. A remote duodenal ulcer is noted in the history, while the recent endoscopy attributed the bleeding to gastr... |
| rec-0fc631ab7e83:evidence:0 | sentence_3 | true | true (0.88) | false (0.93) | Assessment: Permanent AF at OP f/u. No CP or SOB; pulse 80, irreg. Pt and cardiology accept AF; no further rhythm-control attempts planned. Cont metoprolol and apixaban; no bleeding reported. |
| rec-11298e1cb2eb:evidence:0 | sentence_3 | true | true (0.74) | false (0.85) | Telephone follow-up for established type 2 diabetes with diabetic neuropathy. The patient reports that tingling is unchanged since the last office visit. The neuropathy subtype is not specified, and no examination was performed during this call. Continue the c... |
| rec-02c29f06cca5:evidence:0 | sentence_4 | true | true (0.83) | false (0.79) | Telephone follow-up addressed permanent atrial fibrillation. The patient reports mild fatigue during longer walks, unchanged since the last clinic visit. Home pulse readings have ranged from 78 to 92 beats per minute. The patient and cardiologist previously ag... |
| rec-2aded491c04b:evidence:0 | sentence_2 | true | true (0.90) | false (0.78) | Reflux problem: Established gastroesophageal reflux disease causes occasional sour taste when bending after lunch. Esophagitis is not present, as confirmed on the reviewed upper endoscopy report. Clinic plan: Use smaller lunches and continue the current reflux... |
| rec-2db7c9c440ba:evidence:0 | sentence_3 | true | true (0.82) | false (0.75) | Telephone follow-up was completed for established chronic kidney disease. The patient reports no new urinary concerns. The disease stage is not recorded in the current chart. A repeat renal panel and an office follow-up were arranged. |

Evidence regressions by sentence position: sentence_2: 1, sentence_3: 3, sentence_4: 3.

### Ideas to test next (hypotheses, not established causes)

- **OUT_OF_SCOPE:** add more outside-scope training examples whose chapter appears in the root options, or allow the router to enter a chapter and decline at a lower level. Compare on development before testing. A larger model is another option, but it should be tested rather than assumed to help.
- **Abbreviation-heavy notes:** add abbreviation variants of existing cases and check whether routing errors on shorthand notes fall.
- **Evidence:** oversample multi-sentence evidence examples with positive qualifier sentences, and check precision as well as recall.
- **Contradictory notes:** inspect whether a conservative `NEEDS_REVIEW` answer is acceptable for history-versus-current conflicts before treating it as an error.

Row-level files are saved with this report: `finetuned_test_errors.csv` (with note excerpts), `finetuned_test_regressions.csv` and `report_slice_table.csv`.

---

## 7. Limits of these results

- The data are synthetic, and the generator and reviewer were model-based. No independent clinician review is reported.
- No shared case IDs were found across splits, but family-level and semantic overlap were not checked.
- Questions from the same case are correlated, so question counts overstate independent evidence.
- The evaluation uses supplied reference-position questions, not a live controller choosing a path from ROOT.
- One model size, one training run and one test split were evaluated.
- Calibration used the development set. The separate calibration partition from the dataset builder was not used.
- Per-code results are not reported, and some codes have only a few test cases.
- The saved `kev.compare` output reports 2,414 bootstrap groups for 332 test cases, so its intervals may cluster by record. The interval in Section 1 resamples whole cases.

## 8. Suggested next checks

1. Read every fine-tuned error against its note and option list, using the saved CSV.
2. Run a ROOT-to-termination controller with candidate selection that does not use the gold answer.
3. Re-split calibration and development by scenario family and calibrate on the calibration partition.
4. Add per-code results, outcome confusion matrices, and evidence precision and recall.
5. Repeat training with other seeds to see how much results vary.
6. Evaluate independently reviewed or clinician-authored notes before making any real-world claim.
