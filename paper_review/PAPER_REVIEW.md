# Paper review: IEEE format, factual verification, figures, AI usage

Paper: *Explainable Customer Churn Prediction and Retention Intelligence System* (6 pp., `yes_formatted.pdf`).
Every number below was checked against: `notebooks/01-03`, `results/*`, `models/`, `app/app.py`, and the public IBM Telco data
(downloaded from IBM's GitHub; the 21-column version, which has every column the EDA claims use).

---

## 1. Fix these first (will be noticed by any reviewer)

| # | Where | Problem | Fix |
|---|-------|---------|-----|
| 1 | III-B, p.2 | Yellow-highlighted placeholder: "indicating that **[words missing in the original file]** early-tenure attrition." | See text fix T1. |
| 2 | Figs. 1, 2, 3 | Three figures are still "[Insert figure]" placeholders. | Use `figures/fig1`, `fig2`, Fig. 3 (see section 4). |
| 3 | V-C | Says `scale_pos_weight` was used to handle imbalance. **Only the untuned baseline XGBoost used it** (`02_modeling` cell 61). The tuned/final model has no `scale_pos_weight` (search space and final params contain none); imbalance is handled by moving the threshold. | Text fix T2. This is a factual error about your own method. |
| 4 | Abstract / VII | Abstract says three models were "trained and compared", but **no comparison results appear anywhere**. | Add Table VII + Fig. 5 (section 3-4). |
| 5 | Table IV | Column is labelled "Original Feature", but the values are **single one-hot columns** (Contract 0.654 = `Contract_Month-to-month`; Dependents 0.462 = `Dependents_No`; Online Security 0.215 = `Online Security_No`; Internet Service 0.191 = `Fiber optic`). Per-feature SHAP would sum its dummies first. | Relabel (T4) or regenerate with `scripts/make_data_dependent_figures.py`. |
| 6 | VI | Threshold 0.30 was chosen by sweeping **on the test set**, then performance is reported on that same test set. Selection bias. | Disclose (T5); better, choose it on out-of-fold training predictions and re-report. |
| 7 | References | [7] never cited; order of first citation is not sequential (10, 11, 6, 4, 1, 2, 9, 3, 8, 5); TreeExplainer attributed to [4] but belongs to [5]; [11] (Davis & Goadrich) doesn't support the claim it's attached to. | Section 5. |
| 8 | PDF | Fonts not embedded (Times New Roman, Arial). IEEE PDF eXpress rejects this. Arial shouldn't be present. | Re-export from Word with "embed fonts" / save as PDF/A. |

## 2. Fact check

**Verified correct (matches source to stated precision):**
- 7,043 customers; 1,869 churn (26.5%) / 5,174 stay; 80/20 stratified split, seed 42, 5,634 / 1,409; 374 churners in test.
- Churn by contract 42.7 / 11.3 / 2.8%; by internet 19.0 / 41.9 / 7.4%.
- All of Table II (mean/median/min/max, both groups); 11 blank Total Charges, all tenure 0.
- 4 numeric + 16 categorical -> 47 columns after one-hot; 15 actionable + 5 non-actionable features.
- LR `max_iter=1000`, RF 300 trees, Table III search space and best values, 20 iterations x 5 folds = 100 fits.
- Threshold table (0.30: P 54.0, R 77.8, F1 63.7; 0.50: recall 54.0) and ROC-AUC 85.5%.
- Confusion matrix 787 / 248 / 83 / 291; 291 of 374 found; 83 missed.
- Table VI figures (A 93.5%, B 0.17% -> "0.2%", C 30.02%) and every SHAP value quoted.
- Max upload 20,000 rows; top-5 SHAP bar chart.

**Wrong or misleading:**
- V-C `scale_pos_weight` (above).
- V-D "Six-dimensional **grid** search ... 20 **random** iterations": it is a random search, not a grid (the grid has 324 combinations).
- VIII-A "20 **binary** variables": they are 20 features, not binary. Non-actionable list omits CLTV (it is the 5th) and calls Dependents "marital status".
- Table V lists 5 rules; the engine has **15**. Table VI cites rules not in Table V (internet-plan review, billing-history review).
- Table VI omits one recommendation each for A (billing history) and C (lines plan); fine if you say "top actions", but the table doesn't.
- IX "**Parallel** processing of the entire cohort": the code is vectorised, not parallel (no `n_jobs`/multiprocessing in the batch path).
- IV-B "standardization ... makes SHAP interpretability much simpler": unsupported, and tree models are scale-invariant, so scaling gives SHAP no simplification (the app explains using raw input values). Delete.
- X-A "limiting fatal false negatives to a **minimal** 83": 83 is 22.2% of churners. Say the number.
- X-A "very profitable trade-off ... CAC:CRC ratios are much higher": no data or citation; you never quantify cost. Soften or cite.
- III-B "churners have **always** paid more": it is a mean difference, not "always".
- II-B "very few [papers] have used it [SHAP] ... at an individual customer level": uncited novelty claim; cite or soften.
- Section VI "F1 peaked **identically** at 0.30 and 0.35": 63.75% vs 63.72%; they round the same.

**Not reported but should be:**
- Accuracy at the deployed threshold is **76.5%** (below the 80.3% at 0.50). Report it, next to the **73.5% majority-class baseline**. This is the honest answer to the accuracy question your evaluator raised; ROC-AUC is the threshold-free measure.
- Best 5-fold CV ROC-AUC = 0.865 vs test 0.8555 (supports "no overfit").
- Logistic Regression ROC-AUC is 0.849 vs 0.8555 for tuned XGBoost: a **0.007 gap on one split, no confidence interval**. Don't claim more than "comparable discrimination"; the case for XGBoost rests on threshold flexibility and TreeExplainer.

**Repo inconsistencies to tell Gracy about (not changed by me):**
- `results/model_comparison.csv`: LR row is hand-rounded (0.80/0.64/0.57/0.61) and its ROC-AUC is **blank**. `02_modeling` printed the exact values (0.8020 / 0.6426 / 0.5722 / 0.6054 / **0.8489**). Table VII below uses the exact ones.
- `02_modeling` cell 21 re-fits the same parameters and reports AUC 0.8553 / precision 0.5409 / F1 0.6382, slightly different from the saved pickle (0.8555 / 0.5399 / 0.6375). The paper matches the pickle (what the app runs), which is right. Mention in the README that the pickle is canonical.

### Table VII (new): model comparison, test set, exact notebook values

| Model | Acc. | Prec. | Rec. | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80.2 | 64.3 | 57.2 | 60.5 | 84.9 |
| Random Forest | 79.4 | 63.7 | 52.1 | 57.4 | 84.0 |
| XGBoost (baseline, `scale_pos_weight`) | 75.7 | 52.8 | 77.0 | 62.7 | 84.9 |
| Tuned XGBoost, threshold 0.50 | 80.3 | 65.8 | 54.0 | 59.3 | 85.5 |
| **Tuned XGBoost, threshold 0.30 (deployed)** | 76.5 | 54.0 | 77.8 | 63.7 | 85.5 |

Majority-class accuracy: 73.5%. Add "(%)" in the caption.

## 3. IEEE format checklist

| Item | Status |
|---|---|
| A4, two columns, Times 10 pt, Roman-numeral section heads, italic subsections | OK |
| Abstract: single paragraph, 213 words (limit ~150-250), no citations | OK |
| Equations numbered (1)-(3), right-aligned; math is standard and correct | OK |
| Table captions above (TABLE I, small caps), figure captions below ("Fig. 1.") | OK |
| Fonts embedded | **No** (see 1.8) |
| Author block: "1st Gracy" has no surname; no email/ORCID line (IEEE template line 5; roll numbers are not a substitute) | Fix |
| Index Terms: customary alphabetical order; capitalise consistently | Minor |
| Every table and figure called out in text before it appears | **Fix**: Table I, III, IV and Figs. 2, 3, 4 are never cited in the body |
| Abbreviations defined at first use | **Fix**: XAI (used in Intro before being defined in II-B), ROC-AUC (never), CLTV (used in Table I, defined in VI), CRM (never) |
| Table II: units in headers ("Tenure (months)", "Monthly charge (USD)") | Fix |
| Numbers: spell out one-nine ("The 4 continuous variables" -> "The four") | Minor |
| "33 initial features" -> "33 columns" (they include ID, location and target columns) | Minor |
| Fig. 4 is a table drawn as a figure | Replace with `fig4` heatmap |
| Acknowledgments section | **Missing** (needed for AI disclosure, section 6) |
| Code/data availability (repo URL as a reference or footnote) | Recommended |
| Informal wording: "huge churn rate", "very optimized", "Serious ... efforts", "one step further", "surgical" | Fix (section 6) |
| Reference formatting | Section 5 |

## 4. Figures: what to drop in where

All in `figures/` as 300-dpi PNG **and** vector PDF (fonts embedded, Times-compatible, 3.5 in / 7.16 in widths, markers/hatching so they survive black-and-white printing). Regenerate with `python3 paper_review/scripts/make_paper_figures.py`.

| File | Use | Source of numbers |
|---|---|---|
| `fig1_churn_by_contract_internet` | Replaces Fig. 1 placeholder. Caption: "Churn rate by (a) contract type and (b) internet service. M2M: month-to-month." | Computed from the real data; identical to notebook 01 |
| `fig2_threshold_sweep` | Replaces Fig. 2 placeholder | `results/threshold_analysis.csv` |
| `fig3_shap_vs_gain` | New figure next to Table IV (visualises SHAP vs gain; labelled at one-hot level, which is what the data is) | `results/shap_feature_importance.csv`, `xgboost_feature_importance.csv` |
| Fig. 3 beeswarm | Your placeholder wants a beeswarm. `results/shap_summary.png` is real but has raw `cat__` labels. **Run `make_data_dependent_figures.py`** for a clean one (`fig3_shap_beeswarm_clean`). | needs dataset |
| `fig4_confusion_matrix` | Replaces the table-style "Fig. 4" | notebook 02 cell 83 |
| `fig5_model_comparison` (two-column) | New; goes with Table VII | notebook 02 exact outputs |
| `fig6_case_study_drivers` (two-column) | New; goes with Table VI | `results/example_customer_explanations.json` |
| `fig7_roc_curves` | New; needs dataset | `make_data_dependent_figures.py` |

`make_data_dependent_figures.py` also recomputes **Table IV per original feature** and first checks the saved model reproduces 787/248/83/291. If that check prints `False`, the data or split differs and nothing it produces should be used. I tested the script on synthetic data to confirm it runs end to end (and that the check fails on wrong data); I deleted those outputs. **I could not run it on the real data because the dataset is gitignored and not in this environment.** Run it locally, or upload the xlsx and I will.

Number figures in order of first mention in the text; add the missing callouts ("as shown in Fig. 2", "Table IV lists...").

## 5. References

**Problems in the current list**
- Journal titles must use IEEE abbreviations ("J. Mach. Learn. Res.", "Nat. Mach. Intell."), and conference entries need the city ([3], [4], [8]-[11]).
- [1] dataset: needs year and URL ("[Online]. Available: ..."). [6] book: needs edition, year, publisher/URL. [4]: add page range. [7]: add access date, and cite it.
- Section II-A makes four literature claims ("widely studied", "early papers", "RF and GBM are more efficient", "ensembles are opaque") with **zero citations**, and no churn-prediction paper appears in the list at all. Section II-B mentions LIME with no citation.
- [11] Davis & Goadrich is about PR-vs-ROC curves; it does not show false negatives cost more than false positives. Cost-sensitive thresholds are Elkan's topic. Davis & Goadrich *is* a fine citation for using precision-recall analysis under class imbalance (Section VI).

**Added sources I verified exist** (checked by web search; read them before citing): Elkan (IJCAI 2001, pp. 973-978), Ribeiro et al. (LIME, KDD 2016, pp. 1135-1144), Ahmad et al. (J. Big Data 2019, art. 28, doi 10.1186/s40537-019-0191-6; an XGBoost-best telecom churn study). I looked at Vafeiadis et al. (Simul. Model. Pract. Theory, 2015) but could not confirm volume/pages from here, so it is **not** in the list below.

**Proposed list, in order of first citation** (assumes the citations are placed as described; old number in brackets):

1. F. Doshi-Velez and B. Kim, "Towards a rigorous science of interpretable machine learning," 2017, arXiv:1702.08608. *(old 10; cite where "two customers at 70% risk" is argued)*
2. C. Elkan, "The foundations of cost-sensitive learning," in *Proc. 17th Int. Joint Conf. Artif. Intell. (IJCAI)*, Seattle, WA, USA, 2001, pp. 973-978. *(new; replaces old 11 in the Introduction)*
3. C. Molnar, *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*, 2nd ed., 2022. [Online]. Available: https://christophm.github.io/interpretable-ml-book/ *(old 6; **confirm edition, year, URL**)*
4. S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Proc. Adv. Neural Inf. Process. Syst. (NIPS)*, vol. 30, Long Beach, CA, USA, 2017, pp. 4765-4774. *(old 4)*
5. A. K. Ahmad, A. Jafar, and K. Aljoumaa, "Customer churn prediction in telecom using machine learning in big data platform," *J. Big Data*, vol. 6, Art. no. 28, 2019, doi: 10.1186/s40537-019-0191-6. *(new; II-A)*
6. M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, San Francisco, CA, USA, 2016, pp. 1135-1144. *(new; II-B, LIME)*
7. S. M. Lundberg *et al.*, "From local explanations to global understanding with explainable AI for trees," *Nat. Mach. Intell.*, vol. 2, no. 1, pp. 56-67, 2020. *(old 5; also the correct cite for TreeExplainer in II-B)*
8. IBM, "Telco customer churn," IBM Cognos Analytics sample dataset, [year]. [Online]. Available: [URL you downloaded it from]. *(old 1; authors must fill year/URL, I could not reach IBM's page)*
9. F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825-2830, 2011. *(old 2)*
10. L. Breiman, "Random forests," *Mach. Learn.*, vol. 45, no. 1, pp. 5-32, 2001. *(old 9)*
11. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining*, San Francisco, CA, USA, 2016, pp. 785-794. *(old 3)*
12. J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," *J. Mach. Learn. Res.*, vol. 13, pp. 281-305, 2012. *(old 8)*
13. J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," in *Proc. 23rd Int. Conf. Mach. Learn. (ICML)*, Pittsburgh, PA, USA, 2006, pp. 233-240. *(old 11; move to Section VI)*
14. "SHAP documentation." [Online]. Available: https://shap.readthedocs.io (accessed Nov. 15, 2026). *(old 7; cite in VII: "implemented with the `shap` Python package [14]")*

Old -> new: 10->1, 11->13 (moved), 6->3, 4->4, 1->8, 2->9, 9->10, 3->11, 8->12, 5->7, 7->14.
I confirmed [2], [5], [6] by search. The rest are from standard bibliographic knowledge; spot-check pages against the publisher page before submission.

## 6. Suggested text fixes (adapt to your own voice)

**T1 (III-B, the highlighted sentence).** "Churned customers also paid more per month on average (USD 74.44 versus USD 61.27), which, together with their short median tenure, suggests that price may contribute to early-tenure attrition."

**T2 (V-C, last sentence).** "The baseline XGBoost model used `scale_pos_weight` (about 2.8, the ratio of non-churners to churners) to counter class imbalance. The tuned model did not; for it, imbalance is handled by moving the decision threshold (Section VI)."

**T3 (V-D).** "We ran a random search over a six-dimensional space (Table III) with 20 sampled configurations and 5-fold stratified cross-validation (100 fits), scored by ROC-AUC. The best configuration reached a cross-validated ROC-AUC of 0.865 and 0.8555 on the held-out test set."

**T4 (Table IV).** Caption: "Top one-hot features: mean |SHAP| versus XGBoost gain." First column header "Feature (level)", rows e.g. "Contract = Month-to-month". Or regenerate per original feature with the script.

**T5 (VI / XI).** Add to Limitations: "The 0.30 threshold was selected from a sweep on the test set. The reported test metrics at that threshold are therefore optimistic; selecting the threshold on out-of-fold training predictions would remove this bias." Also: "Results come from a single 80/20 split without confidence intervals."

**T6 (X-A).** "At the 0.30 threshold, the model identifies 291 of 374 churners (recall 77.8%) and misses 83 (22.2%), at the cost of 248 false alarms (precision 54.0%). Accuracy is 76.5%, against 73.5% for always predicting 'stay'; ROC-AUC (85.5%) is reported because it does not depend on the threshold." Drop "minimal" and "very profitable"; if you want a cost argument, state the assumed retention cost and customer value and compute it.

**T7 (IX).** "Batch mode scores the whole file with vectorised operations (up to 20,000 rows)". Table V caption: "Representative rules (15 in total)".

**T8 (VIII-A).** "The model uses 20 features: 15 actionable (e.g., contract, services, billing) and 5 non-actionable (gender, senior-citizen status, partner, dependents, CLTV)."

**T9 (Acknowledgments, new).** See next section.

## 7. AI-usage review (read this part carefully)

**What I can and cannot tell you.** There is no reliable way to compute an "AI percentage" for text. Detectors give false positives, especially on technical writing and on non-native English, and I won't hand you a number that looks like evidence but isn't. What I could measure:

- **Overlap with the repository's AI-drafted prose** (`docs/xai_methodology.md`, README, notebook markdown, code comments): the paper shares **one** 8-word run with them (0.2% of the paper), and it is the app's real explanation template ("... is contributing to a higher predicted churn risk"). So the paper was **not** pasted from the repo's methodology doc. That does not say whether AI helped draft the paper itself; only the authors know that.
- **Style markers** (counts in the paper): "very" x8, "highly" x6, "end-to-end" x4, "mathematically" x4, "robust" x3, "comprehensive" x2, "bridging" x2, plus "surgical recommendations", "showcasing the granular utility", "highly functional blueprint", "a very profitable trade-off", "one step further". These are promotional intensifiers. A reviewer will mark them as unsupported whoever wrote them, and removing them (see section 2) also makes the paper more accurate.
- **Uneven voice:** Sections I-III contain plain, idiosyncratic phrasing ("Serious threshold optimization efforts", "very optimized XGBoost"), while VII-X read more polished and templated. A mixed register is something examiners notice. The best fix is for the authors to rewrite the polished passages in their own words, working from the numbers above.

**What I will and won't do.** I'm glad to make the paper more precise and less hyped (the edits above). I won't rewrite it to make AI-assisted text evade detection.

**Disclosure.** IEEE's author guidelines require that AI-generated content ("text, figures, images, and code") be disclosed in the **Acknowledgments**, naming the system and the affected sections and the extent of use; AI used only for grammar/editing is recommended-but-not-required to disclose, and authors remain fully responsible for everything ([IEEE guidance](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/)). Wording varies a little between summaries, so check the current IEEE Author Center page and Chitkara University's own rule.

For this project specifically: the SHAP explanation code, the notebook, the methodology document, and the figure/check scripts in this folder were developed with an AI coding assistant. If that is what you used, the Acknowledgments should say so. Template, to be edited to match what actually happened:

> *Acknowledgment.* An AI coding assistant ([tool name]) was used to help develop the SHAP explanation module, analysis notebooks, and figure-generation scripts. The authors designed the study, reviewed and tested all code and results, wrote the final manuscript, and take full responsibility for its content.

How you handle this with your supervisor is your decision, but the venue's rule is disclosure, and a paper with an accurate disclosure is safer than one without.

## 8. Not done / limits

- I cannot edit your PDF. The source is a Word file, so apply the fixes there; figures are ready to insert.
- Figs. 3 (clean beeswarm) and 7 (ROC) and the per-feature Table IV need the dataset; script provided and tested on synthetic data only.
- I did not touch Gracy's CSV/notebooks. Nothing here is committed or pushed; `paper_review/` is untracked.
- Reference details: three verified by search (Elkan, Ribeiro, Ahmad); the others should be spot-checked. IBM dataset year/URL and Molnar edition must be confirmed by you.
