# Road to NeurIPS: 8-Week Paper Plan

**Goal:** Submit a complete, review-ready JEPA-NAV paper to NeurIPS (Main Track) and maximize acceptance probability.

**Target:** Abstract deadline ~Week 7, full submission ~Week 8 (align with [NeurIPS dates](https://neurips.cc/Conferences/2026/Dates): abstract typically early May, full paper shortly after).

---

## Part 1: What NeurIPS Looks For

### 1.1 Core acceptance criteria

- **Novelty:** Clear contribution (first JEPA-based assistive indoor nav; uncertainty-aware affordance; risk-aware planning).
- **Rigor:** Solid baseline (V-JEPA 2), fair comparison, ablations, and statistical reporting (error bars, seeds, multiple runs).
- **Reproducibility:** Enough detail (and ideally code/data) for others to reproduce main results.
- **Clarity:** Claims in abstract/intro match results; limitations and societal impact discussed.
- **Checklist:** All NeurIPS checklist items answered with justification (no TODO left).

### 1.2 NeurIPS checklist (must complete; desk reject if missing)

| # | Topic | What you need |
|---|--------|----------------|
| 1 | **Claims** | Abstract & intro state contributions and scope; match experiments. |
| 2 | **Limitations** | Dedicated limitations section: assumptions, scope, failure modes, compute. |
| 3 | **Theory** | If no theorems, answer NA; else full assumptions + proof. |
| 4 | **Reproducibility** | Full disclosure of setup: data splits, hyperparameters, optimizer, seeds. |
| 5 | **Code & data** | Open access or clear justification; instructions to reproduce. |
| 6 | **Experimental details** | Training/test details, hyperparameter choice, optimizer. |
| 7 | **Statistical significance** | Error bars / CIs / significance tests; state what they represent. |
| 8 | **Compute** | GPU/CPU, memory, time per run and total. |
| 9 | **Code of ethics** | Confirm adherence to NeurIPS ethics guidelines. |
| 10 | **Broader impact** | Positive and negative societal impact discussed. |
| 11 | **Safeguards** | If releasing models/data with misuse risk, describe safeguards (or NA). |
| 12 | **Licenses** | Credit and license for used code/data; state versions. |
| 13+ | **New assets, crowdsourcing, IRB, LLM** | Answer per guidelines (NA where not applicable). |

Plan experiments and writing so each checklist item can be answered **Yes** or **NA** with a short justification; **No** only with strong justification.

---

## Part 2: 8-Week Roadmap

Assume **Week 1** = start of the 8-week countdown; **Week 8** = submission week.

### Week 1: Baseline locked & full pipeline running

**Experiments**

- Finalize V-JEPA 2 baseline: train/eval on Habitat PointNav (Gibson or Matterport3D).
- Log and save: success rate, SPL, collision rate (mean ± std over seeds, e.g. 3–5 seeds).
- Get JEPA-NAV full pipeline running end-to-end: V-JEPA 2 → affordance head → uncertainty → risk-aware planner → policy/actions.
- Even if early results are weak, ensure one full training run completes and metrics are logged.

**Deliverables**

- Baseline results table (with std).
- At least one JEPA-NAV training curve (reward/success vs steps).
- Scripts: `train_baseline.sh`, `train_jepa_nav.sh`, `eval_*.sh` with fixed seeds.

**Checklist prep**

- Document: data splits, number of scenes/episodes, train vs eval.
- Note compute: GPU type, hours per run (baseline + JEPA-NAV).

---

### Week 2: Ablations & main comparison

**Experiments**

- **Main table:** V-JEPA 2 baseline vs JEPA-NAV (same env, same metrics). Multiple seeds (e.g. 3–5); report mean ± std.
- **Ablations (pick 2–3):**
  - JEPA-NAV w/o uncertainty (affordance only).
  - JEPA-NAV w/o risk-aware planner (affordance + uncertainty but standard planner).
  - Optional: JEPA-NAV with frozen vs finetuned V-JEPA 2.
- **Affordance accuracy:** If you have ground-truth walkable/obstacle (e.g. from depth/semantics in sim), report accuracy or IoU for affordance head.
- **Uncertainty calibration:** If applicable, reliability diagram or ECE for affordance confidence.

**Deliverables**

- Main results table (baseline vs JEPA-NAV vs ablations).
- Affordance and/or calibration metrics.
- Figures: training curves (baseline + JEPA-NAV), optional calibration plot.

**Checklist prep**

- Document hyperparameters, optimizer, learning rate schedule.
- Document how error bars are computed (e.g. std over seeds).

---

### Week 3: Generalization & robustness

**Experiments**

- **Unseen scenes:** Eval on scenes not seen during training (separate split or different dataset, e.g. train on Gibson, eval on Matterport3D or held-out Gibson).
- Report success rate, SPL, collision rate for “seen” vs “unseen.”
- Optional: different episode lengths, different goal distances, or lighting/occlusion if the sim supports it.

**Deliverables**

- Generalization table or plot (seen vs unseen).
- Short paragraph on failure cases (for Limitations).

**Checklist prep**

- Clearly state train vs test scene split; state that test scenes are never used for training.

---

### Week 4: Paper draft (full structure)

**Writing**

- **Sections to complete in draft form:** Abstract, Introduction, Related Work, Method (with equations/figures for pipeline), Experiments (setup, main results, ablations, generalization), Discussion, Limitations, Conclusion.
- **Figures:** Pipeline diagram; main results table; training curves; 1–2 qualitative examples (e.g. trajectory overlay, affordance map).
- **Appendix (start):** Training details, hyperparameters, extra ablations, dataset stats.

**Deliverables**

- First full draft in `neurips_2025.tex` (or copy to `jepa_nav_paper.tex`).
- All main figures in `figures/` and included in draft.
- References in `references.bib`; cite all compared methods and datasets.

**Checklist prep**

- Draft Limitations section (assumptions, sim-only, compute, failure modes).
- Draft Broader impact (positive: assistive tech; negative: reliance on vision, accessibility of deployment).

---

### Week 5: Experiments polished & appendix

**Experiments**

- Re-run any missing seeds or ablations.
- Ensure all numbers in the paper match logged results (no placeholder “TBD”).
- Add 1–2 supplementary experiments if time (e.g. different encoder size, different number of frames for V-JEPA 2).

**Writing**

- Finalize Experiments section: all tables and figures with captions.
- Complete Appendix: full hyperparameters, data splits, compute, extra results.
- Implement or document exact commands to reproduce each table/figure.

**Deliverables**

- Paper draft with all experimental numbers filled in.
- Appendix with reproducibility details.
- `README.md` in repo: install, data, commands for baseline and JEPA-NAV.

**Checklist prep**

- Fill in “Experimental setting/details” and “Statistical significance” from appendix and main text.
- Fill in “Compute resources” from your logs.

---

### Week 6: Checklist, ethics, impact & revision

**Writing**

- **NeurIPS checklist:** Answer every item (Claims, Limitations, Reproducibility, Code & data, etc.); add 1–2 sentence justification for each; remove all \answerTODO{}.
- **Broader impact:** Expand positive (independence for visually impaired) and negative (dependence on vision system, need for validation with users).
- **Limitations:** Explicit list: simulation only, single dataset/sim, no real-user study, compute cost, possible failure modes.
- **Code & data:** Public repo link; anonymized if required. Instructions for code, data, and checkpoints (or clear “available on request” + justification if not open).
- **Licenses:** Cite V-JEPA 2, Habitat, datasets; state licenses (or “see appendix”).

**Deliverables**

- Checklist fully filled in the LaTeX source.
- Impact and limitations sections ready for submission.
- Repo README with reproducibility instructions (and anonymization if needed).

---

### Week 7: Abstract deadline & internal review

**Submission**

- **Abstract submission:** Submit abstract by NeurIPS abstract deadline (check neurips.cc for exact date).
- **Internal review:** Swap with teammate or colleague; fix clarity, typos, and any claim that overstates results.

**Writing**

- Short abstract (one paragraph): problem, gap, method, main result (e.g. “JEPA-NAV improves over V-JEPA 2 baseline by X% success with lower collision rate and better generalization to unseen scenes”).
- Ensure abstract claims are supported by the results in the paper.

**Deliverables**

- Abstract submitted.
- Revised draft incorporating internal feedback.

---

### Week 8: Full submission & supplements

**Submission**

- **Full paper:** Final PDF (US Letter, within page limit, no anonymous violations).
- **Supplementary:** Single PDF or ZIP if needed (appendix, code snapshot, or extra results).
- **Code:** Public or anonymized repo link in paper and submission form.

**Final checks**

- [ ] All checklist items answered; no TODO.
- [ ] All figures and tables have captions and are referenced in text.
- [ ] References consistent; no broken citations.
- [ ] PDF fonts: Type 1 or embedded TrueType (pdflatex default).
- [ ] Acknowledgments/funding excluded from anonymized version (use \texttt{ack} or comment out).
- [ ] Abstract and intro match the reported contributions and numbers.

**Deliverables**

- Final PDF submitted.
- Supplementary and code link submitted as required.
- Local copy of submitted PDF and source for camera-ready if accepted.

---

## Part 3: Paper Structure (NeurIPS 9-page limit)

Suggested section lengths (in pages, approximate):

| Section | Length | Content |
|--------|--------|--------|
| Abstract | 0.25 | Problem, gap, method, main result. |
| Introduction | 0.75–1 | Motivation, problem definition, contributions (bulleted). |
| Related Work | 0.75–1 | Assistive nav; JEPA/predictive vision; gap you fill. |
| Method | 2–2.5 | Pipeline (V-JEPA 2 → affordance → uncertainty → planner); equations; figure. |
| Experiments | 2–2.5 | Setup, main table, ablations, generalization; training curves. |
| Discussion / Limitations | 0.5 | Short; or fold into Experiments. |
| Conclusion | 0.25 | Summary and future work. |
| References | — | Not counted in limit. |
| Appendix | No limit | Details, extra ablations, checklist evidence. |

Use **booktabs** for tables; reference figures as “Figure 1”, “Table 1” in text.

---

## Part 4: Success Metrics for Acceptance

- **Strong comparison:** JEPA-NAV beats V-JEPA 2 baseline on success rate and/or collision rate and/or SPL with consistent trends across seeds.
- **Generalization:** Clear improvement or at least no large drop on unseen scenes vs baseline.
- **Ablations:** At least one ablation showing the role of uncertainty or risk-aware planning.
- **Reproducibility:** Code + README + configs; main results reproducible with stated commands.
- **Checklist:** Every item answered; no desk reject for missing checklist.
- **Clarity:** Reviewer can understand problem, method, and experiments in one read.

---

## Part 5: Risk Mitigation

| Risk | Mitigation |
|------|------------|
| V-JEPA 2 not available or hard to run | Identify fallback: same pipeline with another JEPA-style or video SSL encoder; state clearly in limitations. |
| JEPA-NAV not better than baseline | Report honestly; stress generalization, affordance accuracy, or calibration; frame as “first step” and ablation analysis. |
| Too little compute | Use smaller splits, fewer seeds, or shorter training; report compute and call it a limitation. |
| Missing generalization eval | At minimum, hold out some scenes for test; report seen vs unseen. |
| Checklist incomplete | Allocate Week 6 specifically to checklist + impact + limitations; do not leave for last day. |

---

## Part 6: Weekly Checklist (summary)

| Week | Focus | Key deliverable |
|------|--------|------------------|
| 1 | Baseline + full JEPA-NAV pipeline | Baseline table; one JEPA-NAV run; scripts |
| 2 | Main comparison + ablations | Main table; ablation table; curves |
| 3 | Generalization | Unseen-scene results; failure-case notes |
| 4 | Full paper draft | Complete draft + figures + appendix start |
| 5 | Numbers locked + appendix | All results in; README; reproducibility |
| 6 | Checklist + impact + limitations | Checklist filled; impact/limitations done |
| 7 | Abstract submit + internal review | Abstract in; revised draft |
| 8 | Full submission | Final PDF + supplements + code link |

---

*Update this plan as you go: tick off deliverables, adjust deadlines to match the official NeurIPS abstract and full-paper dates, and keep the checklist and reproducibility at the center of the last three weeks.*
