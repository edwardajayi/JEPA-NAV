# What This Work Is — and What We're Doing Now

## Image or video?

- **V-JEPA 2** is a **video** model (trained on video; learns predictive temporal representations).
- **At runtime** you can feed either:
  - **Single frames (image-based):** one RGB frame per step → V-JEPA 2 → embedding. Simpler; often enough for PointNav.
  - **Short clips (video-based):** last K frames → V-JEPA 2 → embedding. Can help if the model expects a sequence.
- **Recommendation for midterm:** Start **image-based** (one frame per step). Move to video (frame stack) only if you need it or have time.

So: **representation = video-trained; deployment can be image or video.** For the baseline, image-per-step is fine.

---

## What we're looking for NOW (midterm)

**Midterm = V-JEPA 2 baseline only.** You are **not** detecting "stuff in-house" (walkable, obstacles, hazards) yet. That comes in JEPA-NAV (Phase 2).

| Now (midterm) | Later (JEPA-NAV) |
|---------------|------------------|
| **Task:** PointNav in Habitat (go to goal, avoid collisions) | Same + explicit affordance maps |
| **Input:** RGB (one frame or K frames) | Same |
| **Model:** V-JEPA 2 → embedding → **policy** (actions) | V-JEPA 2 → **affordance head** → uncertainty → **risk-aware planner** → actions |
| **Output:** Actions (forward, turn, stop) | Same + walkable/obstacle/hazard maps |
| **No** explicit detection of "stuff" (floor, furniture, hazards) | **Yes** — predict walkable zones, obstacles, hazards |

So for midterm you are **not** building an in-house "detector" of objects or regions. You are:

1. Getting **V-JEPA 2** loaded and producing embeddings from RGB.
2. Plugging that into **Habitat** (e.g. Replica) for **PointNav**.
3. Training a **policy** (e.g. PPO) that maps embedding + goal → actions.
4. Reporting **success rate, SPL, collision rate**.

"Detecting stuff in-house" (walkable vs obstacle vs hazard) is the **affordance prediction** module you add **after** midterm for JEPA-NAV.

---

## Checklist: what to get done for midterm

- [ ] Habitat installed; Replica (or Gibson) downloaded and loading.
- [ ] V-JEPA 2 loaded; wrapper: RGB frame(s) → embedding.
- [ ] Baseline agent: embedding + goal → policy → actions; trains with PPO (or IL).
- [ ] At least one full training run; log success rate, collision rate, SPL.
- [ ] Midterm report (see MIDTERM.md): baseline description, results table, reproducibility (commands, config).

Once this baseline works, you add the **affordance head** (detecting walkable/obstacles/hazards) and the **risk-aware planner** for the full JEPA-NAV pipeline.
