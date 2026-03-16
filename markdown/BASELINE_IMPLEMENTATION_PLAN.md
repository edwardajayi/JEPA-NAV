# JEPA-NAV Baseline Implementation Plan (Midterm)

**Baseline = V-JEPA 2.** You build on it; you don’t replace it with something else. This plan is for implementing the **V-JEPA 2 baseline** first, then adding JEPA-NAV (affordance + uncertainty + risk-aware planning) on top.

---

## 1. Role of V-JEPA 2 and JEPA-NAV

From **proposal.tex**:

- **Baseline**: **V-JEPA 2** (Assran et al.) — self-supervised video model that gives predictive latent representations. This is your visual backbone and the system you compare against.
- **JEPA-NAV (your method)**: Same backbone (V-JEPA 2) + **affordance prediction** (walkable / obstacles / hazards) + **uncertainty** + **risk-aware path planning**.

So:

| System        | What it is |
|---------------|------------|
| **Baseline**  | V-JEPA 2 embeddings → policy (or simple planner) for PointNav. |
| **JEPA-NAV**  | V-JEPA 2 embeddings → **affordance head** → **uncertainty** → **risk-aware planner** → navigation. |

You are **not** building a ResNet or “random” baseline. You are implementing the V-JEPA 2 pipeline first, then extending it.

---

## 2. What to Implement for the Baseline (V-JEPA 2)

Implement a minimal **V-JEPA 2–based** navigation pipeline:

1. **Load V-JEPA 2**  
   Use the official V-JEPA 2 model (or a compatible checkpoint). Input: RGB (and optionally video). Output: latent embeddings per frame (or sequence).

2. **Plug into Habitat**  
   Simulator: Habitat (Habitat-Lab) with Gibson / Matterport3D / Replica. Task: **PointNav** (goal position, reach without collision). Observations: RGB (and optionally depth) from the agent.

3. **Baseline agent**  
   - Observation (RGB) → **V-JEPA 2 encoder** → embedding.  
   - Embedding (+ goal encoding) → **policy** (e.g. MLP or LSTM) → actions (forward, turn left/right, stop, etc.).  
   - Train with PPO (or IL from shortest path) for PointNav.  
   No affordance head, no uncertainty, no risk-aware planner yet — that’s JEPA-NAV.

4. **Metrics**  
   Same as in the proposal: **success rate**, **collision rate**, **SPL**. Same eval episodes you’ll use for JEPA-NAV.

This gives you a **V-JEPA 2 baseline** that you will directly extend (same encoder, add heads and planner).

---

## 3. Implementation Order

### Phase 1: V-JEPA 2 baseline (midterm)

1. **V-JEPA 2 setup**  
   - Get V-JEPA 2 code and weights (e.g. from the paper/repo).  
   - Wrapper: RGB frame(s) → V-JEPA 2 → embedding vector (or sequence).  
   - Optionally support video (e.g. last K frames) if the model expects it.

2. **Habitat setup**  
   - Install Habitat-Lab (+ Habitat-Sim).  
   - Download a small dataset (e.g. Gibson or Matterport3D).  
   - PointNav config: RGB (and optionally depth) observations.

3. **Baseline agent**  
   - In the training loop: get RGB from Habitat → V-JEPA 2 (frozen or finetuned) → embedding → policy → action.  
   - Train with PPO (or IL).  
   - Eval on a fixed set of episodes; log success rate, collision rate, SPL.

4. **Midterm report**  
   - **Baseline**: “V-JEPA 2 embeddings + policy for PointNav in Habitat.”  
   - **Results**: Table with success rate, collision rate, SPL.  
   - **Reproducibility**: commands, config, dataset, seeds.

### Phase 2: JEPA-NAV (after midterm)

- Keep V-JEPA 2 encoder.  
- Add: **affordance head** (walkable / obstacles / hazards), **uncertainty** (e.g. ensemble or probabilistic head), **risk-aware planner** (cost map from affordances + uncertainty → path or policy).  
- Same metrics + **affordance accuracy**, **uncertainty calibration**.  
- Compare **JEPA-NAV vs V-JEPA 2 baseline** (same encoder, same env, same metrics).

---

## 4. Suggested Code Layout

```
JEPANAV/
├── README.md
├── requirements.txt
├── configs/
│   └── pointnav_vjepa2.yaml       # Habitat + V-JEPA 2 baseline
├── src/
│   ├── env/
│   │   └── habitat_env.py         # Habitat env wrapper
│   ├── encoders/
│   │   └── vjepa2.py             # V-JEPA 2 loader + forward (your baseline encoder)
│   ├── models/
│   │   ├── affordance_head.py     # JEPA-NAV only (Phase 2)
│   │   └── policy.py             # policy network (baseline: emb + goal → action)
│   ├── agents/
│   │   ├── baseline_agent.py     # V-JEPA 2 + policy (baseline)
│   │   └── jepa_nav_agent.py     # V-JEPA 2 + affordance + uncertainty + planner (Phase 2)
│   ├── train.py
│   └── eval.py
├── scripts/
│   ├── train_baseline.sh          # train V-JEPA 2 baseline
│   └── eval_baseline.sh
└── data/                          # dataset paths / V-JEPA 2 checkpoints
```

The **baseline** is everything in `encoders/vjepa2.py` + `policy.py` + `baseline_agent.py`. JEPA-NAV adds the rest.

---

## 5. Dependencies

- **V-JEPA 2**: Official repo / checkpoints (e.g. PyTorch); see Assran et al. and any released code.
- **Habitat**: `habitat-lab`, `habitat-sim`.
- **PyTorch**, and whatever V-JEPA 2 requires (e.g. `timm`, specific torch versions).

---

## 6. Summary

| Item | Action |
|------|--------|
| **Baseline** | **V-JEPA 2** embeddings + policy for PointNav in Habitat. |
| **Not baseline** | ResNet, random agents, or other encoders you won’t use later. |
| **Midterm** | Implement and report V-JEPA 2 baseline (metrics: success, collision, SPL). |
| **Next** | Add affordance, uncertainty, risk-aware planning on top of the same V-JEPA 2 pipeline = JEPA-NAV. |

Your baseline is the same representation you will use in JEPA-NAV; the only difference is that the baseline has no affordance, no uncertainty, and no risk-aware planner. No extra “throwaway” baselines.
