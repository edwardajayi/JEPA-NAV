# JEPA-NAV: Midterm Report  
## A Vision-Based Joint Embedding Architecture for Safe Indoor Navigation

**Authors:** Edward Ajayi, Emrys Lasidlaus  
**Affiliation:** Carnegie Mellon University Africa, Kigali, Rwanda

---

## Abstract

Navigating indoor environments is challenging for visually impaired individuals due to cluttered layouts, dynamic obstacles, and partial observability. Existing vision-based assistive systems often rely on specialized hardware, fail to generalize to unseen environments, and rarely provide safety-aware predictions with quantified uncertainty. We propose **JEPA-NAV**, a vision-first system that leverages Joint Embedding Predictive Architectures (JEPA)—specifically V-JEPA 2—to learn predictive latent representations of indoor scenes. These embeddings support affordance prediction (walkable areas, obstacles, and risky zones) with associated uncertainty, integrated into a risk-aware navigation planner. By combining predictive latent vision with uncertainty estimation, JEPA-NAV aims to improve navigation safety and generalization across previously unseen indoor layouts. Our midterm (1) establishes the **V-JEPA 2 encoder pipeline** and validates it on **ReplicaCAD** (encoder output shape and statistics); (2) implements the **full baseline**: V-JEPA 2 + MLP policy for PointNav in Habitat-Sim on ReplicaCAD, with training script and SLURM job. ReplicaCAD requires Habitat-Sim built with Bullet for scene and navmesh loading; we are completing that install. Once done, we will run baseline training and report **Success Rate and SPL** as the baseline result. Experiments use the ReplicaCAD dataset (84 scenes) and Habitat-Sim.

---

## Introduction

### Motivation

Indoor navigation remains a significant challenge for visually impaired individuals seeking independent mobility. Effective assistive systems must perceive obstacles, identify walkable paths, and anticipate hazards while accounting for uncertainty. Navigation becomes particularly difficult when models encounter unfamiliar environments, as traditional approaches often fail to generalize and rarely provide safety-aware predictions.

### Problem Definition

We address the problem of **safe, vision-based indoor navigation** that:

1. **Generalizes** to previously unseen indoor layouts without task-specific retraining.  
2. **Predicts** environmental affordances (walkable zones, obstacles, hazards) from visual input.  
3. **Quantifies uncertainty** in these predictions to support risk-aware path planning.  
4. **Operates** in simulation (and potentially on low-cost devices) using primarily RGB input, with optional depth or other modalities.

We use **V-JEPA 2** as our visual backbone and baseline: a policy that maps V-JEPA 2 embeddings (and goal) to navigation actions. Our full method, **JEPA-NAV**, extends this baseline by adding affordance prediction, uncertainty modeling, and a risk-aware planner.

---

## Related Works

### Vision-Based Assistive Navigation

Previous work has explored vision-based support for indoor navigation for visually impaired users. Li et al. proposed a system that constructs semantic indoor maps, performs real-time obstacle detection using an RGB-D camera, and provides multimodal feedback via audio and haptics, demonstrating improved navigation safety and independence. Na et al. focused on walking path generation for a robotic SmartCane, integrating human motion constraints and motion primitives to guide users in indoor environments. These studies highlight the importance of real-time perception and path planning but rely heavily on specialized hardware and system-specific configurations.

### Predictive Visual Representations and JEPA

Recent advances in predictive visual representations offer new opportunities for generalizable navigation. JEPA-VLA integrates predictive video embeddings into vision-language-action models to capture task-relevant temporal dynamics while ignoring unpredictable factors, improving sample efficiency and generalization. **V-JEPA 2** (Assran et al.) is a self-supervised video model trained on large-scale video data to learn joint predictive embeddings capable of understanding, predicting, and planning in physical environments. These representations form the basis for our baseline and for JEPA-NAV.

### Gaps We Address

- **First application of JEPA-based (V-JEPA 2) predictive embeddings for assistive indoor navigation** in a reproducible pipeline.  
- **Uncertainty-aware affordance prediction** on top of JEPA embeddings for risk-aware path planning.  
- **Vision-first, generalizable pipeline** evaluated in standard simulators (Habitat) with public datasets, establishing a reproducible baseline for future research.

---

## Dataset Description

We use **ReplicaCAD** (Habitat) for the midterm experiments. ReplicaCAD is an artist-created 3D indoor dataset based on the FRL apartment from Replica; it is designed for Habitat and supports embodied navigation and interaction.

| Dataset / Simulator | Description | Use in Project |
|---------------------|-------------|----------------|
| **ReplicaCAD** | 84 scene instances (furniture layouts), navmeshes, stages, objects; CC BY 4.0. | RGB frame collection in Habitat; V-JEPA 2 encoder validation. |
| **Habitat-Sim** | 3D simulator; loads ReplicaCAD via `replicaCAD.scene_dataset_config.json`. | Rendering RGB observations; future PointNav training. |

**Dataset statistics (midterm):**

- **Scenes:** 84 ReplicaCAD scene instances (e.g. `apt_0`, `v3_sc0_staging_00` … `v3_sc3_staging_20`).  
- **Frames collected:** 80 RGB frames (256×256) from one scene (`apt_0`) for encoder validation.  
- **Source:** [Hugging Face: ai-habitat/ReplicaCAD_dataset](https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset). Full layout and usage are documented in `markdown/REPLICACAD_DATASET.md`.

*Optional: add a figure with 4–6 sample frames from `data/replicacad_vjepa2_frames/` as Figure 1.*

---

## Baseline Implementation and Results

### Baseline Definition

Our **baseline** is a V-JEPA 2–based navigation agent:

- **Input:** RGB observation (and optionally a short sequence of frames) from the Habitat agent.  
- **Encoder:** V-JEPA 2 (pretrained), producing a latent embedding per frame (or sequence).  
- **Policy:** A network (e.g., MLP or LSTM) that takes the embedding and goal encoding (e.g., relative goal vector) and outputs navigation actions (e.g., forward, turn left, turn right, stop).  
- **Training:** Reinforcement learning (e.g., PPO) or imitation learning from shortest-path demonstrations for the PointNav task in Habitat.  
- **No** affordance head, **no** explicit uncertainty module, and **no** risk-aware planner in the baseline.

This baseline is the same representation we will use in JEPA-NAV; we only add affordance prediction, uncertainty, and risk-aware planning on top.

### What We Have Done So Far (Midterm)

| Phase | What we did | Status |
|-------|-------------|--------|
| **A** | V-JEPA 2 sanity check: `run_vjepa2_on_frames.py` on synthetic + 4 Replica-Dataset images → encoder (1, 2048, 1024). | Done. |
| **B** | ReplicaCAD + V-JEPA 2: `run_replicacad_vjepa2.py` collects 80 RGB frames from Habitat (scene `apt_0`), runs V-JEPA 2; encoder stats validated. SLURM job run successfully. | Done. |
| **C** | **Baseline PointNav agent:** `train_pointnav_vjepa2.py` — Habitat-Sim + ReplicaCAD, V-JEPA 2 (frozen) → pooled 1024-d embedding + goal (distance, θ) → MLP policy → actions (forward / turn left / turn right / stop). Policy gradient training; episode generator and SLURM script `train_pointnav_vjepa2.slurm` in place. | Implemented; **blocked on Habitat-Sim with Bullet** (ReplicaCAD scene + navmesh require Bullet; conda build has no Bullet). We are installing Habitat-Sim from source with Bullet; once installed, we will run training and report Success Rate (and SPL). |

**Implementation details:**

- **V-JEPA 2:** `facebook/vjepa2-vitl-fpc64-256` (ViT-L); 16-frame clip (256×256) → mean-pooled 1024-d vector.  
- **Policy:** MLP(1024 + 2 → 256 → 4 actions), actor–critic; goal = (distance to target, angle θ).  
- **Task:** PointNav on ReplicaCAD (`apt_0`); success = within 0.36 m of goal.  
- **Reproducibility:** `scripts/install_habitat_sim_with_bullet.sh` documents Bullet install; `scripts/slurm/train_pointnav_vjepa2.slurm` runs training on a GPU node.

### Metrics: What We Report and Why

We use the **standard PointNav metrics** (Anderson et al.; Habitat benchmarks) so our results are comparable to the literature and our experiment is valid:

- **Success Rate (%):** Fraction of episodes where the agent reaches the goal (within 0.36 m). Standard in Habitat PointNav.
- **SPL (Success weighted by Path Length):** (1/N) Σ S_i × (l_i / max(p_i, l_i)) with S_i = success, l_i = geodesic shortest path length, p_i = actual path length. Standard efficiency metric; we compute it using the simulator pathfinder for l_i and tracked agent positions for p_i.
- **Collisions per episode (mean):** Count of collision events per episode (from Habitat-Sim `step()` return); we report the mean over episodes. Relevant for assistive/safe navigation.

All three are **implemented** in `train_pointnav_vjepa2.py` (training and eval). They work for our experiment because we use the same task (PointNav), same success threshold (0.36 m), and the same definition of SPL; collision is provided by the sim. We do **not** use accuracy/precision/recall for the baseline (those apply later to the affordance head).
- **Later (JEPA-NAV, affordance):** When we add the affordance head (walkable / obstacle / hazard), we will have a **classification** problem (per region or per patch). Then we will report **accuracy, precision, recall, F1** (per class or macro), and **uncertainty calibration** (e.g. ECE, reliability diagram).

**How we show the pipeline works (midterm):** (1) Encoder: shape and stats on ReplicaCAD frames ✓. (2) Full loop: run eval episodes (fixed start/goal); report **Success Rate** and **SPL** on a held-out set. (3) Optional: plot Success Rate vs. training steps to show learning. No accuracy/precision/recall are needed for the baseline.

### Results

**1. Encoder validation (done):** V-JEPA 2 encoder output on our inputs:

| Setting | Input | Encoder output shape | Mean | Std |
|---------|--------|------------------------|------|-----|
| Phase A (sanity) | 4 images → 16-frame clip (256×256) | (1, 2048, 1024) | 0.013 | 3.23 |
| ReplicaCAD V-JEPA 2 | 80 Habitat frames → 16-frame clip | (1, 2048, 1024) | 0.019 | 3.03 |

**2. Baseline PointNav:** We run `train_pointnav_vjepa2.py` on ReplicaCAD scene `apt_0` (SLURM job on H100). Baseline results are **pending completion of Habitat-Sim built with Bullet** on our cluster (ReplicaCAD requires Bullet for scene and navmesh loading; the install job has been submitted). The table below will be filled as soon as the run completes; all code, metrics, and scripts are in place.

| Metric | Baseline (V-JEPA 2 + policy) |
|--------|-----------------------------|
| Success Rate (%) | Pending (run after Bullet install) |
| SPL | Pending (run after Bullet install) |
| Collisions per episode (mean) | Pending (run after Bullet install) |

**If the Bullet install did not complete before the report deadline:** We report the baseline **by design and implementation**: same metrics (Success Rate, SPL, Collisions/ep) are implemented in `train_pointnav_vjepa2.py`; the SLURM job and checkpointing are ready; only the cluster environment (Habitat-Sim with Bullet) was blocking the run. Baseline numbers will be added to the camera-ready or a short follow-up once the run completes. This does not affect the validity of the methodology or the JEPA-NAV plan.

**Conclusion so far:** The V-JEPA 2 encoder is validated on ReplicaCAD RGB (Phase A and B). The full baseline (encoder + policy for PointNav) is implemented and **collision counting is wired** (Habitat-Sim `step()` returns `collided`; we aggregate per episode). As soon as Habitat-Sim with Bullet is available, we run training and fill the table above, giving a **reproducible baseline** for the NeurIPS comparison (JEPA-NAV vs. baseline on same metrics).

---

## Methodology

### How JEPA-NAV Extends the Baseline

JEPA-NAV keeps the **same V-JEPA 2 encoder** as the baseline and adds:

1. **Affordance prediction:** A head on top of V-JEPA 2 embeddings that predicts spatial affordances: walkable regions, obstacles, and hazard zones (e.g., as a spatial map or per-patch labels).  
2. **Uncertainty modeling:** Probabilistic output or ensemble over the affordance head so that each prediction has an associated confidence or uncertainty (e.g., entropy or variance).  
3. **Risk-aware path planning:** A planner that uses the affordance map and uncertainty to produce paths that minimize collision risk and prefer high-confidence walkable regions. The agent then follows this plan (or a policy conditioned on it).

This directly addresses the gaps identified in related work: generalizable predictive representations (V-JEPA 2) plus explicit affordance and uncertainty for safer, assistive indoor navigation.

### Pipeline Overview

```
RGB (video) → V-JEPA 2 → embeddings
                ↓
         Affordance head → walkable / obstacle / hazard map + uncertainty
                ↓
         Risk-aware planner → path or cost map
                ↓
         Policy / controller → actions
```

*[ADD: A simple diagram or figure of the full JEPA-NAV pipeline if available.]*

### Planned Evaluation (Post-Midterm)

- Same metrics as baseline: Success Rate, SPL, Collision Rate.  
- Additional: Affordance prediction accuracy (vs. ground truth from simulator), uncertainty calibration (e.g., reliability diagram or ECE).  
- Comparison: JEPA-NAV vs. V-JEPA 2 baseline on the same eval episodes and scenes.

---

## Division of Work

| Team Member | Contributions |
|-------------|----------------|
| **Edward Ajayi** | *[FILL: e.g., literature review, V-JEPA 2 integration, ReplicaCAD pipeline, SLURM/experiments, report writing.]* |
| **Emrys Lasidlaus** | *[FILL: e.g., Habitat/ReplicaCAD setup, dataset docs, baseline design, report writing.]* |

*[FILL: One sentence on how you split tasks and synced (e.g. “We split encoder pipeline vs Habitat/dataset; merged for ReplicaCAD run and report.”)]*

---

## Reproducibility and Code

All code, configs, and instructions needed to reproduce the baseline (and later JEPA-NAV) results are available in the following public repository:

- **GitHub:** *[ADD: e.g., https://github.com/your-org/JEPA-NAV]*

The repository includes:

- *[ADD: e.g., README with setup (Habitat, V-JEPA 2, dependencies), data download links, training and evaluation commands, config files, and seed information.]*

*[ADD: If the repo is not yet public, state “Repository will be made public upon course deadline” and add the link once it is available.]*

---

## References

- Assran et al., V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv, 2025.  
- Li et al., Vision-based mobile indoor assistive navigation aid for blind people. IEEE TMC, 2018.  
- Na et al., Improving walking path generation through biped constraint in indoor navigation system for visually impaired individuals. IEEE TNSRE, 2024.  
- Miao et al., JEPA-VLA: Video Predictive Embedding is Needed for VLA Models. arXiv, 2026.  
- Yadav et al., Habitat-Matterport 3D semantics dataset. CVPR, 2023.  
- Straub et al., The Replica Dataset: A Digital Replica of Indoor Spaces. arXiv, 2019.  
- Yokoyama et al., Benchmarking augmentation methods for learning robust navigation agents (iGibson challenge). IROS, 2022.

---

---

## NeurIPS Readiness: Where the Midterm Stands

This project targets a **NeurIPS submission** (vision + embodied AI / assistive tech). For that, the midterm must establish a **solid, reproducible baseline** and a **clear path** to the full method and experiments.

**What is already strong:**

| Criterion | Status |
|-----------|--------|
| **Problem & scope** | Clearly defined: safe, vision-based indoor navigation for visually impaired; generalizes to unseen layouts; affordance + uncertainty + risk-aware planning. |
| **Baseline definition** | Unambiguous: V-JEPA 2 (frozen) + policy for PointNav; no affordance/uncertainty/planner. Same encoder used in JEPA-NAV. |
| **Encoder validation** | Done: shape and statistics on ReplicaCAD; pipeline runs. |
| **Full baseline implementation** | Done: training script, collision counting, SLURM job; only blocked on Habitat-Sim + Bullet install. |
| **Metrics** | Success Rate, SPL, Collisions per episode—defined and implemented; later: affordance accuracy, uncertainty calibration. |
| **Methodology** | JEPA-NAV extension (affordance head → uncertainty → risk-aware planner) described; comparison plan (same episodes, same metrics) stated. |
| **Reproducibility** | Install script, data source (ReplicaCAD/Hugging Face), commands documented; GitHub placeholder. |

**What to complete for a solid midterm → NeurIPS bridge:**

1. **Baseline results:** Once the Habitat-Sim Bullet install finishes, run training and fill Success Rate, SPL, Collisions/ep in the results table. If the run is not ready by the deadline, the report still stands (implementation and metrics are complete; numbers pending).
2. **Fill [FILL] placeholders:** Division of work, GitHub link (or “will be made public by [date]”).
3. **Optional but recommended:** One figure (e.g. sample ReplicaCAD frames or pipeline diagram); seed and compute noted for the baseline run.
4. **Post-midterm:** Follow the 8-week roadmap in `markdown/PAPER_PLAN.md` (baseline lock → JEPA-NAV pipeline → ablations → main table → writing → checklist). The midterm baseline is **Week 1** of that plan.

**Bottom line:** The midterm plan is **structurally sound** for NeurIPS: clear baseline, same metrics for future comparison, collision (safety) measured, and a defined extension (JEPA-NAV). Delivering the baseline numbers and filling the placeholders will make it a strong foundation for the full paper.

---

*Before submission: fill [FILL] placeholders (division of work, GitHub link). Optionally add a figure from `data/replicacad_vjepa2_frames/` in `tex/figures/` or `figures/` and reference it in Dataset or Results.*
