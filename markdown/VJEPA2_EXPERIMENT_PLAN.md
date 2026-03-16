# V-JEPA 2 experiment plan on ReplicaCAD

Goal: **Test V-JEPA 2 on our dataset** (ReplicaCAD / Habitat RGB frames) and get embeddings for the baseline (and later JEPA-NAV).

Reference: [Hugging Face – V-JEPA 2](https://huggingface.co/docs/transformers/model_doc/vjepa2#transformers.VJEPA2Model)

---

## 1. What we need from the docs

- **Model (feature extraction):** `AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")`  
  - Optional: `skip_predictor=True` in forward if we only need encoder features.
- **Processor:** `AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")`  
  - Input: video tensor shape `(num_frames, channels, height, width)` → T×C×H×W (e.g. 64×3×256×256).  
  - Processor returns dict with `pixel_values_videos` for the model.
- **Input to model:** `pixel_values_videos`: `(batch_size, num_frames, num_channels, frame_size, frame_size)`.
- **Output:**  
  - `outputs.last_hidden_state` → encoder features, shape `(batch_size, sequence_length, hidden_size)` (e.g. 1024-d for ViT-L).  
  - `outputs.predictor_output.last_hidden_state` → predictor output (optional).
- **Config:** `frames_per_clip=64`, `crop_size=256` by default. We can use fewer frames (e.g. 16) and repeat or pad; processor may handle it.

---

## 2. Data source: ReplicaCAD

- ReplicaCAD is **3D scenes** for Habitat. We get **RGB frames** by running Habitat and reading the agent’s observation each step.
- So “testing V-JEPA 2 on our dataset” = **Habitat renders ReplicaCAD → we get RGB (and optional depth) → we feed frames to V-JEPA 2 → we get embeddings**.

---

## 3. Experiment setup (phased)

### Phase A: Sanity check without Habitat (quick)

- **Input:** Synthetic or saved images (e.g. random tensor, or a few PNGs from any source).
- **Logic:** Build a short “video” of 64 frames (repeat single image 64×, or stack 64 different images). Run processor → model → read `last_hidden_state`.
- **Goal:** Confirm model loads, processor runs, output shape is as expected (e.g. batch=1, seq_len=…, hidden_size=1024). No ReplicaCAD yet.
- **Using the older Replica-Dataset repo:** Run `scripts/clone_replica_dataset.sh` to clone it into `data/Replica-Dataset`. After optionally running `download.sh`, scene textures (HDR) are in `replica_flat_*/textures/`; convert a few to PNG, or put any PNG/JPG from the repo in a folder and run: `python scripts/run_vjepa2_on_frames.py --frames_dir data/Replica-Dataset/path/to/images/`.

---

## Result of Phase A (what the experiment showed)

**What we set out to do:** Check that we can load V-JEPA 2, feed it image-like input (a short “video” clip), and get back a usable embedding tensor—without using Habitat yet.

**What we did:**
- Took **4 PNG images** from the Replica-Dataset repo (`data/Replica-Dataset/assets/`: ReplicaViewer, ReplicaDataset, etc.).
- Resized them to 256×256 and **padded to 16 frames** (repeating the last frame) so the input has shape **(16, 3, 256, 256)** — i.e. a 16-frame “video” of 256×256 RGB.
- Passed that through the **V-JEPA 2 processor** (normalization, etc.) and then the **V-JEPA 2 encoder** (Vision Transformer over space-time), with the predictor skipped.

**What we got (the result):**
- **Output shape:** **(1, 2048, 1024)**.
  - **1** = batch size (one clip).
  - **2048** = number of *tokens* (spatiotemporal patches) the model outputs. So the encoder represents the clip as 2048 vectors.
  - **1024** = *hidden size* of each vector (ViT-L). So each token is a 1024-dimensional embedding.
- **Statistics:** mean ≈ 0.01, standard deviation ≈ 3.23 over all 1×2048×1024 values. So the embeddings are roughly centered and have moderate spread; no NaNs or zeros, so the model is producing non-degenerate features.

**What this means:**
1. **Pipeline works:** Loading the model, preprocessing images into a clip, and running the encoder succeeds. We can reliably get **2048 × 1024-d** features from a 16-frame, 256×256 clip.
2. **We have a fixed representation size:** For downstream use (e.g. a policy), we will typically **pool** these 2048 tokens into one vector (e.g. mean over tokens → one 1024-d vector per clip). That single vector is what we will feed to a policy or an affordance head later.
3. **Phase A is a sanity check only:** The input was four *static* Replica asset images, not yet live Habitat/ReplicaCAD views. So we have *not* shown that the embeddings are good for navigation—only that the model runs and produces sensible-looking numbers. Phase B (real Habitat frames) and Phase C (policy trained with these embeddings) are where we actually validate usefulness for navigation.

**In one sentence:** Phase A shows that V-JEPA 2 turns a 16-frame, 256×256 RGB clip into 2048 token embeddings of dimension 1024, with stable statistics, so we can confidently use this encoder as the visual backbone for the next phases.

---

### Phase B: ReplicaCAD frames via Habitat

- **Input:** RGB observations from Habitat (ReplicaCAD scene).  
  - Option 1: Collect a small set of frames (e.g. 100–500) by stepping a random or scripted agent in one ReplicaCAD scene; save as numpy or PNG.  
  - Option 2: Run Habitat in the loop: at each step, take current RGB → form clip (e.g. last 16 or 64 frames, or repeat current frame) → run V-JEPA 2 → use embedding for policy (later).
- **Goal:** Run V-JEPA 2 on real ReplicaCAD visuals; inspect embedding stats (mean, std, a few norms) and optionally t-SNE/UMAP to see if different rooms/angles separate.

### Phase C: Baseline agent (PointNav)

- **Scripts:** `scripts/train_pointnav_vjepa2.py`, `scripts/slurm/train_pointnav_vjepa2.slurm`
- At each step: RGB → V-JEPA 2 → pooled embedding (1024-d) + goal (distance, theta) → policy MLP → action (forward / turn_left / turn_right / stop).
- Train with policy gradient (REINFORCE-style) on ReplicaCAD; success = within 0.36 m of goal. Metrics: Success Rate (SPL can be added).
- Episodes: sampled from ReplicaCAD navmesh (start/goal) in script, or pre-generated with `scripts/generate_replicacad_pointnav_episodes.py`.

---

## 4. Implementation checklist

- [ ] **Env:** Python 3.9 + `transformers`, `torch`, `habitat-lab`, `habitat-sim` (and ReplicaCAD data path).
- [ ] **Phase A:** Script that loads model + processor, runs on synthetic/single-image “video”, prints `last_hidden_state.shape` and optional norm. (See `scripts/run_vjepa2_on_frames.py`.)
- [ ] **Phase B:**  
  - [ ] Habitat loads ReplicaCAD scene; sample RGB (random agent or scripted path).  
  - [ ] Convert frames to T×C×H×W (e.g. 16 or 64), run processor → model → embeddings.  
  - [ ] Save a few embeddings and/or plot (e.g. histogram, simple 2D projection).
- [ ] **Phase C:** Integrate into training loop (embedding → policy → action); train and eval.

---

## 5. File layout

- `markdown/VJEPA2_EXPERIMENT_PLAN.md` — this plan.  
- `scripts/run_vjepa2_on_frames.py` — Phase A: load V-JEPA 2, run on a single image or (T,C,H,W) clip, print or return embeddings.  
- `scripts/run_replicacad_vjepa2.py` — **ReplicaCAD V-JEPA 2** (Phase B): collect RGB frames from Habitat (ReplicaCAD), then run V-JEPA 2 on them; use with `scripts/slurm/run_replicacad_vjepa2.slurm` for SLURM.  
- `scripts/clone_replica_dataset.sh` — clone the original [Replica-Dataset](https://github.com/facebookresearch/Replica-Dataset) repo into `data/Replica-Dataset`; use images from it for Phase A.  
- Later: training script that uses embeddings from `run_replicacad_vjepa2` (or equivalent) per step for Phase C.

---

## 6. Model and processor details (from docs)

- **Model ID:** `facebook/vjepa2-vitl-fpc64-256` (ViT-L, 64 frames per clip, 256 crop).  
- **Encoder output:** `last_hidden_state` → use for navigation (pool over time/spatial tokens or take a single token if we use repeated frame).  
- **Optional:** `torchcodec` is used in HF example for decoding MP4; we don’t need it for Habitat (we get numpy/torch from sim).  
- **Device:** Prefer GPU; `device_map="auto"` or `.to("cuda")` for the model.

---

## 7. Next step

Run **Phase A**:

```bash
cd /ocean/projects/cis250225p/eajayi1/JEPANAV
conda activate ./.conda_env   # or your env with transformers + torch
python scripts/run_vjepa2_on_frames.py
```

Then run **ReplicaCAD V-JEPA 2** (Phase B): `sbatch scripts/slurm/run_replicacad_vjepa2.slurm` or `python scripts/run_replicacad_vjepa2.py --num_frames 50`.
