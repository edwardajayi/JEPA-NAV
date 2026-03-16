# SLURM jobs for JEPA-NAV

- **`run_replicacad_vjepa2.slurm`** / **`run_replicacad_vjepa2_v100.slurm`** — ReplicaCAD V-JEPA 2 (frame collection + encoder).
- **`train_pointnav_vjepa2.slurm`** — Full baseline (200 ep, 500 steps). **~2–4 h** on V100.
- **`train_pointnav_vjepa2_h100.slurm`** — Same full run on **H100**. **~1–2.5 h** (submit on the side; use for final numbers).
- **`train_pointnav_vjepa2_quicktest.slurm`** — **Quick test** (15 ep, 150 steps). **~20–40 min** on V100; use for report if full run is still queued.

## Submit ReplicaCAD V-JEPA 2

**Batch (V100, often faster allocation):**
```bash
cd /ocean/projects/cis250225p/eajayi1/JEPANAV
sbatch scripts/slurm/run_replicacad_vjepa2_v100.slurm
```

**Batch (H100):**
```bash
sbatch scripts/slurm/run_replicacad_vjepa2.slurm
```

**Interactive (V100, 2 hours):** Get a shell on a V100 node, then run the script manually.
```bash
interact -p GPU-shared --gres=gpu:v100-32:1 -t 2:00:00 -A cis250225p
# once allocated:
cd /ocean/projects/cis250225p/eajayi1/JEPANAV
module load anaconda3
source activate /ocean/projects/cis250225p/eajayi1/JEPANAV/.conda_env
python scripts/run_replicacad_vjepa2.py --scene_id apt_0 --num_frames 80 --out_dir data/replicacad_vjepa2_frames --device cuda
```

## Logs and output

- **SLURM stdout:** `logs/replicacad_vjepa2_<jobid>.out`
- **SLURM stderr:** `logs/replicacad_vjepa2_<jobid>.err`
- **Script log (JSON line):** `logs/replicacad_vjepa2_<jobid>.log`
- **Collected frames:** `data/replicacad_vjepa2_frames/*.png`

## Customize

Edit the SLURM script or pass args by wrapping the python call, e.g.:

```bash
# In the .slurm file, change:
python scripts/run_replicacad_vjepa2.py --scene_id v3_sc0_staging_00 --num_frames 100 ...
```

Or run locally (no SLURM):

```bash
python scripts/run_replicacad_vjepa2.py --num_frames 50 --out_dir data/replicacad_vjepa2_frames --log_file logs/replicacad_vjepa2_manual.log
```

---

## PointNav baseline (Phase C): timing and subset

**Rough times (V100, single scene `apt_0`):** Each step runs V-JEPA 2 (~0.5–2 s). So per episode (up to 500 steps) ~5–15 min worst case; typically many episodes finish earlier (success or max steps).

| Run | Episodes | Max steps | Approx. time |
|-----|----------|-----------|--------------|
| **Quick test** | 15 | 150 | ~20–40 min (V100) |
| **Short** | 30 | 200 | ~40–90 min (V100) |
| **Full (V100)** | 200 | 500 | ~2–4 h |
| **Full (H100)** | 200 | 500 | **~1–2.5 h** (faster GPU) |

**Test on a subset first:** Run the quick-test job so the full run doesn’t waste time if something breaks:

```bash
# 1) Quick test (~20–40 min)
sbatch scripts/slurm/train_pointnav_vjepa2_quicktest.slurm

# 2) If logs look good (Success rate, SPL, Collisions printed), run full
sbatch scripts/slurm/train_pointnav_vjepa2.slurm
```

**Custom subset (fewer episodes/steps):**
```bash
python scripts/train_pointnav_vjepa2.py --num_episodes 20 --max_steps 100 --device cuda
```

---

## Alternative: simple .glb scene (no Bullet)

If ReplicaCAD + Bullet is not available, use a **simple mesh scene** (no articulated objects):

- **Script:** `scripts/train_pointnav_vjepa2_simple_scene.py`
- **SLURM:** `scripts/slurm/train_pointnav_simple_scene.slurm`

Uses a single .glb file; Habitat-Sim builds the navmesh from the mesh. Default scene: `tmp/habitat-sim/data/test_assets/scenes/stage_floor1.glb` (if present). Same V-JEPA 2 + policy and metrics (Success Rate, SPL, Collisions/ep).

```bash
# Default test scene (if tmp/habitat-sim/ exists)
sbatch scripts/slurm/train_pointnav_simple_scene.slurm

# Or with your own .glb (e.g. Gibson / Habitat test scenes)
python scripts/train_pointnav_vjepa2_simple_scene.py --scene_glb /path/to/scene.glb --num_episodes 30
```

**Downloading alternative scenes**

- **Habitat test scenes (no form, small):** 3 .glb scenes for quick runs.
  ```bash
  python scripts/download_alternative_scenes.py --habitat-test-scenes
  ```
  Scenes go to `data/scene_datasets/habitat-test-scenes/`. Then run:
  ```bash
  python scripts/train_pointnav_vjepa2_simple_scene.py --scene_glb data/scene_datasets/habitat-test-scenes/<scene>.glb --num_episodes 30
  ```

- **Gibson:** You must agree to terms and get the zip from the [Gibson download page](https://github.com/StanfordVL/GibsonEnv#database) (“Gibson Dataset for Habitat”, e.g. `gibson_habitat.zip` ~1.5GB). Extract to `data/scene_datasets/gibson/`, then:
  ```bash
  python scripts/download_alternative_scenes.py --gibson-config
  python scripts/train_pointnav_vjepa2_simple_scene.py --scene_glb data/scene_datasets/gibson/<SceneName>.glb --num_episodes 30
  ```

**Getting stats for your report (e.g. midterm):**

1. **Submit the job** (after downloading Habitat test scenes):  
   `sbatch scripts/slurm/train_pointnav_simple_scene.slurm`
2. **When the job finishes**, metrics are in two places:
   - **Stdout:** `tail -n 5 logs/pointnav_simple_<JOBID>.out` — last line is:  
     `Success rate: X/Y = Z% | SPL: ... | Collisions/ep: ...`
   - **Results log (appended each run):** `cat logs/pointnav_simple_results.txt` — one line per run with scene, episodes, Success rate %, SPL, Collisions/ep.
3. **For the report table:** use **Success Rate (%)**, **SPL**, and **Collisions per episode (mean)** from that run. You can cite the run as “PointNav + V-JEPA 2 on Habitat test scenes (van-gogh-room / skokloster-castle / apartment_1)” if not using ReplicaCAD.
