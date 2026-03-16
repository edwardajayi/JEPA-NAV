# ReplicaCAD Dataset — What You Have

This document describes the **ReplicaCAD** dataset in your project (downloaded from [Hugging Face](https://huggingface.co/datasets/ai-habitat/ReplicaCAD_dataset)).

---

## 1. What ReplicaCAD is

- **ReplicaCAD** = artist-made 3D indoor scenes based on the scanned “FRL apartment” from the Replica dataset.
- Designed for the **Habitat** simulator: embodied AI, object rearrangement, navigation.
- **License:** CC BY 4.0.
- **Citation:** Szot et al., “Habitat 2.0: Training Home Assistants to Rearrange their Habitat,” NeurIPS 2021.

---

## 2. Where it lives in your project

- **Path:** `data/ReplicaCAD/` (or the path you passed when running `scripts/download_replicacad.sh`).
- **Main config:** `data/ReplicaCAD/replicaCAD.scene_dataset_config.json`  
  Habitat uses this file to find all scenes, stages, objects, navmeshes, and lighting.

---

## 3. Directory structure (what you have)

| Path | Contents |
|------|----------|
| `replicaCAD.scene_dataset_config.json` | **Main dataset config.** Point Habitat’s `scene_dataset_config_file` at this (use absolute path). |
| `configs/stages/` | Stage (background) configs: empty stage, FRL apartment, and 5 macro variants (e.g. `frl_apartment_stage`, `Stage_v3_sc0_staging`, …). |
| `configs/scenes/` | **84 scene instances** (different furniture layouts). Names like `apt_0` … `apt_5`, `v3_sc0_staging_00` … `v3_sc3_staging_20`. Each has a `.scene_instance.json`. |
| `configs/objects/` | Object configs (furniture, props). |
| `configs/lighting/` | Lighting setups. |
| `configs/ssd/` | Semantic scene descriptor (e.g. `replicaCAD_semantic_lexicon.json`). |
| `navmeshes/` | Precomputed navmeshes for each scene (agent radius 0.3 m, Fetch-style). |
| `navmeshes_default/` | Navmeshes for Habitat default agent. |
| `stages/` | Stage mesh/assets. |
| `objects/` | Object assets. |
| `urdf/` | Articulated objects (e.g. fridge, cabinet, doors). |

---

## 4. Scenes you can load

- **6 “apartment” stages:** `apt_0` … `apt_5` (navmesh names; scene instances may use the same or map to stages).
- **84 layout variants:** `v3_sc0_staging_00` … `v3_sc0_staging_20`, `v3_sc1_staging_00` … `v3_sc1_staging_20`, same for `v3_sc2_*`, `v3_sc3_*`.  
  So **84 scene instances** in total (21 × 4 macro variants).
- In Habitat you set **`scene_id`** to one of these (e.g. `apt_0` or `v3_sc0_staging_00`). The exact list is in `replicaCAD.scene_dataset_config.json` under `scene_instances` / `navmesh_instances`.

---

## 5. How to use it with Habitat

1. **Set dataset config (absolute path):**
   ```text
   scene_dataset_config_file = "/ocean/projects/cis250225p/eajayi1/JEPANAV/data/ReplicaCAD/replicaCAD.scene_dataset_config.json"
   ```
2. **Pick a scene:** e.g. `scene_id = "apt_0"` or `scene_id = "v3_sc0_staging_00"`.
3. Create the simulator with this config and an agent that has an **RGB sensor**; then you can step the sim and read RGB observations for Experiment B.

---

## 6. What you *don’t* have in this download

- **ReplicaCAD Baked Lighting** (525 MB) is a separate Hugging Face dataset; you have the **Interactive** (132 MB) version. Baked lighting gives more photorealistic rendering; optional for Exp B.
- The **21 withheld test scenes** (1 macro + 20 micro) are not in this release (used for challenge evaluation).

---

## 7. Quick reference

| Item | Value |
|------|--------|
| Location | `data/ReplicaCAD/` |
| Main config | `replicaCAD.scene_dataset_config.json` |
| Number of scene instances | 84 |
| Example scene_ids | `apt_0`, `v3_sc0_staging_00`, `v3_sc1_staging_05` |
| Use in code | Set `scene_dataset_config_file` and `scene_id` in Habitat simulator config. |
