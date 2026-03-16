#!/usr/bin/env python3
"""
ReplicaCAD V-JEPA 2: Collect RGB frames from Habitat (ReplicaCAD), then run V-JEPA 2 on them.
Run locally or via SLURM; logs go to stdout and optionally to a file.

Usage:
  python scripts/run_replicacad_vjepa2.py --num_frames 50 --out_dir data/replicacad_vjepa2_frames
  python scripts/run_replicacad_vjepa2.py --num_frames 100 --scene_id v3_sc0_staging_00 --log_file logs/replicacad_vjepa2.log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPLICACAD_ROOT = PROJECT_ROOT / "data" / "ReplicaCAD"
SCENE_DATASET_CONFIG = REPLICACAD_ROOT / "replicaCAD.scene_dataset_config.json"


def collect_frames_habitat(
    scene_id: str,
    num_frames: int,
    out_dir: Path,
    height: int = 256,
    width: int = 256,
    seed: int = 42,
) -> int:
    """Load ReplicaCAD in Habitat-sim, sample random navigable points, save RGB frames. Returns count saved."""
    import habitat_sim
    import numpy as np

    if not SCENE_DATASET_CONFIG.exists():
        raise FileNotFoundError(
            f"ReplicaCAD config not found at {SCENE_DATASET_CONFIG}. Run scripts/download_replicacad.sh first."
        )

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(SCENE_DATASET_CONFIG)
    sim_cfg.scene_id = scene_id
    sim_cfg.gpu_device_id = 0
    sim_cfg.requires_textures = True

    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "color_sensor"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [height, width]
    rgb_sensor.position = [0.0, 1.5, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    rng = np.random.default_rng(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        print("Warning: No navmesh loaded; using default agent position only.", file=sys.stderr)
        saved = 0
        for i in range(num_frames):
            obs = sim.get_sensor_observations()
            if obs and "color_sensor" in obs:
                rgb = obs["color_sensor"]
                out_path = out_dir / f"frame_{i:05d}.png"
                try:
                    import cv2
                    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    saved += 1
                except Exception as e:
                    print(f"Save failed {out_path}: {e}", file=sys.stderr)
            sim.step("turn_right")  # vary view a bit
        sim.close()
        return saved

    saved = 0
    for i in range(num_frames):
        nav_point = pathfinder.get_random_navigable_point()
        agent = sim.get_agent(0)
        agent_state = habitat_sim.agent.AgentState()
        agent_state.position = nav_point
        agent.set_state(agent_state)
        obs = sim.get_sensor_observations()
        if obs and "color_sensor" in obs:
            rgb = obs["color_sensor"]
            out_path = out_dir / f"frame_{i:05d}.png"
            try:
                import cv2
                cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                saved += 1
            except Exception as e:
                print(f"Save failed {out_path}: {e}", file=sys.stderr)
    sim.close()
    return saved


def run_vjepa2_on_collected(frames_dir: Path, num_frames: int = 16, device: str | None = None) -> dict:
    """Run V-JEPA 2 on collected frames; return dict with shape, mean, std."""
    from transformers import AutoModel, AutoVideoProcessor
    import torch
    import numpy as np
    from PIL import Image

    model_id = "facebook/vjepa2-vitl-fpc64-256"
    processor = AutoVideoProcessor.from_pretrained(model_id)
    use_cuda = (device or "cuda") == "cuda" and torch.cuda.is_available()
    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.float16 if use_cuda else torch.float32,
        attn_implementation="sdpa",
    )
    dev = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(dev)

    paths = sorted(frames_dir.glob("*.png"))[:num_frames]
    if not paths:
        return {"error": "no frames found", "shape": None, "mean": None, "std": None}
    target_h = target_w = 256
    frames = []
    for p in paths:
        img = np.array(Image.open(p).convert("RGB"))
        img = np.transpose(img, (2, 0, 1))
        h, w = img.shape[1], img.shape[2]
        if h != target_h or w != target_w:
            pil_img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            pil_img = pil_img.resize((target_w, target_h), Image.BILINEAR)
            img = np.transpose(np.array(pil_img), (2, 0, 1))
        frames.append(img)
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    video = np.stack(frames[:num_frames], axis=0)

    inputs = processor(video, return_tensors="pt")
    inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, skip_predictor=True)
    emb = outputs.last_hidden_state
    stats = {
        "shape": list(emb.shape),
        "mean": emb.float().mean().item(),
        "std": emb.float().std().item(),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="ReplicaCAD V-JEPA 2: ReplicaCAD frames + V-JEPA 2 encoder")
    parser.add_argument("--scene_id", type=str, default="apt_0", help="ReplicaCAD scene id")
    parser.add_argument("--num_frames", type=int, default=50, help="Number of frames to collect")
    parser.add_argument("--out_dir", type=str, default=None, help="Output dir for frames (default: data/replicacad_vjepa2_frames)")
    parser.add_argument("--vjepa_frames", type=int, default=16, help="Frames per clip for V-JEPA 2")
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file path")
    parser.add_argument("--no_habitat", action="store_true", help="Skip Habitat; only run V-JEPA 2 on existing --out_dir")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu for V-JEPA 2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or str(PROJECT_ROOT / "data" / "replicacad_vjepa2_frames"))
    log_file = Path(args.log_file) if args.log_file else None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        print(msg)
        if log_file:
            with open(log_file, "a") as f:
                f.write(msg + "\n")

    log("==========================================================")
    log(f"ReplicaCAD V-JEPA 2 started at {datetime.now().isoformat()}")
    log(f"Project root: {PROJECT_ROOT}")
    log("==========================================================")

    if not args.no_habitat:
        log(f"Step 1: Collecting up to {args.num_frames} RGB frames from Habitat (scene_id={args.scene_id})")
        try:
            saved = collect_frames_habitat(
                scene_id=args.scene_id,
                num_frames=args.num_frames,
                out_dir=out_dir,
                seed=42,
            )
            log(f"Saved {saved} frames to {out_dir}")
        except Exception as e:
            log(f"Habitat collection failed: {e}")
            import traceback
            log(traceback.format_exc())
            return 1
    else:
        log(f"Step 1 skipped (--no_habitat). Using existing frames in {out_dir}")
        saved = len(list(out_dir.glob("*.png")))
        log(f"Found {saved} existing frames.")

    log("Step 2: Running V-JEPA 2 on collected frames")
    try:
        stats = run_vjepa2_on_collected(out_dir, num_frames=args.vjepa_frames, device=args.device)
        log(f"V-JEPA 2 encoder output shape: {stats.get('shape')}")
        log(f"  mean = {stats.get('mean')}, std = {stats.get('std')}")
        if log_file:
            with open(log_file, "a") as f:
                f.write(json.dumps({"experiment": "replicacad_vjepa2", "stats": stats, "frames_collected": saved}) + "\n")
    except Exception as e:
        log(f"V-JEPA 2 failed: {e}")
        import traceback
        log(traceback.format_exc())
        return 1

    log("==========================================================")
    log(f"ReplicaCAD V-JEPA 2 finished at {datetime.now().isoformat()}")
    log("==========================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
