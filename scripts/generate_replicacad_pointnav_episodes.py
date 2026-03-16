#!/usr/bin/env python3
"""
Generate PointNav episode dataset for ReplicaCAD.
Uses Habitat-Sim to sample navigable points (start, goal) and writes
a JSON dataset that Habitat-Lab's PointNavDatasetV1 can load.

Usage:
  python scripts/generate_replicacad_pointnav_episodes.py --scene_id apt_0 --num_episodes 100 --out data/datasets/pointnav/replicacad/train/train.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPLICACAD_ROOT = PROJECT_ROOT / "data" / "ReplicaCAD"
SCENE_DATASET_CONFIG = REPLICACAD_ROOT / "replicaCAD.scene_dataset_config.json"


def generate_episodes(
    scene_id: str,
    num_episodes: int,
    min_goal_distance: float = 1.0,
    max_goal_distance: float = 10.0,
    seed: int = 42,
) -> list[dict]:
    """Sample start/goal from ReplicaCAD navmesh; return list of episode dicts."""
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

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    pathfinder = sim.pathfinder

    if not pathfinder.is_loaded:
        sim.close()
        raise RuntimeError(f"Navmesh not loaded for scene {scene_id}. Cannot sample navigable points.")

    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(num_episodes):
        start = pathfinder.get_random_navigable_point()
        goal = pathfinder.get_random_navigable_point()
        dist = np.linalg.norm(np.array(goal) - np.array(start))
        while dist < min_goal_distance or dist > max_goal_distance:
            goal = pathfinder.get_random_navigable_point()
            dist = np.linalg.norm(np.array(goal) - np.array(start))

        # Habitat PointNav episode: start_rotation is [yaw] in radians (around Y); we use 0 for simplicity
        episode = {
            "episode_id": f"{scene_id}_{i}",
            "scene_id": scene_id,
            "start_position": [float(start[0]), float(start[1]), float(start[2])],
            "start_rotation": [0.0, float(rng.uniform(0, 2 * np.pi)), 0.0],
            "goals": [{"position": [float(goal[0]), float(goal[1]), float(goal[2])]}],
        }
        episodes.append(episode)
    sim.close()
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Generate ReplicaCAD PointNav episodes")
    parser.add_argument("--scene_id", type=str, default="apt_0", help="ReplicaCAD scene id")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to generate")
    parser.add_argument("--out", type=str, default=None, help="Output path (e.g. data/datasets/pointnav/replicacad/train/train.json or .json.gz)")
    parser.add_argument("--min_goal_distance", type=float, default=1.0)
    parser.add_argument("--max_goal_distance", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.out or str(PROJECT_ROOT / "data" / "datasets" / "pointnav" / "replicacad" / "train" / "train.json.gz"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    episodes = generate_episodes(
        scene_id=args.scene_id,
        num_episodes=args.num_episodes,
        min_goal_distance=args.min_goal_distance,
        max_goal_distance=args.max_goal_distance,
        seed=args.seed,
    )
    # Habitat-Lab PointNavDatasetV1 _load_from_file expects top-level "episodes" and uses scenes_dir for scene paths.
    # For ReplicaCAD we pass scene_dataset_config in task config; episode scene_id is just the id.
    data = {"episodes": episodes}
    json_str = json.dumps(data, indent=2)

    if out_path.suffix == ".gz" or str(out_path).endswith(".json.gz"):
        with gzip.open(out_path, "wt") as f:
            f.write(json_str)
        print(f"Wrote {len(episodes)} episodes to {out_path} (gzip)")
    else:
        with open(out_path, "w") as f:
            f.write(json_str)
        print(f"Wrote {len(episodes)} episodes to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
