#!/usr/bin/env python3
"""
PointNav baseline with V-JEPA 2 on a **simple .glb scene** (alternative to ReplicaCAD).

Uses a single mesh file (e.g. Habitat test scene, Gibson, or any .glb). No Bullet required;
Habitat-Sim builds the navmesh from the mesh. Same policy and metrics as the ReplicaCAD script.

Usage:
  python scripts/train_pointnav_vjepa2_simple_scene.py --scene_glb path/to/scene.glb --num_episodes 30
  python scripts/train_pointnav_vjepa2_simple_scene.py   # uses default test scene if present

Default scene (if no --scene_glb): tmp/habitat-sim/data/test_assets/scenes/stage_floor1.glb
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse policy, encoder, episode logic from ReplicaCAD script
from scripts.train_pointnav_vjepa2 import (
    ACTION_FORWARD,
    ACTION_STOP,
    PointNavPolicy,
    get_optimal_path_length,
    load_vjepa_encoder,
    run_episode,
)

DEFAULT_SCENE_GLB = PROJECT_ROOT / "tmp/habitat-sim/data/test_assets/scenes/stage_floor1.glb"


def _register_stop_action_once():
    """Register a no-op 'stop' action so the policy can use ACTION_STOP."""
    import habitat_sim
    from habitat_sim.agent.controls.controls import ActuationSpec, SceneNodeControl
    from habitat_sim.registry import registry

    @registry.register_move_fn(body_action=True, name="stop")
    class StopAction(SceneNodeControl):
        def __call__(self, scene_node, actuation_spec: ActuationSpec):
            pass  # no-op: agent does not move

    _register_stop_action_once._stop_class = StopAction  # keep ref so not gc'd


def create_sim_simple(scene_glb_path: str | Path, height: int = 256, width: int = 256):
    """Create Habitat-Sim from a single .glb file. No ReplicaCAD, no Bullet. Navmesh built from mesh."""
    import habitat_sim
    from habitat_sim.utils import settings as hab_settings

    path = Path(scene_glb_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Scene not found: {path}")

    _register_stop_action_once()

    # Use Habitat's settings API: scene = path to .glb, no dataset config -> navmesh built automatically
    cfg_dict = hab_settings.default_sim_settings.copy()
    cfg_dict["scene"] = str(path)
    cfg_dict["height"] = height
    cfg_dict["width"] = width
    cfg_dict["color_sensor"] = True
    cfg_dict["depth_sensor"] = False
    cfg_dict["semantic_sensor"] = False
    # Do not pass scene_dataset_config_file so we load raw .glb
    if "scene_dataset_config_file" in cfg_dict:
        del cfg_dict["scene_dataset_config_file"]
    cfg_dict["default_agent_navmesh"] = True

    hab_cfg = hab_settings.make_cfg(cfg_dict)
    # make_cfg only adds move_forward, turn_left, turn_right. Policy needs "stop" too.
    hab_cfg.agents[0].action_space["stop"] = habitat_sim.agent.ActionSpec(
        "stop", habitat_sim.agent.ActuationSpec(amount=0.0)
    )
    return habitat_sim.Simulator(hab_cfg)


def train(
    scene_glb: str | Path,
    num_episodes: int = 50,
    max_steps: int = 500,
    lr: float = 3e-4,
    device_name: str = "cuda",
    out_dir: Path | None = None,
    seed: int = 42,
    checkpoint_path: Path | str | None = None,
):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    out_dir = out_dir or (PROJECT_ROOT / "checkpoints" / "pointnav_vjepa2_simple")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    sim = create_sim_simple(scene_glb)
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        print("Navmesh not loaded. Ensure the .glb path is valid and the scene has navigable area.", file=sys.stderr)
        sim.close()
        return 1

    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(num_episodes):
        start = pathfinder.get_random_navigable_point()
        goal = pathfinder.get_random_navigable_point()
        d = np.linalg.norm(np.array(goal) - np.array(start))
        while d < 1.0 or d > 8.0:
            goal = pathfinder.get_random_navigable_point()
            d = np.linalg.norm(np.array(goal) - np.array(start))
        episodes.append((list(start), [0, rng.uniform(0, 2 * np.pi), 0], list(goal)))

    print("Loading V-JEPA 2 encoder and policy ...", flush=True)
    processor, vjepa_model = load_vjepa_encoder(device)
    vjepa_model.eval()
    policy = PointNavPolicy().to(device)
    if checkpoint_path is not None:
        ckpt = torch.load(Path(checkpoint_path), map_location=device)
        policy.load_state_dict(ckpt["policy"])
        print(f"Resumed policy from {checkpoint_path} (trained {ckpt.get('episodes', '?')} episodes).", flush=True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    print("Encoder and policy ready.", flush=True)

    success_count = 0
    total_collisions = 0
    spl_list = []
    print(f"Starting training: {num_episodes} episodes, max {max_steps} steps/episode", flush=True)
    for ep_idx in range(num_episodes):
        print(f"Starting episode {ep_idx+1}/{num_episodes} ...", flush=True)
        start_pos, start_rot, goal_pos = episodes[ep_idx]
        rewards, log_probs, values, success, num_collisions, actual_path_len, optimal_path_len = run_episode(
            sim, processor, vjepa_model, policy, start_pos, start_rot, goal_pos, device, max_steps, deterministic=False
        )
        if success:
            success_count += 1
        total_collisions += num_collisions
        if optimal_path_len > 0 and not math.isinf(optimal_path_len):
            spl_i = (1.0 if success else 0.0) * (optimal_path_len / max(actual_path_len, optimal_path_len))
            spl_list.append(spl_i)
        if not log_probs:
            continue
        n = len(values)  # rewards can be n+1 when episode ends on success (terminal reward, no value)
        returns = torch.tensor(rewards[:n], device=device, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values).squeeze()
        advantage = returns - values_t.detach()
        actor_loss = -(log_probs_t * advantage.detach()).mean()
        critic_loss = F.mse_loss(values_t, returns)
        loss = actor_loss + 0.5 * critic_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()
        if (ep_idx + 1) % 10 == 0 or num_episodes <= 10:
            mean_coll = total_collisions / (ep_idx + 1)
            print(f"Episode {ep_idx+1}/{num_episodes} success={success_count} recent_success={success} collisions/ep={mean_coll:.2f} loss={loss.item():.4f}", flush=True)

    sim.close()
    ckpt_path = out_dir / "policy_last.pt"
    torch.save({"policy": policy.state_dict(), "episodes": num_episodes}, ckpt_path)
    mean_collisions = total_collisions / num_episodes if num_episodes else 0
    mean_spl = np.mean(spl_list) if spl_list else 0.0
    success_pct = 100 * success_count / num_episodes if num_episodes else 0.0
    summary = f"Saved {ckpt_path}. Success rate: {success_count}/{num_episodes} = {success_pct:.1f}% | SPL: {mean_spl:.3f} | Collisions/ep: {mean_collisions:.2f}"
    print(summary, flush=True)
    # Write one-line results for report (e.g. midterm)
    results_log = PROJECT_ROOT / "logs" / "pointnav_simple_results.txt"
    results_log.parent.mkdir(parents=True, exist_ok=True)
    with open(results_log, "a") as f:
        from datetime import datetime
        f.write(f"{datetime.now().isoformat()} | scene={scene_glb.name} | episodes={num_episodes} | Success rate: {success_pct:.1f}% | SPL: {mean_spl:.3f} | Collisions/ep: {mean_collisions:.2f}\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="PointNav + V-JEPA 2 on a simple .glb scene (no Bullet).")
    parser.add_argument("--scene_glb", type=str, default=None, help="Path to .glb scene. Default: Habitat test scene if present.")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume: path to policy_last.pt")
    args = parser.parse_args()

    scene_glb = args.scene_glb
    if scene_glb is None:
        scene_glb = DEFAULT_SCENE_GLB
    scene_glb = Path(scene_glb)
    if not scene_glb.is_absolute():
        scene_glb = (PROJECT_ROOT / scene_glb).resolve()
    if not scene_glb.exists():
        print(f"Scene not found: {scene_glb}", file=sys.stderr)
        print(f"Pass --scene_glb path/to/scene.glb or add the default at {DEFAULT_SCENE_GLB}", file=sys.stderr)
        return 1

    return train(
        scene_glb=scene_glb,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        lr=args.lr,
        device_name=args.device,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        seed=args.seed,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
    )


if __name__ == "__main__":
    sys.exit(main())
