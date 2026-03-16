#!/usr/bin/env python3
"""
Phase C: Train PointNav baseline with V-JEPA 2 encoder + policy on ReplicaCAD.

Loop: Habitat-Sim (ReplicaCAD) -> RGB + goal -> V-JEPA 2 -> embedding -> policy -> action.
Training: PPO (or simple policy gradient). Metrics: Success Rate, SPL.

Usage:
  python scripts/train_pointnav_vjepa2.py --scene_id apt_0 --num_episodes 50 --max_steps 500
  python scripts/train_pointnav_vjepa2.py --eval_only --checkpoint path/to/model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPLICACAD_ROOT = PROJECT_ROOT / "data" / "ReplicaCAD"
SCENE_DATASET_CONFIG = REPLICACAD_ROOT / "replicaCAD.scene_dataset_config.json"

# Action indices (match Habitat)
ACTION_FORWARD = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_STOP = 3
NUM_ACTIONS = 4

VJEPA_HIDDEN = 1024
GOAL_DIM = 2  # distance, theta (radians)
SUCCESS_DISTANCE = 0.36
TURN_ANGLE = 10 * math.pi / 180  # 10 degrees per turn
MOVE_DISTANCE = 0.25


def load_vjepa_encoder(device: torch.device, dtype: torch.dtype = torch.float32):
    """Load V-JEPA 2 encoder; return processor, model. Encoder output: (B, seq, 1024)."""
    from transformers import AutoModel, AutoVideoProcessor
    model_id = "facebook/vjepa2-vitl-fpc64-256"
    processor = AutoVideoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model = model.to(device)
    return processor, model


def rgb_to_clip(rgb: np.ndarray, num_frames: int = 16, crop: int = 256) -> np.ndarray:
    """(H,W,3) or (H,W,4) uint8 -> (T,3,H,W) for V-JEPA; repeat frame to fill T."""
    from PIL import Image
    # Habitat may return RGBA; processor expects 3 channels
    if rgb.shape[-1] == 4:
        rgb = np.ascontiguousarray(rgb[..., :3])
    if rgb.shape[0] != crop or rgb.shape[1] != crop:
        pil = Image.fromarray(rgb)
        pil = pil.resize((crop, crop), Image.BILINEAR)
        rgb = np.array(pil)
    # (H,W,3) -> (3,H,W)
    x = np.transpose(rgb, (2, 0, 1))
    clip = np.repeat(x[np.newaxis, ...], num_frames, axis=0)
    return clip.astype(np.uint8)


def encode_observation(processor, model, rgb: np.ndarray, device: torch.device, num_frames: int = 16) -> torch.Tensor:
    """RGB (H,W,3) -> pooled embedding (1, 1024)."""
    clip = rgb_to_clip(rgb, num_frames=num_frames)  # (T, C, H, W)
    inputs = processor(clip, return_tensors="pt", input_data_format="channels_first")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, skip_predictor=True)
    emb = out.last_hidden_state  # (1, 2048, 1024)
    pooled = emb.mean(dim=1)  # (1, 1024)
    return pooled


class PointNavPolicy(nn.Module):
    """MLP: (V-JEPA embedding + goal) -> action logits and value."""
    def __init__(self, embed_dim: int = VJEPA_HIDDEN, goal_dim: int = GOAL_DIM, hidden: int = 256, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + goal_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, num_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, embedding: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([embedding, goal], dim=-1)
        x = self.fc(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


def _register_stop_action_once():
    """Register no-op 'stop' so policy ACTION_STOP is in agent action space."""
    from habitat_sim.agent.controls.controls import ActuationSpec, SceneNodeControl
    from habitat_sim.registry import registry
    if "stop" in registry._mapping.get("move_fn", {}):
        return
    @registry.register_move_fn(body_action=True, name="stop")
    class _StopAction(SceneNodeControl):
        def __call__(self, scene_node, actuation_spec: ActuationSpec):
            pass
    _register_stop_action_once._stop_class = _StopAction


def create_sim(scene_id: str, height: int = 256, width: int = 256):
    """Create Habitat-Sim instance with ReplicaCAD scene and RGB sensor."""
    import habitat_sim
    if not SCENE_DATASET_CONFIG.exists():
        raise FileNotFoundError(f"ReplicaCAD not found at {SCENE_DATASET_CONFIG}")
    _register_stop_action_once()
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
    # Default action_space has move_forward, turn_left, turn_right only; policy needs "stop" too.
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)),
        "stop": habitat_sim.agent.ActionSpec("stop", habitat_sim.agent.ActuationSpec(amount=0.0)),
    }
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def get_goal_vector(agent_state, goal_pos: list) -> tuple[float, float]:
    """Return (distance, theta) in radians relative to agent."""
    pos = agent_state.position
    # Vector from agent to goal in world XZ
    dx = goal_pos[0] - pos[0]
    dz = goal_pos[2] - pos[2]
    distance = math.sqrt(dx * dx + dz * dz)
    # Agent rotation is quaternion; we need yaw. For simplicity use atan2(dz, dx) and subtract agent yaw.
    # habitat_sim rotation: quaternion (w,x,y,z). Yaw from quat: yaw = atan2(2*(w*y + z*x), 1 - 2*(x*x + y*y))
    q = agent_state.rotation
    w, x, y, z = q.w, q.x, q.y, q.z
    agent_yaw = math.atan2(2 * (w * y + z * x), 1 - 2 * (x * x + y * y))
    goal_angle = math.atan2(dz, dx)
    theta = goal_angle - agent_yaw
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return distance, theta


ACTION_NAMES = ["move_forward", "turn_left", "turn_right", "stop"]


def apply_action_sim(sim, action: int):
    """Execute one discrete action; returns observation dict from step (includes 'collided' when Bullet/pathfinder in use)."""
    return sim.step(ACTION_NAMES[action])


def get_optimal_path_length(pathfinder, start_pos, goal_pos) -> float:
    """Geodesic shortest path length (for SPL). Returns inf if no path."""
    try:
        start_pos = np.array(start_pos, dtype=np.float32)
        goal_pos = np.array(goal_pos, dtype=np.float32)
        if pathfinder.is_loaded:
            start_pos = pathfinder.snap_point(start_pos)
            goal_pos = pathfinder.snap_point(goal_pos)
        path = habitat_sim.ShortestPath()
        path.requested_start = start_pos
        path.requested_end = goal_pos
        if pathfinder.find_path(path):
            return float(path.geodesic_distance)
    except Exception:
        pass
    return float("inf")


def run_episode(
    sim,
    processor,
    vjepa_model,
    policy: PointNavPolicy,
    start_pos,
    start_rot,
    goal_pos,
    device: torch.device,
    max_steps: int,
    deterministic: bool = False,
    optimal_path_length: float | None = None,
):
    """Run one episode; return rewards, log_probs, values, success, num_collisions, actual_path_length.
    If optimal_path_length is None, it is computed from pathfinder (for SPL).
    """
    agent = sim.get_agent(0)
    pathfinder = sim.pathfinder
    state = agent.get_state()
    state.position = np.array(start_pos)
    from habitat_sim.utils import quat_from_angle_axis
    state.rotation = quat_from_angle_axis(start_rot[1], np.array([0, 1, 0]))
    agent.set_state(state)

    rewards = []
    log_probs_list = []
    values_list = []
    success = False
    num_collisions = 0
    positions = [np.array(start_pos)]
    if optimal_path_length is None and pathfinder.is_loaded:
        optimal_path_length = get_optimal_path_length(pathfinder, start_pos, goal_pos)
    if optimal_path_length is None:
        optimal_path_length = float("inf")
    policy.eval() if deterministic else policy.train()

    obs = sim.get_sensor_observations()
    for step in range(max_steps):
        rgb = obs["color_sensor"]  # (H,W,3)
        dist, theta = get_goal_vector(agent.get_state(), goal_pos)
        if dist <= SUCCESS_DISTANCE:
            success = True
            rewards.append(1.0)
            positions.append(agent.get_state().position)
            break

        goal_tensor = torch.tensor([[dist, theta]], dtype=torch.float32, device=device)
        with torch.no_grad() if deterministic else torch.enable_grad():
            emb = encode_observation(processor, vjepa_model, rgb, device)
            logits, value = policy(emb.float(), goal_tensor)
            dist_probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = logits.argmax(dim=-1).item()
                log_prob = F.log_softmax(logits, dim=-1)[0, action]
            else:
                m = torch.distributions.Categorical(dist_probs)
                action = m.sample().item()
                log_prob = m.log_prob(torch.tensor(action, device=device))
            values_list.append(value.squeeze())
            log_probs_list.append(log_prob)

        obs = apply_action_sim(sim, action)
        positions.append(agent.get_state().position)
        if obs.get("collided", False):
            num_collisions += 1
        rewards.append(-0.01 * dist)
        if action == ACTION_STOP and dist <= SUCCESS_DISTANCE:
            success = True
            break

    # Actual path length = sum of segment lengths
    actual_path_length = 0.0
    for i in range(1, len(positions)):
        actual_path_length += np.linalg.norm(positions[i] - positions[i - 1])

    if success and rewards:
        rewards[-1] = 1.0
    return rewards, log_probs_list, values_list, success, num_collisions, actual_path_length, optimal_path_length


def train(
    scene_id: str = "apt_0",
    num_episodes: int = 100,
    max_steps: int = 500,
    lr: float = 3e-4,
    device_name: str = "cuda",
    out_dir: Path | None = None,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    out_dir = out_dir or (PROJECT_ROOT / "checkpoints" / "pointnav_vjepa2")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sim and generate episodes
    sim = create_sim(scene_id)
    pathfinder = sim.pathfinder
    if not pathfinder.is_loaded:
        print("Navmesh not loaded; cannot run PointNav.", file=sys.stderr)
        print(
            "ReplicaCAD needs Habitat-Sim built WITH Bullet (articulated objects). "
            "Conda builds often have no Bullet. Options: (1) Build habitat-sim from source with --bullet, "
            "or (2) Use a dataset that does not use articulated objects (e.g. Gibson / Matterport3D).",
            file=sys.stderr,
        )
        return 1
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(num_episodes):
        start = pathfinder.get_random_navigable_point()
        goal = pathfinder.get_random_navigable_point()
        d = np.linalg.norm(np.array(goal) - np.array(start))
        while d < 1.0 or d > 8.0:
            goal = pathfinder.get_random_navigable_point()
            d = np.linalg.norm(np.array(goal) - np.array(start))
        episodes.append((list(start), [0, rng.uniform(0, 2 * np.pi), 0], list(goal)))

    processor, vjepa_model = load_vjepa_encoder(device)
    vjepa_model.eval()
    policy = PointNavPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    success_count = 0
    total_collisions = 0
    spl_list = []  # SPL per episode for reporting
    for ep_idx in range(num_episodes):
        start_pos, start_rot, goal_pos = episodes[ep_idx]
        rewards, log_probs, values, success, num_collisions, actual_path_len, optimal_path_len = run_episode(
            sim, processor, vjepa_model, policy, start_pos, start_rot, goal_pos, device, max_steps, deterministic=False
        )
        if success:
            success_count += 1
        total_collisions += num_collisions
        # SPL = success * (optimal / max(actual, optimal)); standard PointNav metric
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
        if (ep_idx + 1) % 10 == 0:
            mean_coll = total_collisions / (ep_idx + 1)
            print(f"Episode {ep_idx+1}/{num_episodes} success={success_count} recent_success={success} collisions/ep={mean_coll:.2f} loss={loss.item():.4f}")
    sim.close()

    ckpt_path = out_dir / "policy_last.pt"
    torch.save({"policy": policy.state_dict(), "episodes": num_episodes}, ckpt_path)
    mean_collisions_per_episode = total_collisions / num_episodes if num_episodes else 0
    mean_spl = np.mean(spl_list) if spl_list else 0.0
    print(f"Saved {ckpt_path}. Success rate: {success_count}/{num_episodes} = {100*success_count/num_episodes:.1f}% | SPL: {mean_spl:.3f} | Collisions/ep: {mean_collisions_per_episode:.2f}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_id", type=str, default="apt_0")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.eval_only:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        policy = PointNavPolicy().to(device)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device)
            policy.load_state_dict(ckpt["policy"])
        processor, vjepa_model = load_vjepa_encoder(device)
        sim = create_sim(args.scene_id)
        pathfinder = sim.pathfinder
        if not pathfinder.is_loaded:
            print("Navmesh not loaded")
            return 1
        rng = np.random.default_rng(42)
        successes = 0
        total_collisions = 0
        spl_list = []
        for i in range(20):
            start = pathfinder.get_random_navigable_point()
            goal = pathfinder.get_random_navigable_point()
            d = np.linalg.norm(np.array(goal) - np.array(start))
            while d < 1.0 or d > 8.0:
                goal = pathfinder.get_random_navigable_point()
                d = np.linalg.norm(np.array(goal) - np.array(start))
            _, _, _, ok, num_coll, actual_len, optimal_len = run_episode(sim, processor, vjepa_model, policy, list(start), [0, rng.uniform(0, 2*np.pi), 0], list(goal), device, args.max_steps, deterministic=True)
            if ok:
                successes += 1
            total_collisions += num_coll
            if optimal_len > 0 and not math.isinf(optimal_len):
                spl_list.append((1.0 if ok else 0.0) * (optimal_len / max(actual_len, optimal_len)))
        sim.close()
        mean_coll = total_collisions / 20
        mean_spl = np.mean(spl_list) if spl_list else 0.0
        print(f"Eval success rate: {successes}/20 = {100*successes/20:.1f}% | SPL: {mean_spl:.3f} | Collisions/ep: {mean_coll:.2f}")
        return 0

    return train(
        scene_id=args.scene_id,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        lr=args.lr,
        device_name=args.device,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
