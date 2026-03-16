#!/usr/bin/env python3
"""
Download scene data for PointNav when not using ReplicaCAD (no Bullet).

Option A: Habitat test scenes (small, no form) – 3 .glb scenes for testing.
Option B: Gibson – requires agreeing to terms; then download zip and run this to set up config.

Usage:
  # Download Habitat test scenes (recommended first; ~few MB)
  python scripts/download_alternative_scenes.py --habitat-test-scenes

  # After you have Gibson zip from the form, extract to data/scene_datasets/gibson/
  # then run this to add the Habitat config:
  python scripts/download_alternative_scenes.py --gibson-config
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
SCENE_DATASETS = DATA_ROOT / "scene_datasets"
GIBSON_DIR = SCENE_DATASETS / "gibson"
GIBSON_CONFIG_URL = "http://dl.fbaipublicfiles.com/habitat/gibson/config_v1/gibson_semantic.scene_dataset_config.json"


def download_habitat_test_scenes() -> int:
    """Download Habitat test scenes (van-gogh-room, skokloster-castle, etc.) via Habitat's utility."""
    try:
        import habitat_sim
    except ImportError:
        print("habitat_sim not found. Activate your conda env and run again.", file=sys.stderr)
        return 1
    SCENE_DATASETS.mkdir(parents=True, exist_ok=True)
    data_path = str(DATA_ROOT) + os.sep
    cmd = [
        sys.executable,
        "-m",
        "habitat_sim.utils.datasets_download",
        "--uids", "habitat_test_scenes",
        "--data-path", data_path,
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.call(cmd)
    if r != 0:
        return r
    out = SCENE_DATASETS / "habitat-test-scenes"
    if out.exists():
        glbs = list(out.glob("*.glb"))
        print(f"Done. Scenes in {out}: {[p.name for p in glbs]}")
        print(f"Run: python scripts/train_pointnav_vjepa2_simple_scene.py --scene_glb {out}/<scene>.glb --num_episodes 30")
    return 0


def download_gibson_config() -> int:
    """Download Gibson scene_dataset_config.json into data/scene_datasets/gibson/."""
    try:
        import urllib.request
    except ImportError:
        print("urllib not available", file=sys.stderr)
        return 1
    GIBSON_DIR.mkdir(parents=True, exist_ok=True)
    dest = GIBSON_DIR / "gibson_semantic.scene_dataset_config.json"
    print(f"Downloading {GIBSON_CONFIG_URL} -> {dest}")
    try:
        urllib.request.urlretrieve(GIBSON_CONFIG_URL, dest)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1
    print(f"Saved {dest}. Put extracted Gibson .glb files in {GIBSON_DIR} then use --scene_glb with one of them.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Download alternative scene data for PointNav (no Bullet).")
    ap.add_argument("--habitat-test-scenes", action="store_true", help="Download Habitat test scenes (small, no form).")
    ap.add_argument("--gibson-config", action="store_true", help="Download Gibson config only (you must get the zip from the form).")
    args = ap.parse_args()
    if not (args.habitat_test_scenes or args.gibson_config):
        ap.print_help()
        print("\nGibson full dataset: agree to terms at https://github.com/StanfordVL/GibsonEnv#database")
        print("  then get 'Gibson Dataset for Habitat' (e.g. gibson_habitat.zip ~1.5GB or gibson_habitat_trainval.zip ~11GB),")
        print("  extract to data/scene_datasets/gibson/ and run: python scripts/download_alternative_scenes.py --gibson-config")
        return 0
    if args.habitat_test_scenes:
        return download_habitat_test_scenes()
    if args.gibson_config:
        return download_gibson_config()
    return 0


if __name__ == "__main__":
    sys.exit(main())
