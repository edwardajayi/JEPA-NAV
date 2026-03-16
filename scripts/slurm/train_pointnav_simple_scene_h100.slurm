#!/bin/bash
# Same as train_pointnav_simple_scene.slurm but requests H100 (sometimes shorter queue than V100).
# PointNav + V-JEPA 2 on a simple .glb scene (no ReplicaCAD, no Bullet).
#SBATCH --job-name=JEPANAV_pointnav_simple
#SBATCH --output=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_simple_%j.out
#SBATCH --error=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_simple_%j.err
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -A cis250225p

set -e
CONDA_ENV=/ocean/projects/cis250225p/eajayi1/JEPANAV/.conda_env
module load anaconda3
source activate "$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$CONDA_ENV/lib/python3.9/site-packages:$PYTHONPATH"
export PYTHONUNBUFFERED=1
cd /ocean/projects/cis250225p/eajayi1/JEPANAV
mkdir -p logs checkpoints/pointnav_vjepa2_simple

# Prefer Habitat test scenes (download with: python scripts/download_alternative_scenes.py --habitat-test-scenes)
SCENE_GLB=""
for f in data/scene_datasets/habitat-test-scenes/van-gogh-room.glb \
         data/scene_datasets/habitat-test-scenes/skokloster-castle.glb \
         data/scene_datasets/habitat-test-scenes/apartment_1.glb; do
  if [[ -f "$f" ]]; then SCENE_GLB="$f"; break; fi
done
SCENE_ARGS=()
if [[ -n "$SCENE_GLB" ]]; then
  echo "Using scene: $SCENE_GLB"
  SCENE_ARGS=(--scene_glb "$SCENE_GLB")
fi

echo "PointNav simple scene (H100) - $(date)"
"$CONDA_ENV/bin/python" -u scripts/train_pointnav_vjepa2_simple_scene.py \
  "${SCENE_ARGS[@]}" \
  --num_episodes 30 \
  --max_steps 300 \
  --lr 3e-4 \
  --device cuda \
  --out_dir checkpoints/pointnav_vjepa2_simple \
  --seed 42
echo "Done - $(date)"
