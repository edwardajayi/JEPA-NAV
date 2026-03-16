#!/bin/bash
# Quick ~15 min run: 10 episodes, 150 max steps. Same code as train_pointnav_simple_scene_h100.slurm.
#SBATCH --job-name=JEPANAV_quick
#SBATCH --output=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_simple_quick_%j.out
#SBATCH --error=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_simple_quick_%j.err
#SBATCH --mem=48G
#SBATCH --time=0:20:00
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
mkdir -p logs checkpoints/pointnav_vjepa2_simple_quick

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

echo "PointNav simple (H100) QUICK - 10 ep, 150 steps - $(date)"
"$CONDA_ENV/bin/python" -u scripts/train_pointnav_vjepa2_simple_scene.py \
  "${SCENE_ARGS[@]}" \
  --num_episodes 10 \
  --max_steps 150 \
  --lr 3e-4 \
  --device cuda \
  --out_dir checkpoints/pointnav_vjepa2_simple_quick \
  --seed 42
echo "Done - $(date)"
