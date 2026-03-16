#!/bin/bash
# ~8 min total: 4 episodes, 40 steps/ep. For midterm baseline numbers.
#SBATCH --job-name=JEPANAV_8min
#SBATCH --output=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_simple_8min_%j.out
#SBATCH --error=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_simple_8min_%j.err
#SBATCH --mem=48G
#SBATCH --time=0:10:00
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
mkdir -p logs checkpoints/pointnav_vjepa2_simple_8min

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

echo "PointNav simple (H100) 8min - 4 ep, 40 steps - $(date)"
"$CONDA_ENV/bin/python" -u scripts/train_pointnav_vjepa2_simple_scene.py \
  "${SCENE_ARGS[@]}" \
  --num_episodes 4 \
  --max_steps 40 \
  --lr 3e-4 \
  --device cuda \
  --out_dir checkpoints/pointnav_vjepa2_simple_8min \
  --seed 42
echo "Done - $(date)"
