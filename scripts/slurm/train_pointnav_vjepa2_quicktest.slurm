#!/bin/bash
# Quick test: fewer episodes and steps to verify pipeline in ~20-40 min.
# Use this first; then run full train_pointnav_vjepa2.slurm (200 ep, 500 steps).
#SBATCH --job-name=JEPANAV_pointnav_quick
#SBATCH --output=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_quick_%j.out
#SBATCH --error=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_quick_%j.err
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -A cis250225p

set -e
CONDA_ENV=/ocean/projects/cis250225p/eajayi1/JEPANAV/.conda_env
module load anaconda3
source activate "$CONDA_ENV"
# Force env's libs and Python packages (so Bullet-built habitat_sim is used, not module's)
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$CONDA_ENV/lib/python3.9/site-packages:$PYTHONPATH"
cd /ocean/projects/cis250225p/eajayi1/JEPANAV
mkdir -p logs checkpoints/pointnav_vjepa2

echo "Quick test: 15 episodes, 150 max steps - $(date)"
echo "Using Python: $(which python) ($CONDA_ENV)"
"$CONDA_ENV/bin/python" scripts/train_pointnav_vjepa2.py \
  --scene_id apt_0 \
  --num_episodes 15 \
  --max_steps 150 \
  --lr 3e-4 \
  --device cuda \
  --out_dir checkpoints/pointnav_vjepa2 \
  --seed 42
echo "Done - $(date)"
