#!/bin/bash
# Full baseline training on H100 (200 episodes, 500 max steps).
# Expect ~1–2.5 h on H100 (faster than V100). Submit on the side while you use quicktest for the report.
#SBATCH --job-name=JEPANAV_pointnav_full
#SBATCH --output=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_vjepa2_full_%j.out
#SBATCH --error=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_vjepa2_full_%j.err
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -A cis250225p

set -e
CONDA_ENV=/ocean/projects/cis250225p/eajayi1/JEPANAV/.conda_env
module load anaconda3
source activate "$CONDA_ENV"
# Force env's libs and Python packages (so Bullet-built habitat_sim is used)
export LD_LIBRARY_PATH="$CONDA_ENV/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$CONDA_ENV/lib/python3.9/site-packages:$PYTHONPATH"
cd /ocean/projects/cis250225p/eajayi1/JEPANAV
mkdir -p logs checkpoints/pointnav_vjepa2

echo "Full run (H100): 200 episodes, 500 max steps - $(date)"
echo "Using Python: $(which python) ($CONDA_ENV)"
"$CONDA_ENV/bin/python" scripts/train_pointnav_vjepa2.py \
  --scene_id apt_0 \
  --num_episodes 200 \
  --max_steps 500 \
  --lr 3e-4 \
  --device cuda \
  --out_dir checkpoints/pointnav_vjepa2 \
  --seed 42
echo "Done - $(date)"
