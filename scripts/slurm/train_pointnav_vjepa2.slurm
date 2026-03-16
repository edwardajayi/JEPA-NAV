#!/bin/bash
#SBATCH --job-name=JEPANAV_pointnav
#SBATCH --output=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_vjepa2_%j.out
#SBATCH --error=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/pointnav_vjepa2_%j.err
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -A cis250225p

set -e

module load anaconda3
source activate /ocean/projects/cis250225p/eajayi1/JEPANAV/.conda_env

echo "=========================================================="
echo "Phase C: PointNav V-JEPA 2 training - Job Start: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "=========================================================="

cd /ocean/projects/cis250225p/eajayi1/JEPANAV
mkdir -p logs checkpoints/pointnav_vjepa2

python scripts/train_pointnav_vjepa2.py \
  --scene_id apt_0 \
  --num_episodes 200 \
  --max_steps 500 \
  --lr 3e-4 \
  --device cuda \
  --out_dir checkpoints/pointnav_vjepa2 \
  --seed 42

echo "=========================================================="
echo "Job End: $(date)"
echo "=========================================================="
