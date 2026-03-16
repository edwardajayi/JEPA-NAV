#!/bin/bash
#SBATCH --job-name=JEPANAV_replicacad_vjepa2
#SBATCH --output=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/replicacad_vjepa2_%j.out
#SBATCH --error=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/replicacad_vjepa2_%j.err
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH -A cis250225p

set -e

module load anaconda3
source activate /ocean/projects/cis250225p/eajayi1/JEPANAV/.conda_env

echo "=========================================================="
echo "Job Start: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo "=========================================================="

cd /ocean/projects/cis250225p/eajayi1/JEPANAV

mkdir -p logs
mkdir -p data/replicacad_vjepa2_frames

# ReplicaCAD V-JEPA 2: Collect ReplicaCAD frames in Habitat, then run V-JEPA 2
LOG_EXTRA="logs/replicacad_vjepa2_${SLURM_JOB_ID}.log"
python scripts/run_replicacad_vjepa2.py \
  --scene_id apt_0 \
  --num_frames 80 \
  --out_dir data/replicacad_vjepa2_frames \
  --vjepa_frames 16 \
  --log_file "$LOG_EXTRA" \
  --device cuda

echo "=========================================================="
echo "Job End: $(date)"
echo "=========================================================="
