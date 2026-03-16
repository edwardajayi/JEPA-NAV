#!/bin/bash
# Run Habitat-Sim install with Bullet ON A COMPUTE NODE so the same .conda_env
# that training jobs use gets the Bullet build. Run once; then PointNav jobs will work.
#SBATCH --job-name=JEPANAV_install_bullet
#SBATCH --output=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/install_habitat_bullet_%j.out
#SBATCH --error=/ocean/projects/cis250225p/eajayi1/JEPANAV/logs/install_habitat_bullet_%j.err
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH -A cis250225p

set -e
module load anaconda3
source activate /ocean/projects/cis250225p/eajayi1/JEPANAV/.conda_env
cd /ocean/projects/cis250225p/eajayi1/JEPANAV
mkdir -p logs

echo "Installing Habitat-Sim with Bullet on compute node - $(date)"
bash scripts/install_habitat_sim_with_bullet.sh
echo "Install finished - $(date)"
echo "Verifying habitat_sim and Bullet (load ReplicaCAD scene, check pathfinder)..."
python -c "
import habitat_sim
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = '/ocean/projects/cis250225p/eajayi1/JEPANAV/data/ReplicaCAD/scenes/apt_0/habitat/apt_0.glb'
cfg.scene_dataset_config_file = '/ocean/projects/cis250225p/eajayi1/JEPANAV/data/ReplicaCAD/replicaCAD.scene_dataset_config.json'
cfg.physics_config_file = ''
agent_cfg = habitat_sim.AgentConfiguration()
sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
loaded = sim.pathfinder.is_loaded
sim.close()
print('pathfinder.is_loaded:', loaded)
assert loaded, 'Navmesh not loaded - Bullet build may have failed'
print('OK: Habitat-Sim with Bullet is working.')
"
echo "All done - $(date)"
