#!/bin/bash
# Install Habitat-Sim WITH Bullet via conda (official aihabitat packages).
# Use this instead of building from source if your env is Python 3.9.
# Run from project root with conda env activated:
#   conda activate /ocean/projects/cis250225p/eajayi1/JEPANAV/.conda_env
#   bash scripts/install_habitat_sim_conda_bullet.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "Installing Habitat-Sim with Bullet via conda (headless)"
echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python)"
echo "=========================================================="

# Remove existing habitat-sim so we get the withbullet build
echo "Removing existing habitat-sim (conda/pip)..."
conda remove habitat-sim --force -y 2>/dev/null || true
pip uninstall -y habitat-sim 2>/dev/null || true

# Official command from Habitat-Sim README: withbullet + headless for clusters
echo "Installing habitat-sim withbullet headless from aihabitat..."
conda install -y habitat-sim withbullet headless -c conda-forge -c aihabitat

echo "=========================================================="
echo "Done. Verify with:"
echo "  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
echo "  python -c \"import habitat_sim; print(habitat_sim.__file__)\""
echo "  python -c \"from scripts.train_pointnav_vjepa2 import create_sim; s=create_sim('apt_0'); print('pathfinder:', s.pathfinder.is_loaded); s.close()\""
echo "=========================================================="
