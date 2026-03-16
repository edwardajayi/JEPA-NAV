#!/bin/bash
# Install Habitat-Sim from source WITH Bullet so ReplicaCAD PointNav (navmesh) works.
# Run from project root with conda env activated:
#   conda activate /ocean/projects/cis250225p/eajayi1/JEPANAV/.conda_env
#   bash scripts/install_habitat_sim_with_bullet.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================================="
echo "Installing Habitat-Sim from source (with Bullet)"
echo "Project root: $PROJECT_ROOT"
echo "Python: $(which python)"
echo "=========================================================="

# Ensure build deps (cmake)
if ! command -v cmake &>/dev/null; then
  echo "Installing cmake via conda..."
  conda install -y cmake
fi

# Remove existing habitat-sim so we replace with source build (Bullet enabled)
echo "Removing existing habitat-sim (conda/pip)..."
conda remove habitat-sim --force -y 2>/dev/null || true
pip uninstall -y habitat-sim 2>/dev/null || true

# Clone habitat-sim (stable) if not present
HABITAT_SIM_SRC="$PROJECT_ROOT/tmp/habitat-sim"
if [ ! -d "$HABITAT_SIM_SRC" ]; then
  echo "Cloning habitat-sim (stable)..."
  mkdir -p "$(dirname "$HABITAT_SIM_SRC")"
  git clone --depth 1 --branch stable https://github.com/facebookresearch/habitat-sim.git "$HABITAT_SIM_SRC"
fi
cd "$HABITAT_SIM_SRC"

# Clean previous failed/partial build so magnum-bindings .so is built fresh
echo "Cleaning previous build dir (if any)..."
rm -rf build

# Step 1: Run native build first so _corrade.*.so exists before pip's metadata phase.
# Build WITHOUT CUDA so it works on login nodes (no nvcc there). Rendering on GPU nodes still uses OpenGL/EGL.
# To build with CUDA, run this script from a GPU node or after: module load cuda
echo "Step 1/2: Native build (headless + Bullet, no CUDA). This may take 15-30 min..."
echo "  CWD: $(pwd)"
python setup.py build_ext --inplace --headless --bullet -j 4

# Step 2: pip install so habitat_sim and magnum bindings are installed (magnum .so already exists)
echo "Step 2/2: pip install..."
pip install . --no-build-isolation

cd "$PROJECT_ROOT"
echo "=========================================================="
echo "Done. Verify with: python -c \"import habitat_sim; print(habitat_sim.__file__)\""
echo "=========================================================="
