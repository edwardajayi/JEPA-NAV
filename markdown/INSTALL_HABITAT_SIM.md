# Installing habitat-sim (separate from pip)

`habitat-sim` often has **no PyPI wheel** on Linux. The conda channel has prebuilt packages only for **Python 3.9** (and older). If your env is **Python 3.10+**, conda will fail with a conflict.

---

## Option 1: New conda env with Python 3.9 (recommended)

Create a dedicated env that matches habitat-sim’s prebuilt builds:

**On HPC (e.g. Bridges2) use a persistent path** so the env is not in `/tmp` and lost on logout:

```bash
# Persistent env inside your project (survives /tmp cleanup)
cd /ocean/projects/cis250225p/eajayi1/JEPANAV
conda create --prefix ./.conda_env python=3.9 -y
conda activate ./.conda_env

# Install habitat-sim from conda (headless for servers without display)
# For ReplicaCAD/PointNav you need Bullet; use withbullet:
conda install -c conda-forge -c aihabitat habitat-sim withbullet headless -y

# Install the rest from your project
pip install -r requirements.txt
```

Activate later with: `conda activate /ocean/projects/cis250225p/eajayi1/JEPANAV/.conda_env`

**If you don't need a persistent path** (e.g. local machine):

```bash
conda create -n jepa_nav python=3.9 -y
conda activate jepa_nav
conda install -c conda-forge -c aihabitat habitat-sim withbullet headless -y
cd /ocean/projects/cis250225p/eajayi1/JEPANAV
pip install -r requirements.txt
```

---

## Option 2: Keep Python 3.10 (e.g. `jayi`) — build habitat-sim from source

If you must stay on Python 3.10, you have to **build habitat-sim from source**; there is no prebuilt conda package for 3.10.

1. Install build deps (Linux, e.g. Bridges2):
   ```bash
   # If you have sudo: apt-get install -y libglm-dev libglfw3-dev libmagic-dev
   # On HPC, use a module or ask support for: CMake, Ninja, Bullet, Magnum, etc.
   ```
2. Clone and build:
   ```bash
   git clone https://github.com/facebookresearch/habitat-sim.git
   cd habitat-sim
   pip install -r requirements.txt
   python setup.py install --headless --with-cuda
   ```
   See [habitat-sim: Installation](https://github.com/facebookresearch/habitat-sim#installation) for full dependencies and options.

---

## ReplicaCAD and PointNav: Bullet required

**ReplicaCAD** scenes use **articulated objects** (fridge, doors, cabinets). Habitat-sim needs to be built **with Bullet** to load them. If Bullet is missing:

- Scene load fails for those objects and the **navmesh is not attached**.
- The training script exits with: `Navmesh not loaded; cannot run PointNav.`

**Options for Bullet (ReplicaCAD PointNav):**

1. **Conda install with Bullet (easiest, Python 3.9 only)**  
   The aihabitat channel provides builds **with** Bullet. Use:
   ```bash
   conda install habitat-sim withbullet headless -c conda-forge -c aihabitat
   ```
   See [Habitat-Sim README](https://github.com/facebookresearch/habitat-sim#conda-install-habitat-sim). Script: `scripts/install_habitat_sim_conda_bullet.sh`.

2. **Build habitat-sim from source with Bullet**  
   See [habitat-sim install](https://github.com/facebookresearch/habitat-sim#installation); use the `--bullet` build flag. Script: `scripts/install_habitat_sim_with_bullet.sh`.

3. **Use a different dataset for PointNav**  
   Use a scene dataset that does not rely on articulated objects (e.g. Gibson, Matterport3D, or Habitat test scenes) so the conda build is sufficient.

---

## Summary

| Your Python | What to do |
|-------------|------------|
| **3.9**     | `conda install -c conda-forge -c aihabitat habitat-sim withbullet headless` (for ReplicaCAD) |
| **3.10+**   | Use a **Python 3.9** env (Option 1) or **build from source** (Option 2). |

After `habitat-sim` is installed, `habitat-lab` (from requirements.txt) will work with it.
