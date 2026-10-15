#!/usr/bin/env bash
# One-time setup of every tool the pipeline needs (tested on macOS 27, Apple Silicon).
# Linux works too with the obvious substitutions (apt instead of brew, CUDA instead of MPS).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD

echo "== system packages (Homebrew)"
brew install dcm2niix mpfr gmp git-lfs uv

echo "== main Python env (.venv)"
uv venv -q --allow-existing -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -q -r pipeline/requirements.txt

mkdir -p tools
echo "== FastSurfer v2.6.0 (brain segmentation; seg-only mode needs no FreeSurfer licence)"
[ -d tools/FastSurfer ] || git clone -q --depth 1 --branch v2.6.0 https://github.com/Deep-MI/FastSurfer.git tools/FastSurfer
uv venv -q --allow-existing -p 3.12 tools/venv-fastsurfer
VIRTUAL_ENV=tools/venv-fastsurfer uv pip install -q -r tools/FastSurfer/requirements.txt

echo "== SimNIBS 4.6 (charm head segmentation)"
# SimNIBS normally installs into conda; here we use its own wheels from GitHub instead.
uv venv -q --allow-existing -p 3.11 tools/venv-simnibs
VIRTUAL_ENV=tools/venv-simnibs uv pip install -q \
  "https://github.com/oulap/samseg_wheels/releases/download/dev/samseg-0.5a0-cp311-cp311-macosx_14_0_arm64.whl" \
  "https://github.com/simnibs/cortech/releases/download/v0.1/cortech-0.1-cp311-cp311-macosx_11_0_arm64.whl" \
  "https://github.com/simnibs/fmm3dpy/releases/download/v1.0.4/fmm3dpy-1.0.4-cp311-cp311-macosx_14_0_arm64.whl" \
  "https://github.com/simnibs/petsc4py/releases/download/v3.22.2/petsc4py-3.22.2-cp311-cp311-macosx_14_0_arm64.whl" \
  "brainnet @ git+https://github.com/simnibs/brainnet@v0.2" "brainsynth @ git+https://github.com/simnibs/brainsynth@v0.1" \
  gmsh==4.14.0 h5py==3.15.1 nibabel==5.3.3 numba==0.64.0 numpy==2.3.0 pygpc==0.4.4 pyqt5==5.15.11 scipy==1.17.1 torch==2.6.0 \
  jsonschema matplotlib mock pillow pyopengl requests \
  "https://github.com/simnibs/simnibs/releases/download/v4.6.0/simnibs-4.6.0-cp311-cp311-macosx_11_0_arm64.whl"
SP=tools/venv-simnibs/lib/python3.11/site-packages
# The CGAL extensions expect conda's libmpfr/libgmp: point them at Homebrew's copies.
for f in "$SP"/simnibs/mesh_tools/cgal/*.so; do
  for p in "$(brew --prefix mpfr)/lib" "$(brew --prefix gmp)/lib" /usr/lib; do
    install_name_tool -add_rpath "$p" "$f" 2>/dev/null || true
  done
  codesign -f -s - "$f"
done
# python-mumps is conda-only and only used for FEM field simulations, which we don't run.
echo '"""Stub: python-mumps is only needed for FEM solves, not for charm segmentation."""' > "$SP/mumps.py"

echo "== SynthStrip (skull stripping) via Docker"
docker pull --platform linux/amd64 freesurfer/synthstrip:latest

echo "Setup complete. Next: pipeline/run_all.sh"
