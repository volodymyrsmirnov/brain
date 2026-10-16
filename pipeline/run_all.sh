#!/usr/bin/env bash
# Full pipeline: anonymized DICOM zip -> segmentations -> web model in docs/data.
# Each step is skipped when its output already exists; delete work/ to start over.
# Runtime on an M5 MacBook: ~3 h from scratch (verified Oct 2026), dominated by SimNIBS charm (twice) and HypVINN.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
W=$ROOT/work
D=$ROOT/data/derived
PY="$ROOT/.venv/bin/python -I"
mkdir -p "$W" "$D"

T1=9_t1_mprage_sag_p2_iso_101        # 3D T1 MPRAGE, 0.6 x 0.5 x 0.5 mm
T2TSE=14_pd-t2_tse_tra_101_e2         # 2D T2 TSE (full head, helps charm find bone)
T2SPACE=13_t2_space_tra_iso_126       # 3D T2 SPACE 0.3 mm (inner ear)
SWI=17_t2_qswi_tra_p2_101_SWI         # susceptibility-weighted (veins)
SWIPHA=16_t2_qswi_tra_p2_101_Pha_ph   # SWI phase (flags artefact-prone areas near air)

step() { echo; echo "== $*"; }

step "1/8 DICOM -> NIfTI (dcm2niix)"
if [ ! -f "$W/nifti/$T1.nii.gz" ]; then
  rm -rf "$W/dicom" && mkdir -p "$W/dicom" "$W/nifti"
  unzip -q data/raw/mri_study_anonymized.zip -d "$W/dicom"
  dcm2niix -z y -f "%s_%d" -o "$W/nifti" "$W/dicom" > "$W/dcm2niix.log"
  cp "$W/nifti/$T1.nii.gz" "$D/t1.nii.gz"
fi

step "2/8 skull stripping (SynthStrip)"
if [ ! -f "$D/brain_mask.nii.gz" ]; then
  mkdir -p "$W/synthstrip" && cp "$D/t1.nii.gz" "$W/synthstrip/t1.nii.gz"
  docker run --rm --platform linux/amd64 -v "$W/synthstrip":/data freesurfer/synthstrip:latest \
    -i /data/t1.nii.gz -o /data/t1_brain.nii.gz -m /data/brain_mask.nii.gz
  cp "$W/synthstrip/brain_mask.nii.gz" "$D/brain_mask.nii.gz"
fi

step "3/8 brain segmentation (FastSurfer, GPU via MPS if available)"
if [ ! -f "$D/aparc.DKTatlas+aseg.deep.mgz" ]; then
  ( export PATH="$ROOT/tools/venv-fastsurfer/bin:$PATH"; cd tools/FastSurfer
    ./run_fastsurfer.sh --t1 "$D/t1.nii.gz" --sid head --sd "$W/fastsurfer" --seg_only \
      --viewagg_device cpu --vox_size min --threads 8 --py python3 --no_hypothal ) > "$W/fastsurfer.log" 2>&1 || true
  cp "$W/fastsurfer/head/mri/aparc.DKTatlas+aseg.deep.mgz" "$D/"
fi

step "3b/8 extra FastSurfer modules: hypothalamus (HypVINN), cerebellar lobules (CerebNet), corpus callosum (FastSurfer-CC)"
pipeline/seg_fastsurfer_extras.sh

step "4/8 head tissues incl. skull (SimNIBS charm, T1 + T2)"
if [ ! -f "$D/charm_tissues.nii.gz" ]; then
  LBL="$W/m2m_head/label_prep/tissue_labeling_upsampled.nii.gz"
  ( cd "$W" && "$ROOT/tools/venv-simnibs/bin/charm" head "$D/t1.nii.gz" "$W/nifti/$T2TSE.nii.gz" \
      --registerT2 --initatlas --segment --forcerun ) > "$W/charm.log" 2>&1 &
  CHARM=$!
  # --segment goes on to reconstruct cortical surfaces (slow, not needed here):
  # stop charm as soon as the tissue label map has been written.
  until [ -f "$LBL" ] || ! kill -0 $CHARM 2>/dev/null; do sleep 10; done
  sleep 15; kill $CHARM 2>/dev/null || true
  cp "$LBL" "$D/charm_tissues.nii.gz"
fi

step "4b/8 skull parts (charm bone + FastSurfer landmarks)"
if [ ! -f "$D/skull_parts.nii.gz" ]; then
  $PY pipeline/seg_skull_parts.py "$D/charm_tissues.nii.gz" "$D/aparc.DKTatlas+aseg.deep.mgz" "$D"
fi

step "4c/8 optic nerves + eye muscles (T1)"
if [ ! -f "$D/orbit_labels.nii.gz" ]; then
  $PY pipeline/seg_orbit.py "$D/t1.nii.gz" "$D/charm_tissues.nii.gz" "$D/aparc.DKTatlas+aseg.deep.mgz" "$D" --per-muscle
fi

step "4d/8 dental X-ray bent onto the jaw (only if data/raw/dental_panoramic.jpg exists)"
if [ -f data/raw/dental_panoramic.jpg ] && [ ! -f "$D/dental_xray.npz" ]; then
  # --chin-z: charm's jaw stops short of the real chin on this scan; -105 mm was read off the midline T1.
  $PY pipeline/build_dental_xray.py data/raw/dental_panoramic.jpg "$D/skull_parts.nii.gz" "$D/dental_xray.npz" --chin-z -105
fi

step "5/8 inner ear (T2 SPACE, labyrinths found automatically)"
if [ ! -f "$D/inner_ear_labels.nii.gz" ]; then
  $PY pipeline/seg_inner_ear.py "$W/nifti/$T2SPACE.nii.gz" "$W/ear" --aseg "$D/aparc.DKTatlas+aseg.deep.mgz"
  cp "$W/ear/inner_ear_labels.nii.gz" "$D/"
fi

step "6/8 veins (SWI) + dural sinuses (charm)"
if [ ! -f "$D/veins.nii.gz" ]; then
  $PY pipeline/seg_veins.py "$W/nifti/$SWI.nii.gz" "$W/veins" "$D/brain_mask.nii.gz" 99.3 97.0 40 0.12 \
    --phase "$W/nifti/$SWIPHA.nii.gz" --charm "$D/charm_tissues.nii.gz"
  cp "$W/veins/veins.nii.gz" "$D/"
fi

step "7/8 cortical surfaces (SimNIBS charm TopoFit, T1 only) + inflation, DKT labels, sulcal depth"
if [ ! -f "$D/surfaces/rh.inflated.gii" ]; then
  # A second, T1-only charm run: --segment does not build surfaces in SimNIBS 4.6, --surfaces does.
  mkdir -p "$W/cortical-surfaces"
  ( cd "$W/cortical-surfaces" && "$ROOT/tools/venv-simnibs/bin/charm" head_surf "$D/t1.nii.gz" --initatlas --segment --forcerun \
      && "$ROOT/tools/venv-simnibs/bin/charm" head_surf --surfaces ) > "$W/cortical-surfaces/charm.log" 2>&1
  $PY pipeline/build_surfaces.py "$W/cortical-surfaces/m2m_head_surf/surfaces" "$D/aparc.DKTatlas+aseg.deep.mgz" "$D/surfaces" \
      --t1 "$D/t1.nii.gz"
fi

step "8/8 meshes + web data"
$PY pipeline/build_model.py "$D" docs/data "$W/meshes"
cp "$W/meshes/head_all.glb" "$D/head_all.glb"

echo; echo "Done. Preview: python3 -m http.server -d docs 8000  ->  http://localhost:8000"
