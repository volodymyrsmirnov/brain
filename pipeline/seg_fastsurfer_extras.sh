#!/usr/bin/env bash
# FastSurfer v2.6 extra modules on top of the existing whole-brain segmentation:
#   HypVINN   - hypothalamus subregions, mammillary bodies, fornix, optic nerves/chiasm/tracts,
#               pituitary, pineal, anterior commissure, 3rd ventricle   (T1 only, see below)
#   CerebNet  - cerebellum lobules I-X, crus I/II, vermis, cerebellar white matter
#   FastSurfer-CC - corpus callosum in five segments (front to back) + fornix
# None of these needs a FreeSurfer licence or FreeSurfer binaries.
#
# The main segmentation (data/derived/aparc.DKTatlas+aseg.deep.mgz, ~10 min) is reused, not re-run:
# orig.mgz is re-conformed from data/derived/t1.nii.gz (bit-identical grid, checked) and
# aseg.auto_noCCseg.mgz + mask.mgz are re-derived exactly as FastSurferCNN does, then
# run_fastsurfer.sh --no_asegdkt runs the bias-field correction and the three modules.
#
# Why T1 only for HypVINN: it was trained on 3D T1w/T2w. Our only 3D "T2-like" scan covering the
# hypothalamus is a FLAIR (dark CSF: a different contrast), the T2 TSE has 4 mm slices, and the
# 0.3 mm T2 SPACE slab stops below the hypothalamus. HypVINN's T1-only mode is the supported option.
#
# Outputs (T1 world space, 0.5 mm conformed grid, label + JSON {label: {name, color}}):
#   data/derived/{hypothalamus_labels,cerebellum_lobules,corpus_callosum}.nii.gz/.json
# Idempotent: finished steps are skipped; FORCE=1 re-runs everything. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
D="$ROOT/data/derived"
W="$ROOT/work/fastsurfer-extras"
S="$W/head"
M="$S/mri"
FS="$ROOT/tools/FastSurfer"
VENV="$ROOT/tools/venv-fastsurfer"
THREADS="${THREADS:-6}"
HELPER="$ROOT/pipeline/export_fastsurfer_extras.py"
OUTS=("$D"/hypothalamus_labels.nii.gz "$D"/cerebellum_lobules.nii.gz "$D"/corpus_callosum.nii.gz)

if [[ "${FORCE:-0}" == "1" ]]; then rm -rf "$S"; rm -f "$D"/{hypothalamus_labels,cerebellum_lobules,corpus_callosum}.*; fi
all_done=1; for f in "${OUTS[@]}"; do [[ -f "$f" ]] || all_done=0; done
if [[ "$all_done" == "1" ]]; then echo "fastsurfer extras: outputs exist, skipping (FORCE=1 to redo)"; exit 0; fi

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$FS"
mkdir -p "$M"

# 1. conformed T1 on the same 0.5 mm grid as the existing segmentation
if [[ ! -f "$M/orig.mgz" ]]; then
  python3 "$FS/FastSurferCNN/data_loader/conform.py" -i "$D/t1.nii.gz" -o "$M/orig.mgz" --vox_size min
fi
# 2. aseg.auto_noCCseg.mgz + mask.mgz from the existing asegdkt (also copies it into the subject)
if [[ ! -f "$M/aseg.auto_noCCseg.mgz" ]]; then
  python3 "$HELPER" prep "$D/aparc.DKTatlas+aseg.deep.mgz" "$M"
fi
# 3. checkpoints (~240 MB, into tools/FastSurfer/checkpoints)
( cd "$FS" && python3 FastSurferCNN/download_checkpoints.py --cerebnet --cc --hypvinn )

# 4. CC + CerebNet through run_fastsurfer.sh (also writes orig_nu.mgz, the bias-corrected T1);
#    skip any whose output already exists. HypVINN runs separately in step 5.
flags=(--no_hypothal)
[[ -f "$M/cerebellum.CerebNet.nii.gz" ]] && flags+=(--no_cereb)
[[ -f "$M/aparc.DKTatlas+aseg.deep.withCC.mgz" ]] && flags+=(--no_cc)
if [[ "${#flags[@]}" -lt 3 || ! -f "$M/orig_nu.mgz" ]]; then
  # helper shell steps (stats) may fail on odd paths; the label files are checked below
  ( cd "$FS" && ./run_fastsurfer.sh --t1 "$D/t1.nii.gz" --sid head --sd "$W" --seg_only --no_asegdkt \
      --vox_size min --threads "$THREADS" --viewagg_device cpu --py python3 "${flags[@]}" ) \
      > "$W/fastsurfer_extras.log" 2>&1 || echo "WARNING: run_fastsurfer.sh exited non-zero, see $W/fastsurfer_extras.log"
fi
# 5. HypVINN (T1 only) on the bias-corrected T1, as run_fastsurfer.sh would call it, but through the
#    helper that keeps the optic tracts (see export_fastsurfer_extras.py "hypvinn").
if [[ ! -f "$M/hypothalamus.HypVINN.nii.gz" ]]; then
  python3 "$HELPER" hypvinn --sd "$W" --sid head --threads "$THREADS" --async_io --batch_size 1 \
      --device auto --viewagg_device cpu --t1 "$M/orig_nu.mgz" \
      --seg_log "$S/scripts/hypvinn_seg.log" >> "$W/fastsurfer_extras.log" 2>&1
fi
for f in hypothalamus.HypVINN.nii.gz cerebellum.CerebNet.nii.gz aparc.DKTatlas+aseg.deep.withCC.mgz; do
  [[ -f "$M/$f" ]] || { echo "ERROR: $M/$f missing, see $W/fastsurfer_extras.log"; exit 1; }
done

# 6. export to data/derived
python3 "$HELPER" export "$S" "$D"
