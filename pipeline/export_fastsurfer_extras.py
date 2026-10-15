#!/usr/bin/env python3
"""Helpers for the FastSurfer extra modules (HypVINN, CerebNet, FastSurfer-CC).

Two sub-commands, both called by pipeline/seg_fastsurfer_extras.sh:

  prep <aparc.mgz> <subject_mri_dir>
      Re-creates the inputs FastSurfer's later modules expect next to an existing
      `aparc.DKTatlas+aseg.deep.mgz`, so the 10-minute main segmentation is not re-run:
      `aseg.auto_noCCseg.mgz` and `mask.mgz`, made exactly as FastSurferCNN/run_prediction.py
      makes them (reduce_to_aseg + create_mask(5, 4) + flip_wm_islands). `orig.mgz` must
      already exist (conform.py --vox_size min) and share the aparc grid; this is checked.
      Must run with the FastSurfer venv and PYTHONPATH=tools/FastSurfer.

  hypvinn <HypVINN/run_prediction.py arguments...>
      Runs HypVINN with one post-processing fix. HypVINN keeps only the largest connected blob of
      {optic nerves, optic tracts} after a 7-voxel dilation, chiasm excluded. On the 0.5 mm grid
      7 voxels = 3.5 mm do not bridge the chiasm, so the nerves and tracts form separate blobs and
      the optic tracts were silently dropped. The fix counts the chiasm as part of the optic
      system for that connectivity test only (the chiasm labels themselves come from the other
      branch as before). Must run with the FastSurfer venv and PYTHONPATH=tools/FastSurfer.

  export <subject_dir> <out_dir>
      Converts the module outputs into label volumes in the T1 world space (the conformed
      0.5 mm grid, whose affine maps to scanner RAS like every other file in data/derived)
      plus a JSON {label: {name, color}} with plain-English names:
        hypothalamus_labels.nii.gz  <- mri/hypothalamus.HypVINN.nii.gz
        cerebellum_lobules.nii.gz   <- mri/cerebellum.CerebNet.nii.gz
        corpus_callosum.nii.gz      <- CC 251-255 from mri/aparc.DKTatlas+aseg.deep.withCC.mgz
                                       (+ fornix 250 from mri/callosum.CC.orig.mgz, if present)
      Needs only nibabel + numpy (the project .venv works).
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------------------------
# Label tables. Keys are the module's own label values (kept, so they can be traced back to the
# FastSurfer LUTs); names are for non-experts; colours are hand-picked to be distinguishable
# side by side and to read well on a dark background. Left/right partners share a hue family.
# ---------------------------------------------------------------------------------------------

# HypVINN writes FreeSurfer LUT ids 961-984 (HypVINN/config/hypvinn_global_var.py). Its
# globus pallidus classes (13/20) are deleted by HypVINN itself; the aseg pallidum covers them.
HYPOTHALAMUS = {
    961: ("Optic nerve (R)", "#f2c14e"),
    962: ("Optic nerve (L)", "#e0a526"),
    967: ("Optic chiasm (R half)", "#f7dc6f"),
    968: ("Optic chiasm (L half)", "#ecc94b"),
    964: ("Optic tract (R)", "#f5a65b"),
    965: ("Optic tract (L)", "#e8873a"),
    963: ("Mammillary body (R)", "#ff6b81"),
    966: ("Mammillary body (L)", "#e84a6a"),
    969: ("Anterior commissure", "#9b8cff"),
    970: ("Third ventricle", "#5dade2"),
    971: ("Fornix (R)", "#e8e8e8"),
    972: ("Fornix (L)", "#c9c9d6"),
    973: ("Pineal gland", "#c39bd3"),
    974: ("Pituitary gland", "#ff9ff3"),
    975: ("Pituitary stalk", "#f368e0"),
    976: ("Tuberal hypothalamus", "#26c6da"),
    981: ("Medial hypothalamus (R)", "#66bb6a"),
    977: ("Medial hypothalamus (L)", "#43a047"),
    982: ("Lateral hypothalamus (R)", "#9ccc65"),
    978: ("Lateral hypothalamus (L)", "#7cb342"),
    983: ("Anterior hypothalamus (R)", "#4db6ac"),
    979: ("Anterior hypothalamus (L)", "#26a69a"),
    984: ("Posterior hypothalamus (R)", "#aed581"),
    980: ("Posterior hypothalamus (L)", "#8bc34a"),
}

# CerebNet writes FreeSurfer LUT ids (CerebNet/config/CerebNet2FreeSurfer.json): Cbm_* 601-631 and
# the cerebellar white matter as 7 / 46. Left hemisphere = cool hues, right = warm, vermis = violet.
CEREBELLUM = {
    601: ("Cerebellum lobules I-IV (L)", "#4e79a7"),
    602: ("Cerebellum lobules I-IV (R)", "#e15759"),
    603: ("Cerebellum lobule V (L)", "#59a5d8"),
    604: ("Cerebellum lobule V (R)", "#f28e2b"),
    605: ("Cerebellum lobule VI (L)", "#76b7b2"),
    607: ("Cerebellum lobule VI (R)", "#edc948"),
    608: ("Cerebellum crus I (L)", "#59a14f"),
    610: ("Cerebellum crus I (R)", "#ff9d76"),
    611: ("Cerebellum crus II (L)", "#8cd17d"),
    613: ("Cerebellum crus II (R)", "#d37295"),
    614: ("Cerebellum lobule VIIb (L)", "#2e86ab"),
    616: ("Cerebellum lobule VIIb (R)", "#f6aa1c"),
    617: ("Cerebellum lobule VIIIa (L)", "#86bcb6"),
    619: ("Cerebellum lobule VIIIa (R)", "#fabfd2"),
    620: ("Cerebellum lobule VIIIb (L)", "#499894"),
    622: ("Cerebellum lobule VIIIb (R)", "#b07aa1"),
    623: ("Cerebellum lobule IX (L)", "#a0cbe8"),
    625: ("Cerebellum lobule IX (R)", "#ffbe7d"),
    626: ("Cerebellum lobule X, flocculus (L)", "#3b5b92"),
    628: ("Cerebellum lobule X, flocculus (R)", "#c2185b"),
    606: ("Vermis VI", "#e0c3ff"),
    630: ("Vermis VII", "#c9a0f5"),
    631: ("Vermis VIII", "#b07fe6"),
    624: ("Vermis IX", "#9a63d6"),
    627: ("Vermis X (nodulus)", "#8249c2"),
    7: ("Cerebellar white matter (L)", "#d9d4c7"),
    46: ("Cerebellar white matter (R)", "#c8c2b2"),
}

# FreeSurfer CC label ids 251 (posterior) .. 255 (anterior). FastSurfer-CC fills them with its
# default "shape" Hofer-Frahm subdivision (cuts at 1/6, 1/2, 2/3, 3/4 of the CC length, front to
# back): I genu/prefrontal, II premotor, III motor, IV sensory (~isthmus), V splenium. 250 = fornix,
# which FastSurfer-CC segments as well. Order is checked against the centroids at export.
CORPUS_CALLOSUM = {
    255: ("Corpus callosum: front (genu)", "#ff7f50"),
    254: ("Corpus callosum: front of body", "#ffb347"),
    253: ("Corpus callosum: middle of body", "#f9e076"),
    252: ("Corpus callosum: back of body", "#9fd8a4"),
    251: ("Corpus callosum: back (splenium)", "#6ec6e8"),
    250: ("Fornix (FastSurfer-CC)", "#e8e8e8"),
}


def prep(aparc_path: Path, mri_dir: Path) -> None:
    import FastSurferCNN.reduce_to_aseg as rta

    aparc = nib.load(aparc_path)
    orig = nib.load(mri_dir / "orig.mgz")
    if aparc.shape != orig.shape or not np.allclose(aparc.affine, orig.affine, atol=1e-4):
        sys.exit(f"ERROR: {aparc_path} is not on the grid of {mri_dir/'orig.mgz'}")
    pred = np.asarray(aparc.dataobj).astype(np.int16)
    bm = rta.create_mask(pred, 5, 4)
    aseg = rta.reduce_to_aseg(pred)
    aseg[bm == 0] = 0
    aseg = rta.flip_wm_islands(aseg)

    def save(arr, name, dtype):
        img = nib.MGHImage(arr.astype(dtype), orig.affine, orig.header)
        img.set_data_dtype(dtype)
        nib.save(img, mri_dir / name)

    save(pred, "aparc.DKTatlas+aseg.deep.mgz", np.int16)
    save(bm, "mask.mgz", np.uint8)
    save(aseg, "aseg.auto_noCCseg.mgz", np.uint8)
    print("prep: wrote aparc copy, mask.mgz, aseg.auto_noCCseg.mgz in", mri_dir)


def hypvinn(argv: list[str]) -> None:
    import runpy

    import HypVINN.utils.img_processing_utils as ipu

    original = ipu.get_clean_mask

    def get_clean_mask(segmentation, optic=False):
        if not optic:
            return original(segmentation, optic)
        chiasm = np.isin(segmentation, [7, 8])  # HypVINN-internal ids of the chiasm halves
        seg = segmentation.copy()
        seg[chiasm] = 1  # temporarily an "optic nerve" voxel, so it links nerves and tracts
        clean, labels_cc, savemask = original(seg, optic=True)
        clean[chiasm] = 0  # the chiasm itself is kept by the non-optic branch
        return clean, labels_cc, savemask

    ipu.get_clean_mask = get_clean_mask
    script = Path(ipu.__file__).resolve().parents[1] / "run_prediction.py"
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


def _write(labels: np.ndarray, ref, table: dict, out_dir: Path, stem: str) -> None:
    present = [int(v) for v in np.unique(labels) if v != 0]
    missing = [v for v in present if v not in table]
    if missing:
        sys.exit(f"ERROR: {stem}: no name for labels {missing}")
    lab = labels.astype(np.int16)
    img = nib.Nifti1Image(lab, ref.affine)
    img.set_qform(ref.affine, code=1)
    img.set_sform(ref.affine, code=1)
    img.set_data_dtype(np.int16)
    nib.save(img, out_dir / f"{stem}.nii.gz")
    vox = float(np.prod(ref.header.get_zooms()[:3]))
    meta = {str(v): {"name": table[v][0], "color": table[v][1]} for v in sorted(present)}
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{stem}: {len(present)} labels")
    for v in sorted(present):
        print(f"  {v:4d} {table[v][0]:40s} {np.count_nonzero(lab == v) * vox:9.1f} mm3")


def _check_side(labels: np.ndarray, affine, table: dict, stem: str) -> None:
    """Sanity check: every '(L)' label must lie at more negative RAS x than its '(R)' partner."""
    xs = {}
    for v in np.unique(labels):
        if v == 0 or int(v) not in table:
            continue
        ijk = np.argwhere(labels == v).mean(0)
        xs[table[int(v)][0]] = (affine @ np.r_[ijk, 1])[0]
    for name, x in xs.items():
        if "(L" in name:
            partner = name.replace("(L", "(R")
            if partner in xs and not x < xs[partner]:
                sys.exit(f"ERROR: {stem}: '{name}' x={x:.1f} is not left of '{partner}' x={xs[partner]:.1f}")


def _cerebellum_on_aparc_grid(cer, aparc) -> np.ndarray:
    """CerebNet 1 mm labels -> 0.5 mm aparc grid, restricted to (and filling) aseg cerebellum 7/8/46/47.

    Each label's one-hot map is trilinearly interpolated (smoother than nearest neighbour), the
    most likely label wins; aseg-cerebellum voxels no CerebNet label reaches take the nearest one.
    """
    from scipy import ndimage as ndi

    A = np.asarray(aparc.dataobj)
    mask = np.isin(A, [7, 8, 46, 47])
    idx = np.argwhere(mask)
    lo, hi = idx.min(0) - 2, idx.max(0) + 3
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    grid = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1)
    vox2vox = np.linalg.inv(cer.affine) @ aparc.affine
    coords = (grid.reshape(-1, 3) @ vox2vox[:3, :3].T + vox2vox[:3, 3]).T
    C = np.asarray(cer.dataobj).astype(np.int16)
    ids = [int(v) for v in np.unique(C) if v != 0]
    best = np.zeros(coords.shape[1], np.float32)
    out = np.zeros(coords.shape[1], np.int16)
    for v in ids:
        prob = ndi.map_coordinates((C == v).astype(np.float32), coords, order=1, mode="constant")
        better = prob > best
        best[better], out[better] = prob[better], v
    out = out.reshape(sub.shape)
    out[~sub] = 0
    hole = sub & (best.reshape(sub.shape) < 0.05)
    out[hole] = 0
    if hole.any():
        _, near = ndi.distance_transform_edt(out == 0, return_indices=True)
        out[hole] = out[tuple(n[hole] for n in near)]
    full = np.zeros(A.shape, np.int16)
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = out
    print(f"cerebellum: {hole.sum() / 8:.0f} mm3 of the aseg cerebellum filled from the nearest CerebNet label")
    return full


def export(subject_dir: Path, out_dir: Path) -> None:
    mri = subject_dir / "mri"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- HypVINN: written on the grid of its T1 input (orig_nu.mgz = conformed grid) ---------
    hyp = nib.load(mri / "hypothalamus.HypVINN.nii.gz")
    hlab = np.asarray(hyp.dataobj).astype(np.int16)
    _check_side(hlab, hyp.affine, HYPOTHALAMUS, "hypothalamus_labels")
    _write(hlab, hyp, HYPOTHALAMUS, out_dir, "hypothalamus_labels")

    # --- CerebNet ---------------------------------------------------------------------------------
    # CerebNet always works on a 1 mm grid. Lift it onto the 0.5 mm grid and fit it to the 0.5 mm
    # whole-cerebellum outline of the main segmentation, so the lobules tile exactly the cerebellum
    # that is already in the model.
    cer = nib.load(mri / "cerebellum.CerebNet.nii.gz")
    clab = _cerebellum_on_aparc_grid(cer, nib.load(mri / "aparc.DKTatlas+aseg.deep.mgz"))
    ref = nib.load(mri / "orig.mgz")
    _check_side(clab, ref.affine, CEREBELLUM, "cerebellum_lobules")
    _write(clab, ref, CEREBELLUM, out_dir, "cerebellum_lobules")

    # --- Corpus callosum: subdivisions painted into the asegdkt by paint_cc_into_pred.py --------
    wcc = nib.load(mri / "aparc.DKTatlas+aseg.deep.withCC.mgz")
    w = np.asarray(wcc.dataobj)
    cc = np.where((w >= 251) & (w <= 255), w, 0).astype(np.int16)
    fx_path = mri / "callosum.CC.orig.mgz"
    if fx_path.exists():
        fx = np.asarray(nib.load(fx_path).dataobj)
        cc[(fx == 250) & (cc == 0)] = 250
    ys = [(wcc.affine @ np.r_[np.argwhere(cc == v).mean(0), 1])[1] for v in (251, 252, 253, 254, 255)]
    if not all(a < b for a, b in zip(ys, ys[1:])):
        sys.exit(f"ERROR: corpus_callosum: 251..255 are not ordered back to front (RAS y {np.round(ys, 1)})")
    _write(cc, wcc, CORPUS_CALLOSUM, out_dir, "corpus_callosum")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "prep":
        prep(Path(sys.argv[2]), Path(sys.argv[3]))
    elif len(sys.argv) > 2 and sys.argv[1] == "hypvinn":
        hypvinn(sys.argv[2:])
    elif len(sys.argv) == 4 and sys.argv[1] == "export":
        export(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.exit(__doc__)
