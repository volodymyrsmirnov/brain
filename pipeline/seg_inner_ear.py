"""Segment both inner-ear labyrinths (fluid-filled, bright on T2 SPACE).

Usage: python seg_inner_ear.py <t2_space.nii.gz> <out_dir>
ROIs/cut planes were picked by inspecting thin-slab MIPs of this scan
(canonical RAS voxel indices of the 0.32 x 0.32 x 0.3 mm grid).
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage import filters

src, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
img = nib.as_closest_canonical(nib.load(src))
a = img.get_fdata(dtype=np.float32)

# ROI per side (x, y, z ranges) and which x-face is medial (where the internal
# auditory canal, IAC, enters the ROI from the brain side).
sides = {
    "ear_x_low":  dict(x=(85, 180), y=(185, 265), z=(60, 116), medial="high"),
    "ear_x_high": dict(x=(330, 432), y=(180, 266), z=(68, 126), medial="low"),
}
vox = float(np.mean(img.header.get_zooms()[:3]))

full = np.zeros(a.shape, np.uint8)
for i, (name, s) in enumerate(sides.items(), 1):
    (x0, x1), (y0, y1), (z0, z1) = s["x"], s["y"], s["z"]
    roi = ndi.gaussian_filter(a[x0:x1, y0:y1, z0:z1], 0.6)
    thr = filters.threshold_otsu(roi)
    m = roi > thr

    # IAC/CSF: thick fluid reaching the medial face. Labyrinth canals are ~1 mm wide,
    # so a 1.1 mm core radius never survives inside them.
    edt = ndi.distance_transform_edt(m, sampling=vox)
    core_lab, _ = ndi.label(edt > 1.1)
    face = core_lab[-1] if s["medial"] == "high" else core_lab[0]
    iac_core = np.isin(core_lab, np.unique(face[face > 0]))
    iac = ndi.distance_transform_edt(~iac_core, sampling=vox) <= 1.6
    m &= ~iac

    # Drop anything touching the ROI border (truncated CSF / IAC remnants).
    lab, n = ndi.label(m)
    border = np.zeros_like(m)
    border[[0, -1]] = border[:, [0, -1]] = border[:, :, [0, -1]] = True
    touching = np.unique(lab[border & m])
    sizes = ndi.sum(m, lab, range(1, n + 1))
    ok = [k + 1 for k in range(n) if (k + 1) not in touching]
    big = max((sizes[k - 1] for k in ok), default=0)
    main = lab == max(ok, key=lambda k: sizes[k - 1])
    near = ndi.distance_transform_edt(~main, sampling=vox) <= 1.5
    keep = main | np.isin(lab, [k for k in ok if sizes[k - 1] > 0.04 * big and near[lab == k].any()])
    full[x0:x1, y0:y1, z0:z1][keep] = i
    vol_mm3 = keep.sum() * np.prod(img.header.get_zooms()[:3])
    print(name, "thr", round(float(thr), 1), "kept", len(ok), "volume mm3", round(float(vol_mm3), 1))

nib.save(nib.Nifti1Image(full, img.affine), out / "inner_ear_labels.nii.gz")
