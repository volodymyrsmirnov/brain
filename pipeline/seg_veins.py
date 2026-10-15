"""Extract cerebral veins (dark tubes on SWI) with a Frangi vesselness filter.

Usage: python seg_veins.py <swi.nii.gz> <out_dir> <brain_mask.nii.gz> [high_pct low_pct min_mm3 max_density]
Writes veins.nii.gz on a 0.7 mm isotropic grid (with a correct affine).
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, frangi

src, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
img = nib.as_closest_canonical(nib.load(src))
zooms = np.array(img.header.get_zooms()[:3], float)
iso = zooms.min()
fac = zooms / iso
a = ndi.zoom(img.get_fdata(dtype=np.float32), fac, order=1)
aff = img.affine.copy()
aff[:3, :3] = img.affine[:3, :3] @ np.diag(1 / fac)

# Brain mask (SynthStrip, T1 space) moved onto this grid; shrunk a little so the
# brain/CSF edge itself isn't detected as a "vessel". Skull and scalp are noisy on SWI.
bm = nib.load(sys.argv[3])
mask = resample_from_to(bm, (a.shape, aff), order=0).get_fdata() > 0.5
mask = ndi.distance_transform_edt(mask, sampling=iso) > 1.5

# Flatten slow intensity variations, then look for dark tubes 0.7–3 mm wide.
bg = ndi.gaussian_filter(a, 6 / iso)
norm = np.where(mask, a / np.maximum(bg, 1e-3), 1).astype(np.float32)
cache = out / "vesselness.npy"
if cache.exists():
    v = np.load(cache)
else:
    v = frangi(norm, sigmas=np.array([0.5, 0.8, 1.2, 1.8]) / iso, black_ridges=True,
               alpha=0.5, beta=0.5, gamma=None).astype(np.float32)
    np.save(cache, v)
v[~mask] = 0

hi_pct, lo_pct, min_mm3 = (float(x) for x in (sys.argv[4:7] if len(sys.argv) > 6 else (99.3, 97.0, 40)))
# Hysteresis: confident vessel voxels seed growth along weaker vesselness, which
# keeps veins continuous across the thick (1.7 mm) SWI slices.
hi, lo = np.percentile(v[mask], [hi_pct, lo_pct])
veins = apply_hysteresis_threshold(np.where(norm < 0.9, v, 0), lo, hi)
# Real veins are sparse threads; susceptibility artefacts near the skull base show up
# as a dense mesh. Drop voxels where the local vessel fraction is too high.
density = ndi.gaussian_filter(veins.astype(np.float32), 3 / iso)
veins &= density < float(sys.argv[7]) if len(sys.argv) > 7 else density < 0.12
lab, n = ndi.label(veins, structure=np.ones((3, 3, 3)))
sizes = ndi.sum(veins, lab, range(1, n + 1)) * iso ** 3
veins = np.isin(lab, 1 + np.where(sizes > min_mm3)[0])  # drop specks
print("components kept", int((sizes > min_mm3).sum()), "of", n, "volume mm3", round(float(veins.sum() * iso ** 3)))

nib.save(nib.Nifti1Image(veins.astype(np.uint8), aff), out / "veins.nii.gz")
