"""Compare two data/derived folders, e.g. a fresh pipeline run against the committed one.

Usage: python compare_derived.py <reference_derived_dir> <new_derived_dir>

Label volumes: per-label Dice (mean, worst), plus voxel agreement. Intensity volumes: identical or max difference.
Surfaces: per-vertex distance when both have the same vertex count, and agreement of per-vertex arrays.
Deep-learning tools on a GPU and atlas fitting are not bit-exact between runs, so small differences are expected;
the point is to catch steps that break or drift.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ref, new = Path(sys.argv[1]), Path(sys.argv[2])


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * np.logical_and(a, b).sum() / s


def labels(path_a, path_b):
    A, B = nib.load(path_a), nib.load(path_b)
    a, b = np.asarray(A.dataobj), np.asarray(B.dataobj)
    if a.shape != b.shape or not np.allclose(A.affine, B.affine, atol=1e-3):
        return "different grid"
    labs = sorted((set(np.unique(a)) | set(np.unique(b))) - {0})
    ds = {int(l): dice(a == l, b == l) for l in labs}
    worst = min(ds, key=ds.get) if ds else None
    fg = (a > 0) | (b > 0)
    agree = (a[fg] == b[fg]).mean() if fg.any() else 1.0
    return (f"{len(labs):3d} labels  mean Dice {np.mean(list(ds.values())):.3f}  worst {ds[worst]:.3f} (label {worst})"
            f"  voxel agreement {agree:.3f}") if ds else "empty"


rows = []
for p in sorted(ref.glob("*.nii.gz")) + sorted(ref.glob("*.mgz")):
    q = new / p.name
    if not q.exists():
        rows.append((p.name, "MISSING in new run")); continue
    if p.name in ("t1.nii.gz",):
        a, b = np.asarray(nib.load(p).dataobj), np.asarray(nib.load(q).dataobj)
        rows.append((p.name, "identical" if a.shape == b.shape and np.array_equal(a, b) else f"max diff {np.abs(a.astype(float) - b).max():.3g}"))
    else:
        rows.append((p.name, labels(p, q)))

for hemi in ("lh", "rh"):
    for kind in ("white", "pial", "inflated"):
        p, q = ref / "surfaces" / f"{hemi}.{kind}.gii", new / "surfaces" / f"{hemi}.{kind}.gii"
        if not (p.exists() and q.exists()):
            rows.append((p.name, "missing")); continue
        va, vb = nib.load(p).darrays[0].data, nib.load(q).darrays[0].data
        if len(va) == len(vb):
            d = np.linalg.norm(va - vb, axis=1)
            rows.append((f"surfaces/{p.name}", f"{len(va)} verts  vertex distance median {np.median(d):.2f} mm, 95% {np.percentile(d, 95):.2f} mm"))
        else:
            rows.append((f"surfaces/{p.name}", f"vertex counts differ ({len(va)} vs {len(vb)})"))
    for kind in ("dkt", "sulc"):
        p, q = ref / "surfaces" / f"{hemi}.{kind}.npy", new / "surfaces" / f"{hemi}.{kind}.npy"
        if p.exists() and q.exists():
            a, b = np.load(p), np.load(q)
            if a.shape != b.shape:
                rows.append((f"surfaces/{p.name}", "different length")); continue
            rows.append((f"surfaces/{p.name}", f"same label {(a == b).mean():.3f}" if kind == "dkt"
                         else f"correlation {np.corrcoef(a, b)[0, 1]:.3f}"))

w = max(len(r[0]) for r in rows)
for name, msg in rows:
    print(f"{name:{w}s}  {msg}")
