"""Extract cerebral veins (dark tubes on SWI) with a Frangi vesselness filter.

Usage:
  python seg_veins.py <swi.nii.gz> <out_dir> <brain_mask.nii.gz> [high_pct low_pct min_mm3 max_density]
                      [--phase <swi_phase.nii.gz>|auto|none] [--charm <charm_tissues.nii.gz>|auto|none]
  auto (default): the phase image next to the SWI (*_Pha*.nii*), charm_tissues.nii.gz next to the brain mask.

Writes veins.nii.gz on a 0.7 mm isotropic grid (with a correct affine):
  1 = veins found on the SWI, 2 = large dural sinuses (SimNIBS charm "blood" label that is also dark on
  the SWI). Viewers that don't care can use `> 0`.

Steps (see README "Veins"):
  1. Normalise the SWI by a 6 mm local mean; Frangi vesselness for dark tubes (cached: vesselness.npy).
  2. Phase reliability: near air (sinuses, mastoids) the SWI phase is wrapped all over the place and the
     magnitude has dark streaks that look like veins. The local density of phase wraps inside the brain
     marks those zones; vesselness there is down-weighted (needs ~3x the evidence).
  3. Hysteresis thresholding (keeps veins continuous across the 1.7 mm slices) + local-density filter.
  4. Shape filter: in risky zones, keep only clearly elongated components.
  5. Gap bridging: skeleton end points are joined to a nearby vein straight ahead (cone search) when the
     vesselness along the bridge supports it.
  6. Dural sinuses from charm label 9, kept only where the SWI is dark (drops arteries / neck vessels).
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import apply_hysteresis_threshold, frangi
from skimage.morphology import skeletonize

args = sys.argv[1:]
opts = {}
for flag in ("--phase", "--charm"):
    if flag in args:
        i = args.index(flag)
        opts[flag] = args[i + 1]
        del args[i:i + 2]
src, out, bm_path = args[0], Path(args[1]), args[2]
hi_pct, lo_pct, min_mm3, max_density = (float(x) for x in (args[3:7] if len(args) > 6 else (99.3, 97.0, 40, 0.12)))
out.mkdir(parents=True, exist_ok=True)

img = nib.as_closest_canonical(nib.load(src))
zooms = np.array(img.header.get_zooms()[:3], float)
iso = zooms.min()
fac = zooms / iso
a = ndi.zoom(img.get_fdata(dtype=np.float32), fac, order=1)
aff = img.affine.copy()
aff[:3, :3] = img.affine[:3, :3] @ np.diag(1 / fac)
grid = (a.shape, aff)

# Brain mask (SynthStrip, T1 space) moved onto this grid; shrunk a little so the
# brain/CSF edge itself isn't detected as a "vessel". Skull and scalp are noisy on SWI.
bm = nib.load(bm_path)
mask = resample_from_to(bm, grid, order=0).get_fdata() > 0.5
mask = ndi.distance_transform_edt(mask, sampling=iso) > 1.5

# Flatten slow intensity variations, then look for dark tubes 1–3.6 mm wide.
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
hi, lo = np.percentile(v[mask], [hi_pct, lo_pct])

# ---- 2. phase reliability (fraction of in-plane phase wraps within ~4 mm, inside the brain)
phase = opts.get("--phase", "auto")
if phase == "auto":
    cands = sorted(Path(src).parent.glob("*_Pha*.nii*"))
    phase = str(cands[0]) if cands else "none"
risk = np.zeros(a.shape, np.float32)
if phase != "none":
    p = nib.as_closest_canonical(nib.load(phase))
    assert p.shape == img.shape and np.allclose(p.affine, img.affine, atol=1e-3), "phase must match the SWI grid"
    p = p.get_fdata(dtype=np.float32)
    span = float(p.max() - p.min())  # Siemens: -4096..4095 = -pi..pi
    wrap = np.zeros(p.shape, bool)
    for ax in (0, 1):  # in-plane only (slices are thick)
        d = np.abs(np.diff(p, axis=ax)) > span / 2
        lo_sl, hi_sl = [slice(None)] * 3, [slice(None)] * 3
        lo_sl[ax], hi_sl[ax] = slice(0, -1), slice(1, None)
        wrap[tuple(lo_sl)] |= d
        wrap[tuple(hi_sl)] |= d
    wrap = ndi.zoom(wrap.astype(np.float32), fac, order=1)
    s = 4 / iso
    wd = ndi.gaussian_filter(wrap * mask, s) / np.maximum(ndi.gaussian_filter(mask.astype(np.float32), s), 0.05)
    risk = np.clip((wd - 0.02) / 0.06, 0, 1).astype(np.float32)
    print("phase:", Path(phase).name, "risky brain fraction", round(float((risk[mask] > 0.5).mean()), 3))

# ---- 3. hysteresis + density
# Confident vessel voxels seed growth along weaker vesselness, which keeps veins continuous
# across the thick (1.7 mm) SWI slices. In risky zones the evidence must be ~3x stronger.
vw = np.where(norm < 0.9, v / (1 + 2 * risk), 0)
veins = apply_hysteresis_threshold(vw, lo, hi)
# Real veins are sparse threads; susceptibility artefacts show up as a dense mesh.
density = ndi.gaussian_filter(veins.astype(np.float32), 3 / iso)
veins &= density < max_density

# ---- 4. components: size + shape
lab, n = ndi.label(veins, structure=np.ones((3, 3, 3)))
keep = np.zeros(n + 1, bool)
for k, o in enumerate(ndi.find_objects(lab), 1):
    m = lab[o] == k
    if m.sum() * iso ** 3 < 8:  # specks; the real size filter runs after gap bridging
        continue
    if risk[o][m].mean() > 0.3:
        ev = np.sort(np.maximum(np.linalg.eigvalsh(np.cov((np.argwhere(m) * iso).T)), 1e-4))
        if np.sqrt(ev[2] / ev[1]) < 3:  # not tube-like
            continue
    keep[k] = True
veins = keep[lab]
print("pieces kept", int(keep.sum()), "of", n, "volume mm3", round(float(veins.sum() * iso ** 3)))

# ---- 5. gap bridging along the vessel direction
def bridge(veins, support, max_mm=5.0, cos_min=0.8, support_min=0.55):
    lab, _ = ndi.label(veins, structure=np.ones((3, 3, 3)))
    sk = skeletonize(veins)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant") - 1
    ends = np.argwhere(sk & (nb == 1))
    skp = np.argwhere(sk)
    sk_tree = cKDTree(skp)
    pts = np.argwhere(veins)
    tree = cKDTree(pts)
    r = max_mm / iso
    added = np.zeros_like(veins)
    nb_made = 0
    for e in ends:
        # direction: from the local skeleton centroid (≤3 mm back) towards the end point
        near = skp[sk_tree.query_ball_point(e, 3 / iso)]
        near = near[lab[tuple(near.T)] == lab[tuple(e)]]
        if len(near) < 3:
            continue
        d = e - near.mean(0)
        if np.linalg.norm(d) < 1e-6:
            continue
        d /= np.linalg.norm(d)
        best = None
        for j in tree.query_ball_point(e, r):
            q = pts[j]
            if lab[tuple(q)] == lab[tuple(e)]:
                continue
            w = q - e
            L = np.linalg.norm(w)
            if L < 1.5 or w @ d / L < cos_min:
                continue
            if best is None or L < best[0]:
                best = (L, q)
        if best is None:
            continue
        L, q = best
        t = np.linspace(0, 1, int(np.ceil(L * 2)) + 1)[:, None]
        line = np.round(e + t * (q - e)).astype(int)
        if support[tuple(line.T)].mean() < support_min:
            continue
        added[tuple(line.T)] = True
        nb_made += 1
    added = ndi.binary_dilation(added) & ~veins  # ~2 voxel thick bridge
    return veins | added, nb_made


support = ndi.maximum_filter(vw > np.percentile(v[mask], 90), size=3)
veins, nb_made = bridge(veins, support)
lab, n = ndi.label(veins, structure=np.ones((3, 3, 3)))
sizes = np.bincount(lab.ravel()) * iso ** 3
sizes[0] = 0
veins = sizes[lab] >= min_mm3  # drop specks
print("bridges", nb_made, "components", int((sizes >= min_mm3).sum()), "volume mm3", round(float(veins.sum() * iso ** 3)))

# ---- 6. dural sinuses from charm (label 9 = blood), confirmed dark on the SWI
result = veins.astype(np.uint8)
charm = opts.get("--charm", "auto")
if charm == "auto":
    charm = str(Path(bm_path).parent / "charm_tissues.nii.gz")
    charm = charm if Path(charm).exists() else "none"
if charm != "none":
    ch = nib.load(charm)
    C = resample_from_to(ch, grid, order=0).get_fdata().astype(np.uint8)
    head = C > 0
    s = 8 / iso
    local = ndi.gaussian_filter(np.where(head, a, 0), s) / np.maximum(ndi.gaussian_filter(head.astype(np.float32), s), 1e-3)
    blood = C == 9
    dark = blood & (a < 0.8 * local)
    # keep large dark pieces; then take the whole charm tube around them (fills speckle)
    dl, _ = ndi.label(ndi.binary_closing(dark, np.ones((3, 3, 3))) & blood, structure=np.ones((3, 3, 3)))
    sz = np.bincount(dl.ravel()) * iso ** 3
    sz[0] = 0
    core = sz[dl] >= 300
    sinus = blood & (ndi.distance_transform_edt(~core, sampling=iso) <= 1.5)
    # nothing below the brain (neck veins/arteries)
    bz = np.where(mask.any((0, 1)))[0]
    sinus[:, :, :max(bz.min() - int(5 / iso), 0)] = False
    # The carotid siphons / cavernous sinus are in the label too and their fast flow is also dark.
    # They sit off the midline in front of the brain's centre; the dural sinuses we want
    # (superior sagittal, straight, transverse, sigmoid) are on the midline or behind it.
    xyz = np.argwhere(sinus)
    w = (aff @ np.c_[xyz, np.ones(len(xyz))].T).T[:, :3]
    cen = aff @ np.r_[np.argwhere(mask)[::97].mean(0), 1]
    off = (np.abs(w[:, 0] - cen[0]) > 8) & (w[:, 1] > cen[1] + 12)
    sinus[tuple(xyz[off].T)] = False
    sl, _ = ndi.label(sinus, structure=np.ones((3, 3, 3)))
    sz = np.bincount(sl.ravel()) * iso ** 3
    sz[0] = 0
    sinus = sz[sl] >= 500
    result[sinus & ~veins] = 2
    print("sinuses mm3", round(float((result == 2).sum() * iso ** 3)))

nib.save(nib.Nifti1Image(result, aff), out / "veins.nii.gz")
