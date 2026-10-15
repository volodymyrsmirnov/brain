"""Segment both inner-ear labyrinths (fluid-filled, bright on T2 SPACE).

Usage: python seg_inner_ear.py <t2_space.nii.gz> <out_dir> [--aseg <aparc.DKTatlas+aseg.deep.mgz>]
                               [--roi-left x0:x1,y0:y1,z0:z1] [--roi-right x0:x1,y0:y1,z0:z1]

Writes inner_ear_labels.nii.gz on the T2 grid: 1 = left labyrinth, 2 = right labyrinth.

Finding the ears (no hand-picked coordinates):
  1. The brainstem centre (FastSurfer label 16, from --aseg; default data/derived/...) gives a search box
     22–62 mm to each side of it. Without an aseg, each half of the scan is searched.
  2. In each box: Otsu threshold, remove thick fluid (core radius > 1.1 mm: CSF cisterns and the internal
     auditory canal, IAC), and score the remaining bright pieces of labyrinth size (100–1500 mm³) by how
     dark (bony) their 0.5–1.5 mm surroundings are. The labyrinth is the only bright, curly, canal-sized
     structure wrapped in bone.
  3. A 32 × 28 × 20 mm ROI centred on the winner is then segmented as before.
  --roi-left/--roi-right override step 1–2 with canonical (RAS) voxel index ranges, e.g. the old hand-picked
  ROIs of this scan: --roi-left 85:180,185:265,60:116 --roi-right 330:432,180:266,68:126.

Segmenting one ROI:
  - Otsu threshold; the IAC (thick fluid reaching the medial ROI face) is removed with a 1.6 mm margin.
  - Keep the piece found in step 2 plus nearby (≤ 1.5 mm) pieces that don't touch the ROI border.
  - The IAC touches the cochlea and vestibule. Its thin end and the fluid between its nerves survive the
    thickness filter and stay attached (the old "stub" on the right, flat "fins" on both sides). The IAC is
    modelled as a cylinder (axis = principal direction of its thick core near the fundus, the core voxel
    closest to the vestibule) and everything within 5 mm of the axis and medial of fundus + 1 mm is cut.
    The labyrinth lies lateral to the fundus, so cochlea, vestibule and canals are untouched.
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage import filters

args = sys.argv[1:]
opts = {}
for flag in ("--aseg", "--roi-left", "--roi-right"):
    if flag in args:
        i = args.index(flag)
        opts[flag] = args[i + 1]
        del args[i:i + 2]
src, out = args[0], Path(args[1])
out.mkdir(parents=True, exist_ok=True)
img = nib.as_closest_canonical(nib.load(src))
a = img.get_fdata(dtype=np.float32)
zooms = np.array(img.header.get_zooms()[:3], float)
mm = lambda x: np.round(np.asarray(x, float) / zooms).astype(int)  # mm -> voxels per axis


def parse_roi(s):
    return tuple(slice(*map(int, r.split(":"))) for r in s.split(","))


def clip_box(lo, hi):
    lo = np.clip(np.asarray(lo, int), 0, a.shape)
    hi = np.clip(np.asarray(hi, int), 0, a.shape)
    return tuple(slice(l, h) for l, h in zip(lo, hi))


def thick_fluid(m, big_mm3=15.0):
    """Thick (core radius > 1.1 mm) fluid pieces bigger than big_mm3: CSF / IAC, never labyrinth."""
    edt = ndi.distance_transform_edt(m, sampling=zooms)
    cl, n = ndi.label(edt > 1.1)
    cs = np.bincount(cl.ravel()) * zooms.prod()
    cs[0] = 0
    return cs[cl] > big_mm3


def find_labyrinth(box):
    """Return (score, mask in full-volume coords) of the best labyrinth candidate in box."""
    roi = ndi.gaussian_filter(a[box], 0.6)
    thr = filters.threshold_otsu(roi)
    m = roi > thr
    csf = ndi.distance_transform_edt(~thick_fluid(m), sampling=zooms) <= 1.6
    r = m & ~csf
    lab, n = ndi.label(r)
    sizes = np.bincount(lab.ravel()) * zooms.prod()
    best = (0.0, None)
    for k, o in enumerate(ndi.find_objects(lab), 1):
        if o is None or not 100 <= sizes[k] <= 1500:
            continue
        pad = tuple(slice(max(s.start - 6, 0), s.stop + 6) for s in o)
        comp = lab[pad] == k
        d = ndi.distance_transform_edt(~comp, sampling=zooms)
        shell = (d > 0.5) & (d <= 1.5)
        dark = float((roi[pad][shell] < 0.5 * thr).mean())
        if dark > best[0]:
            full = np.zeros(a.shape, bool)
            full[box][pad][comp] = True
            best = (dark, full)
    return best


# ------------------------------------------------------------------ locate --
sides = {}  # name -> (label, box, medial face 'high'/'low', candidate mask or None)
aseg = opts.get("--aseg", str(Path(__file__).resolve().parents[1] / "data/derived/aparc.DKTatlas+aseg.deep.mgz"))
centre = None
if Path(aseg).exists():
    sg = nib.load(aseg)
    S = np.asarray(sg.dataobj)
    w = sg.affine @ np.r_[np.argwhere(S == 16).mean(0), 1]  # brainstem centre, world mm
    centre = (np.linalg.inv(img.affine) @ w)[:3]
    print("brainstem centre (T2 voxels)", centre.round(1))
mid_x = centre[0] if centre is not None else a.shape[0] / 2

for name, lab_id, sgn in (("left", 1, -1), ("right", 2, 1)):
    medial = "high" if sgn < 0 else "low"  # canonical RAS: x grows towards the patient's right
    key = f"--roi-{name}"
    if key in opts:
        sides[name] = (lab_id, parse_roi(opts[key]), medial, None)
        continue
    if centre is not None:
        x0, x1 = sorted(centre[0] + sgn * mm([22, 62, 0])[:2])
        lo = [x0, centre[1] + mm([0, -22, 0])[1], centre[2] + mm([0, 0, -22])[2]]
        hi = [x1, centre[1] + mm([0, 25, 0])[1], centre[2] + mm([0, 0, 18])[2]]
        box = clip_box(np.floor(lo), np.ceil(hi))
    else:
        half = int(mid_x)
        box = clip_box([0 if sgn < 0 else half, 0, 0], [half if sgn < 0 else a.shape[0], a.shape[1], a.shape[2]])
    score, cand = find_labyrinth(box)
    if cand is None or score < 0.6:
        sys.exit(f"{name} inner ear not found (best score {score:.2f}); pass {key} x0:x1,y0:y1,z0:z1")
    idx = np.argwhere(cand)
    c = (idx.min(0) + idx.max(0)) / 2
    half = mm([16, 14, 10])
    roi = clip_box(np.round(c - half), np.round(c + half))
    print(f"{name}: found, bone-shell score {score:.2f}, ROI {key} "
          + ",".join(f"{s.start}:{s.stop}" for s in roi))
    sides[name] = (lab_id, roi, medial, cand)

# ---------------------------------------------------------------- segment --
full = np.zeros(a.shape, np.uint8)
IAC_R, IAC_END = 5.0, 1.0  # mm (tuned on previews: 4.0 left a flap of IAC fluid, 3.0 kept the stub)
for name, (lab_id, box, medial, cand) in sides.items():
    roi = ndi.gaussian_filter(a[box], 0.6)
    thr = filters.threshold_otsu(roi)
    m = roi > thr

    # IAC/CSF: thick fluid reaching the medial face. Labyrinth canals are ~1 mm wide,
    # so a 1.1 mm core radius never survives inside them.
    edt = ndi.distance_transform_edt(m, sampling=zooms)
    core_lab, _ = ndi.label(edt > 1.1)
    face = core_lab[-1] if medial == "high" else core_lab[0]
    iac_core = np.isin(core_lab, np.unique(face[face > 0]))
    m &= ndi.distance_transform_edt(~iac_core, sampling=zooms) > 1.6

    def pick(m, ref):
        """Main piece (most overlap with ref, else largest not touching the border) + near pieces."""
        lab, n = ndi.label(m)
        border = np.zeros_like(m)
        border[[0, -1]] = border[:, [0, -1]] = border[:, :, [0, -1]] = True
        touching = set(np.unique(lab[border & m]))
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        if ref is not None and ref.any():
            ov = np.bincount(lab[ref], minlength=n + 1)
            ov[0] = 0
            main_k = int(ov.argmax())
        else:
            main_k = max((k for k in range(1, n + 1) if k not in touching), key=lambda k: sizes[k])
        main = lab == main_k
        near = ndi.distance_transform_edt(~main, sampling=zooms) <= 1.5
        extra = [k for k in range(1, n + 1) if k != main_k and k not in touching
                 and sizes[k] > 0.04 * sizes[main_k] and near[lab == k].any()]
        return main | np.isin(lab, extra), 1 + len(extra)

    keep, _ = pick(m, cand[box] if cand is not None else None)

    # Stub/fin: the thin end of the IAC fluid (and a rim along its wall) stays attached to the
    # cochlea and vestibule. Model the IAC as a cylinder: axis = principal direction of the thick
    # IAC core within 8 mm of the fundus (the core voxel closest to the vestibule, i.e. the thickest
    # part of the labyrinth); remove everything within IAC_R mm of the axis that is medial of the
    # fundus + IAC_END mm.
    if iac_core.any():
        cx = np.argwhere(iac_core) * zooms
        thick = ndi.distance_transform_edt(keep, sampling=zooms)
        vest = np.array(np.unravel_index(thick.argmax(), thick.shape)) * zooms
        f = cx[((cx - vest) ** 2).sum(1).argmin()]
        loc = cx[((cx - f) ** 2).sum(1) <= 8 ** 2]
        u = np.linalg.eigh(np.cov(loc.T))[1][:, -1]
        u *= np.sign(u @ (f - loc.mean(0))) or 1  # points laterally, towards the labyrinth
        P = np.stack(np.meshgrid(*[np.arange(n) * z for n, z in zip(m.shape, zooms)], indexing="ij"), -1) - f
        t = P @ u
        r = np.linalg.norm(P - t[..., None] * u, axis=-1)
        cut = (t < IAC_END) & (r < IAC_R)
        print(name, "IAC axis", u.round(2), "cut voxels", int((keep & cut).sum()))
        keep, _ = pick(keep & ~cut, keep & ~cut & (thick > 0))
    full[box][keep] = lab_id
    print(name, "thr", round(float(thr), 1),
          "volume mm3", round(float(keep.sum() * zooms.prod()), 1))

nib.save(nib.Nifti1Image(full, img.affine), out / "inner_ear_labels.nii.gz")
