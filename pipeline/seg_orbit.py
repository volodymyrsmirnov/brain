"""Segment the optic nerves, optic chiasm and extraocular muscles behind each eye.

Usage: python seg_orbit.py <t1.nii.gz> <charm_tissues.nii.gz> <aparc.DKTatlas+aseg.deep.mgz> <out_dir> [--per-muscle]
Writes <out_dir>/orbit_labels.nii.gz (uint8 on the charm grid) and orbit_labels.json
({label: {name, color}}).

On T1 the orbit is filled with bright fat, and the optic nerve and the eye muscles are
dark bands running through it from the eyeball to the apex of the orbit.
  * Prior: charm models the optic nerves and chiasm as "white matter" outside the brain
    (its atlas has Optic-Nerve / Optic-Chiasm), which gives the rough path, but it stops
    short of the eyeball and is a few mm off in places.
  * Nerve centre line: cheapest path (skimage MCP) from the back of the eyeball to the
    chiasm. Steps are cheap near the charm prior and, inside the orbit, through dark T1;
    bone, eyeball, air and brain are forbidden.
  * Optic canal: where the path is closest to bone. In front of it (inside the orbit) the
    nerve is the dark tissue within 2.6 mm of the centre line; behind it (canal and
    cistern) the part of the charm prior within 1.8 mm of it; always a 1.2 mm core tube.
  * Globe: charm's eyeball misses the dark sclera, so each globe's radius is re-measured
    from the radial T1 profile (dark inside, bright fat outside).
  * Orbit: soft tissue in a 27-degree cone from the optic canal to the globe, connected to
    the nerve. Muscles are the dark tissue in it (smoothed T1 < min(Otsu, 0.58 x fat level)),
    > 1.5 mm from the nerve, > 0.75 mm from bone; next to the globe only a thin shell
    (tendons), then opened and cleaned of specks < 40 mm3 (vessels).
  * --per-muscle: split each side's muscles by their angle around the orbit axis
    (superior rectus + levator, lateral, inferior, medial, superior oblique in the
    superomedial corner); each muscle is closed by 1 mm and cleaned of specks.
Labels: 1/2 optic nerve L/R, 3 chiasm; 11-15 left muscles, 21-25 right muscles (or 11/21
"Eye muscles (L/R)" without --per-muscle). L/R = patient's left/right (RAS x < 0 is left).
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage import filters
from skimage.graph import MCP_Geometric

args = [a for a in sys.argv[1:] if not a.startswith("--")]
per_muscle = "--per-muscle" in sys.argv
t1_path, charm_path, aseg_path, out = args[0], args[1], args[2], Path(args[3])
out.mkdir(parents=True, exist_ok=True)

ct = nib.load(charm_path)
Lfull = np.asarray(ct.dataobj).squeeze().astype(np.uint8)
A0 = ct.affine
vox = np.sqrt((A0[:3, :3] ** 2).sum(0))
VOXV = float(np.prod(vox))

# ------------------------------------------------------------------ eyes, crop --
eyes_lab, n = ndi.label(Lfull == 6)
sizes = np.bincount(eyes_lab.ravel())[1:]
eye_ks = list(np.argsort(sizes)[-2:] + 1)
eye_vox = [np.argwhere(eyes_lab == k).mean(0) for k in eye_ks]
eye_w = [(A0 @ np.r_[c, 1])[:3] for c in eye_vox]
order = np.argsort([w[0] for w in eye_w])            # RAS: smaller x = patient's left
eye_ks = [eye_ks[i] for i in order]
eye_vox = [eye_vox[i] for i in order]
eye_w = [eye_w[i] for i in order]
mid_w = (eye_w[0] + eye_w[1]) / 2
# crop: 30 mm lateral of the eyes, 75 mm behind to 20 mm in front, 40 mm below to 35 mm above
lo_w = np.array([min(w[0] for w in eye_w) - 30, mid_w[1] - 75, mid_w[2] - 40])
hi_w = np.array([max(w[0] for w in eye_w) + 30, mid_w[1] + 20, mid_w[2] + 35])
corners = np.array([[x, y, z] for x in (lo_w[0], hi_w[0]) for y in (lo_w[1], hi_w[1]) for z in (lo_w[2], hi_w[2])])
cv = (np.linalg.inv(A0) @ np.c_[corners, np.ones(8)].T).T[:, :3]
lo = np.maximum(np.floor(cv.min(0)).astype(int), 0)
hi = np.minimum(np.ceil(cv.max(0)).astype(int) + 1, Lfull.shape)
crop = tuple(slice(a, b) for a, b in zip(lo, hi))
L = Lfull[crop]
eyes = eyes_lab[crop]
A = A0.copy()
A[:3, 3] = (A0 @ np.r_[lo, 1])[:3]                  # affine of the crop
shape = L.shape


def resample(img, order):
    M = np.linalg.inv(img.affine) @ A
    return ndi.affine_transform(np.asarray(img.dataobj, dtype=np.float32), M[:3, :3], M[:3, 3],
                                output_shape=shape, order=order)


T1 = resample(nib.load(t1_path), 1)
S = resample(nib.load(aseg_path), 0).astype(np.int32)
brain = S > 0
bone = np.isin(L, [4, 7, 8])
air = L == 0
eye = L == 6
W = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
W = (W @ A[:3, :3].T + A[:3, 3]).reshape(shape + (3,)).astype(np.float32)  # world mm per voxel
print("crop", shape)

# ---------------------------------------------------------------- charm prior --
prior = (L == 1) & ~brain
plab, pn = ndi.label(prior)
psz = np.bincount(plab.ravel()) * VOXV
psz[0] = 0
prior = plab == np.argmax(psz)                       # the optic nerves + chiasm piece
print(f"charm optic prior {psz.max():.0f} mm3")
ex = eye_w[1] - eye_w[0]; ex /= np.linalg.norm(ex)
xs = (W - mid_w) @ ex                                # signed distance from the midline
pidx = np.argwhere(prior)
mid_pts = pidx[np.abs(xs[prior]) < 4]
chiasm = mid_pts.mean(0) if len(mid_pts) else pidx[np.argmin(np.abs(xs[prior]))]
print("chiasm (crop voxel)", chiasm.round(1))

# ------------------------------------------------------------------- intensities --
# Fat level: bright T1 within 15 mm behind the eyes, outside bone/eye/brain.
dist_eye = ndi.distance_transform_edt(~eye, sampling=vox)
grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).astype(np.float32)

# The eyeball wall (sclera) is as dark as muscle, so measure each globe's real radius from
# the radial T1 profile (dark inside, bright fat outside) and treat that ball as eye.
globe = eye.copy()
for s in (0, 1):
    c = eye_vox[s] - lo
    d = np.linalg.norm((grid - c) * vox, axis=-1)
    rr = np.arange(8, 18, 0.5)
    prof = np.array([np.median(T1[(d >= r) & (d < r + 0.5)]) for r in rr])
    half = (prof[0] + prof[-6:].max()) / 2
    R = rr[np.argmax(prof > half)]
    globe |= d <= R + 0.5
    print(f"globe {'LR'[s]}: radius {R:.1f} mm")
dist_globe = ndi.distance_transform_edt(~globe, sampling=vox)
soft = ~(bone | globe | air | brain)
near = soft & (dist_globe < 15) & (W[..., 1] < mid_w[1])
fat = np.percentile(T1[near], 90)
T1n = T1 / fat
T1s = ndi.gaussian_filter(T1n, 0.4 / vox)        # light smoothing for the muscle threshold
print(f"fat level {fat:.0f}")

bone_d = ndi.distance_transform_edt(~bone, sampling=vox)
prior_d = ndi.distance_transform_edt(~prior, sampling=vox)

out_lab = np.zeros(shape, np.uint8)
labels = {}
NERVE = {0: 1, 1: 2}
CHIASM = 3
SIDES = {0: "L", 1: "R"}
nerves_cl = []

for s in (0, 1):
    side = SIDES[s]
    g = eyes == eye_ks[s]
    ec = eye_vox[s] - lo
    sign = -1 if s == 0 else 1                       # lateral direction in xs
    # --- centre line: back of the eyeball -> chiasm
    cost = 1.0 + 6.0 * (prior_d > 1.5) + 8.0 * np.clip((T1n - 0.45) / 0.4, 0, 1) * (prior_d > 0.5)
    forbid = bone | globe | air | brain
    cost[forbid] = np.inf
    shell = (dist_globe > 0) & (dist_globe <= 1.0) & ~forbid
    behind = (W[..., 1] < (A @ np.r_[ec, 1])[1] - 3)
    starts = np.argwhere(shell & behind & (eyes_lab[crop] == 0) & (np.abs(xs) * 1 > 0)
                         & (np.sign(xs) == sign))
    # only the eye of this side: within 16 mm of its centre
    starts = starts[np.linalg.norm((starts - ec) * vox, axis=1) < 16]
    m = MCP_Geometric(cost, sampling=tuple(vox))
    tgt = tuple(np.round(chiasm).astype(int))
    cum, _ = m.find_costs([tgt])
    st = starts[np.argmin(cum[tuple(starts.T)])]
    path = np.array(m.traceback(tuple(st)))
    path_mm = path * vox
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path_mm, axis=0), axis=1))]
    # --- optic canal: the closest approach to bone in the middle of the path
    bd = ndi.uniform_filter1d(bone_d[tuple(path.T)], 9)
    midsec = (arc > 15) & (arc < arc[-1] - 5)
    canal_i = np.where(midsec)[0][np.argmin(bd[midsec])]
    print(f"{side}: path {arc[-1]:.1f} mm, start {st}, canal at {arc[canal_i]:.1f} mm (bone {bd[canal_i]:.2f} mm)")
    # --- nerve mask
    cl = np.zeros(shape, bool)
    cl[tuple(path.T)] = True
    d_cl, near_i = ndi.distance_transform_edt(~cl, sampling=vox, return_indices=True)
    # which path point is nearest -> orbital (before the canal) or not
    pos = np.full(shape, -1, np.int32)
    pos[tuple(path.T)] = np.arange(len(path))
    seg = pos[tuple(near_i)]
    orbital = seg <= canal_i
    dark = T1n < 0.62
    nerve = (d_cl <= 1.2) | (orbital & (d_cl <= 2.6) & dark) | (~orbital & (d_cl <= 1.8) & prior)
    nerve &= ~(bone | globe | air) & ~(brain & ~prior)
    nerve &= np.abs(xs) > 3.5                        # chiasm handled separately
    nerve = ndi.binary_opening(nerve, iterations=1) | (d_cl <= 1.2)
    nerve &= ~(bone | globe | air)
    lab_n, nn = ndi.label(nerve)
    nerve = lab_n == lab_n[tuple(path[len(path) // 3])]
    out_lab[nerve & (out_lab == 0)] = NERVE[s]
    labels[NERVE[s]] = (f"Optic nerve ({side})", "#f1d54a")
    nerves_cl.append((path, canal_i, st))
    print(f"{side}: nerve {nerve.sum() * VOXV:.0f} mm3")

# ---------------------------------------------------------------------- chiasm --
cdist = np.linalg.norm((grid - chiasm) * vox, axis=-1)
ch = prior & (cdist < 9) & (np.abs(xs) <= 7)
for path, _, _ in nerves_cl:                          # include the tube where the nerves meet
    tail = path[-6:]
    tube = np.zeros(shape, bool); tube[tuple(tail.T)] = True
    ch |= (ndi.distance_transform_edt(~tube, sampling=vox) <= 1.8) & (np.abs(xs) <= 7)
ch &= ~(bone | air | globe)
out_lab[ch & (out_lab == 0)] = CHIASM
labels[CHIASM] = ("Optic chiasm", "#e9c33d")
print(f"chiasm {ch.sum() * VOXV:.0f} mm3")

# ---------------------------------------------------------------------- muscles --
MUSCLES = [("Superior rectus + levator", 90, "#c8564b"), ("Lateral rectus", 0, "#b8433c"),
           ("Inferior rectus", 270, "#cf6a52"), ("Medial rectus", 180, "#a93a36"),
           ("Superior oblique", 135, "#d7806a")]
for s in (0, 1):
    side = SIDES[s]
    path, canal_i, st = nerves_cl[s]
    apex = path[canal_i].astype(float)
    ec = eye_vox[s] - lo
    axis = (ec - apex) * vox
    depth = np.linalg.norm(axis); axis /= depth
    rel = (grid - apex) * vox
    t = rel @ axis                                   # along the orbit axis, 0 at the canal
    radial = rel - t[..., None] * axis
    r = np.linalg.norm(radial, axis=-1)
    cone = (t > -2) & (t < depth) & (r < 3 + np.clip(t, 0, None) * np.tan(np.radians(27)))
    cand = cone & soft & ~(out_lab > 0)
    orbit_lab, _ = ndi.label(cand)
    # the orbit = the component(s) touching the nerve's orbital part
    nerve_orb = np.zeros(shape, bool); nerve_orb[tuple(path[:canal_i].T)] = True
    ks = np.unique(orbit_lab[ndi.binary_dilation(nerve_orb, iterations=4) & cand])
    orbit = np.isin(orbit_lab, ks[ks > 0])
    # Otsu splits fat from the rest; capped at 0.58 x fat (90th percentile of charm's
    # muscle cores here) so partial-volume rims and the periorbita don't bloat the muscles.
    thr = min(filters.threshold_otsu(T1n[orbit]), 0.58)
    mus = orbit & (T1s < thr) & (dist_globe > 1.0) & (bone_d > 0.75)
    # Beside the eyeball the recti thin out into tendons hugging it; further out it is
    # orbital wall, lacrimal gland or eyelid, so keep only a 3.5 mm shell there.
    # (allowed distance from the globe grows smoothly further back, no hard step)
    mus &= dist_globe <= 3.5 + 2.0 * np.clip(depth - 8 - t, 0, None)
    mus &= ndi.distance_transform_edt(out_lab != NERVE[s], sampling=vox) > 1.5  # nerve sheath
    mus = ndi.binary_opening(mus, iterations=1)
    ml, mn = ndi.label(mus)
    msz = np.bincount(ml.ravel()) * VOXV
    keep = [k for k in range(1, mn + 1) if msz[k] > 40]
    mus = np.isin(ml, keep)
    print(f"{side}: orbit {orbit.sum() * VOXV / 1000:.1f} cm3, otsu {thr:.2f}, muscles {mus.sum() * VOXV:.0f} mm3 "
          f"in {len(keep)} pieces")
    base = 11 if s == 0 else 21
    if not per_muscle:
        out_lab[mus] = base
        labels[base] = (f"Eye muscles ({side})", "#c0504d")
        continue
    # angle around the axis: 90 = up (superior), 0 = lateral
    up = np.array([0, 0, 1.0])
    up = up - (up @ axis) * axis; up /= np.linalg.norm(up)
    lat = ex * (1 if s == 1 else -1)
    lat = lat - (lat @ axis) * axis - (lat @ up) * up; lat /= np.linalg.norm(lat)
    ang = np.degrees(np.arctan2(radial @ up, radial @ lat)) % 360
    centres = np.array([c for _, c, _ in MUSCLES], float)
    dif = np.abs((ang[mus][:, None] - centres[None, :] + 180) % 360 - 180)
    which = np.argmin(dif, axis=1)
    per = np.zeros(shape, np.uint8)
    per[mus] = 1 + which
    # Tidy each muscle: close 1 mm gaps (distance transforms), keep its main body plus
    # pieces > 30 mm3; dropped bits stay unlabelled.
    for i in range(len(MUSCLES)):
        m = per == i + 1
        m = (ndi.distance_transform_edt(ndi.distance_transform_edt(~m, sampling=vox) <= 1.0,
                                        sampling=vox) > 1.0) & orbit & (out_lab == 0)
        ml_, mn_ = ndi.label(m)
        if mn_ == 0:
            continue
        sz_ = np.bincount(ml_.ravel()) * VOXV
        sz_[0] = 0
        m = np.isin(ml_, [k for k in range(1, mn_ + 1) if k == np.argmax(sz_) or sz_[k] > 30])
        out_lab[m & (out_lab == 0)] = base + i
    for i, (nm, _, col) in enumerate(MUSCLES):
        labels[base + i] = (f"{nm} ({side})", col)

full = np.zeros(Lfull.shape, np.uint8)
full[crop] = out_lab
nib.save(nib.Nifti1Image(full, A0), out / "orbit_labels.nii.gz")
present = set(np.unique(full).tolist())
lut = {str(k): {"name": n, "color": c} for k, (n, c) in sorted(labels.items()) if k in present}
json.dump(lut, open(out / "orbit_labels.json", "w"), indent=1)
for k, v in lut.items():
    print(f"  {k:>2} {v['name']:32s} {(full == int(k)).sum() * VOXV:7.0f} mm3")
