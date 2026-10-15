"""Split the charm skull (bone labels 4, 7, 8) into named parts.

Usage: python seg_skull_parts.py <charm_tissues.nii.gz> <aparc.DKTatlas+aseg.deep.mgz> <out_dir>
Writes <out_dir>/skull_parts.nii.gz (uint8, charm grid) and skull_parts.json ({label: {name, color}}).
Every bone voxel gets exactly one part, so the parts together replace the single skull mesh.

Bone is dark on MRI and charm's skull is an atlas-guided estimate, so the split uses
anatomical landmarks and the bone's own shape rather than sutures (which MRI can't see):
  * head frame: origin between the eyeball centres (charm label 6), x along the eyes,
    z = scanner up made orthogonal to x, y = anterior;
  * spinal cord axis: charm's "white matter" below the brain (charm labels the cord as WM);
  * foramen magnum level ~ lowest point of the cerebellum (FastSurfer labels 8/47);
  * spine: bone cores (distance-transform erosion) near the cord axis below the foramen
    magnum that come apart from the skull core at some erosion radius = spine seeds;
  * mandible / face / skull seeds from coarse anatomical zones in the head frame;
  * seeds grow back over the full bone by watershed on the bone thickness, so the borders
    fall where the bone is thinnest (jaw joint, cranio-cervical joints, thin necks);
  * the teeth touch, so inside the dental arch the jaws are split by the occlusal plane,
    found as the minimum of the axial bone-area profile per front-back band;
  * face vs cranium has no joint, so that border is a fixed geometric cut in the head
    frame (roughly the frontozygomatic / frontonasal sutures and the pterygoid plates);
  * calvaria (skull cap) vs skull base: a plane from above the brows to above the inion;
  * spine: C1 (atlas, no body) = top cores right under the skull; everything below = one
    part, because charm fuses neighbouring arches too much to number C2..C7 reliably.
    Spine voxels go to the nearest seed by a geodesic distance where up/down steps cost 3x;
  * hyoid: a small separate bone below the chin, labelled only if charm found one (it
    usually doesn't).
Labels: 1 calvaria, 2 skull base, 3 facial skeleton, 4 mandible, 5 hyoid, 10 C1, 11 C2-.
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric
from skimage.segmentation import watershed

charm_path, aseg_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

CALVARIA, BASE, FACE, MANDIBLE, HYOID = 1, 2, 3, 4, 5
VERT0 = 10  # 10 = C1 (atlas), 11 = C2 and below
LUT = {CALVARIA: ("Skull cap (calvaria)", "#efe9d6"),
       BASE: ("Skull base", "#e2d2a8"),
       FACE: ("Facial skeleton", "#f4dcc4"),
       MANDIBLE: ("Mandible (lower jaw)", "#e8c79a"),
       HYOID: ("Hyoid bone", "#d9c49a")}
VERT_NAMES = ["C1 vertebra (atlas)", "Cervical vertebrae (C2 and below)"]
VERT_COLORS = ["#dfe3e8", "#cfd8e2"]

ct = nib.load(charm_path)
L = np.asarray(ct.dataobj).squeeze().astype(np.uint8)
A = ct.affine
vox = np.sqrt((A[:3, :3] ** 2).sum(0))
bone = np.isin(L, [4, 7, 8])

# FastSurfer labels onto the charm grid (nearest neighbour, only the bits we need).
fs = nib.load(aseg_path)
M = np.linalg.inv(fs.affine) @ A
S = ndi.affine_transform(np.asarray(fs.dataobj).astype(np.int16), M[:3, :3], M[:3, 3],
                         output_shape=L.shape, order=0)
brain = S > 0

# ------------------------------------------------------------------ head frame --
eyes_lab, n = ndi.label(L == 6)
sizes = ndi.sum(L == 6, eyes_lab, range(1, n + 1))
two = np.argsort(sizes)[-2:] + 1
eye_c = [(A @ np.r_[ndi.center_of_mass(eyes_lab == k), 1])[:3] for k in two]
eye_c.sort(key=lambda p: p[0])                      # RAS: smaller x = patient's left
E = (eye_c[0] + eye_c[1]) / 2
ex = eye_c[1] - eye_c[0]; ex /= np.linalg.norm(ex)
ez = np.array([0, 0, 1.0]) - ex[2] * ex; ez /= np.linalg.norm(ez)
ey = np.cross(ez, ex)
R = np.stack([ex, ey, ez])                          # world -> frame rotation


def frame(idx):
    """Voxel indices (N,3) -> head-frame mm (N,3)."""
    w = idx @ A[:3, :3].T + A[:3, 3]
    return (w - E) @ R.T


def zf_of_voxel_z(k):
    return frame(np.array([[L.shape[0] / 2, L.shape[1] / 2, k]]))[0, 2]


# Spinal cord: charm WM outside the FastSurfer brain, the component reaching lowest.
wm_out = (L == 1) & ~brain
lab, n = ndi.label(wm_out)
objs = ndi.find_objects(lab)
wsz = np.bincount(lab.ravel())
cord_k = min((k for k in range(1, n + 1) if wsz[k] * np.prod(vox) > 250), key=lambda k: objs[k - 1][2].start)
cord = frame(np.argwhere(lab == cord_k))
zs = np.arange(np.floor(cord[:, 2].min()), np.ceil(cord[:, 2].max()) + 1, 2.0)
cx = np.array([cord[np.abs(cord[:, 2] - z) < 2, 0].mean() if (np.abs(cord[:, 2] - z) < 2).any() else np.nan for z in zs])
cy = np.array([cord[np.abs(cord[:, 2] - z) < 2, 1].mean() if (np.abs(cord[:, 2] - z) < 2).any() else np.nan for z in zs])
ok = ~np.isnan(cx)
px, py = np.polyfit(zs[ok], cx[ok], 1), np.polyfit(zs[ok], cy[ok], 1)  # straight cord axis


def cord_xy(z):
    return np.polyval(px, z), np.polyval(py, z)


cb = frame(np.argwhere(np.isin(S, [8, 47])))
z_fm = cb[:, 2].min()                               # foramen magnum ~ cerebellar tonsils
print(f"eyes {np.round(eye_c, 1).tolist()}  cord axis x={px.round(3)} y={py.round(3)}  foramen magnum z'={z_fm:.1f}")

# ------------------------------------------------------------------ bone frame --
sl = ndi.find_objects(bone.astype(np.uint8))[0]
sl = tuple(slice(max(s.start - 2, 0), s.stop + 2) for s in sl)
b = bone[sl]
off = np.array([s.start for s in sl])
idx = np.argwhere(b)
F = frame(idx + off)
thick = ndi.distance_transform_edt(b, sampling=vox).astype(np.float32)
T = thick[b]
ccx, ccy = cord_xy(F[:, 2])
rcord = np.hypot(F[:, 0] - ccx, F[:, 1] - ccy)      # distance from the cord axis
dy_cord = F[:, 1] - ccy                              # >0: in front of the cord

VOXV = float(np.prod(vox))


def core_components(r, region):
    """Connected bone cores (thickness > r mm) inside `region` (bool array on the crop)."""
    clab, _ = ndi.label((thick > r) & region)
    return clab, np.bincount(clab.ravel()) * VOXV


# ------------------------------------------------------------------- spine seed --
# Bone cores near the cord axis below the foramen magnum that come apart from the main
# skull core at some erosion radius. The vertebrae detach at different radii (C1 is
# usually the last, its lateral masses sit right on the occipital condyles), so the seeds
# are the union over a range of radii.
in_zone = (rcord < 32) & (F[:, 2] < z_fm + 15)
spine_seed = np.zeros(len(F), bool)
for r in [1.0, 1.25, 1.5, 1.75, 2.0]:
    clab, csz = core_components(r, b)
    main = np.argmax(csz[1:]) + 1
    cvals = clab[b]
    new = np.zeros(len(F), bool)
    for k in np.unique(cvals[in_zone]):
        if k in (0, main) or csz[k] < 150:
            continue
        sel = cvals == k
        zc = F[sel, 2].mean()
        if in_zone[sel].mean() > 0.8 and zc < z_fm + 6 and np.median(dy_cord[sel]) < 25:  # not jaw / hyoid
            new |= sel
            if csz[k] > 1000:
                print(f"     core {csz[k]:.0f} mm3, centre z' {zc:.1f}")
    spine_seed |= new
    leak = ((cvals == main) & (rcord < 20) & (F[:, 2] < z_fm - 15)).sum() * VOXV
    print(f"  r={r}: detached spine cores {new.sum() * VOXV:.0f} mm3, "
          f"skull core below foramen magnum {leak:.0f} mm3")
SPINE = 9

# ----------------------------------------------------- skull / face / jaw seeds --
core1 = thick[b] > 0.75
front = dy_cord > 30                                   # well in front of the spine
# chin = lowest substantial bone in front of the spine (ignore small specks such as the
# hyoid or charm noise in the neck: use the largest bone component reaching the front)
comp, _ = ndi.label(b)
cvals = comp[b]
csz = np.bincount(cvals) * VOXV
csz[0] = 0
big = csz[cvals] > 20000
z_chin = np.percentile(F[front & big & (np.abs(F[:, 0]) < 20), 2], 0.5)
print(f"chin z'={z_chin:.1f}")
seed = np.zeros(len(F), np.int32)
# skull: vault far above the eyes, and occipital bone behind/above the foramen magnum
seed[(F[:, 2] > 35) | ((F[:, 2] > z_fm + 15) & (dy_cord < -20))] = BASE
# face: maxilla body around the nasal cavity and maxillary sinuses
seed[(F[:, 2] > -45) & (F[:, 2] < -15) & (F[:, 1] > -10) & (F[:, 1] < 25) & (np.abs(F[:, 0]) < 30)] = FACE
# mandible: lowest bone well in front of the spine (chin and body)
seed[front & (F[:, 2] < z_chin + 18) & (np.abs(F[:, 0]) < 45)] = MANDIBLE
seed[~core1] = 0
seed[spine_seed] = SPINE


def grow(seed_vals):
    mk = np.zeros(b.shape, np.int32)
    mk[b] = seed_vals
    return watershed(-thick, mk, mask=b)[b]


print("watershed 1 ...")
lab = grow(seed)

# ---------------------------------------------------------------- occlusal plane --
# Upper and lower teeth touch, so a watershed alone lets one jaw flood into the other.
# Where the teeth meet, the bone of the dental arch is narrowest: per 6 mm front-back band
# of the arch, the bite level is the minimum of the axial bone-area profile between the
# chin and the palate. A line z' = a + b*y' through those minima is the occlusal plane;
# inside the dental arch the jaws are split by that plane, elsewhere (rami, condyles,
# coronoid processes) by the watershed.
in_arch = front & (np.abs(F[:, 0]) < 30) & (F[:, 1] > 5)
zb = np.arange(z_chin + 12, z_chin + 55, 1.0)
pts = []
for y0 in np.arange(5, F[in_arch, 1].max() - 3, 6.0):
    s = in_arch & (F[:, 1] >= y0) & (F[:, 1] < y0 + 6)
    if s.sum() * VOXV < 300:
        continue
    h = np.histogram(F[s, 2], bins=np.r_[zb - 0.5, zb[-1] + 0.5])[0].astype(float)
    h = ndi.gaussian_filter1d(h, 1.5)
    inner = slice(3, len(h) - 3)                       # a real minimum, not the window edge
    k = np.argmin(h[inner]) + 3
    if h[k] < 0.8 * min(h[:k].max(), h[k:].max()):
        pts.append((y0 + 3, zb[k], s.sum()))
pts = np.array(pts)
if len(pts) >= 2:
    # Theil-Sen slope (median of pairwise slopes, robust to one bad band), kept within
    # +-14 degrees, and flat when the pairwise slopes disagree in sign; intercept = median.
    sl_ = [(pts[j, 1] - pts[i, 1]) / (pts[j, 0] - pts[i, 0]) for i in range(len(pts)) for j in range(i + 1, len(pts))]
    q1, q3 = np.percentile(sl_, [25, 75])
    slope = 0.0 if q1 < 0 < q3 else np.clip(np.median(sl_), -0.25, 0.25)  # flat if unsure
    occ = np.array([slope, 0.0])
    occ[1] = np.median(pts[:, 1] - occ[0] * pts[:, 0])
else:
    occ = np.array([0.0, z_chin + 30])
print(f"occlusal plane z' = {occ[1]:.1f} + {occ[0]:.2f} y'   (minima at "
      + ", ".join(f"y'{p[0]:.0f}:z'{p[1]:.0f}" for p in pts) + ")")
dz = F[:, 2] - np.polyval(occ, F[:, 1])
seed2 = seed.copy()
seed2[in_arch & (dz > 3) & (dz < 15) & core1] = FACE
seed2[in_arch & (dz < -3) & (dz > -15) & core1] = MANDIBLE
print("watershed 2 ...")
lab = grow(seed2)
jaw = in_arch & np.isin(lab, [FACE, MANDIBLE])
lab[jaw] = np.where(dz[jaw] > 0, FACE, MANDIBLE)

# ------------------------------------------------------------------------ hyoid --
# A separate bone below the mandible, in front of the spine; if charm found it, it is a
# small component of its own.
comp, _ = ndi.label(b)
cvals = comp[b]
csz = np.bincount(cvals) * VOXV
for k in np.unique(cvals):
    sel = cvals == k
    if 150 < csz[k] < 3000 and np.median(dy_cord[sel]) > 10 and F[sel, 2].mean() < z_chin + 25 \
            and abs(F[sel, 0].mean()) < 15:
        lab[sel] = HYOID
        print(f"  hyoid candidate: {csz[k]:.0f} mm3")

# ----------------------------------------------------------- face vs cranium cut --
# No joint between them, so this is a geometric cut in the head frame: below the brow line
# (higher in the middle, where the nasal bones reach the nasion) and in front of the
# pterygoid plates / middle of the zygomatic arch.
z_top = np.where(np.abs(F[:, 0]) < 12, 12.0, 9.0)
y_back = np.where(np.abs(F[:, 0]) < 28, -32.0, -26.0)
in_face = (F[:, 2] < z_top) & (F[:, 1] > y_back) & (F[:, 2] > z_fm)
lab[(lab == BASE) & in_face] = FACE
lab[(lab == FACE) & ~in_face] = BASE

# Calvaria (skull cap) vs skull base: plane from above the brows (y'=+10, z'=+28) to above
# the inion (y'=-110, z'=+2).
p0, p1 = np.array([10.0, 28.0]), np.array([-110.0, 2.0])
nrm = np.array([-(p1 - p0)[1], (p1 - p0)[0]])
nrm *= np.sign(nrm[1]) / np.linalg.norm(nrm)
lab[(lab == BASE) & (((F[:, 1:] - p0) @ nrm) > 0)] = CALVARIA

# ------------------------------------------------------------------- vertebrae --
# Neighbouring vertebral arches often touch in charm's bone, but the vertebral bodies are
# stacked in front of the cord with discs between them. So: one seed per body core (bone in
# front of the cord, erosion radius giving the most separate cores) plus C1, which has no
# body: the top core of the whole spine if it sits clearly above the first body (the dens of
# C2 belongs to the C2 body core).
spine = np.zeros(b.shape, bool)
spine[b] = lab == SPINE
dyg = np.zeros(b.shape, np.float32)
dyg[b] = dy_cord


def best_cores(region, min_mm3):
    best = None
    for r in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
        clab, csz = core_components(r, region)
        ks = [k for k in range(1, len(csz)) if csz[k] > min_mm3]
        if best is None or len(ks) > len(best[2]):
            best = (r, clab, ks)
    r, clab, ks = best
    cz = {k: frame(np.argwhere(clab == k) + off)[:, 2].mean() for k in ks}
    return r, clab, sorted(ks, key=lambda k: -cz[k]), cz


rb, blab, bks, bz = best_cores(spine & (dyg > 3), 200)
ra, alab, aks, az = best_cores(spine, 300)
mk = np.zeros(b.shape, np.int32)
seeds_z = []
if aks:
    # C1 = the top core(s) of the spine, right under the skull (both lateral masses / arches),
    # plus any body-zone core at the same height (its anterior arch).
    c1 = [k for k in aks if az[k] > az[aks[0]] - 6]
    mk[np.isin(alab, c1)] = 1
    seeds_z.append(az[aks[0]])
    mk[np.isin(blab, [k for k in bks if abs(bz[k] - seeds_z[0]) < 8]) & (mk == 0)] = 1
# Below C1, charm's spine is too coarse to number the vertebrae reliably (neighbouring
# arches are fused), so C2 and everything below form one part.
rest = [k for k in bks if not seeds_z or bz[k] < seeds_z[0] - 8]
if rest:
    mk[np.isin(blab, rest) & (mk == 0)] = 2
    seeds_z.append(bz[rest[0]])
# Grow each seed through the spine's bone with up/down steps 3x as expensive as sideways
# ones, so arches and spinous processes go to the vertebra at their own level (charm
# often fuses neighbouring arches, which a plain watershed would hand to one vertebra).
ssl = ndi.find_objects(spine.astype(np.uint8))[0]
cost = np.where(spine[ssl], 1.0, np.inf)
best_d = np.full(cost.shape, np.inf)
vert_c = np.zeros(cost.shape, np.int32)
for i in np.unique(mk[mk > 0]):
    m = MCP_Geometric(cost, sampling=(vox[0], vox[1], 3 * vox[2]))
    d, _ = m.find_costs(np.argwhere(mk[ssl] == i))
    better = d < best_d
    best_d[better] = d[better]
    vert_c[better] = i
vert = np.zeros(b.shape, np.int32)
vert[ssl] = vert_c
vb = vert[b]
lab[lab == SPINE] = np.where(vb > 0, VERT0 - 1 + vb, 0)[lab == SPINE]
nvert = len(seeds_z)
print(f"vertebrae: body cores r={rb} ({len(bks)}), {nvert} vertebrae at z' "
      + ", ".join(f"{z:.0f}" for z in seeds_z))

# Anything unseeded (tiny isolated bits) takes the nearest labelled bone voxel.
res = np.zeros(b.shape, np.uint8)
res[b] = lab
if (res[b] == 0).any():
    _, near = ndi.distance_transform_edt(res == 0, return_indices=True)
    res[b] = res[tuple(n[b] for n in near)]

# Clean-up: stray pieces of a part (not its largest piece, < 2 cm3) that touch other bone
# join the part they touch most.
for k in [k for k in np.unique(res) if k not in (0, HYOID)]:
    pl, pn = ndi.label(res == k)
    if pn < 2:
        continue
    psz = np.bincount(pl.ravel()) * VOXV
    psz[0] = 0
    for j, bb in enumerate(ndi.find_objects(pl), 1):
        if j == np.argmax(psz) or psz[j] > 2000:
            continue
        bb = tuple(slice(max(x.start - 1, 0), x.stop + 1) for x in bb)
        piece = pl[bb] == j
        sub = res[bb]
        ring = ndi.binary_dilation(piece) & ~piece & (sub > 0)
        if ring.any():
            sub[piece] = np.bincount(sub[ring]).argmax()

full = np.zeros(L.shape, np.uint8)
full[sl] = res
nib.save(nib.Nifti1Image(full, A), out / "skull_parts.nii.gz")
lut = {str(k): {"name": n, "color": c} for k, (n, c) in LUT.items()}
for i in range(len(VERT_NAMES)):
    lut[str(VERT0 + i)] = {"name": VERT_NAMES[i], "color": VERT_COLORS[i]}
present = set(np.unique(full).tolist())
lut = {k: v for k, v in lut.items() if int(k) in present}
json.dump(lut, open(out / "skull_parts.json", "w"), indent=1)
for k, v in lut.items():
    print(f"  {k:>2} {v['name']:28s} {(full == int(k)).sum() * VOXV / 1000:7.1f} cm3")
