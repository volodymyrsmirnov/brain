"""Turn the segmentations into meshes + a volume for the web viewer.

Everything is placed in world RAS millimetres (scanner coordinates, shared by all
series of the session) minus the centre of the T1 volume.

Usage: python build_model.py <derived_dir> <web_data_dir> [<meshes_dir>]
Expects in derived_dir:
  t1.nii.gz                       T1 MPRAGE
  aparc.DKTatlas+aseg.deep.mgz    FastSurfer labels
  charm_tissues.nii.gz            SimNIBS charm labels (optional)
  inner_ear_labels.nii.gz         inner ears (optional)
  veins.nii.gz                    SWI veins (optional)
Writes to web_data_dir:
  manifest.json                   layer list (names, colours, groups, sizes)
  <layer>.bin                     gzip: int16 vertices (x100 mm) + uint32 triangle indices
  volume.bin                      gzip: uint8 T1 volume, 1 mm, x fastest
Optionally writes OBJ files per layer and head_all.glb to meshes_dir.
"""
import gzip, json, sys
from pathlib import Path

import fast_simplification
import nibabel as nib
import numpy as np
import trimesh
from nibabel.processing import resample_to_output
from scipy import ndimage as ndi
from skimage import filters, measure

src, out = Path(sys.argv[1]), Path(sys.argv[2])
mesh_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else None
out.mkdir(parents=True, exist_ok=True)

t1 = nib.load(src / "t1.nii.gz")
shape = np.array(t1.shape[:3])
C = (t1.affine @ np.r_[(shape - 1) / 2, 1])[:3]  # world centre of the T1 volume


def write_gz(path, *arrays):
    path.write_bytes(gzip.compress(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays), 9, mtime=0))


def mesh_from_mask(mask, affine, sigma_mm=0.6, faces=60_000, upsample=1, smooth=8):
    """Marching cubes on a (cropped) binary mask -> trimesh in model space, or None."""
    idx = np.argwhere(mask)
    if len(idx) < 20:
        return None
    lo = np.maximum(idx.min(0) - 4, 0)
    hi = np.minimum(idx.max(0) + 5, mask.shape)
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    vox = np.sqrt((affine[:3, :3] ** 2).sum(0))
    if upsample > 1:
        sub = ndi.zoom(sub, upsample, order=1)
    sub = ndi.gaussian_filter(sub, sigma_mm / vox * upsample)
    if sub.max() < 0.5:
        return None
    v, f, _, _ = measure.marching_cubes(sub, 0.5)
    v = v / upsample + lo
    v = (affine @ np.c_[v, np.ones(len(v))].T).T[:, :3] - C
    if np.linalg.det(affine[:3, :3]) < 0:
        f = f[:, ::-1]
    if len(f) > faces:
        v, f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), 1 - faces / len(f))
    m = trimesh.Trimesh(v, f, process=True)
    if smooth:
        trimesh.smoothing.filter_taubin(m, iterations=smooth)
    return m


layers = []  # dicts: id, group, name, hemi, color, mesh


def add(id_, group, name, hemi, color, m):
    if m is None:
        print("  skip (empty)", id_)
        return
    layers.append(dict(id=id_, group=group, name=name, hemi=hemi, color=color, mesh=m))
    print(f"  {id_:28s} {len(m.faces):7d} faces")


# ---------------------------------------------------------------- skin (T1) --
print("skin")
t075 = resample_to_output(t1, voxel_sizes=0.75, order=1)
a = t075.get_fdata(dtype=np.float32)
a /= np.percentile(a[a > 0], 99.5)
sm = ndi.gaussian_filter(np.clip(a, 0, 1), 1.0)
head = sm > filters.threshold_otsu(sm[sm > 0.02]) * 0.35
lab, n = ndi.label(head)
head = lab == (np.argmax(ndi.sum(head, lab, range(1, n + 1))) + 1)
dist_out = ndi.distance_transform_edt(~head, sampling=0.75)
head = ndi.distance_transform_edt(dist_out <= 4, sampling=0.75) > 4  # closing, 4 mm
for ax in range(3):
    head = np.moveaxis(np.stack([ndi.binary_fill_holes(s) for s in np.moveaxis(head, ax, 0)]), 0, ax)
add("skin", "head", "Skin", "mid", "#e3b39a", mesh_from_mask(head, t075.affine, 1.0, 250_000))

# ------------------------------------------------------- skull & eyes (charm) --
charm = src / "charm_tissues.nii.gz"
if charm.exists():
    print("skull (charm)")
    ct = nib.load(charm)
    L = np.asarray(ct.dataobj).squeeze()
    add("skull", "head", "Skull", "mid", "#efe9d6",
        mesh_from_mask(np.isin(L, [4, 7, 8]), ct.affine, 1.0, 350_000, smooth=20))
    eyes = L == 6
    lab, n = ndi.label(eyes)
    for k in range(1, n + 1):
        part = lab == k
        if part.sum() * np.prod(ct.header.get_zooms()[:3]) < 1000:
            continue
        x = (ct.affine @ np.r_[np.argwhere(part).mean(0), 1])[0]
        hemi = "R" if x > C[0] else "L"
        add(f"eye_{hemi}", "head", "Eyeball " + ("(R)" if hemi == "R" else "(L)"),
            hemi, "#f4f1ea", mesh_from_mask(part, ct.affine, 0.8, 20_000))
else:
    print("no charm output, skipping skull/eyes")

# ---------------------------------------------------------- brain (FastSurfer) --
print("brain")
fs = nib.load(src / "aparc.DKTatlas+aseg.deep.mgz")
S = np.asarray(fs.dataobj).astype(np.int32)

LOBES = {
    "frontal":   ("Frontal lobe", "#e98f8f"),
    "parietal":  ("Parietal lobe", "#e8c46f"),
    "temporal":  ("Temporal lobe", "#7fc8a9"),
    "occipital": ("Occipital lobe", "#8fa8e8"),
    "cingulate": ("Cingulate gyrus", "#c99be0"),
    "insula":    ("Insula", "#f0a85c"),
}
DKT = {  # label % 1000 -> (lobe, name)
    2: ("cingulate", "Caudal anterior cingulate"), 3: ("frontal", "Caudal middle frontal"),
    5: ("occipital", "Cuneus"), 6: ("temporal", "Entorhinal cortex"), 7: ("temporal", "Fusiform gyrus"),
    8: ("parietal", "Inferior parietal"), 9: ("temporal", "Inferior temporal"),
    10: ("cingulate", "Isthmus cingulate"), 11: ("occipital", "Lateral occipital"),
    12: ("frontal", "Lateral orbitofrontal"), 13: ("occipital", "Lingual gyrus"),
    14: ("frontal", "Medial orbitofrontal"), 15: ("temporal", "Middle temporal"),
    16: ("temporal", "Parahippocampal"), 17: ("frontal", "Paracentral lobule"),
    18: ("frontal", "Pars opercularis (Broca)"), 19: ("frontal", "Pars orbitalis"),
    20: ("frontal", "Pars triangularis (Broca)"), 21: ("occipital", "Pericalcarine (primary visual)"),
    22: ("parietal", "Postcentral (somatosensory)"), 23: ("cingulate", "Posterior cingulate"),
    24: ("frontal", "Precentral (primary motor)"), 25: ("parietal", "Precuneus"),
    26: ("cingulate", "Rostral anterior cingulate"), 27: ("frontal", "Rostral middle frontal"),
    28: ("frontal", "Superior frontal"), 29: ("parietal", "Superior parietal"),
    30: ("temporal", "Superior temporal"), 31: ("parietal", "Supramarginal"),
    34: ("temporal", "Transverse temporal (auditory)"), 35: ("insula", "Insula"),
}
HEMI = {"L": "(L)", "R": "(R)"}


def shade(hex_color, k):
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], float)
    c = np.clip(c * (0.85 + 0.3 * k), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c)


rng = np.random.default_rng(7)
for base, hemi in ((1000, "L"), (2000, "R")):
    for code, (lobe, name) in DKT.items():
        add(f"ctx_{hemi}_{code}", "lobe_" + lobe, f"{name} {HEMI[hemi]}", hemi,
            shade(LOBES[lobe][1], rng.random()),
            mesh_from_mask(S == base + code, fs.affine, 0.5, 14_000))

SUB = [  # id, names, labels (L, R or single), color, group
    ("wm", "White matter", (2, 41), "#f1ede4", "deep"),
    ("ventricles", "Lateral ventricle", (4, 43), "#5fb7e8", "deep"),
    ("ventricle3", "Third ventricle", (14,), "#5fb7e8", "deep"),
    ("ventricle4", "Fourth ventricle", (15,), "#5fb7e8", "deep"),
    ("thalamus", "Thalamus", (10, 49), "#58a6a6", "deep"),
    ("caudate", "Caudate nucleus", (11, 50), "#d7b46a", "deep"),
    ("putamen", "Putamen", (12, 51), "#e48aa7", "deep"),
    ("pallidum", "Globus pallidus", (13, 52), "#9f8fd6", "deep"),
    ("hippocampus", "Hippocampus", (17, 53), "#f2cc4c", "deep"),
    ("amygdala", "Amygdala", (18, 54), "#7ac36a", "deep"),
    ("accumbens", "Nucleus accumbens", (26, 58), "#e9925f", "deep"),
    ("ventraldc", "Ventral diencephalon", (28, 60), "#b08c6e", "deep"),
    ("brainstem", "Brainstem", (16,), "#c7a27c", "deep"),
    ("cerebellum", "Cerebellum (cortex)", (8, 47), "#e3a0c8", "cerebellum"),
    ("cerebellum_wm", "Cerebellum (white matter)", (7, 46), "#f3e6ee", "cerebellum"),
]
for id_, name, labs, color, group in SUB:
    faces = 160_000 if id_ in ("wm", "cerebellum") else 30_000
    if len(labs) == 2:
        for lab_, hemi in zip(labs, ("L", "R")):
            add(f"{id_}_{hemi}", group, f"{name} {HEMI[hemi]}", hemi, color,
                mesh_from_mask(S == lab_, fs.affine, 0.5, faces))
    else:
        add(id_, group, name, "mid", color, mesh_from_mask(S == labs[0], fs.affine, 0.5, faces))

# -------------------------------------------------------------- inner ears --
ear = src / "inner_ear_labels.nii.gz"
if ear.exists():
    print("inner ears")
    e = nib.load(ear)
    E = np.asarray(e.dataobj)
    for k in (1, 2):
        part = E == k
        x = (e.affine @ np.r_[np.argwhere(part).mean(0), 1])[0]
        hemi = "R" if x > C[0] else "L"
        add(f"ear_{hemi}", "ear", f"Inner ear {HEMI[hemi]}", hemi, "#f5d76e",
            mesh_from_mask(part, e.affine, 0.25, 40_000, upsample=2, smooth=5))

# -------------------------------------------------------------------- veins --
veins = src / "veins.nii.gz"
if veins.exists():
    print("veins")
    vimg = nib.load(veins)
    add("veins", "vessels", "Veins (SWI)", "mid", "#4a78d0",
        mesh_from_mask(np.asarray(vimg.dataobj) > 0, vimg.affine, 0.4, 300_000, smooth=4))

# ------------------------------------------------------------ export files --
if mesh_dir:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for L_ in layers:
        L_["mesh"].export(mesh_dir / f"{L_['id']}.obj")
        m = L_["mesh"].copy()
        m.visual.face_colors = [int(L_["color"][i:i + 2], 16) for i in (1, 3, 5)] + [255]
        scene.add_geometry(m, node_name=L_["id"])
    scene.export(mesh_dir / "head_all.glb")

# ------------------------------------------------- volume for slices/render --
v1 = resample_to_output(t1, voxel_sizes=1.0, order=1)
assert np.allclose(np.diag(v1.affine)[:3], 1.0), v1.affine
vol = v1.get_fdata(dtype=np.float32)
vol = np.clip(vol / np.percentile(vol[vol > 0], 99.5), 0, 1)
vol8 = (vol * 255).astype(np.uint8)
write_gz(out / "volume.bin", vol8.transpose(2, 1, 0))  # x fastest

for old in out.glob("*.bin"):
    if old.name != "volume.bin" and old.stem not in {L_["id"] for L_ in layers}:
        old.unlink()

manifest = {
    "lobes": {k: v[0] for k, v in LOBES.items()},
    "volume": {"file": "volume.bin", "dims": list(vol8.shape), "spacing": 1.0,
               "origin": (v1.affine[:3, 3] - C).tolist()},
    "layers": [],
}
for L_ in layers:
    v = np.asarray(L_["mesh"].vertices)
    assert np.abs(v).max() < 327, L_["id"]  # fits int16 at 0.01 mm
    q = np.round(v * 100).astype(np.int16)
    f = np.asarray(L_["mesh"].faces, dtype=np.uint32)
    write_gz(out / f"{L_['id']}.bin", q, f)
    manifest["layers"].append({k: L_[k] for k in ("id", "group", "name", "hemi", "color")}
                              | {"file": f"{L_['id']}.bin", "verts": len(q), "faces": len(f),
                                 "bytes": (out / f"{L_['id']}.bin").stat().st_size})
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
total = sum(p.stat().st_size for p in out.glob("*.bin"))
print("layers", len(layers), "faces", sum(len(L_["mesh"].faces) for L_ in layers),
      "web data MB", round(total / 1e6, 1))
