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
  <name>.nii.gz + <name>.json     optional label volumes, one mesh per label (see LABEL_SETS);
                                  JSON = {"<label>": {"name": ..., "color": "#rrggbb"}}
  surfaces/{lh,rh}.{pial,inflated}.gii + {lh,rh}.{dkt,sulc}.npy   optional cortical surfaces
Writes to web_data_dir:
  manifest.json                   groups (titles, order, info, defaults) + layers; schema at MANIFEST below
  <layer>.drc                     Draco mesh (positions on a <= 0.05 mm grid; surfaces add morph/dkt/sulc attributes)
  volume.bin                      gzip: uint8 T1 volume, 1 mm, x fastest, delta-coded along x
Optionally writes OBJ files per layer and head_all.glb to meshes_dir.
"""
import gzip, json, math, re, sys
from pathlib import Path

import DracoPy
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

SULC_MM = 7.0  # sulcal depth that maps to full dark/light (FreeSurfer sulc spans about -15..+21 mm)
QUANT_STEP = 0.05  # mm grid per axis: Draco rounds to it, so the error is <= 0.025 mm per axis, <= 0.043 mm in 3D


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


# ------------------------------------------------------------------ groups --
# The viewer builds its layer panel from these (in this order). Only groups that end up with layers are written.
#   visible: shown in "free mode"/"everything on" (a layer can override with its own "visible")
#   explode: moved apart by the "Split brain" slider (by each layer's hemi)
GROUPS = [
    dict(key="head", title="Head", visible=True, explode=False,
         info="The outside of the head: skin, skull and eyes."),
    dict(key="bones", title="Skull bones", visible=True, explode=False,
         info="The skull split into its main parts. Bone is dark on MRI, so these shapes are an atlas-guided estimate."),
    dict(key="eyes", title="Eyes & optic nerves", visible=True, explode=False,
         info="The cables and tiny muscles behind the eyeballs."),
    dict(key="surface", title="Brain surface", visible=False, explode=True,
         info="The outer surface of the cortex as one smooth sheet. It can be inflated like a balloon to show the hidden folds."),
    dict(key="cortex", title="Cortex (by lobe)", visible=True, explode=True,
         info="The wrinkled outer layer of the brain, split into regions."),
    dict(key="deep", title="Deep structures", visible=True, explode=True,
         info="Structures hidden beneath the cortex."),
    dict(key="cc", title="Corpus callosum", visible=True, explode=False,
         info="The bridge of nerve fibres joining the two halves of the brain."),
    dict(key="hypothalamus", title="Hypothalamus area", visible=True, explode=False,
         info="Small structures at the base of the brain that run hormones, hunger, sleep and memory circuits."),
    dict(key="cerebellum", title="Cerebellum", visible=True, explode=True,
         info="The “little brain” at the back: movement, balance and timing."),
    dict(key="cerebellum_lobules", title="Cerebellum lobules", visible=False, explode=True,
         info="The cerebellum divided into its numbered lobules."),
    dict(key="ear", title="Inner ear", visible=True, explode=False,
         info="Hearing and balance organs, each about 1 cm across."),
    dict(key="vessels", title="Veins", visible=False, explode=False,
         info="Veins draining blood from the brain."),
    dict(key="xray", title="Dental X-ray", visible=False, explode=False,
         info="My panoramic dental X-ray, bent back onto the curve of the jaw."),
]

# Generic label volumes: <stem>.nii.gz + <stem>.json -> one mesh per label. Skipped when absent.
LABEL_SETS = [
    # budget = total triangles for the whole set, shared by label surface area (~ voxels^(2/3))
    dict(stem="skull_parts", group="bones", prefix="bone", sigma=1.0, budget=360_000, smooth=20, opacity=0.45,
         replaces=["skull"]),
    dict(stem="orbit_labels", group="eyes", prefix="orbit", sigma=0.5, budget=60_000, smooth=6, upsample=2,
         skip=r"chiasm"),  # the hypothalamus set has a finer optic chiasm (0.5 mm T1)
    # skip: labels that duplicate meshes we already build from other sources
    dict(stem="corpus_callosum", group="cc", prefix="cc", sigma=0.5, budget=60_000, smooth=8, skip=r"fornix"),
    dict(stem="hypothalamus_labels", group="hypothalamus", prefix="hyp", sigma=0.3, budget=60_000, smooth=6,
         # optic nerves come from orbit_labels instead: they run the whole way from the eye to the chiasm
         skip=r"globus pallidus|third ventricle|optic nerve"),
    dict(stem="cerebellum_lobules", group="cerebellum_lobules", prefix="cbl", sigma=0.5, budget=320_000, smooth=8,
         skip=r"white matter"),
]

layers = []  # dicts: id, group, name, hemi, color, mesh (+ optional extra manifest keys)


def add(id_, group, name, hemi, color, m, **extra):
    if m is None:
        print("  skip (empty)", id_)
        return
    layers.append(dict(id=id_, group=group, name=name, hemi=hemi, color=color, mesh=m, extra=extra))
    print(f"  {id_:28s} {len(m.faces):7d} faces")


HEMI = {"L": "(L)", "R": "(R)"}

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
add("skin", "head", "Skin", "mid", "#e3b39a", mesh_from_mask(head, t075.affine, 1.0, 250_000), opacity=0.18)
del a, sm, head, lab, dist_out

# ------------------------------------------------------- skull & eyes (charm) --
replaced = {r for s in LABEL_SETS if (src / f"{s['stem']}.nii.gz").exists() for r in s.get("replaces", [])}
charm = src / "charm_tissues.nii.gz"
if charm.exists():
    print("skull (charm)")
    ct = nib.load(charm)
    L = np.asarray(ct.dataobj).squeeze()
    if "skull" not in replaced:
        add("skull", "head", "Skull", "mid", "#efe9d6",
            mesh_from_mask(np.isin(L, [4, 7, 8]), ct.affine, 1.0, 350_000, smooth=20), opacity=0.45)
    eyes = L == 6
    lab, n = ndi.label(eyes)
    for k in range(1, n + 1):
        part = lab == k
        if part.sum() * np.prod(ct.header.get_zooms()[:3]) < 1000:
            continue
        x = (ct.affine @ np.r_[np.argwhere(part).mean(0), 1])[0]
        hemi = "R" if x > C[0] else "L"
        add(f"eye_{hemi}", "head", "Eyeball " + HEMI[hemi], hemi, "#f4f1ea", mesh_from_mask(part, ct.affine, 0.8, 20_000))
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


def shade(hex_color, k):
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], float)
    c = np.clip(c * (0.85 + 0.3 * k), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c)


# one colour per region (same for both hemispheres and for the surface parcels)
REGION_COLOR = {code: shade(LOBES[lobe][1], np.random.default_rng(code).random()) for code, (lobe, _) in DKT.items()}
for base, hemi in ((1000, "L"), (2000, "R")):
    for code, (lobe, name) in DKT.items():
        add(f"ctx_{hemi}_{code}", "cortex", f"{name} {HEMI[hemi]}", hemi, REGION_COLOR[code],
            mesh_from_mask(S == base + code, fs.affine, 0.5, 14_000),
            row="lobe_" + lobe, rowName=LOBES[lobe][0], lobe=lobe, code=code)

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
HIDDEN = {"wm", "cerebellum_wm"}  # not part of "everything on"
for id_, name, labs, color, group in SUB:
    faces = 160_000 if id_ in ("wm", "cerebellum") else 30_000
    extra = {"visible": False} if id_ in HIDDEN else {}
    if len(labs) == 2:
        for lab_, hemi in zip(labs, ("L", "R")):
            add(f"{id_}_{hemi}", group, f"{name} {HEMI[hemi]}", hemi, color,
                mesh_from_mask(S == lab_, fs.affine, 0.5, faces), **extra)
    else:
        add(id_, group, name, "mid", color, mesh_from_mask(S == labs[0], fs.affine, 0.5, faces), **extra)
del S

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
    VL = np.asarray(vimg.dataobj)  # 1 = SWI veins, 2 = dural sinuses (charm label 9, checked on SWI)
    add("veins", "vessels", "Veins (SWI)", "mid", "#4a78d0",
        mesh_from_mask((VL > 0) & (VL != 2), vimg.affine, 0.4, 300_000, smooth=4))
    if (VL == 2).any():
        add("sinuses", "vessels", "Dural sinuses", "mid", "#b8434f",
            mesh_from_mask(VL == 2, vimg.affine, 0.6, 120_000, smooth=8))
    del VL


# ----------------------------------------------------- generic label volumes --
def hemi_of(name):
    """'(L)', 'Left …', 'lh …', '… left' -> L/R, otherwise mid."""
    if re.search(r"\(L( half)?\)|\bleft\b|\blh\b|^L[ _-]|[ _-]L$", name, re.I):
        return "L"
    if re.search(r"\(R( half)?\)|\bright\b|\brh\b|^R[ _-]|[ _-]R$", name, re.I):
        return "R"
    return "mid"


def slug(s):
    s = re.sub(r"\((L|R)( half)?\)|\b(left|right|lh|rh)\b", "", s, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "part"


def label_color(i):
    h = (i * 0.618034) % 1  # golden-ratio hues for labels without a colour
    r, g, b = [int(255 * (0.55 + 0.35 * math.cos(2 * math.pi * (h + o)))) for o in (0, 1 / 3, 2 / 3)]
    return "#%02x%02x%02x" % (r, g, b)


for spec in LABEL_SETS:
    nii = src / f"{spec['stem']}.nii.gz"
    if not nii.exists():
        print(f"no {nii.name}, skipping group {spec['group']}")
        continue
    print(f"labels: {nii.name}")
    img = nib.load(nii)
    Lv = np.asarray(img.dataobj).squeeze().astype(np.int32)
    js = src / f"{spec['stem']}.json"
    meta = json.loads(js.read_text()) if js.exists() else {}
    ks, counts = np.unique(Lv[Lv != 0], return_counts=True)
    names = {int(k): meta.get(str(int(k)), {}).get("name") or f"{spec['group'].replace('_', ' ').capitalize()} label {k}"
             for k in ks}
    keep_ = [(int(k), c) for k, c in zip(ks, counts) if not (spec.get("skip") and re.search(spec["skip"], names[int(k)], re.I))]
    for k in set(names) - {k for k, _ in keep_}:
        print("  skip (duplicate)", names[k])
    weight = {k: c ** (2 / 3) for k, c in keep_}
    seen = set()
    for i, (k, _) in enumerate(keep_):
        info = meta.get(str(k), {})
        name = names[k]
        faces = int(max(3000, spec["budget"] * weight[k] / sum(weight.values())))
        hemi = hemi_of(name)
        if hemi != "mid" and not re.search(r"\((L|R)( half)?\)$", name):  # normalise to "Name (L)"
            name = re.sub(r"\s*\b(left|right|lh|rh)\b\s*", " ", name, flags=re.I).strip() + " " + HEMI[hemi]
        id_ = f"{spec['prefix']}_{slug(name)}" + ("" if hemi == "mid" else "_" + hemi)
        while id_ in seen:
            id_ += "_"
        seen.add(id_)
        extra = {"opacity": spec["opacity"]} if "opacity" in spec else {}
        add(id_, spec["group"], name, hemi, info.get("color") or label_color(i),
            mesh_from_mask(Lv == k, img.affine, spec["sigma"], faces, spec.get("upsample", 1),
                           spec.get("smooth", 8)), label=k, **extra)
    del Lv

# --------------------------------------------------------- cortical surfaces --
surfaces = []  # dicts: id, hemi, pial (N,3), faces, inflated (N,3), dkt (N,), sulc (N,)
sdir = src / "surfaces"


def read_gii(path):
    g = nib.load(path)
    pts = next(d.data for d in g.darrays if d.intent == nib.nifti1.intent_codes["NIFTI_INTENT_POINTSET"])
    tri = next(d.data for d in g.darrays if d.intent == nib.nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"])
    return np.asarray(pts, np.float64), np.asarray(tri, np.int64)


if all((sdir / f"{h}.{k}").exists() for h in ("lh", "rh") for k in ("pial.gii", "inflated.gii", "dkt.npy")):
    print("cortical surfaces")
    for h, H in (("lh", "L"), ("rh", "R")):
        pial, faces = read_gii(sdir / f"{h}.pial.gii")
        infl, _ = read_gii(sdir / f"{h}.inflated.gii")
        dkt = np.load(sdir / f"{h}.dkt.npy").astype(np.int64) % 1000
        sulc = np.load(sdir / f"{h}.sulc.npy") if (sdir / f"{h}.sulc.npy").exists() else np.zeros(len(pial))
        assert len(pial) == len(infl) == len(dkt) == len(sulc), h
        surfaces.append(dict(id=f"surf_{H}", hemi=H, pial=pial - C, faces=faces, inflated=infl - C,
                             dkt=dkt, sulc=np.asarray(sulc, np.float64)))
    # Inflated hemispheres (scaled to the pial size, centred on the pial centroid) nearly touch at the midline:
    # push them apart along x so the morph opens a clear gap.
    xm = (surfaces[0]["pial"][:, 0].max() + surfaces[1]["pial"][:, 0].min()) / 2  # midline (L = -x in RAS)
    gap = 10.0
    for s in surfaces:
        x = s["inflated"][:, 0]
        s["inflated"][:, 0] += (xm - gap / 2 - x.max()) if s["hemi"] == "L" else (xm + gap / 2 - x.min())
    for s in surfaces:
        m = trimesh.Trimesh(s["pial"], s["faces"], process=False)
        add(s["id"], "surface", "Brain surface " + HEMI[s["hemi"]], s["hemi"], "#d9c2bd", m, type="surface")
        s["layer"] = layers[-1]
else:
    print("no cortical surfaces (surfaces/{lh,rh}.{pial,inflated}.gii + .dkt.npy), skipping")

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
# Air around the head is pure noise and costs ~30% of the file: zero it. "Air" = 3 mm-smoothed intensity < 12/255,
# connected to the volume border (so sinuses and airways inside the head keep their values).
air, _ = ndi.label(ndi.gaussian_filter(vol8.astype(np.float32), 3) < 12)
edge = np.unique(np.concatenate([np.moveaxis(air, ax, 0)[i].ravel() for ax in range(3) for i in (0, -1)]))
vol8[np.isin(air, edge[edge > 0])] = 0
del air
xyz = vol8.transpose(2, 1, 0)  # (z, y, x) memory order = x fastest
delta = np.diff(xyz.astype(np.int16), axis=2, prepend=0).astype(np.uint8)  # wraps mod 256; viewer prefix-sums rows
write_gz(out / "volume.bin", delta)


# ------------------------------------------------------------- draco files --
def draco_bits(v):
    rng = float(np.ptp(v, 0).max())
    return int(min(16, max(11, math.ceil(math.log2(rng / QUANT_STEP + 1)))))


def unique_ids(buf):
    """Map our generic-attribute keys to Draco unique ids (the decoder needs them)."""
    d = DracoPy.decode(buf)
    ids = {"position": next(a["unique_id"] for a in d.attributes if a["attribute_type"] == 0)}
    return ids


# ------------------------------------------------- dental X-ray (optional) --
# data/derived/dental_xray.npz (pipeline/build_dental_xray.py): a ribbon along the jaw, world RAS + UVs into
# data/raw/dental_panoramic.jpg. Shipped as a textured mesh; the image goes to docs/data/dental.jpg.
xray_npz, xray_jpg = src / "dental_xray.npz", src.parent / "raw" / "dental_panoramic.jpg"
xray_uv = None
if xray_npz.exists() and xray_jpg.exists():
    print("dental X-ray")
    d = np.load(xray_npz)
    m = trimesh.Trimesh(d["verts"] - C, d["faces"], process=False)    # keep vertex order: UVs are per vertex
    xray_uv = d["uv"]
    add("dental_xray", "xray", "Panoramic dental X-ray", "mid", "#ffffff", m, type="image")
    from PIL import Image
    im = Image.open(xray_jpg).convert("L")
    im.thumbnail((2048, 2048))
    im.save(out / "dental.jpg", quality=88)

keep = {"manifest.json", "volume.bin"} | ({"dental.jpg"} if xray_uv is not None else set())
manifest_layers = []
for L_ in layers:
    v = np.asarray(L_["mesh"].vertices, np.float64)
    f = np.asarray(L_["mesh"].faces, np.uint32)
    assert np.abs(v).max() < 1000, L_["id"]
    bits = draco_bits(v)
    entry = {k: L_[k] for k in ("id", "group", "name", "hemi", "color")} | L_["extra"]
    if L_["extra"].get("type") == "surface":
        s = next(s for s in surfaces if s["layer"] is L_)
        lo = s["inflated"].min(0)
        step = 0.05
        morph = np.round((s["inflated"] - lo) / step).astype(np.uint16)
        sulc = np.clip(np.round(s["sulc"] / SULC_MM * 127 + 128), 0, 255).astype(np.uint8)  # ±7 mm full scale
        dkt = np.clip(s["dkt"], 0, 255).astype(np.uint8)
        buf = DracoPy.encode(v, f, quantization_bits=bits, compression_level=7,
                             generic_attributes={1: morph, 2: dkt[:, None], 3: sulc[:, None]})
        ids = unique_ids(buf) | {"morph": 1, "dkt": 2, "sulc": 3}
        entry |= {"attributes": ids, "morph": {"origin": lo.round(4).tolist(), "step": step},
                  "sulcScale": SULC_MM / 127}
    elif L_["extra"].get("type") == "image":
        uv16 = np.round(np.clip(xray_uv, 0, 1) * 65535).astype(np.uint16)
        buf = DracoPy.encode(v, f, quantization_bits=bits, compression_level=7, generic_attributes={1: uv16})
        entry |= {"attributes": unique_ids(buf) | {"uv": 1}, "image": "dental.jpg", "uvScale": 1 / 65535}
    else:
        buf = DracoPy.encode(v, f, quantization_bits=bits, compression_level=7)
    name = f"{L_['id']}.drc"
    (out / name).write_bytes(buf)
    keep.add(name)
    entry |= {"file": name, "verts": len(v), "faces": len(f), "bytes": len(buf), "bits": bits}
    manifest_layers.append(entry)

for old in list(out.glob("*.bin")) + list(out.glob("*.drc")):
    if old.name not in keep:
        old.unlink()

# MANIFEST (version 2), read by docs/index.html:
#   lobes    {lobe: title}            regions {dkt code: {name, lobe, color}}
#   bounds   {center, size}           model-space box of the skin (camera framing before meshes arrive)
#   volume   {file, dims, spacing, origin, encoding: "delta-x", bytes}
#   groups   [{key, title, info, visible, explode, order}]   panel sections, in order
#   layers   [{id, group, name, hemi: L|R|mid, color, file, verts, faces, bytes, bits,
#              optional: opacity, visible, row/rowName (panel row), lobe/code (DKT region), label (source label),
#              type: "surface" + attributes {position, morph, dkt, sulc: Draco unique ids},
#              morph {origin, step} (uint16 -> mm), sulcScale,
#              type: "image" + attributes {position, uv}, image (texture file), uvScale (uint16 -> 0..1)}]
used = {L_["group"] for L_ in layers}
skin = next((L_ for L_ in layers if L_["id"] == "skin"), layers[0])
sv = np.asarray(skin["mesh"].vertices)
manifest = {
    "version": 2,
    "lobes": {k: v[0] for k, v in LOBES.items()},
    "regions": {str(c): {"name": n, "lobe": lobe, "color": REGION_COLOR[c]} for c, (lobe, n) in DKT.items()},
    "bounds": {"center": ((sv.min(0) + sv.max(0)) / 2).round(3).tolist(), "size": np.ptp(sv, 0).round(3).tolist()},
    "volume": {"file": "volume.bin", "dims": list(vol8.shape), "spacing": 1.0,
               "origin": (v1.affine[:3, 3] - C).tolist(), "encoding": "delta-x",
               "bytes": (out / "volume.bin").stat().st_size},
    "groups": [g | {"order": i} for i, g in enumerate(G for G in GROUPS if G["key"] in used)],
    "layers": manifest_layers,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
total = sum((out / n).stat().st_size for n in keep)
print("layers", len(layers), "faces", sum(len(L_["mesh"].faces) for L_ in layers),
      "web data MB", round(total / 1e6, 1), "(volume", round((out / "volume.bin").stat().st_size / 1e6, 1), "MB)")
