# MRI Head Explorer

**An interactive 3D model of a real human head, built from one routine brain MRI with free, open-source research tools.**

Rotate it, slice it open, take the skull off, pull the hemispheres apart, find the tiny inner ear, and click any part of the brain to learn what it does. It runs in a web browser, and you don't need a medical background.

👉 **Live demo:** https://volodymyrsmirnov.github.io/brain/ (after GitHub Pages is enabled, see [Publishing](#publishing))

| | | |
|---|---|---|
| ![Raw MRI as a 3D cloud](docs/img/volume.jpg) | ![See-through head](docs/img/glass-head.jpg) | ![Skull](docs/img/skull.jpg) |
| *The raw scan as a 3D cloud* | *Glass head* | *Skull (atlas-based estimate)* |
| ![Brain map](docs/img/brain-map.jpg) | ![Virtual cut](docs/img/slice.jpg) | ![Deep structures](docs/img/deep.jpg) |
| *62 cortical regions* | *Virtual cut with the real MRI section* | *Deep structures in a glass brain* |

> **Not a medical tool.** This is a visualization for curiosity and education. Automatic segmentation makes mistakes,
> and nothing here says anything about anyone's health. The person who was scanned published it themselves.

---

## What's in the model

| Layer | Source scan | Method | Notes |
|---|---|---|---|
| Raw volume | T1 MPRAGE | direct volume rendering (ray marching, in the browser) | 1 mm, "Solid" and "X-ray" styles, cut planes with a real MRI section on the cut face |
| Skin | T1 MPRAGE | intensity threshold + morphology | the face is recognisable |
| Skull, eyes | T1 + T2 TSE | SimNIBS **charm** (atlas-based Bayesian segmentation) | bone is dark on MRI, so this is an *estimate* |
| Cortex: 31 regions per hemisphere, grouped into 6 lobes | T1 MPRAGE | **FastSurfer** (deep learning, DKT atlas) | hover/click for names and plain-language descriptions |
| Deep structures: ventricles, thalamus, caudate, putamen, pallidum, hippocampus, amygdala, accumbens, ventral DC, brainstem, white matter, cerebellum | T1 MPRAGE | FastSurfer | |
| Inner ears (cochlea, vestibule, semicircular canals) | T2 SPACE, 0.3 mm | threshold + canal-width filtering | each is about 1 cm |
| Veins | SWI | Frangi vesselness + hysteresis + density filter | some artefacts near the skull base |

Total: 96 meshes, ~3 million triangles, 28 MB of web data.

## Try it locally

```bash
git clone https://github.com/volodymyrsmirnov/brain.git && cd brain
python3 -m http.server -d docs 8000
# open http://localhost:8000
```

The viewer is a single static page (`docs/index.html`) plus binary data (`docs/data/`). It needs a local web server because browsers don't let pages `fetch()` files from `file://` URLs. A Git LFS checkout is **not** needed just to view it.

## Repository layout

```
docs/                     the website (GitHub Pages root)
  index.html              viewer: three.js + three-mesh-bvh, no build step
  data/manifest.json      layer list: names, colours, groups, sizes
  data/*.bin              gzip'd meshes (int16 vertices at 0.01 mm + uint32 triangles) and the volume
  img/                    screenshots for this README
data/                     ⚠ Git LFS
  raw/mri_study_anonymized.zip   the full study: 2,058 DICOM files, 23 series, de-identified
  derived/                intermediate results so you can skip the slow steps
    t1.nii.gz             T1 MPRAGE as NIfTI
    brain_mask.nii.gz     SynthStrip brain mask
    aparc.DKTatlas+aseg.deep.mgz   FastSurfer labels
    charm_tissues.nii.gz  SimNIBS charm labels (skin, bone, CSF, eyes, …)
    inner_ear_labels.nii.gz
    veins.nii.gz
    head_all.glb          every mesh in one glTF file, for Blender etc.
pipeline/
  setup.sh                installs every tool (Homebrew, uv, Docker)
  run_all.sh              the whole pipeline, step by step, resumable
  anonymize_dicom.py      DICOM de-identification
  seg_inner_ear.py        inner-ear segmentation
  seg_veins.py            vein extraction from SWI
  build_model.py          meshes + web data
```

## Methodology

### 1. Source data

A routine clinical brain MRI ("RM Crânio-Encefálico") on a **Siemens MAGNETOM Skyra 3T** with a 20-channel head/neck coil:
23 series, 2,058 DICOM images. The clinic's patient portal (Biotronics3D 3Dnet) shows images only in a web viewer,
but it offers a hidden "download study" job, which the viewer's own JavaScript uses. Calling it returned the original
DICOM files, not screenshots.

Series used in this project:

| Series | Type | Voxel size (mm) | TR / TE | Used for |
|---|---|---|---|---|
| 9 `t1_mprage_sag_p2_iso` | 3D T1-weighted MPRAGE | 0.6 × 0.5 × 0.5 | 2300 / 3.4 ms, TI 900 ms | brain, skin, volume rendering, reference space |
| 14 `pd-t2_tse_tra` (echo 2) | 2D T2-weighted TSE | 0.45 × 0.45 × 3.96 | 4550 / 82 ms | helps charm separate bone from fluid |
| 13 `t2_space_tra_iso` | 3D T2 SPACE | 0.32 × 0.32 × 0.30 | 1400 / 154 ms | inner ear |
| 17 `t2_qswi_tra` (SWI) | susceptibility-weighted | 0.7 × 0.7 × 1.7 | 27 / 20 ms | veins |

The other series (FLAIR, diffusion/ADC, coronal T2, ear reconstructions, SWI phase/mIP) are in the archive but not used yet.

### 2. De-identification ([`anonymize_dicom.py`](pipeline/anonymize_dicom.py))

Before anything was committed, every DICOM header was scrubbed, roughly following the DICOM PS3.15 Basic profile:
- all person names were blanked;
- patient ID, birth date, address and other identifiers were removed;
- institution, station and device serial number were removed;
- accession and study IDs were removed;
- all dates were blanked;
- every UID was replaced with a fresh one;
- all private vendor tags were stripped.

Sex, age and every acquisition parameter were kept. The portal's `DICOMDIR` and `Readme.html` were dropped.

A byte-level search of the output confirmed that none of the original identifiers remain, and converting the anonymized files gives **voxel-identical** NIfTI volumes.

> The images themselves still show the face (it is a 3D head scan). That is a deliberate choice of the person scanned.

### 3. DICOM → NIfTI

[`dcm2niix`](https://github.com/rordenlab/dcm2niix) converts each series to compressed NIfTI. All series share the scanner's patient coordinate system, so they line up without registration. Every later step works in **world RAS millimetres**.

### 4. Skull stripping

[SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) (FreeSurfer, run in Docker) predicts a brain mask from the T1 scan. It works on any contrast and resolution. The veins step uses this mask.

### 5. Brain segmentation

[FastSurfer](https://github.com/Deep-MI/FastSurfer) v2.6 (`--seg_only`, native high-resolution mode, run on the Apple GPU via PyTorch MPS) labels ~100 structures. Its labels follow FreeSurfer's `aparc.DKTatlas+aseg` scheme:
- 31 cortical regions per hemisphere from the Desikan–Killiany–Tourville atlas;
- subcortical structures, ventricles, cerebellum and brainstem.

The cortical regions are grouped into frontal, parietal, temporal and occipital lobes, plus the cingulate gyrus and the insula.

### 6. Head tissues and skull

Bone has almost no MRI signal. A first attempt that thresholded the dark band between scalp and brain produced only a rough cap.

[SimNIBS](https://simnibs.github.io/simnibs/) 4.6 `charm` does much better:
1. It registers the T2 TSE scan to the T1.
2. It fits a probabilistic head atlas with a Bayesian (samseg) model.
3. The result labels skin, compact and spongy bone, CSF, eyes, muscle and more.

The skull mesh is compact plus spongy bone. Ripples on the back of the head come from the T2 scan's 4 mm slices.

### 7. Inner ear ([`seg_inner_ear.py`](pipeline/seg_inner_ear.py))

On the 0.3 mm T2 SPACE scan, the fluid-filled labyrinth is bright inside dark bone. The method:
1. Pick a region of interest around each petrous bone. **These ROIs are hand-picked for this scan;** see the comments in the script.
2. Apply an Otsu threshold.
3. Remove the internal auditory canal. It is the thick (> 1.1 mm core radius) fluid channel entering from the brain side; the labyrinth's canals are only ~1 mm wide, so they survive this filter.
4. Remove fragments cut off at the region borders.
5. Keep pieces within 1.5 mm of the main labyrinth.

The result shows the cochlea, the vestibule and the three semicircular canals.

### 8. Veins ([`seg_veins.py`](pipeline/seg_veins.py))

On SWI, deoxygenated blood is dark. The steps:
1. Resample the SWI to 0.7 mm isotropic and normalise it by a 6 mm local mean.
2. Compute Frangi vesselness for dark tubes 0.5–1.8 mm wide, inside the SynthStrip brain mask.
3. Apply hysteresis thresholding (99.3th / 97th percentile), so veins stay continuous across the thick SWI slices.
4. Apply a local-density filter: real veins are sparse threads, while the susceptibility artefacts near the sinuses form a dense mesh.
5. Drop specks under 40 mm³.

Several parameter sets were compared visually; the trade-offs are documented in the script.

### 9. Surfaces and web export ([`build_model.py`](pipeline/build_model.py))

Each label becomes a surface:
1. Crop to the label and apply a light Gaussian blur.
2. Run marching cubes ([scikit-image](https://scikit-image.org)).
3. Decimate the mesh ([fast-simplification](https://github.com/pyvista/fast-simplification)).
4. Apply Taubin smoothing ([trimesh](https://trimesh.org)).
5. Transform the vertices to world millimetres through each volume's affine, then centre them on the T1 volume.

Vertices are stored as int16 at 0.01 mm and triangles as uint32, gzip-compressed into one file per layer. The T1 volume is resampled to 1 mm, normalised to uint8 and stored the same way.

### 10. Viewer ([`docs/index.html`](docs/index.html))

Plain HTML + [three.js](https://threejs.org), with no build step:
- **Hover and click** run on [three-mesh-bvh](https://github.com/gkjohnson/three-mesh-bvh), so picking stays fast across 3M triangles.
- **Volume rendering** is a custom GLSL ray marcher over a 3D texture. "Solid" composites front to back with gradient shading; "X-ray" accumulates absorption.
- **Cut planes** clip both the meshes and the volume. Where a ray leaves the removed half, it shows the MRI intensity, so the cut face looks like a real slice.
- **Shading:** screen-space ambient occlusion (GTAO), switched off automatically for translucent scenes.

## Reproduce it

You need macOS on Apple Silicon (tested), Homebrew, Docker (OrbStack works), and ~20 GB of disk space.

```bash
git lfs install && git lfs pull      # fetch data/ (~0.5 GB)
pipeline/setup.sh                    # tools: dcm2niix, FastSurfer, SimNIBS, SynthStrip
pipeline/run_all.sh                  # ~1 hour; resumable, writes data/derived and docs/data
```

To regenerate only the web model from the committed intermediate results (5 minutes, Python only):

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -r pipeline/requirements.txt
.venv/bin/python pipeline/build_model.py data/derived docs/data work/meshes
```

To process **your own** MRI:
1. Anonymize it with `anonymize_dicom.py`.
2. Adjust the series names at the top of `run_all.sh`.
3. Re-pick the inner-ear ROIs.

Without an SWI or a thin-slice T2 scan, the veins and inner-ear steps simply won't apply.

## Limitations

- **Skull:** atlas-guided estimate from MRI, not CT. The overall shape is right; thin bones and sutures are not.
- **Veins:** artefacts near the skull base and some broken vessels, because of SWI's 1.7 mm slices.
- **Inner ear:** hand-picked ROIs; the right side has a small extra stub (likely a nerve or vessel).
- **Labels:** automatic segmentation can be off by a millimetre or two at region borders.

## Publishing

1. Create an empty GitHub repository named `brain`.
2. Push:
   ```bash
   git remote add origin git@github.com:volodymyrsmirnov/brain.git
   git push -u origin main          # LFS objects upload automatically (~0.5 GB)
   ```
3. On GitHub, open **Settings → Pages** and set **Deploy from a branch → `main` / `/docs`**.

GitHub Pages can't serve Git LFS files, which is why `docs/` is in plain git (28 MB) while only `data/` uses LFS. Free GitHub accounts include a limited amount of LFS storage and bandwidth; check your quota before sharing widely.

## Credits

- **FastSurfer:** Henschel et al., *NeuroImage* 2020; Henschel et al., *NeuroImage* 2022 (VINN, high-resolution)
- **SimNIBS charm:** Puonti et al., *NeuroImage* 2020
- **SynthStrip:** Hoopes et al., *NeuroImage* 2022
- **DKT atlas:** Klein & Tourville, *Frontiers in Neuroscience* 2012
- **Frangi vesselness:** Frangi et al., *MICCAI* 1998
- **dcm2niix:** Li et al., *J Neurosci Methods* 2016
- three.js, three-mesh-bvh, scikit-image, trimesh, nibabel, pydicom

## License

- **Code** (`pipeline/`, `docs/index.html`): [MIT](LICENSE)
- **Imaging data and 3D model** (`data/`, `docs/data/`): [CC BY 4.0](LICENSE-DATA). Please don't try to re-identify the person or draw conclusions about their health.
