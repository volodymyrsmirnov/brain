# CLAUDE.md

Context and hard-won lessons for working on this repo. Read before changing the pipeline or the viewer.

## What this is

An interactive 3D model of the owner's own head, built from one clinical brain MRI (Siemens Skyra 3T, Oct 2026,
Affidea Lisbon) and published as open source for non-medical people to explore.

- `docs/` is the GitHub Pages site: `index.html` (single file, three.js via CDN importmap, no build step) +
  `docs/data/` (manifest.json + gzip'd `.bin` per layer + `volume.bin`). Plain git, NOT LFS (Pages can't serve LFS).
- `data/raw/` + `data/derived/` are in Git LFS (`.gitattributes`). Anything new under `data/` goes to LFS automatically.
  Licence files etc. must live outside `data/` (that's why `LICENSE-DATA` is at the root).
- `pipeline/`: `setup.sh` (tools), `run_all.sh` (resumable steps 1–7), and Python scripts.
- `work/`, `.venv/`, `tools/` are gitignored and recreated by the scripts.

The owner speaks Russian and English; the viewer UI and all structure names must stay in **English**.

## Everyday commands

```bash
python3 -m http.server -d docs 8000                       # view locally (fetch() fails on file://)
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -r pipeline/requirements.txt
.venv/bin/python -I pipeline/build_model.py data/derived docs/data work/meshes   # rebuild web data (~5 min)
pipeline/setup.sh && pipeline/run_all.sh                  # full pipeline from the DICOM zip (~1 h)
```

Most viewer work only needs `docs/index.html` and the existing `docs/data`. Most model tweaks only need
`build_model.py` + `data/derived` — no FastSurfer/SimNIBS required.

## Coordinate system (easy to get wrong)

- Every volume is mapped through its own NIfTI affine to **world RAS mm** (scanner space). All series of the session
  share it, so no registration is needed. Model space = world RAS − centre of the T1 volume (`C` in build_model.py).
- Mesh vertices are stored as int16 = mm × 100 (asserted |v| < 327 mm), triangles as uint32, gzip'd together.
- The volume is the T1 resampled to 1 mm with `nibabel.processing.resample_to_output` (axis-aligned RAS, positive
  diagonal asserted), uint8, written x-fastest (`transpose(2,1,0)` of the (X,Y,Z) array). `manifest.volume.origin` is
  the model-space position of voxel (0,0,0).
- In the viewer, `root` has a fixed matrix mapping RAS → three.js: x = −R (patient's left shows on screen right when
  facing them), y = S, z = A. Explode offsets are applied in RAS x on each mesh's `position`.
- The "left" camera view looks at the patient's left side; it sits at three.js +x. This was swapped once; don't reswap.

## Pipeline lessons

**Getting the data.** The Affidea patient portal (Biotronics3D 3Dnet, `cloudpacs-pt.affidea.com`) opens the viewer
in a pop-up (blocked in the built-in browser). The authenticated session can call the JSON API directly:
`POST /B3DNet.svc/json/BeginDownloadDataJob` with `{request:{Studies:[{StudyDBID:<id>,SeriesOUIDs:null}],JobType:0,
OrgId:<org>,ExcludeViewer:true,ImageQuality:100,...}}` → poll `CheckExportDataJobProgress` → `GetExportDataJobResult` →
`/HttpDownloadHandler.ashx?filename=<zip>&dad=1&dtype=1`. Gives original DICOM (DIR000/…, DICOMDIR, Readme.html).

**Anonymization** (`anonymize_dicom.py`): verified by byte-searching the output for every original identifier
(0 hits) and by checking dcm2niix output is voxel-identical to the original. Re-run that check after any change.
`DeidentificationMethod` is VR LO, so keep it ≤ 64 characters. The face stays visible by the owner's choice.

**dcm2niix** naming `-f "%s_%d"` gives stable names like `9_t1_mprage_sag_p2_iso_101`. Series used: 9 T1 MPRAGE,
14 PD/T2 TSE echo 2 (`_e2`), 13 T2 SPACE 0.3 mm (inner ear), 17 SWI. The `1023…1049 OUVIDOS` series are ear MPRs.

**SynthStrip**: `freesurfer/synthstrip` Docker image is amd64-only; on Apple Silicon it runs under emulation (~8 min).
OrbStack must be running (`orb start`).

**FastSurfer v2.6** runs natively (Python 3.12 venv) with `--device mps`/auto, `--seg_only` (no FreeSurfer licence
needed), `--vox_size min` (0.5 mm). Pitfalls:
- Its helper shell steps break on paths containing spaces. The segmentation still succeeds; only the stats steps fail.
  That's why `run_all.sh` has `|| true` and then checks the output file.
- Output: `work/fastsurfer/head/mri/aparc.DKTatlas+aseg.deep.mgz` (DKT cortical labels 1000+/2000+ and aseg labels).

**SimNIBS 4.6 `charm`** without conda needs three hacks (all in `setup.sh`):
1. Install the extra wheels from GitHub URLs (fmm3dpy, samseg, cortech, petsc4py) listed in the release's
   `environment_macos.yml`.
2. The CGAL `.so` files reference `@rpath/libmpfr.6.dylib` and `@rpath/libgmp`. DYLD_* env vars are ignored
   (hardened Python), so `install_name_tool -add_rpath $(brew --prefix mpfr)/lib …` then `codesign -f -s -`.
3. Add a stub `mumps.py` (python-mumps is conda-only and only needed for FEM).

Charm's `--segment` continues into slow cortical-surface reconstruction we don't need. Wait for
`m2m_head/label_prep/tissue_labeling_upsampled.nii.gz`, then kill charm. Labels: 1 WM, 2 GM, 3 CSF, 4 bone (unused),
5 scalp, 6 eyes, 7 compact bone, 8 spongy bone, 9 blood, 10 muscle. Skull = {4,7,8}. Using the 2D T2 TSE (4 mm slices)
as T2 causes the ripples on the back of the skull. The skull mesh uses sigma 1.0 mm and 20 Taubin iterations to hide
them. The first attempt (thresholding the dark band on T1) gave only a blob; charm is vastly better.

**Inner ear** (`seg_inner_ear.py`): ROIs are hand-picked for THIS scan (found with thin-slab MIPs of series 13).
The internal auditory canal touches the cochlea, so a straight x-cut fails. Instead, remove the thick fluid
(distance-transform core > 1.1 mm) reaching the medial ROI face, drop border-touching pieces, and keep pieces within
1.5 mm of the main labyrinth. The right ear still has a small stub.

**Veins** (`seg_veins.py`, final params `99.3 97.0 40 0.12`):
- The SWI is only skull-stripped in the lower slices; upper slices include scalp and skull, which wrecked the
  percentiles. Always mask with the SynthStrip brain mask resampled onto the SWI grid (`resample_from_to`).
- Single threshold → fragments; hysteresis keeps veins continuous across 1.7 mm slices.
- Local-density filter (`gaussian(veins, 3 mm) < 0.12`) removes the dense artefact mesh near the skull base.
  At 0.08 it chops the superior sagittal sinus. Some basal artefacts remain (documented limitation).
- Vesselness is cached in `work/veins/vesselness.npy` (Frangi takes ~4 min); delete it if the filter changes.

**Morphology**: never use `binary_dilation` with big ball structuring elements on 0.5–0.75 mm grids (took >10 min).
Use distance transforms (`dil`/`ero`/`close_`/`open_` helpers pattern).

**Rendering previews** headless: pyvista with `pv.OFF_SCREEN=True` → PNG, great for checking meshes before the viewer.

## Viewer lessons (`docs/index.html`)

- The canvas needs explicit `width:100%; height:100%`. With `inset:0` alone it renders at device-pixel size and the
  model appears shifted to the bottom-right.
- The side panel covers the left of the screen, so `camera.setViewOffset` centres the scene in the free area
  (recomputed in `resize()`). Raycasting and `project()` stay correct with the offset. On portrait screens,
  `viewDir` backs off by `0.8/aspect`.
- GTAO (ambient occlusion) turns translucent layers into grey mud. `updateAO()` disables it whenever a visible mesh has
  opacity < 1, a cut is active, or volume rendering is on. With GTAO, set `scene.background` explicitly; otherwise
  the background renders grey.
- "Glass brain" needs cortex group opacity ~0.05 (0.12 still looks opaque: many overlapping double-sided layers).
- Volume ray marcher: unit-cube local space; camera and clip plane are transformed with the inverse of
  `vrBox.matrixWorld`. Where a ray exits the clipped half, it writes the raw intensity ("cap"), so cuts look like real
  MRI slices. Modes: 0 = Solid (front-to-back compositing + gradient shading), 1 = X-ray (absorption sum).
  Defaults: threshold 0.08, density 0.45 (owner liked the low threshold).
- Picking uses three-mesh-bvh (`computeBoundsTree` per geometry); `pick()` ignores meshes with opacity ≤ 0.3, so you
  can hover "through" the faint skin.
- Hover tooltip = name + one-line description from `infoFor()` (REGION_INFO by DKT code, LOBE_INFO, INFO by id
  prefix). Click = caption card. When adding layers, add a plain-language INFO entry too.
- Data files are gzip but named `.bin` (so no server sets Content-Encoding). `inflate()` checks the gzip magic bytes
  anyway.
- `window.viewer = { scene, camera, controls, items, startTour }` is exposed for debugging and screenshots.
- Testing in the Claude built-in browser: it can't open `file://` or localhost started from Bash, so use
  `preview_start` with a launch config. A 44 MB+ page may need 15–20 s to load; wait before clicking. Screenshots
  can lag one frame behind; use `zoom` or re-screenshot before concluding something isn't drawn.

## Ideas not done yet

- Use the 3D FLAIR (series 5, 0.5 mm) or T2 SPACE in charm instead of the 4 mm T2 TSE, for a smoother skull.
- FastSurfer full surface pipeline (needs a FreeSurfer licence) for real pial/white surfaces, and an inflated brain.
- Corpus callosum labels (FastSurfer CC module), hypothalamus (HypVINN with T2), cerebellum sub-lobules (CerebNet).
- Diffusion series (19/20) are trace/ADC only, so no tractography is possible from this study.
- Optic nerves and eye muscles, labelled facial bones, a guided "story" mode that steps through tours automatically.
- Mesh LOD or Draco compression if load time matters (currently 28 MB).
