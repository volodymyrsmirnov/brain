# CLAUDE.md

Context and hard-won lessons for working on this repo. Read before changing the pipeline or the viewer.

**Keep this file in sync.** The owner wants CLAUDE.md updated in the same change as any code, data or docs change:
new steps, new files, changed parameters, new pitfalls, finished or new ideas. If you learn something the hard way,
write it down here before you finish.

## What this is

An interactive 3D model of the owner's own head, built from one clinical brain MRI (Siemens Skyra 3T, Oct 2026,
Affidea Lisbon) and published as open source for non-medical people to explore. Live: https://volodymyrsmirnov.github.io/brain/
(GitHub Pages from `main` / `docs`). Repo: https://github.com/volodymyrsmirnov/brain.

- `docs/` is the GitHub Pages site: `index.html` (single file, three.js 0.160 + three-mesh-bvh + DRACOLoader via CDN
  importmap, no build step) + `docs/data/` (manifest.json, one Draco `.drc` per mesh layer, `volume.bin`). Plain git,
  NOT LFS (Pages can't serve LFS).
- `data/raw/` + `data/derived/` are in Git LFS (`.gitattributes`). Anything new under `data/` goes to LFS automatically.
  Licence files etc. must live outside `data/` (that's why `LICENSE-DATA` is at the root).
- `pipeline/`: `setup.sh` (tools), `run_all.sh` (resumable steps), and the step scripts (see README "Repository layout").
- `work/`, `.venv/`, `tools/` are gitignored and recreated by the scripts. `work/<agent-name>/` folders from the v2 work
  hold previews, comparisons and debug renders; they are disposable.

The owner speaks Russian and English; the viewer UI and all structure names must stay in **English**.
Voice: the project is told in the **first person** by the owner, Volodymyr ("my head", "my MRI"). He is happy to
say it's his head. Don't write "a real person" / "the person who was scanned". Tone: informal and plain
(it's a side project, not an academic paper), but no forced jokes; the owner finds them cringe. Facts stay correct;
the methodology may be technical. DICOM headers are still scrubbed,
as record-keeping hygiene (IDs, birth date, clinic, doctors), not to hide who it is.

## Everyday commands

```bash
python3 -m http.server -d docs 8000                       # view locally (fetch() fails on file://)
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -r pipeline/requirements.txt
.venv/bin/python -I pipeline/build_model.py data/derived docs/data work/meshes   # rebuild web data (3–15 min)
pipeline/setup.sh && pipeline/run_all.sh                  # full pipeline from the DICOM zip (hours)
```

Most viewer work only needs `docs/index.html` and the existing `docs/data`. Most model tweaks only need
`build_model.py` + `data/derived` — no FastSurfer/SimNIBS required. `?data=<dir>/` in the URL loads another build.

## Coordinate system (easy to get wrong)

- Every volume is mapped through its own NIfTI affine to **world RAS mm** (scanner space). All series of the session
  share it, so no registration is needed. Model space = world RAS − centre of the T1 volume (`C` in build_model.py).
  Cortical surfaces (GIfTI) are stored in world RAS too.
- Meshes are Draco-encoded with quantization bits chosen per mesh so the step is ≤ 0.05 mm (max measured error
  0.041 mm). Draco reorders vertices: per-vertex data (DKT code, sulc) must be Draco generic attributes, and their
  unique ids are written into the manifest (`attributes`).
- The volume is the T1 resampled to 1 mm with `nibabel.processing.resample_to_output` (axis-aligned RAS, positive
  diagonal asserted), uint8, x-fastest, delta-coded along x then gzip (`encoding: "delta-x"`). Air connected to the
  volume border is zeroed; everything inside the head is unchanged. `manifest.volume.origin` = model-space position of
  voxel (0,0,0).
- In the viewer, `root` has a fixed matrix mapping RAS → three.js: x = −R (patient's left shows on screen right when
  facing them), y = S, z = A. Explode offsets are applied in RAS x on each mesh's `position`.
- The "left" camera view looks at the patient's left side; it sits at three.js +x. This was swapped once; don't reswap.
- Label names use the patient's side: "(L)" = RAS x < 0.

## Pipeline (run_all.sh order) and lessons

**Getting the data.** The Affidea patient portal (Biotronics3D 3Dnet, `cloudpacs-pt.affidea.com`) opens the viewer
in a pop-up (blocked in the built-in browser). The authenticated session can call the JSON API directly:
`POST /B3DNet.svc/json/BeginDownloadDataJob` with `{request:{Studies:[{StudyDBID:<id>,SeriesOUIDs:null}],JobType:0,
OrgId:<org>,ExcludeViewer:true,ImageQuality:100,...}}` → poll `CheckExportDataJobProgress` → `GetExportDataJobResult` →
`/HttpDownloadHandler.ashx?filename=<zip>&dad=1&dtype=1`. Gives original DICOM (DIR000/…, DICOMDIR, Readme.html).

**Anonymization** (`anonymize_dicom.py`): verified by byte-searching the output for every original identifier
(0 hits) and by checking dcm2niix output is voxel-identical to the original. Re-run that check after any change.
`DeidentificationMethod` is VR LO, so keep it ≤ 64 characters.

**dcm2niix** naming `-f "%s_%d"` gives stable names like `9_t1_mprage_sag_p2_iso_101`. Series used: 9 T1 MPRAGE,
14 PD/T2 TSE echo 2 (`_e2`), 13 T2 SPACE 0.3 mm (inner ear), 17 SWI magnitude, 16 SWI phase. 5 is a 3D FLAIR
(tried for charm, not used). 19/20 are diffusion trace/ADC only (no tractography possible). `1023…1049 OUVIDOS` are
ear MPRs.

**setup.sh**: `uv venv` refuses to overwrite an existing venv, hence `--allow-existing`. `brew install` also upgrades
already-installed packages (uv, git-lfs). Verified on a clean `tools/` (Oct 2026).

**SynthStrip**: `freesurfer/synthstrip` Docker image is amd64-only; on Apple Silicon it runs under emulation (~8 min).
OrbStack must be running (`orb start`).

**FastSurfer v2.6** runs natively (Python 3.12 venv) with `--device mps`/auto, `--seg_only` (no FreeSurfer licence
needed), `--vox_size min` (0.5 mm). Pitfalls:
- Its helper shell steps break on paths containing spaces (the project path has none now).
- Output: `aparc.DKTatlas+aseg.deep.mgz` (DKT cortical labels 1000+/2000+ and aseg labels).

**FastSurfer extras** (`seg_fastsurfer_extras.sh` + `export_fastsurfer_extras.py`; idempotent, `THREADS=`, `FORCE=1`):
- Reuses the existing segmentation: `--no_asegdkt` needs `orig.mgz`, `aseg.auto_noCCseg.mgz` and `mask.mgz` in the
  subject dir; `export_fastsurfer_extras.py prep` rebuilds the latter two from the aparc. `conform.py --vox_size min`
  reproduces the aparc grid exactly.
- `--seg_only` runs CC and CerebNet by default. CerebNet always works at 1 mm; the export lifts it to 0.5 mm
  (per-label probability argmax) and clips it to the aseg cerebellum.
- **HypVINN bug at 0.5 mm:** its clean-up keeps only the largest {optic nerve + tract} blob after a 7-voxel dilation,
  which doesn't bridge the chiasm at 0.5 mm, so the optic tracts get dropped silently. The helper's `hypvinn` command
  counts the chiasm as optic for that test only. Worth reporting upstream.
- HypVINN ran on T1 only (FLAIR = wrong contrast, T2 TSE too thick, T2 SPACE covers 31%). With a T2 it needs FreeSurfer
  `mri_coreg` unless `--reg_mode none` with a T2 already on the T1 grid.
- Output label ids are FreeSurfer LUT ids: HypVINN 961–984, CerebNet 601–631 + 7/46, CC 250–255.
- macOS `/bin/bash` 3.2 with `set -u` fails on an empty `"${arr[@]}"`.
- HypVINN takes ~35 min under load (~10–15 idle). Weights (~240 MB) download into `tools/FastSurfer/checkpoints/`.

**SimNIBS 4.6 `charm`** without conda needs three hacks (all in `setup.sh`):
1. Install the extra wheels from GitHub URLs (fmm3dpy, samseg, cortech, petsc4py) listed in the release's
   `environment_macos.yml`.
2. The CGAL `.so` files reference `@rpath/libmpfr.6.dylib` and `@rpath/libgmp`. DYLD_* env vars are ignored
   (hardened Python), so `install_name_tool -add_rpath $(brew --prefix mpfr)/lib …` then `codesign -f -s -`.
3. Add a stub `mumps.py` (python-mumps is conda-only and only needed for FEM).

Charm notes:
- For tissues, `run_all.sh` waits for `m2m_head/label_prep/tissue_labeling_upsampled.nii.gz`, then kills charm.
- `--segment` does **not** build cortical surfaces in 4.6; that is a separate `charm <sub> --surfaces` run.
- Labels: 1 WM, 2 GM, 3 CSF, 5 scalp, 6 eyes, 7 compact bone, 8 spongy bone, 9 blood, 10 muscle (4 is unused).
  Skull = {7,8}.
- Thread count: copy `charm.ini`, change threads, pass `--usesettings`. A charm run is ~1 h idle, 2.5–3.5 h when three
  run in parallel. Don't run several at once.
- The 4 mm T2 TSE causes ripples on the back of the skull. FLAIR and T1-only were compared (roughness RMS, renders):
  FLAIR swaps the steps for finer bumps, holes in the face, a ragged jaw and +70 cm³ of bone; T1-only is in between.
  The T2 TSE version stays. The skull mesh uses sigma 1.0 mm + 20 Taubin iterations.
- Charm's atlas has optic nerves, chiasm and rectus muscles, but merges them into WM/muscle in the final labels.
  "Charm WM outside the FastSurfer brain" is a free prior for optic nerves, chiasm and spinal cord.
- Charm's eyeball (6) excludes the dark sclera (~1 mm too small); re-measure the globe radius from T1 before calling
  dark tissue next to the eye "muscle".

**Skull parts** (`seg_skull_parts.py`): seeds (eyes frame, spinal-cord axis, foramen magnum = lowest cerebellum point,
detached spine cores) + watershed on bone thickness, so borders fall at joints. Teeth touch, so the jaws are cut at the
bite plane (narrowest dental-arch cross-section per 6 mm band). Face/cranium and cap/base are geometric cuts (no
sutures on MRI). C2–C7 numbering was tried and dropped (charm fuses arches). Hyoid (label 5) only if charm finds one.

**Orbit** (`seg_orbit.py --per-muscle`): optic nerve = skimage MCP cheapest path from the eye to the chiasm through dark
T1 + the charm prior (bone/eye/air/brain forbidden); the optic canal = the path's closest approach to bone. Muscles =
dark tissue (≤ min(Otsu, 0.58 × fat)) in a 27° cone, split by angle. Nerves reliable, per-muscle shapes approximate.

**Inner ear** (`seg_inner_ear.py`): automatic. A box 22–62 mm lateral of the brainstem centre (aseg 16); remove thick
fluid; score bright 100–1500 mm³ pieces by the dark-bone fraction of their 0.5–1.5 mm shell. The internal auditory
canal is cut as a 5 mm-radius cylinder along its PCA axis (3 and 4 mm left a stub/fin). `--roi-left/--roi-right`
override. Tuned on this scan only.

**Veins** (`seg_veins.py ... 99.3 97.0 40 0.12 --phase <16> --charm <tissues>`):
- Mask with the SynthStrip brain mask resampled onto the SWI grid: the upper SWI slices include scalp and skull.
- Near air the SWI **phase wraps**; in-brain phase-wrap density is the artefact-risk map (8% of the brain). Hysteresis
  runs on `vesselness / (1 + 2·risk)`. A distance-from-air margin from charm was tried and is bad (it cuts real veins,
  and charm leaves stray 0-voxels inside the brain).
- Density filter `gaussian(veins, 3 mm) < 0.12` (0.08 chops the sagittal sinus). Risky pieces must have PCA
  elongation ≥ 3. Skeleton end points are bridged ≤ 5 mm in a cone with vesselness support; the size filter runs
  after bridging.
- Output values: 1 = SWI veins, 2 = dural sinuses (charm label 9 ∩ dark on SWI, neck cut, carotids removed by
  position: > 8 mm off the midline and > 12 mm anterior; carotid flow voids are dark on SWI too).
- Vesselness is cached in `work/veins/vesselness.npy` (Frangi ~4 min); delete it if the filter changes.
- The committed v1 `veins.nii.gz` could not be reproduced by the v1 code (24.6 vs 18.7 cm³), so don't trust old numbers.

**Cortical surfaces** (`build_surfaces.py`, run_all step 7): a second, T1-only charm run (`--initatlas --segment`,
then `charm head_surf --surfaces`, ~14 min idle) gives TopoFit white/pial GIfTIs in scanner RAS.
- TopoFit meshes are icosahedron subdivisions (finest 245,762 vertices), so white and pial correspond vertex for vertex.
- Inflation: 3000 steps of "move halfway to the neighbour mean", area-rescaled every 5 steps; sulc = summed normal
  displacement (+ = sulcus, mean 0). curv = smoothed mean curvature of white (+ = concave).
- The inflated surface is rescaled to the pial size and centre: a true area-preserving inflated brain is ~1.6× longer
  and bursts out of the head during the morph. Area is therefore not preserved (sulc and shape are fine).
- DKT per vertex: 5 samples white→pial, nearest same-hemisphere cortical voxel within 2 mm, majority vote, 3-pass mode
  filter, islands < 60 vertices merged. 0 = medial wall (~3.4%), shown grey.
- Decimation to 150k vertices/hemi: collapse groups computed once on the mid-thickness surface and applied to all three
  surfaces with group-mean positions. `fast_simplification.replay_simplification` on a different geometry folds
  hundreds of triangles; don't use it.
- Outputs: `data/derived/surfaces/{lh,rh}.{white,pial,inflated}.gii` + `.dkt.npy` (int16, label % 1000), `.sulc.npy`,
  `.curv.npy`. Previews: `pipeline/surf_previews.py` (needs pyvista + matplotlib, in `.venv` but not in
  requirements.txt).
- Under `python -I` a script's own folder isn't on `sys.path`; `build_surfaces.py` adds `pipeline/` itself.
- Untested shortcut: let the step-4 charm run finish, then `charm head --surfaces` in `work/`, to skip the second
  segmentation (~1 h).

**build_model.py**: `LABEL_SETS` meshes any `<stem>.nii.gz` + `.json` (`{"<label>": {"name", "color"}}`) in
`data/derived`, one mesh per label, with a triangle budget per set. `skip=` drops labels that duplicate other sources:
the optic nerves come from `orbit_labels` (eye → chiasm), the chiasm and tracts from HypVINN; the CC fornix, HypVINN's
third ventricle and pallidus, and CerebNet's white matter are skipped. `skull_parts` `replaces` the single skull. Groups
(title, info, default visibility, explode) live in the manifest, not in the JS.

**Morphology**: never use `binary_dilation` with big ball structuring elements on 0.5–0.75 mm grids (took >10 min).
Use distance transforms (`dil`/`ero`/`close_`/`open_` helpers pattern).

**Screenshots** (`pipeline/screenshots.py`, Playwright in `work/pwenv`, ~1 min): serves `docs/`, hides the UI,
opens each guided view, re-frames on the visible meshes (opacity > 0.3, optional focus filter) and writes
`docs/img/*.jpg` plus the 1200×630 link preview `docs/img/og.jpg` (title text overlay, head shifted right with
`setViewOffset`). The og/twitter meta tags in `index.html` point at the absolute GitHub Pages URL. Re-run it after
visual changes so the README matches the app. `window.viewer.THREE` is exposed for it.

**Rendering previews** headless: pyvista with `pv.OFF_SCREEN=True` → PNG (pyvista is in `.venv`). Good preview
scripts: `work/skull-orbit/render_labels.py` (any label map + LUT), `work/veins-ear/render.py`,
`work/fastsurfer-extras/previews/render.py`.

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
- Loading is progressive: `volume.bin` first (the default view), then meshes in the background (6 parallel fetches,
  3 retries each). Python's `http.server` drops connections under parallel load; the retries matter.
- Picking uses three-mesh-bvh (`computeBoundsTree` per geometry); `pick()` ignores meshes with opacity ≤ 0.3, so you
  can hover "through" the faint skin. three-mesh-bvh **reorders the index buffer** when building a BVH, so two BVHs
  must never share one index (the surface pick copy has its own).
- Surfaces: `type:"surface"` layers have a morph target (inflated, uint16 at 0.05 mm) driven by "Inflate brain", and
  per-vertex colours ("Regions" = DKT colours darkened in folds, "Folds" = sulc grey). Picking reports the DKT region of
  the nearest vertex.
- Hover tooltip = name + one-line description from `infoFor()`: REGION_INFO by DKT code, LOBE_INFO, INFO by id prefix,
  then `NAME_INFO` regexes by name (first match wins, so order specific → generic). When adding layers, add a
  plain-language text too.
- Guided views (`TOURS`) double as story steps. A `when` hides a view if its data is missing; `story` overrides the
  text in the story; `inStory:false` keeps it out of the story. Space pauses, ←/→ steps, Esc exits.
- **Caching:** GitHub Pages lets browsers cache files for 10 minutes. After v2 shipped, a browser combined a new
  manifest with a cached v1 `volume.bin` (same name, different encoding), and the volume rendered as a smeared box.
  So `manifest.json` is fetched with `cache: 'no-cache'`, and every data URL gets `?v=<FNV hash of the manifest>`.
  Any data change also changes the manifest (byte sizes), so URLs change with it. Keep this when touching the loader.
- Hidden browser tabs throttle `setTimeout` heavily; don't yield with `setTimeout(0)` in loading code.
- `window.viewer` exposes scene, camera, controls, items, startTour, story controls, setInflate, setSurfColors,
  setExplode, meshesReady, infoFor, pick(x, y) for debugging and screenshots.
- Testing in the Claude built-in browser: it can't open `file://` or localhost started from Bash, so use
  `preview_start` with a launch config (the preview tool may look for `.claude/launch.json` in the session's original
  folder). Screenshots can lag one frame behind; use `zoom` or re-screenshot before concluding something isn't drawn.
  Headless Playwright was used for timing and story runs.

## Ideas not done yet

- Real per-vertebra labels (C2–C7) and a hyoid would need a CT-like bone signal or a better bone model.
- A T2 for HypVINN (properly registered) could sharpen the hypothalamic subregion borders.
- Diffusion tractography is impossible with this study (trace/ADC only).
- The remaining vein artefacts near the sphenoid wing; tracing the sinuses from SWI instead of the charm atlas.
- Per-muscle eye labels are approximate; the inferior oblique isn't labelled.

Done in v2 (Oct 2026): corpus callosum, hypothalamus area, cerebellar lobules, skull parts, optic nerves + eye muscles,
automatic inner ear, vein artefact map + sinuses, Draco + progressive loading, story mode, cortical surfaces + inflation.
