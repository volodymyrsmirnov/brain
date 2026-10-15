"""Cortical surfaces per hemisphere: white, pial and inflated (same vertices) + per-vertex DKT label,
sulcal depth and curvature.

Input surfaces come from SimNIBS charm (`--segment` continues into TopoFit surface reconstruction,
brainnet package; no FreeSurfer licence needed). TopoFit deforms an icosahedral template mesh, so
white and pial already share one topology (genus 0, 245,762 vertices per hemisphere). This script:

1. inflates the white surface (FreeSurfer-style: iterative neighbour averaging with the total area
   rescaled back to the white area); sulcal depth = displacement along the normal accumulated during
   inflation (positive = deep/sulcal, zero-mean), as in FreeSurfer's ?h.sulc;
2. computes mean curvature of the white surface (cotangent Laplacian; positive = concave/sulcal);
3. labels each vertex with its DKT parcel: points along the white->pial segment (inside the cortical
   ribbon) are matched to the nearest cortical voxel of that hemisphere in the FastSurfer
   aparc.DKTatlas+aseg volume; majority vote, mesh majority filter, small islands merged.
   0 = no cortex nearby (medial wall cut through corpus callosum / thalamus / brainstem);
4. decimates one shared topology (quadric collapses computed on the mid-thickness surface, replayed
   by vertex clusters with cluster-mean positions) so all three surfaces keep a 1:1 correspondence.

Output (all vertices in world/scanner RAS mm, the space of the NIfTI affines):
  {lh,rh}.{white,pial,inflated}.gii   GIfTI pointset float32 + triangle int32 (same faces per hemi)
  {lh,rh}.dkt.npy    int16  DKT code = aparc label % 1000 (2..35), 0 = unknown / medial wall
  {lh,rh}.sulc.npy   float32 mm, + = sulcus, - = gyrus (zero-mean)
  {lh,rh}.curv.npy   float32 1/mm, mean curvature of white, + = concave (sulcal fundus)
The inflated surface is scaled to the pial surface's RMS radius and centred on the pial centroid (so a
pial<->inflated morph stays in place); the two inflated hemispheres overlap at the midline, so a viewer
must offset them (e.g. lh along -x, rh along +x).

Usage:
  python build_surfaces.py <charm_surfaces_dir> <aparc.DKTatlas+aseg.mgz> <out_dir>
                           [--t1 t1.nii.gz --previews DIR] [--target-verts 150000]
"""
import argparse
import sys
import time
from pathlib import Path

import fast_simplification
import nibabel as nib
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HEMIS = {"lh": (1000, "CortexLeft"), "rh": (2000, "CortexRight")}
WM_LABELS = (2, 41)  # aseg cerebral white matter


# ---------------------------------------------------------------- mesh helpers
def load_gii(path):
    g = nib.load(str(path))
    arrs = {nib.nifti1.intent_codes.niistring[d.intent]: d.data for d in g.darrays}
    return arrs["NIFTI_INTENT_POINTSET"].astype(np.float64), arrs["NIFTI_INTENT_TRIANGLE"].astype(np.int64)


def save_gii(path, v, f, structure, kind):
    meta = {"AnatomicalStructurePrimary": structure,
            "GeometricType": "Inflated" if kind == "inflated" else "Anatomical"}
    if kind in ("white", "pial"):
        meta["AnatomicalStructureSecondary"] = {"white": "GrayWhite", "pial": "Pial"}[kind]
    scanner = nib.gifti.GiftiCoordSystem(dataspace=3, xformspace=3, xform=np.eye(4))  # NIFTI_XFORM_SCANNER_ANAT
    pts = nib.gifti.GiftiDataArray(v.astype(np.float32), intent="NIFTI_INTENT_POINTSET",
                                   datatype="NIFTI_TYPE_FLOAT32", coordsys=scanner,
                                   meta=nib.gifti.GiftiMetaData(meta))
    tri = nib.gifti.GiftiDataArray(f.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE",
                                   datatype="NIFTI_TYPE_INT32", coordsys=None)
    nib.save(nib.gifti.GiftiImage(darrays=[pts, tri]), str(path))


def adjacency(f, n):
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    A.data[:] = 1.0
    return A


def incidence(f, n):
    """(n_vertices x n_faces) sparse matrix: sum face quantities onto their corners."""
    m = len(f)
    return sparse.csr_matrix((np.ones(3 * m), (f.T.ravel(), np.tile(np.arange(m), 3))), shape=(n, m))


def face_cross(v, f):
    return np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])


def vertex_normals(v, f, inc):
    n = inc @ face_cross(v, f)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def total_area(v, f):
    return 0.5 * np.linalg.norm(face_cross(v, f), axis=1).sum()


def mean_curvature(v, f, inc):
    """Mean curvature via the cotangent Laplacian; + for convex (sphere with outward normals)."""
    n = len(v)
    rows, cols, vals = [], [], []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):  # angle at a, opposite edge b-c
        e1, e2 = v[f[:, b]] - v[f[:, a]], v[f[:, c]] - v[f[:, a]]
        cot = (e1 * e2).sum(1) / np.maximum(np.linalg.norm(np.cross(e1, e2), axis=1), 1e-12)
        cot = np.clip(cot, -10, 10)
        rows += [f[:, b], f[:, c]]
        cols += [f[:, c], f[:, b]]
        vals += [0.5 * cot, 0.5 * cot]
    W = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    L = sparse.diags(np.asarray(W.sum(1)).ravel()) - W
    area = inc @ (0.5 * np.linalg.norm(face_cross(v, f), axis=1)) / 3.0
    Hn = (L @ v) / (2 * area[:, None])  # = H * n (mean curvature normal)
    return (Hn * vertex_normals(v, f, inc)).sum(1)


def smooth_values(x, A, iters):
    S = sparse.diags(1.0 / (np.asarray(A.sum(1)).ravel() + 1)) @ (A + sparse.eye(A.shape[0]))
    for _ in range(iters):
        x = S @ x
    return x


# ---------------------------------------------------------------- inflation
def inflate(v, f, A, inc, n_iter, step=0.5, every=5):
    """FreeSurfer-like inflation: neighbour averaging, total area rescaled to the input's every few
    steps. Returns inflated vertices and sulc = displacement along the normal accumulated during the
    averaging (normals refreshed every `every` steps; the rescaling is not counted), zero-mean."""
    W = sparse.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A
    area0 = total_area(v, f)
    x = v.copy()
    sulc = np.zeros(len(v))
    for it in range(0, n_iter, every):
        nrm = vertex_normals(x, f, inc)
        x0 = x.copy()
        for _ in range(min(every, n_iter - it)):
            x += step * (W @ x - x)
        sulc += ((x - x0) * nrm).sum(1)
        c = x.mean(0)
        x = c + (x - c) * np.sqrt(area0 / total_area(x, f))
    return x, sulc - sulc.mean()


# ---------------------------------------------------------------- DKT labels
def label_vertices(white, pial, f, A, aparc_img, base, max_dist=2.0, fracs=(0.1, 0.3, 0.5, 0.7, 0.9)):
    data = np.asarray(aparc_img.dataobj)
    idx = np.argwhere((data > base) & (data < base + 100))
    codes = (data[tuple(idx.T)] - base).astype(np.int64)
    pts = idx @ aparc_img.affine[:3, :3].T + aparc_img.affine[:3, 3]
    tree = cKDTree(pts)
    n = len(white)
    votes = np.zeros((n, 36))
    for t in fracs:
        d, j = tree.query(white + t * (pial - white), distance_upper_bound=max_dist, workers=6)
        ok = np.isfinite(d)
        lab = np.zeros(n, np.int64)
        lab[ok] = codes[j[ok]]
        votes[np.arange(n), lab] += 1.0
    votes[:, 0] *= 0.9  # ties go to cortex
    lab = votes.argmax(1)
    # mesh majority filter (vertex + its neighbours)
    M = A + sparse.eye(n)
    for _ in range(3):
        onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), lab)), shape=(n, 36))
        cnt = (M @ onehot).toarray()
        cnt[np.arange(n), lab] += 0.5  # keep own label on ties
        lab = cnt.argmax(1)
    # merge small islands (< 60 vertices ~ 25 mm^2) into the surrounding label
    i, j = A.nonzero()
    same = lab[i] == lab[j]
    G = sparse.csr_matrix((np.ones(same.sum()), (i[same], j[same])), shape=(n, n))
    ncomp, comp = connected_components(G, directed=False)
    size = np.bincount(comp)
    lab = np.where(size[comp] < 60, -1, lab)
    while (lab < 0).any():
        known = lab >= 0
        onehot = sparse.csr_matrix((np.ones(known.sum()), (np.where(known)[0], lab[known])), shape=(n, 36))
        cnt = (A @ onehot).toarray()
        fill = (~known) & (cnt.sum(1) > 0)
        if not fill.any():
            lab[lab < 0] = 0
            break
        lab[fill] = cnt[fill].argmax(1)
    return lab.astype(np.int16)


def sample_labels(img, xyz):
    data = np.asarray(img.dataobj)
    ijk = np.rint((xyz - img.affine[:3, 3]) @ np.linalg.inv(img.affine[:3, :3]).T).astype(int)
    ijk = np.clip(ijk, 0, np.array(data.shape) - 1)
    return data[tuple(ijk.T)]


# ---------------------------------------------------------------- decimation with correspondence
def decimate_shared(driver, f, target_verts):
    """Quadric edge collapses computed once on `driver` (fast-simplification). Returns `cluster`
    (original vertex -> decimated vertex) and the decimated faces. Every surface sharing f is reduced
    with the same clusters (cluster-mean positions), so white/pial/inflated keep 1:1 correspondence.
    (Replaying the collapses with each surface's own quadric positions produced folded triangles on
    the other surfaces; cluster means don't.)"""
    n = len(driver)
    if n <= target_verts:
        return np.arange(n), f
    _, _, collapses = fast_simplification.simplify(driver.astype(np.float32), f.astype(np.int32),
                                                   target_count=2 * target_verts - 4, return_collapses=True)
    _, fd, cluster = fast_simplification.replay_simplification(driver.astype(np.float32), f.astype(np.int32),
                                                               collapses)
    return cluster.astype(np.int64), fd.astype(np.int64)


def reduce_mean(x, cluster):
    m = cluster.max() + 1
    cnt = np.bincount(cluster, minlength=m).astype(np.float64)
    if x.ndim == 1:
        return np.bincount(cluster, weights=x, minlength=m) / cnt
    return np.stack([np.bincount(cluster, weights=x[:, k], minlength=m) for k in range(x.shape[1])], 1) / cnt[:, None]


def reduce_mode(lab, cluster, n_classes=36):
    m = cluster.max() + 1
    cnt = sparse.csr_matrix((np.ones(len(lab)), (cluster, lab)), shape=(m, n_classes)).toarray()
    return cnt.argmax(1).astype(lab.dtype)


def flipped_faces(v_full, f_full, cluster, vd, fd, inc_full):
    """Decimated faces whose normal opposes the full-resolution normals of their corner clusters."""
    vn = reduce_mean(vertex_normals(v_full, f_full, inc_full), cluster)
    fn = face_cross(vd, fd)
    return int(((fn * vn[fd].sum(1)).sum(1) <= 0).sum())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("surf_dir", type=Path, help="charm m2m_<sub>/surfaces")
    ap.add_argument("aparc", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--t1", type=Path, help="T1 for the overlay preview")
    ap.add_argument("--previews", type=Path, help="write preview PNGs here")
    ap.add_argument("--target-verts", type=int, default=150_000)
    ap.add_argument("--inflate-iter", type=int, default=3000)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    aparc = nib.load(str(a.aparc))
    out = {}
    for hemi, (base, structure) in HEMIS.items():
        t0 = time.time()
        white, f = load_gii(a.surf_dir / f"{hemi}.white.gii")
        pial, fp = load_gii(a.surf_dir / f"{hemi}.pial.gii")
        assert np.array_equal(f, fp) and white.shape == pial.shape, "white/pial topology differs"
        n = len(white)
        A, inc = adjacency(f, n), incidence(f, n)
        euler = n - len(sparse.triu(A).nonzero()[0]) + len(f)
        print(f"{hemi}: {n} vertices, {len(f)} faces, Euler characteristic {euler}")

        infl, sulc = inflate(white, f, A, inc, a.inflate_iter)
        # FreeSurfer keeps the white surface's total area, which makes the inflated brain ~1.6x longer
        # than the real one. For a pial<->inflated morph in the head we rescale it to the pial's RMS
        # radius and centre it on the pial centroid (shape and sulc are unaffected).
        cp, ci = pial.mean(0), infl.mean(0)
        infl = cp + (infl - ci) * np.sqrt(((pial - cp) ** 2).sum(1).mean() / ((infl - ci) ** 2).sum(1).mean())
        curv = -smooth_values(mean_curvature(white, f, inc), A, 2)  # + = concave (FreeSurfer-like)
        dkt = label_vertices(white, pial, f, A, aparc, base)

        # alignment check against the FastSurfer volume
        nw = vertex_normals(white, f, inc)
        mid = 0.5 * (white + pial)
        in_wm = np.isin(sample_labels(aparc, white - 0.5 * nw), WM_LABELS).mean()
        mid_ctx = np.isin(sample_labels(aparc, mid) - base, np.arange(1, 36)).mean()
        out_wm = np.isin(sample_labels(aparc, pial + 0.5 * vertex_normals(pial, f, inc)), WM_LABELS).mean()
        thick = np.linalg.norm(pial - white, axis=1)
        print(f"  check: white-0.5mm in WM {in_wm:.1%} | mid-thickness in DKT cortex {mid_ctx:.1%} | "
              f"pial+0.5mm in WM {out_wm:.1%} | thickness median {np.median(thick):.2f} mm | "
              f"unlabelled {np.mean(dkt == 0):.1%}")

        cluster, fd = decimate_shared(mid, f, a.target_verts)
        surfs = {k: reduce_mean(s, cluster) for k, s in (("white", white), ("pial", pial), ("inflated", infl))}
        flips = {k: flipped_faces(s, f, cluster, surfs[k], fd, inc) for k, s in
                 (("white", white), ("pial", pial), ("inflated", infl))}
        print(f"  decimated to {len(surfs['white'])} vertices, {len(fd)} faces; flipped faces {flips}")
        dkt_d = reduce_mode(dkt, cluster)
        sulc_d, curv_d = reduce_mean(sulc, cluster), reduce_mean(curv, cluster)
        for kind, v in surfs.items():
            save_gii(a.out_dir / f"{hemi}.{kind}.gii", v, fd, structure, kind)
        np.save(a.out_dir / f"{hemi}.dkt.npy", dkt_d)
        np.save(a.out_dir / f"{hemi}.sulc.npy", sulc_d.astype(np.float32))
        np.save(a.out_dir / f"{hemi}.curv.npy", curv_d.astype(np.float32))
        out[hemi] = dict(faces=fd, dkt=dkt_d, sulc=sulc_d, curv=curv_d, **surfs)
        print(f"  sulc range {np.percentile(sulc, [1, 99]).round(1)} mm, curv range "
              f"{np.percentile(curv, [1, 99]).round(2)} 1/mm, {time.time() - t0:.0f} s")

    if a.previews:
        sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` drops the script dir
        import surf_previews
        surf_previews.render_all(out, a.previews, aparc, nib.load(str(a.t1)) if a.t1 else None)


if __name__ == "__main__":
    main()
