"""Preview PNGs for build_surfaces.py: pial by DKT parcel, inflated by sulcal depth, and white/pial
contours over T1 slices (with the FastSurfer cortical ribbon) to check alignment."""
from pathlib import Path

import numpy as np

# FreeSurfer colour table for the DKT codes
DKT_COLORS = {
    2: (125, 100, 160), 3: (100, 25, 0), 5: (220, 20, 100), 6: (220, 20, 10), 7: (180, 220, 140),
    8: (220, 60, 220), 9: (180, 40, 120), 10: (140, 20, 140), 11: (20, 30, 140), 12: (35, 75, 50),
    13: (225, 140, 140), 14: (200, 35, 75), 15: (160, 100, 50), 16: (20, 220, 60), 17: (60, 220, 60),
    18: (220, 180, 140), 19: (20, 100, 50), 20: (220, 60, 20), 21: (120, 100, 60), 22: (220, 20, 20),
    23: (220, 180, 220), 24: (60, 20, 220), 25: (160, 140, 180), 26: (80, 20, 140), 27: (75, 50, 125),
    28: (20, 220, 160), 29: (20, 180, 140), 30: (140, 220, 220), 31: (80, 160, 20), 34: (150, 150, 200),
    35: (255, 192, 32),
}


def _lut():
    lut = np.full((36, 3), 90, np.uint8)  # 0 / unused = dark grey
    for k, c in DKT_COLORS.items():
        lut[k] = c
    return lut


def _polydata(v, f):
    import pyvista as pv
    return pv.PolyData(np.asarray(v, np.float32), np.c_[np.full(len(f), 3), f].ravel())


def _views(out, hemis, title, offset_inflated=False):
    """2x3 grid: lateral/medial per hemisphere + dorsal + ventral of both."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, shape=(2, 3), window_size=(1800, 1100), border=False)
    meshes = {}
    for h, (v, f, rgb) in hemis.items():
        m = _polydata(v, f)
        m["rgb"] = rgb
        meshes[h] = m
    if offset_inflated:  # pull the inflated hemispheres apart so they don't overlap
        gap = 4.0
        lx, rx = meshes["lh"].bounds, meshes["rh"].bounds
        cx = 0.5 * (meshes["lh"].center[0] + meshes["rh"].center[0])
        meshes["lh"].translate((cx - gap / 2 - lx[1], 0, 0), inplace=True)
        meshes["rh"].translate((cx + gap / 2 - rx[0], 0, 0), inplace=True)
    both_c = np.mean([m.center for m in meshes.values()], axis=0)

    def cam(direction, up, center, dist=420):
        return [tuple(center + np.asarray(direction) * dist), tuple(center), up]

    panels = [
        ((0, 0), ["lh"], (-1, 0, 0), (0, 0, 1), "lh lateral"),
        ((0, 1), ["rh"], (1, 0, 0), (0, 0, 1), "rh lateral"),
        ((0, 2), ["lh", "rh"], (0, 0, 1), (0, 1, 0), "dorsal"),
        ((1, 0), ["lh"], (1, 0, 0), (0, 0, 1), "lh medial"),
        ((1, 1), ["rh"], (-1, 0, 0), (0, 0, 1), "rh medial"),
        ((1, 2), ["lh", "rh"], (0, 0, -1), (0, 1, 0), "ventral"),
    ]
    for (r, c), hs, d, up, label in panels:
        p.subplot(r, c)
        for h in hs:
            p.add_mesh(meshes[h], scalars="rgb", rgb=True, smooth_shading=True, specular=0.15)
        center = both_c if len(hs) == 2 else np.asarray(meshes[hs[0]].center)
        p.camera_position = cam(d, up, center)
        p.camera.parallel_projection = True
        p.reset_camera()
        p.add_text(label, font_size=10)
    p.subplot(0, 0)
    p.add_text(title, position="lower_left", font_size=9)
    p.screenshot(str(out))
    p.close()


def _slice_mesh(v, f, axis, c):
    s = v[:, axis] - c
    sf = s[f]
    cross = (sf.min(1) < 0) & (sf.max(1) > 0)
    f2, sf2 = f[cross], sf[cross]
    segs = np.zeros((len(f2), 2, 3))
    cnt = np.zeros(len(f2), int)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        sa, sb = sf2[:, a], sf2[:, b]
        m = sa * sb < 0
        t = sa[m] / (sa[m] - sb[m])
        pa, pb = v[f2[m, a]], v[f2[m, b]]
        idx = np.where(m)[0]
        segs[idx, np.minimum(cnt[idx], 1)] = pa + t[:, None] * (pb - pa)
        cnt[idx] += 1
    return segs[cnt == 2]


def _sample(img, axis, c, extent, step=0.4, order=1):
    from scipy import ndimage as ndi
    other = [k for k in range(3) if k != axis]
    u = np.arange(extent[0][0], extent[0][1], step)
    w = np.arange(extent[1][0], extent[1][1], step)
    U, Wg = np.meshgrid(u, w)
    xyz = np.zeros(U.shape + (3,))
    xyz[..., other[0]], xyz[..., other[1]], xyz[..., axis] = U, Wg, c
    ijk = (xyz - img.affine[:3, 3]) @ np.linalg.inv(img.affine[:3, :3]).T
    data = np.asarray(img.dataobj)
    return ndi.map_coordinates(data, ijk.reshape(-1, 3).T, order=order, mode="constant").reshape(U.shape)


def _overlay(out, surfs, aparc, t1, centre, zoom_box=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    names = ["sagittal (x = %.0f)", "coronal (y = %.0f)", "axial (z = %.0f)"]
    allv = np.concatenate([surfs[h]["pial"] for h in surfs])
    lo, hi = allv.min(0) - 8, allv.max(0) + 8
    fig, axes = plt.subplots(1, 3, figsize=(21, 8), facecolor="black")
    for axis, ax in enumerate(axes):
        c = centre[axis]
        other = [k for k in range(3) if k != axis]
        ext = [(lo[other[0]], hi[other[0]]), (lo[other[1]], hi[other[1]])]
        if zoom_box is not None:
            ext = [zoom_box[other[0]], zoom_box[other[1]]]
        img = _sample(t1, axis, c, ext)
        lab = _sample(aparc, axis, c, ext, order=0)
        extent = (ext[0][0], ext[0][1], ext[1][0], ext[1][1])
        vmax = np.percentile(img[img > 0], 99.5) if (img > 0).any() else 1
        ax.imshow(img, cmap="gray", origin="lower", extent=extent, vmin=0, vmax=vmax, interpolation="bilinear")
        ctx = np.ma.masked_where(lab < 1000, np.ones_like(lab, float))
        ax.imshow(ctx, cmap="cool", origin="lower", extent=extent, alpha=0.25, interpolation="nearest")
        for h in surfs:
            for kind, col in (("white", "yellow"), ("pial", "red")):
                segs = _slice_mesh(surfs[h][kind], surfs[h]["faces"], axis, c)[:, :, other]
                ax.add_collection(LineCollection(segs, colors=col, linewidths=0.8))
        ax.set_xlim(ext[0]); ax.set_ylim(ext[1]); ax.set_aspect("equal")
        ax.set_title(names[axis] % c + "  — yellow: white, red: pial, cyan tint: FastSurfer cortex",
                     color="white", fontsize=10)
        ax.tick_params(colors="gray")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor="black")
    plt.close(fig)


def render_all(surfs, out_dir, aparc, t1=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lut = _lut()
    _views(out_dir / "pial_dkt.png",
           {h: (s["pial"], s["faces"], lut[s["dkt"]]) for h, s in surfs.items()},
           "pial surface, DKT parcels (FreeSurfer colours, dark grey = unlabelled)")
    _views(out_dir / "white_dkt.png",
           {h: (s["white"], s["faces"], lut[s["dkt"]]) for h, s in surfs.items()},
           "white surface, DKT parcels")

    def grey(x, lim):
        g = np.clip(0.5 - 0.5 * x / lim, 0, 1) * 0.55 + 0.3  # sulci dark, gyri light
        return (np.repeat(g[:, None], 3, 1) * 255).astype(np.uint8)

    _views(out_dir / "inflated_sulc.png",
           {h: (s["inflated"], s["faces"], grey(s["sulc"], np.percentile(np.abs(s["sulc"]), 90)))
            for h, s in surfs.items()},
           "inflated surface, sulcal depth (dark = sulci)", offset_inflated=True)
    _views(out_dir / "inflated_dkt.png",
           {h: (s["inflated"], s["faces"], lut[s["dkt"]]) for h, s in surfs.items()},
           "inflated surface, DKT parcels", offset_inflated=True)
    _views(out_dir / "white_curv.png",
           {h: (s["white"], s["faces"], grey(np.sign(s["curv"]) * 1.0, 1.0)) for h, s in surfs.items()},
           "white surface, binary curvature (dark = concave)")
    if t1 is not None:
        allv = np.concatenate([s["white"] for s in surfs.values()])
        centre = allv.mean(0)
        _overlay(out_dir / "overlay_t1.png", surfs, aparc, t1, centre)
        # zoomed detail on the left hemisphere, a bit lateral
        c2 = centre + np.array([-35.0, 0, 15])
        box = [(c2[0] - 30, c2[0] + 30), (c2[1] - 30, c2[1] + 30), (c2[2] - 30, c2[2] + 30)]
        _overlay(out_dir / "overlay_t1_zoom.png", surfs, aparc, t1, c2, zoom_box=box)
    print(f"previews -> {out_dir}")
