"""Wrap the panoramic dental X-ray onto the jaw of the 3D model.

Usage: python build_dental_xray.py <dental_panoramic.jpg> <skull_parts.nii.gz> <out.npz> [--chin-z MM] [--preview <png>]

A panoramic X-ray (orthopantomogram) is a flattened picture of a curved band that follows the jaw: the tube and
detector orbit the head, and only a "focal trough" along the dental arch stays sharp. So the image has no depth,
but it can be bent back onto that curve. The curve and the scale come from the lower jaw (label 4) of skull_parts:

- the arch: medial line of the jaw seen from above (median radius along rays from a point inside the arch);
- vertical scale: bite height and bottom of the chin at the midline, matched to the image rows of the bite line
  and of the chin's lower border;
- horizontal scale: the midline (incisors) and the back edges of the rami, matched per side to image columns,
  because panoramic magnification differs horizontally and vertically.

The image landmarks (pixels) are for this particular X-ray and were read off the image by hand (IMG below).
charm's lower jaw stops ~13 mm short of the real bottom of the chin on this scan (the marrow continues below it on
the midline T1), so the chin height can be given with --chin-z (world z, mm, read off a midline T1 slice).
The output is a ribbon mesh in world RAS mm with UVs, for build_model.py.
"""
import argparse

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# Hand-picked landmarks on the cropped X-ray (x right, y down). Patient's left is on the image's right ("L" marker).
IMG = dict(mid_x=1010, bite_y=600, chin_y=1048, ramus_right_x=60, ramus_left_x=1940)


def arch_curve(mand_xy, center, step_deg=1.0, max_deg=125):
    """Median radius of the jaw's top-view footprint along rays from `center`. Angle 0 = forward (+y), + = patient right (+x)."""
    pts = []
    rel = mand_xy - center
    ang = np.degrees(np.arctan2(rel[:, 0], rel[:, 1]))
    rad = np.hypot(rel[:, 0], rel[:, 1])
    for a in np.arange(-max_deg, max_deg + 1e-6, step_deg):
        sel = np.abs(ang - a) < step_deg
        if sel.sum() < 30:
            continue
        r = np.median(rad[sel])
        t = np.radians(a)
        pts.append(center + r * np.array([np.sin(t), np.cos(t)]))
    pts = np.array(pts)
    pts = ndi.gaussian_filter1d(pts, 3, axis=0, mode="nearest")
    return pts  # ordered from patient left (-x) to patient right (+x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xray"); ap.add_argument("skull_parts"); ap.add_argument("out")
    ap.add_argument("--chin-z", type=float, help="world z of the bottom of the chin, if charm's jaw is too short")
    ap.add_argument("--preview")
    a = ap.parse_args()

    img = nib.load(a.skull_parts)
    lab = np.asarray(img.dataobj)
    P = nib.affines.apply_affine(img.affine, np.argwhere(lab == 4))   # lower jaw, world RAS mm
    x, y, z = P.T
    xm = 0.5 * (x[x > 20].mean() + x[x < -20].mean())                # midline between the two halves

    front = (np.abs(x - xm) < 3) & (y > y.max() - 12)
    z_bite, z_chin = z[front].max(), z[front].min()
    if a.chin_z is not None:
        z_chin = a.chin_z
    mm_v = (z_bite - z_chin) / (IMG["chin_y"] - IMG["bite_y"])        # vertical mm per pixel

    # Arch from above, using the tooth-bearing height band (bite down to mid-chin) plus the rami.
    band = (z < z_bite + 5) & (z > z_bite - 0.5 * (z_bite - z_chin))
    center = np.array([xm, y.min() + 0.35 * (y.max() - y.min())])
    curve = arch_curve(np.c_[x[band], y[band]], center)
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]                                      # arc length, patient left -> right
    front_pts = np.where(curve[:, 1] > center[1])[0]
    s_mid = s[front_pts[np.argmin(np.abs(curve[front_pts, 0] - xm))]]  # where the curve crosses the midline (front)

    # Image column for each arc position: patient right side = image left.
    s_rel = s - s_mid                                                 # < 0 on patient left, > 0 on patient right
    left_len, right_len = -s_rel.min(), s_rel.max()
    col = np.where(s_rel >= 0,
                   IMG["mid_x"] - s_rel / right_len * (IMG["mid_x"] - IMG["ramus_right_x"]),
                   IMG["mid_x"] + (-s_rel) / left_len * (IMG["ramus_left_x"] - IMG["mid_x"]))

    W, H = Image.open(a.xray).size
    rows = np.linspace(0, H - 1, 48)
    zs = z_bite + (IMG["bite_y"] - rows) * mm_v
    n_s, n_z = len(curve), len(rows)
    verts = np.zeros((n_z, n_s, 3), np.float32)
    verts[..., 0] = curve[None, :, 0]
    verts[..., 1] = curve[None, :, 1]
    verts[..., 2] = zs[:, None]
    uv = np.zeros((n_z, n_s, 2), np.float32)
    uv[..., 0] = (col / (W - 1))[None, :]
    uv[..., 1] = (1 - rows / (H - 1))[:, None]                        # WebGL textures: v = 0 at the bottom
    keep = (col >= 0) & (col <= W - 1)                                # drop curve ends beyond the image
    verts, uv = verts[:, keep], uv[:, keep]
    n_s = verts.shape[1]
    idx = np.arange(n_z * n_s).reshape(n_z, n_s)
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    faces = np.r_[quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]].astype(np.uint32)

    np.savez_compressed(a.out, verts=verts.reshape(-1, 3), uv=uv.reshape(-1, 2), faces=faces)
    print(f"jaw midline x={xm:.1f} bite z={z_bite:.1f} chin z={z_chin:.1f}  vertical {mm_v:.4f} mm/px"
          f"  arch {s[-1]:.0f} mm (left {left_len:.0f}, right {right_len:.0f})"
          f"  horizontal {right_len / (IMG['mid_x'] - IMG['ramus_right_x']):.4f}/{left_len / (IMG['ramus_left_x'] - IMG['mid_x']):.4f} mm/px"
          f"  ribbon {verts.reshape(-1, 3).shape[0]} verts")

    if a.preview:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].scatter(x[band][::50], y[band][::50], s=0.2, c="0.6"); ax[0].plot(curve[:, 0], curve[:, 1], "r-")
        ax[0].plot(*center, "bx"); ax[0].set_aspect("equal"); ax[0].set_title("jaw from above + arch")
        ax[1].scatter(x[::50], z[::50], s=0.2, c="0.6"); ax[1].axhline(z_bite, c="r"); ax[1].axhline(z_chin, c="b")
        ax[1].set_aspect("equal"); ax[1].set_title("jaw from the front: bite (red), chin (blue)")
        plt.tight_layout(); plt.savefig(a.preview, dpi=80)


if __name__ == "__main__":
    main()
