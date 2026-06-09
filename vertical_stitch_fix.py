"""
Re-do vertical stitch from saved row PNGs using translation-only H.

Root cause: SIFT homographies between rows carried a 0.75x scale factor (due
to row 3 being inflated by a 4° deskew). Composing these scales compounds to
0.62x by row 4 — causing it to appear as a narrow wedge in the canvas.

Fix: extract only tx, ty from the SIFT H and force a pure-translation matrix.
This stacks rows at their native scale, positioned by the overlap translation.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from scipy.ndimage import uniform_filter1d

OUTPUT_DIR = "/home/user/yurumemo-johokyoku/map_output/"
ROWS_DIR   = OUTPUT_DIR + "rows_v6/"
MAX_DIM    = 30000   # allow wider canvas this time


def sift_match(gray_a, gray_b, ratio=0.67, nfeat=8000, max_dim=4000):
    ha, wa = gray_a.shape; hb, wb = gray_b.shape
    s = min(1.0, max_dim / max(ha, wa, hb, wb))
    if s < 0.99:
        g_a = cv2.resize(gray_a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        g_b = cv2.resize(gray_b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    else:
        g_a, g_b = gray_a, gray_b
    sift = cv2.SIFT_create(nfeatures=nfeat, contrastThreshold=0.01, edgeThreshold=20)
    kp_a, des_a = sift.detectAndCompute(g_a, None)
    kp_b, des_b = sift.detectAndCompute(g_b, None)
    print(f"    KP: a={len(kp_a)}, b={len(kp_b)}")
    if des_a is None or des_b is None or len(kp_a)<8 or len(kp_b)<8: return None, None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=150))
    raw = flann.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in raw if m.distance < ratio*n.distance]
    print(f"    Good matches: {len(good)}")
    if len(good) < 8: return None, None
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]) / s
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]) / s
    return pts_a, pts_b


def get_translation(img_ref, img_mov, overlap_frac=0.20):
    """
    Find the vertical stitch overlap (ty, tx) using strip SIFT.
    Returns a PURE TRANSLATION matrix H = [[1,0,tx],[0,1,ty],[0,0,1]].
    Scale/rotation from SIFT is discarded to prevent compounding distortion.
    """
    hr, wr = img_ref.shape[:2]; hm, wm = img_mov.shape[:2]
    sw = int(min(hr, hm) * overlap_frac)
    yr = hr - sw
    ref_strip = img_ref[yr:, :]
    mov_strip = img_mov[:sw, :]

    gray_r = cv2.cvtColor(ref_strip, cv2.COLOR_BGR2GRAY)
    gray_m = cv2.cvtColor(mov_strip, cv2.COLOR_BGR2GRAY)

    pts_r, pts_m = sift_match(gray_r, gray_m)
    if pts_r is not None:
        pts_r_full = pts_r + np.float32([0, yr])
        pts_m_full = pts_m.copy()   # already in strip coords (no offset for mov)
        H, mask = cv2.findHomography(pts_m_full, pts_r_full, cv2.RANSAC, 4.0,
                                      maxIters=5000, confidence=0.999)
        if H is not None and mask is not None and mask.sum() >= 10:
            tx, ty = H[0,2], H[1,2]
            n = int(mask.sum())
            print(f"    Inliers: {n}  tx={tx:.0f}  ty={ty:.0f}")
            # Validate translation bounds
            if hr*0.15 < ty < hr*1.1 and abs(tx) < wr*0.4:
                return np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)
            else:
                print(f"    Translation out of range, using fallback")

    # Fallback: centre-aligned with estimated ty
    tx = (wr - wm) / 2
    ty = hr * (1 - overlap_frac)
    print(f"    Fallback: tx={tx:.0f}  ty={ty:.0f}")
    return np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)


def make_content_mask(img, white_thresh=252):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nw = (gray < white_thresh).astype(np.uint8) * 255
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    filled = cv2.morphologyEx(nw, cv2.MORPH_CLOSE, k1)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 200))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k2)
    return filled


def find_best_seam_row(canvas, warped, overlap_mask):
    """Find horizontal seam row with minimum combined edge density."""
    ov_rows = np.where(overlap_mask.any(1))[0]
    if len(ov_rows) < 3:
        return int(ov_rows.mean()) if len(ov_rows) else 0
    r0, r1 = ov_rows[0], ov_rows[-1]
    ec = cv2.Canny(cv2.cvtColor(canvas[r0:r1+1,:],cv2.COLOR_BGR2GRAY),30,90).sum(1).astype(float)
    ew = cv2.Canny(cv2.cvtColor(warped[r0:r1+1,:],cv2.COLOR_BGR2GRAY),30,90).sum(1).astype(float)
    return int(uniform_filter1d(ec+ew, 40).argmin()) + r0


def feather_h(canvas, warped, seam, overlap_mask, px=12):
    """Feather blend at horizontal seam."""
    hc, wc = canvas.shape[:2]
    out = canvas.copy()
    y0 = max(0, seam-px); y1 = min(hc, seam+px+1)
    for y in range(y0, y1):
        row = overlap_mask[y, :]
        if not row.any(): continue
        a = (y-y0)/(y1-y0-1) if y1 > y0+1 else 0.5
        out[y, row] = ((1-a)*canvas[y,row].astype(np.float32)
                       + a*warped[y,row].astype(np.float32)).astype(np.uint8)
    bm = overlap_mask.copy(); bm[:y1, :] = False
    out[bm] = warped[bm]
    return out


def warp_onto_v(canvas, canvas_mask, new_img, H):
    """Place new_img onto canvas via translation H, with horizontal seam."""
    hc, wc = canvas.shape[:2]
    warped = cv2.warpPerspective(new_img, H, (wc, hc), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cm = make_content_mask(new_img)
    warped_valid = cv2.warpPerspective(cm, H, (wc, hc), flags=cv2.INTER_NEAREST) > 127

    only_new = warped_valid & (canvas_mask == 0)
    overlap  = warped_valid & (canvas_mask  > 0)

    out = canvas.copy(); om = canvas_mask.copy()
    out[only_new] = warped[only_new]; om[only_new] = 255

    if overlap.any():
        seam = find_best_seam_row(canvas, warped, overlap)
        out = feather_h(out, warped, seam, overlap, px=12)

    return out, om


def build_vertical_canvas(row_imgs, Hs, max_dim=MAX_DIM):
    # Estimate canvas size
    all_c = []
    for img, H in zip(row_imgs, Hs):
        h, w = img.shape[:2]
        c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        all_c.append(cv2.perspectiveTransform(c, H).reshape(4,2))
    pts = np.vstack(all_c)
    x_min = int(np.floor(pts[:,0].min())); x_max = int(np.ceil(pts[:,0].max()))
    y_min = int(np.floor(pts[:,1].min())); y_max = int(np.ceil(pts[:,1].max()))
    est_w = x_max - x_min + 1; est_h = y_max - y_min + 1
    print(f"  Estimated canvas: {est_w}×{est_h}")

    s = 1.0
    if est_w > max_dim or est_h > max_dim:
        s = max_dim / max(est_w, est_h)
        print(f"  Scale-down: {s:.3f}×")
        row_imgs = [cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    for img in row_imgs]
        S  = np.array([[s,0,0],[0,s,0],[0,0,1]], np.float64)
        Si = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]], np.float64)
        Hs = [S @ H @ Si for H in Hs]
        all_c = []
        for img, H in zip(row_imgs, Hs):
            h, w = img.shape[:2]
            c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
            all_c.append(cv2.perspectiveTransform(c, H).reshape(4,2))
        pts = np.vstack(all_c)
        x_min = int(np.floor(pts[:,0].min())); x_max = int(np.ceil(pts[:,0].max()))
        y_min = int(np.floor(pts[:,1].min())); y_max = int(np.ceil(pts[:,1].max()))
        est_w = x_max - x_min + 1; est_h = y_max - y_min + 1

    tx, ty = max(0, -x_min), max(0, -y_min)
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)
    cw, ch = est_w, est_h
    print(f"  Building canvas: {cw}×{ch}  scale={s:.3f}")

    canvas = np.zeros((ch, cw, 3), np.uint8)
    canvas_mask = np.zeros((ch, cw), np.uint8)
    for i, (img, H) in enumerate(zip(row_imgs, Hs)):
        canvas, canvas_mask = warp_onto_v(canvas, canvas_mask, img, T @ H)
        print(f"  Placed row {i+1}/{len(row_imgs)}")

    rows_idx = np.where(canvas_mask.any(1))[0]
    cols_idx = np.where(canvas_mask.any(0))[0]
    if len(rows_idx) and len(cols_idx):
        canvas = canvas[rows_idx[0]:rows_idx[-1]+1, cols_idx[0]:cols_idx[-1]+1]
    return canvas


if __name__ == "__main__":
    # Load saved row images
    row_imgs = []
    for i in range(1, 5):
        p = ROWS_DIR + f"row_{i}.png"
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        print(f"Row {i}: {img.shape[1]}×{img.shape[0]}")
        row_imgs.append(img)

    # Compute translation-only Hs for vertical stitch
    Hs = [np.eye(3, dtype=np.float64)]
    for i in range(1, len(row_imgs)):
        print(f"\n  Vertical align row {i+1}→{i} (translation only)...")
        H_trans = get_translation(row_imgs[i-1], row_imgs[i], overlap_frac=0.20)
        Hs.append(Hs[-1] @ H_trans)
        print(f"  Chain tx={Hs[-1][0,2]:.0f}  ty={Hs[-1][1,2]:.0f}")

    print(f"\n  Building final canvas...")
    final = build_vertical_canvas(row_imgs, Hs)
    out = OUTPUT_DIR + "final_map_v2.png"
    cv2.imwrite(out, final)
    print(f"\nFINAL MAP v2: {out}  ({final.shape[1]}×{final.shape[0]})")

    # Quick preview
    scale = min(1.0, 1800 / final.shape[1])
    preview = cv2.resize(final, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(OUTPUT_DIR + "final_map_v2_preview.jpg", preview,
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Preview: {preview.shape[1]}×{preview.shape[0]}")
