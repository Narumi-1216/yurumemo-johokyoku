#!/usr/bin/env python3
"""
Reference-guided high-resolution map stitching.

Uses the wide-angle reference photo as geometric base, then registers each of
the 12 close-up photos onto a 3× scaled canvas for super-resolution output.

Strategy:
  1. Perspective-correct the reference wide-angle photo
  2. For each close-up: SIFT match vs corrected reference → H (cu → ref coords)
  3. Warp each close-up onto canvas at SCALE_OUT × reference size
  4. Seam-cut + feather blend (close-ups always win over reference quality)
"""

import cv2
import numpy as np
import os
from pathlib import Path
from scipy.ndimage import uniform_filter1d

UPLOAD_DIR    = "/root/.claude/uploads/1b9fe45d-a299-5f9c-9937-d45d27c12f22/"
OUTPUT_DIR    = "/home/user/yurumemo-johokyoku/map_output/"
REF_FILE      = UPLOAD_DIR + "12b92caa-IMG_6832.jpeg"
SCALE_OUT     = 3.3    # output canvas = SCALE_OUT × corrected_reference_dims
                       # ≈ native close-up resolution (zoom factor ~3.3×)
MATCH_DIM_REF = 4000   # reference max-dim for SIFT (→ s_r ≈ 0.73)
MATCH_DIM_CU  = 2000   # close-up max-dim for SIFT  (→ s_c ≈ 0.35)
                       # scale ratio ~1.6× in overlap — keeps SIFT reliable

CLOSE_UPS = [
    (UPLOAD_DIR + "69a424a3-IMG_6837.jpeg",  "r1_1"),
    (UPLOAD_DIR + "95841777-IMG_6838.jpeg",  "r1_2"),
    (UPLOAD_DIR + "d9b6fffb-IMG_6839.jpeg",  "r1_3"),
    (UPLOAD_DIR + "e316d2e3-IMG_6842.jpeg",  "r2_1"),
    (UPLOAD_DIR + "fd9b9afe-IMG_6843.jpeg",  "r2_2"),
    (UPLOAD_DIR + "213fbf53-IMG_6844.jpeg",  "r2_3"),
    (UPLOAD_DIR + "9a0ccada-IMG_6845.jpeg",  "r3_1"),
    (UPLOAD_DIR + "51c8fcf1-IMG_6846.jpeg",  "r3_2"),
    (UPLOAD_DIR + "801c9748-IMG_6847.jpeg",  "r3_3"),
    (UPLOAD_DIR + "c816dae8-IMG_6848.jpeg",  "r4_1"),
    (UPLOAD_DIR + "d2f6ad4d-IMG_6849.jpeg",  "r4_2"),
    (UPLOAD_DIR + "7ae2ed01-IMG_6850.jpeg",  "r4_3"),
    (UPLOAD_DIR + "ea45a7de-IMG_7029.jpeg",  "r5_1"),
    (UPLOAD_DIR + "4eb93d9e-IMG_7028.jpeg",  "r5_2"),
]


# ── Perspective correction ──────────────────────────────────────────────── #

def order_pts(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def correct_perspective_ref(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect bright (white paper) frame via threshold → contour
    _, bright = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  k)

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("  [REF] No bright contours — skipping correction")
        return img

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:3]:
        if cv2.contourArea(c) < 0.25 * h * w:
            continue
        peri = cv2.arcLength(c, True)
        for eps in [0.02, 0.03, 0.015, 0.01]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) != 4:
                continue
            src = order_pts(approx.reshape(4, 2))
            tl, tr, br, bl = src
            wp = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            hp = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            if wp < w * 0.50 or hp < h * 0.50:
                continue
            if wp * hp < 0.75 * w * h:
                print(f"  [REF] Quad area too small ({wp}×{hp} vs {w}×{h})")
                continue
            dst = np.float32([[0, 0], [wp, 0], [wp, hp], [0, hp]])
            M = cv2.getPerspectiveTransform(src, dst)
            out = cv2.warpPerspective(img, M, (wp, hp), flags=cv2.INTER_LANCZOS4)
            print(f"  [REF] Perspective corrected: {wp}×{hp}")
            return out

    # Fallback: bounding rect
    c = contours[0]
    x, y, bw, bh = cv2.boundingRect(c)
    if bw > w * 0.55 and bh > h * 0.55:
        print(f"  [REF] Fallback crop: ({x},{y}) {bw}×{bh}")
        return img[y:y+bh, x:x+bw]

    print("  [REF] Perspective correction skipped")
    return img


# ── SIFT matching ──────────────────────────────────────────────────────── #


def compute_H_to_ref(ref, cu, tag=""):
    """Compute homography: cu full-res pixel → ref full-res pixel.

    Cascades through Lowe ratios from tight to loose; a high ratio can pass
    noisy matches that mislead RANSAC when map patterns repeat.
    """
    hr, wr = ref.shape[:2]
    hc, wc = cu.shape[:2]

    s_r = min(1.0, MATCH_DIM_REF / max(hr, wr))
    s_c = min(1.0, MATCH_DIM_CU  / max(hc, wc))

    ref_sm = cv2.resize(ref, None, fx=s_r, fy=s_r, interpolation=cv2.INTER_AREA)
    cu_sm  = cv2.resize(cu,  None, fx=s_c, fy=s_c, interpolation=cv2.INTER_AREA)
    gr = cv2.cvtColor(ref_sm, cv2.COLOR_BGR2GRAY)
    gc = cv2.cvtColor(cu_sm,  cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=20000, contrastThreshold=0.004, edgeThreshold=20)
    kp_r, des_r = sift.detectAndCompute(gr, None)
    kp_c, des_c = sift.detectAndCompute(gc, None)
    print(f"    KP: ref={len(kp_r)}, cu={len(kp_c)}")

    if des_r is None or des_c is None or len(kp_r) < 8 or len(kp_c) < 8:
        print(f"  [{tag}] Too few keypoints")
        return None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=8), dict(checks=300))
    raw = flann.knnMatch(des_c, des_r, k=2)

    for ratio in [0.65, 0.60, 0.70, 0.55, 0.72]:
        good = [m for m, n in raw if m.distance < ratio * n.distance]
        if len(good) < 10:
            continue
        pts_c = np.float32([kp_c[m.queryIdx].pt for m in good]) / s_c
        pts_r = np.float32([kp_r[m.trainIdx].pt for m in good]) / s_r
        H, mask = cv2.findHomography(pts_c, pts_r, cv2.RANSAC, 5.0,
                                      maxIters=8000, confidence=0.999)
        n_in = 0 if mask is None else int(mask.sum())
        print(f"    ratio={ratio:.2f}: {len(good)} matches  {n_in} inliers")
        if H is not None and n_in >= 15:
            print(f"  [{tag}] OK  inliers={n_in}  ratio={ratio:.2f}")
            return H

    print(f"  [{tag}] Homography failed across all ratios")
    return None


# ── Canvas / blending ───────────────────────────────────────────────────── #

def make_content_mask(img, white_thresh=252):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nw = (gray < white_thresh).astype(np.uint8) * 255
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    filled = cv2.morphologyEx(nw, cv2.MORPH_CLOSE, k1)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 200))
    return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k2)


def find_seam_row(canvas, warped, overlap, smooth=40):
    ov_rows = np.where(overlap.any(1))[0]
    if len(ov_rows) < 3:
        return int(ov_rows.mean()) if len(ov_rows) else 0
    r0, r1 = ov_rows[0], ov_rows[-1]
    ec = cv2.Canny(cv2.cvtColor(canvas[r0:r1+1, :], cv2.COLOR_BGR2GRAY), 30, 90).sum(1).astype(float)
    ew = cv2.Canny(cv2.cvtColor(warped[r0:r1+1, :], cv2.COLOR_BGR2GRAY), 30, 90).sum(1).astype(float)
    return int(uniform_filter1d(ec + ew, smooth).argmin()) + r0


MASK_REF = 64   # canvas pixel came from reference (low priority)
MASK_CU  = 255  # canvas pixel came from a close-up (high priority)


def blend_onto(canvas, canvas_mask, warped, warped_valid, px=20):
    """
    Blend warped close-up onto canvas.
    - Pixels that are new (mask==0) or only reference (mask==MASK_REF): direct copy.
    - Pixels overlapping an existing close-up (mask==MASK_CU): seam-cut blend.
    """
    only_new  = warped_valid & (canvas_mask == 0)
    ov_ref    = warped_valid & (canvas_mask == MASK_REF)
    ov_cu     = warped_valid & (canvas_mask == MASK_CU)

    out = canvas.copy()
    om  = canvas_mask.copy()

    # New pixels and reference regions: close-up wins directly
    direct = only_new | ov_ref
    out[direct] = warped[direct]
    om[direct]  = MASK_CU

    # Close-up vs close-up: seam-cut blend
    if ov_cu.any():
        hc = canvas.shape[0]
        seam = find_seam_row(canvas, warped, ov_cu)
        y0 = max(0, seam - px)
        y1 = min(hc, seam + px + 1)
        # Below seam: warped wins
        bm = ov_cu.copy(); bm[:y1, :] = False
        out[bm] = warped[bm]
        # Feather zone
        for y in range(y0, y1):
            row = ov_cu[y, :]
            if not row.any():
                continue
            a = (y - y0) / (y1 - y0 - 1) if y1 > y0 + 1 else 0.5
            out[y, row] = ((1 - a) * canvas[y, row].astype(np.float32)
                           + a * warped[y, row].astype(np.float32)).astype(np.uint8)
        om[ov_cu] = MASK_CU

    return out, om


def place_cu_on_canvas(canvas, canvas_mask, cu, H, S_out):
    hc, wc = canvas.shape[:2]
    H_out = S_out @ H

    warped = cv2.warpPerspective(cu, H_out, (wc, hc),
                                  flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cm = make_content_mask(cu)
    warped_valid = cv2.warpPerspective(cm, H_out, (wc, hc),
                                        flags=cv2.INTER_NEAREST) > 127

    return blend_onto(canvas, canvas_mask, warped, warped_valid)


# ── Main ────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load and correct reference
    print("=" * 60)
    print("Loading reference...")
    ref = cv2.imread(REF_FILE)
    if ref is None:
        raise FileNotFoundError(REF_FILE)
    print(f"  Loaded: {ref.shape[1]}×{ref.shape[0]}")

    ref_corr = correct_perspective_ref(ref)
    print(f"  Corrected: {ref_corr.shape[1]}×{ref_corr.shape[0]}")
    cv2.imwrite(OUTPUT_DIR + "ref_corrected.png", ref_corr)

    hr, wr = ref_corr.shape[:2]
    cw = int(wr * SCALE_OUT)
    ch = int(hr * SCALE_OUT)
    S_out = np.diag([SCALE_OUT, SCALE_OUT, 1.0]).astype(np.float64)
    print(f"  Output canvas: {cw}×{ch}  (SCALE_OUT={SCALE_OUT}×)")

    # 2. Initialize canvas with upscaled reference as base layer
    canvas      = np.zeros((ch, cw, 3), np.uint8)
    canvas_mask = np.zeros((ch, cw),    np.uint8)

    ref_scaled = cv2.resize(ref_corr, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
    ref_cm     = make_content_mask(ref_scaled)
    ref_valid  = ref_cm > 127
    canvas[ref_valid]      = ref_scaled[ref_valid]
    canvas_mask[ref_valid] = MASK_REF
    print("  Reference placed as base layer")

    # 3. Match and warp each close-up
    successes = 0
    for i, (path, tag) in enumerate(CLOSE_UPS):
        print(f"\n[{i+1}/{len(CLOSE_UPS)}] {tag}  ({Path(path).name})")
        cu = cv2.imread(path)
        if cu is None:
            print(f"  Cannot read {path}")
            continue
        print(f"  Size: {cu.shape[1]}×{cu.shape[0]}")

        H = compute_H_to_ref(ref_corr, cu, tag)
        if H is None:
            print(f"  SKIPPED — no valid homography")
            continue

        canvas, canvas_mask = place_cu_on_canvas(canvas, canvas_mask, cu, H, S_out)
        print(f"  Placed on canvas  ({successes+1} of {len(CLOSE_UPS)})")
        successes += 1

    print(f"\nSuccessfully placed: {successes}/{len(CLOSE_UPS)}")

    # 4. Crop to content
    rows_idx = np.where(canvas_mask.any(1))[0]
    cols_idx = np.where(canvas_mask.any(0))[0]
    if len(rows_idx) and len(cols_idx):
        canvas = canvas[rows_idx[0]:rows_idx[-1]+1, cols_idx[0]:cols_idx[-1]+1]
        print(f"Cropped to content: {canvas.shape[1]}×{canvas.shape[0]}")

    # 5. Save full-resolution + preview
    out_path = OUTPUT_DIR + "final_map_hires.png"
    print(f"\nSaving {out_path}...")
    cv2.imwrite(out_path, canvas)
    sz_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved: {canvas.shape[1]}×{canvas.shape[0]}  {sz_mb:.0f} MB")

    scale   = min(1.0, 2000 / canvas.shape[1])
    preview = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pv_path = OUTPUT_DIR + "final_map_hires_preview.jpg"
    cv2.imwrite(pv_path, preview, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Preview: {pv_path}  ({preview.shape[1]}×{preview.shape[0]})")
    print("=" * 60)
    print("Done.")
