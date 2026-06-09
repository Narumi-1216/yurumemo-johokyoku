"""
Map stitching v4 — affine-aware strip matching.

Key changes from v3:
- Strip-based matching uses full homography (handles rotation differences)
- Fallback to ECC-based translation for difficult pairs
- Fixed canvas placement bug
- Configurable overlap fraction
"""

import cv2
import numpy as np
import os
from pathlib import Path

UPLOAD_DIR = "/root/.claude/uploads/1b9fe45d-a299-5f9c-9937-d45d27c12f22/"
OUTPUT_DIR = "/home/user/yurumemo-johokyoku/map_output/"
CORRECTED  = OUTPUT_DIR + "corrected_v4/"
ROWS_DIR   = OUTPUT_DIR + "rows_v4/"

for d in [OUTPUT_DIR, CORRECTED, ROWS_DIR]:
    os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────
# Perspective correction (same as v3 — proven to work well)
# ──────────────────────────────────────────────────────────

def order_pts(pts):
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1)
    rect[0] = pts[s.argmin()]; rect[2] = pts[s.argmax()]
    rect[1] = pts[d.argmin()]; rect[3] = pts[d.argmax()]
    return rect


def correct_perspective(img, tag=""):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    bright  = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k_close)
    bright  = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  k_open)

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = None
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for cnt in cnts[:3]:
            area = cv2.contourArea(cnt)
            if area < 0.08 * h * w: continue
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                bw, bh = cv2.minAreaRect(pts)[1]
                if min(bw, bh) > 0 and max(bw, bh)/min(bw, bh) < 4:
                    quad = order_pts(pts)
                    break

    if quad is None:
        ys, xs = np.where(bright > 0)
        if len(xs) == 0:
            return img
        x0, x1 = xs.min() + 2, xs.max() - 2
        y0, y1 = ys.min() + 2, ys.max() - 2
        quad = order_pts(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32))

    pts = quad
    wp = int(max(np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[3])))
    hp = int(max(np.linalg.norm(pts[3]-pts[0]), np.linalg.norm(pts[2]-pts[1])))
    dst = np.float32([[0,0],[wp-1,0],[wp-1,hp-1],[0,hp-1]])
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (wp, hp), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    print(f"  [{tag}] warped: {wp}x{hp}")
    return warped


def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    strip_h = h // 5
    strip = np.vstack([gray[:strip_h], gray[-strip_h:]])
    edges = cv2.Canny(cv2.GaussianBlur(strip, (5,5), 0), 20, 80)
    lines = cv2.HoughLines(edges, 1, np.pi/1800, threshold=int(w * 0.15))
    if lines is None: return img, 0.0
    angles = []
    for line in lines:
        rho, theta = line[0]
        skew = np.degrees(theta) - 90.0
        if abs(skew) < 8: angles.append(skew)
    if not angles: return img, 0.0
    angle = float(np.median(angles))
    if abs(angle) < 0.3: return img, 0.0
    print(f"  Deskew: {angle:.2f}°")
    M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
    cos_a, sin_a = abs(M[0,0]), abs(M[0,1])
    nw = int(h*sin_a + w*cos_a); nh = int(h*cos_a + w*sin_a)
    M[0,2] += (nw-w)/2; M[1,2] += (nh-h)/2
    result = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return result, angle


def preprocess(path, tag):
    print(f"\n[Pre] {tag}  ({Path(path).name})")
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(path)
    print(f"  Loaded: {img.shape[1]}x{img.shape[0]}")
    img = correct_perspective(img, tag)
    img, angle = deskew(img)
    print(f"  Final: {img.shape[1]}x{img.shape[0]}  (skew={angle:.2f}°)")
    cv2.imwrite(CORRECTED + f"{tag}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return img


# ──────────────────────────────────────────────────────────
# Strip-based matching with FULL homography
# ──────────────────────────────────────────────────────────

def sift_match(gray_a, gray_b, ratio=0.68, max_feat=6000):
    """Returns matched point pairs (pts_a, pts_b) or (None, None)."""
    sift = cv2.SIFT_create(nfeatures=max_feat, contrastThreshold=0.01,
                            edgeThreshold=20)
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)
    print(f"    KP: a={len(kp_a)}, b={len(kp_b)}")
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return None, None

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=150))
    raw = flann.knnMatch(des_a, des_b, k=2)
    good = [m for m,n in raw if m.distance < ratio * n.distance]
    print(f"    Good matches: {len(good)}")
    if len(good) < 8: return None, None

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
    return pts_a, pts_b


def find_H_strip(img_left, img_right, overlap_frac=0.40, direction="horizontal"):
    """
    Find homography mapping img_right → img_left coordinate space,
    using only the overlap strip region. Returns (H, n_inliers) or (None, 0).
    """
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]

    if direction == "horizontal":
        strip_w = int(min(w_l, w_r) * overlap_frac)
        x_l = w_l - strip_w
        ref_strip = img_left[:, x_l:]
        mov_strip = img_right[:, :strip_w]
        offset_x_ref, offset_y_ref = x_l, 0
        offset_x_mov, offset_y_mov = 0,   0
    else:  # vertical
        strip_h = int(min(h_l, h_r) * overlap_frac)
        y_l = h_l - strip_h
        ref_strip = img_left[y_l:, :]
        mov_strip = img_right[:strip_h, :]
        offset_x_ref, offset_y_ref = 0, y_l
        offset_x_mov, offset_y_mov = 0, 0

    gray_ref = cv2.cvtColor(ref_strip, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(mov_strip, cv2.COLOR_BGR2GRAY)

    pts_ref, pts_mov = sift_match(gray_ref, gray_mov)
    if pts_ref is None: return None, 0

    # Convert strip coordinates to full-image coordinates
    pts_ref_full = pts_ref + [offset_x_ref, offset_y_ref]
    pts_mov_full = pts_mov + [offset_x_mov, offset_y_mov]

    # Find homography: maps mov full-image coords → ref full-image coords
    H, mask = cv2.findHomography(pts_mov_full, pts_ref_full,
                                  cv2.RANSAC, 4.0,
                                  maxIters=5000, confidence=0.999)
    n = int(mask.sum()) if mask is not None else 0
    print(f"    Strip H inliers: {n}/{len(pts_ref)}")
    return H, n


def validate_H(H, src_shape, max_scale=2.0):
    """Basic sanity check: H should not wildly distort the image."""
    if H is None: return False
    h, w = src_shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    warped  = cv2.perspectiveTransform(corners, H).reshape(4,2)
    x_min, y_min = warped.min(0); x_max, y_max = warped.max(0)
    new_w = x_max - x_min; new_h = y_max - y_min
    if new_w < w/max_scale or new_w > w*max_scale: return False
    if new_h < h/max_scale or new_h > h*max_scale: return False
    # Projective terms should be small
    if abs(H[2,0]) > 5e-4 or abs(H[2,1]) > 5e-4: return False
    return True


def ecc_align(img_ref, img_mov, init_tx=0.0, init_ty=0.0, mode=cv2.MOTION_AFFINE):
    """
    Use ECC to refine alignment starting from (init_tx, init_ty).
    Returns 2x3 warp matrix mapping mov → ref space, or None.
    """
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Resize to speed up ECC
    scale = min(1.0, 800 / max(gray_ref.shape))
    if scale < 1.0:
        gray_ref = cv2.resize(gray_ref, None, fx=scale, fy=scale)
        gray_mov = cv2.resize(gray_mov, None, fx=scale, fy=scale)

    if mode == cv2.MOTION_TRANSLATION:
        warp = np.array([[1,0,init_tx*scale],[0,1,init_ty*scale]], np.float32)
    else:
        warp = np.eye(2, 3, np.float32)
        warp[0,2] = init_tx * scale
        warp[1,2] = init_ty * scale

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-6)
    try:
        cc, warp_out = cv2.findTransformECC(gray_ref, gray_mov, warp, mode, criteria,
                                             inputMask=None, gaussFiltSize=5)
        print(f"    ECC correlation: {cc:.4f}")
        if cc < 0.3: return None
        # Scale back to original size
        warp_out[0,2] /= scale; warp_out[1,2] /= scale
        return warp_out
    except cv2.error as e:
        print(f"    ECC failed: {e}")
        return None


# ──────────────────────────────────────────────────────────
# Canvas compositing
# ──────────────────────────────────────────────────────────

def warp_onto(canvas, canvas_mask, new_img, H):
    """
    Warp new_img using H (3×3) onto canvas. Seam-cut composite.
    """
    hc, wc = canvas.shape[:2]
    hn, wn = new_img.shape[:2]

    warped = cv2.warpPerspective(new_img, H, (wc, hc),
                                  flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_mask = cv2.warpPerspective(
        np.ones((hn, wn), np.uint8)*255, H, (wc, hc), flags=cv2.INTER_NEAREST) > 0

    only_new = warped_mask & (canvas_mask == 0)
    overlap  = warped_mask & (canvas_mask  > 0)

    out = canvas.copy(); out_mask = canvas_mask.copy()
    out[only_new] = warped[only_new]; out_mask[only_new] = 255

    if overlap.any():
        # Determine seam direction
        ov_cols = np.where(overlap.any(0))[0]
        ov_rows = np.where(overlap.any(1))[0]
        col_span = ov_cols[-1]-ov_cols[0] if len(ov_cols) else 0
        row_span = ov_rows[-1]-ov_rows[0] if len(ov_rows) else 0
        if col_span >= row_span:
            seam = int(ov_cols.mean())
            take_new = np.zeros((hc,wc), bool); take_new[:, seam:] = True
        else:
            seam = int(ov_rows.mean())
            take_new = np.zeros((hc,wc), bool); take_new[seam:, :] = True
        out[overlap & take_new] = warped[overlap & take_new]

    return out, out_mask


def build_canvas(images, Hs):
    """
    Place images on canvas using cumulative homographies.
    Hs[i] maps image_i → canvas (image 0) coordinates.
    """
    all_corners = []
    for img, H in zip(images, Hs):
        h, w = img.shape[:2]
        c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        all_corners.append(cv2.perspectiveTransform(c, H).reshape(4,2))
    all_c = np.vstack(all_corners)
    x_min = int(np.floor(all_c[:,0].min())); x_max = int(np.ceil(all_c[:,0].max()))
    y_min = int(np.floor(all_c[:,1].min())); y_max = int(np.ceil(all_c[:,1].max()))
    tx, ty = max(0, -x_min), max(0, -y_min)
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)
    cw, ch = x_max-x_min+1, y_max-y_min+1
    print(f"  Canvas: {cw}x{ch}  offset=({tx},{ty})")
    canvas = np.zeros((ch, cw, 3), np.uint8)
    canvas_mask = np.zeros((ch, cw), np.uint8)
    for i, (img, H) in enumerate(zip(images, Hs)):
        H_adj = T @ H
        canvas, canvas_mask = warp_onto(canvas, canvas_mask, img, H_adj)
        print(f"  Placed image {i+1}")
    rows = np.where(canvas_mask.any(1))[0]; cols = np.where(canvas_mask.any(0))[0]
    if len(rows) and len(cols):
        canvas = canvas[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    return canvas


# ──────────────────────────────────────────────────────────
# Row stitching
# ──────────────────────────────────────────────────────────

def compute_pairwise_H(img_a, img_b, overlap_frac=0.40, direction="horizontal"):
    """
    Compute H mapping img_b → img_a coordinate space.
    Tries: strip SIFT → full-image SIFT → ECC → translation estimate.
    """
    print(f"  Strip SIFT (overlap={overlap_frac:.0%})...")
    H, n = find_H_strip(img_a, img_b, overlap_frac, direction)

    if H is not None and n >= 12 and validate_H(H, img_b.shape):
        return H

    print(f"  Strip failed (n={n}). Trying full-image SIFT...")
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    pts_a, pts_b = sift_match(gray_a, gray_b)
    if pts_a is not None:
        H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 4.0,
                                      maxIters=5000, confidence=0.999)
        n = int(mask.sum()) if mask is not None else 0
        print(f"  Full SIFT H inliers: {n}")
        if H is not None and n >= 12 and validate_H(H, img_b.shape):
            return H

    print(f"  Full SIFT failed (n={n}). Trying ECC...")
    ha, wa = img_a.shape[:2]
    hb, wb = img_b.shape[:2]
    if direction == "horizontal":
        init_tx = wa * (1 - overlap_frac)
        init_ty = (ha - hb) / 2
    else:
        init_tx = (wa - wb) / 2
        init_ty = ha * (1 - overlap_frac)

    # Crop to overlap region for ECC
    if direction == "horizontal":
        sw = int(min(wa, wb) * (overlap_frac + 0.1))
        ref_strip = img_a[:, wa-sw:]
        mov_strip = img_b[:, :sw]
    else:
        sh = int(min(ha, hb) * (overlap_frac + 0.1))
        ref_strip = img_a[ha-sh:, :]
        mov_strip = img_b[:sh, :]

    warp2x3 = ecc_align(ref_strip, mov_strip, 0, 0, cv2.MOTION_AFFINE)
    if warp2x3 is not None:
        # Convert 2×3 affine to 3×3 homography in full-image coords
        # The affine was computed for ref_strip coords; need to add offsets
        if direction == "horizontal":
            rx_off = wa - int(min(wa, wb) * (overlap_frac + 0.1))
        else:
            rx_off = 0
        ry_off = 0 if direction == "horizontal" else ha - int(min(ha, hb) * (overlap_frac + 0.1))

        # warp2x3 maps mov_strip → ref_strip
        # full-image: pt_ref_full = pt_ref_strip + (rx_off, ry_off)
        # In ref_strip: pt_ref_strip = warp2x3 * pt_mov_strip
        # pt_ref_full = warp2x3 * (pt_mov_full - offset_mov) + offset_ref
        # (offset_mov = 0 since mov_strip starts at (0,0) of mov image)
        # pt_ref_full = warp2x3 * pt_mov_full + (rx_off - warp2x3 * [0,0]^T + [rx_off, ry_off]^T)

        # Simpler: build 3x3 homography from the affine
        A = warp2x3[:2, :2]
        t = warp2x3[:2, 2]
        # Adjust translation for offset
        t_adjusted = t + np.array([rx_off, ry_off]) - A @ np.array([0.0, 0.0])
        H_ecc = np.eye(3, dtype=np.float64)
        H_ecc[:2, :2] = A; H_ecc[:2, 2] = t_adjusted
        print(f"  ECC H: t=({t_adjusted[0]:.1f},{t_adjusted[1]:.1f})")
        if validate_H(H_ecc, img_b.shape, max_scale=1.5):
            return H_ecc

    print(f"  ECC failed. Using translation estimate.")
    if direction == "horizontal":
        tx = wa * (1 - overlap_frac)
        ty = (ha - hb) / 2
    else:
        tx = (wa - wb) / 2
        ty = ha * (1 - overlap_frac)
    H_est = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)
    return H_est


def stitch_row(images, tag, direction="horizontal", overlap_frac=0.30):
    if len(images) == 1: return images[0]

    # Try OpenCV Stitcher (SCANS) first
    print(f"\n  Trying cv2.Stitcher (SCANS)...")
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        stitcher.setPanoConfidenceThresh(0.3)
        status, result = stitcher.stitch(images)
        ok = {cv2.Stitcher_OK: "OK",
              cv2.Stitcher_ERR_NEED_MORE_IMGS: "need more images",
              cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "homography fail",
              cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "camera params fail"}
        print(f"  Stitcher: {ok.get(status, status)}")
        if status == cv2.Stitcher_OK:
            print("  cv2.Stitcher succeeded!")
            return result
    except Exception as e:
        print(f"  Stitcher exception: {e}")

    # Manual pairwise stitching
    Hs = [np.eye(3, dtype=np.float64)]  # H_0 = identity
    for i in range(1, len(images)):
        print(f"\n  Aligning image {i+1} → image {i} (H chain)...")
        H_local = compute_pairwise_H(images[i-1], images[i], overlap_frac, direction)
        # Clamp projective components to avoid explosion
        H_local[2,0] = 0.0; H_local[2,1] = 0.0; H_local[2,2] = 1.0
        Hs.append(H_local @ Hs[-1])
        print(f"  H[{i+1}]: tx={H_local[0,2]:.1f}  ty={H_local[1,2]:.1f}")

    print(f"\n  Building canvas...")
    return build_canvas(images, Hs)


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

ROW_FILES = {
    "1": [
        "69a424a3-IMG_6837.jpeg",
        "95841777-IMG_6838.jpeg",
        "d9b6fffb-IMG_6839.jpeg",
    ],
    "2": [],
    "3": [],
    "4": [],
}


def process_row(row_tag, file_names):
    paths = [UPLOAD_DIR + f for f in file_names]
    present = [(p, f"r{row_tag}_{i+1}") for i, p in enumerate(paths)
               if Path(p).exists()]
    if not present:
        print(f"\nRow {row_tag}: no files, skipping.")
        return None
    print(f"\n{'='*55}\n Row {row_tag}  ({len(present)} images)\n{'='*55}")
    imgs = [preprocess(p, tag) for p, tag in present]

    result = stitch_row(imgs, row_tag)

    out = ROWS_DIR + f"row_{row_tag}.png"
    cv2.imwrite(out, result)
    print(f"\nSaved: {out}  ({result.shape[1]}x{result.shape[0]})")
    return result


if __name__ == "__main__":
    row_imgs = {}
    for rtag, files in ROW_FILES.items():
        if not files: continue
        r = process_row(rtag, files)
        if r is not None: row_imgs[rtag] = r

    rows = [row_imgs[k] for k in sorted(row_imgs) if k in row_imgs]
    if rows:
        out = OUTPUT_DIR + "row_1_result.png"
        cv2.imwrite(out, rows[0])
        print(f"\nResult saved: {out}  ({rows[0].shape[1]}x{rows[0].shape[0]})")
    print("\nDone.")
