"""
Map stitching v3 — three-phase robust approach.

Phase 1: Perspective correction per image (detect map frame, warp to rectangle)
Phase 2: Horizontal row stitching using:
         a) cv2.Stitcher SCANS mode (best quality)
         b) Fallback: strip-based translation match
Phase 3: Vertical row stacking
"""

import cv2
import numpy as np
import os
from pathlib import Path

UPLOAD_DIR = "/root/.claude/uploads/1b9fe45d-a299-5f9c-9937-d45d27c12f22/"
OUTPUT_DIR = "/home/user/yurumemo-johokyoku/map_output/"
CORRECTED  = OUTPUT_DIR + "corrected_v3/"
ROWS_DIR   = OUTPUT_DIR + "rows_v3/"
DEBUG_DIR  = OUTPUT_DIR + "debug_v3/"

for d in [OUTPUT_DIR, CORRECTED, ROWS_DIR, DEBUG_DIR]:
    os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────
# 1. Image crop & perspective correction
# ──────────────────────────────────────────────────────────

def order_pts(pts):
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1)
    rect[0] = pts[s.argmin()]   # TL
    rect[2] = pts[s.argmax()]   # BR
    rect[1] = pts[d.argmin()]   # TR
    rect[3] = pts[d.argmax()]   # BL
    return rect


def find_map_quad(img, tag=""):
    """
    Detect the four corners of the physical map's rectangular frame.
    Uses a white-region approach: the map has a bright interior bounded
    by a thin dark frame.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── A: look for the brightest large rectangular region ──
    # Threshold: map surface is bright white (>180)
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # Fill small holes inside the map
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel_close)
    # Remove small specks
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  kernel_open)

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for cnt in cnts[:3]:
            area = cv2.contourArea(cnt)
            if area < 0.08 * h * w:
                continue
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2)
                bw, bh = cv2.minAreaRect(pts)[1]
                if min(bw, bh) > 0 and max(bw, bh)/min(bw, bh) < 4:
                    print(f"  [{tag}] quad found (area={area:.0f})")
                    return order_pts(pts)

    # ── B: bounding-box fallback ──
    ys, xs = np.where(bright > 0)
    if len(xs) == 0:
        print(f"  [{tag}] no bright region, returning None")
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # Shrink by 0.5% to avoid frame bleed
    shrink_x = int((x1-x0) * 0.005)
    shrink_y = int((y1-y0) * 0.005)
    x0 += shrink_x; x1 -= shrink_x
    y0 += shrink_y; y1 -= shrink_y
    print(f"  [{tag}] bbox fallback: ({x0},{y0})-({x1},{y1})")
    return order_pts(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32))


def correct_perspective(img, tag=""):
    """Detect map quad and warp to rectangle. Returns corrected image."""
    quad = find_map_quad(img, tag)
    if quad is None:
        return img
    pts = order_pts(quad)
    tl, tr, br, bl = pts
    w = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
    h = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
    dst = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(255,255,255))
    print(f"  [{tag}] perspective corrected: {w}x{h}")
    return warped


def detect_skew_full(img):
    """
    Detect document skew using the full image Hough transform.
    More robust than strip-only version.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Only look at top 20% and bottom 20% for horizontal lines
    # (where the coordinate labels and borders are cleanest)
    strip_h = h // 5
    strip = np.vstack([gray[:strip_h], gray[h-strip_h:]])
    edges = cv2.Canny(cv2.GaussianBlur(strip, (5,5), 0), 20, 80)
    lines = cv2.HoughLines(edges, 1, np.pi/1800, threshold=int(w * 0.15))
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        rho, theta = line[0]
        # theta near 0 or pi → horizontal line
        deg = np.degrees(theta)
        skew = deg - 90.0   # 0° = perfectly horizontal
        if abs(skew) < 8:
            angles.append(skew)
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(img):
    angle = detect_skew_full(img)
    if abs(angle) < 0.3:
        return img, 0.0
    print(f"  Deskew: {angle:.2f}°")
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
    cos_a, sin_a = abs(M[0,0]), abs(M[0,1])
    nw = int(h*sin_a + w*cos_a)
    nh = int(h*cos_a + w*sin_a)
    M[0,2] += (nw-w)/2; M[1,2] += (nh-h)/2
    result = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255,255,255))
    return result, angle


def trim_white_border(img, margin_frac=0.005):
    """Trim very thin white/empty border after perspective warp."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find rows / cols that are not all white
    row_min = gray.min(axis=1)   # minimum per row (dark = map content)
    col_min = gray.min(axis=0)
    threshold = 240
    rows = np.where(row_min < threshold)[0]
    cols = np.where(col_min < threshold)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    r0, r1 = rows[0], rows[-1]+1
    c0, c1 = cols[0], cols[-1]+1
    return img[r0:r1, c0:c1]


def preprocess(path, tag):
    print(f"\n[Pre] {tag}  ({Path(path).name})")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    print(f"  Loaded: {img.shape[1]}x{img.shape[0]}")

    # Step 1: perspective correction
    img = correct_perspective(img, tag)

    # Step 2: trim artifact borders
    img = trim_white_border(img)
    print(f"  After trim: {img.shape[1]}x{img.shape[0]}")

    # Step 3: deskew
    img, angle = deskew(img)
    print(f"  Final: {img.shape[1]}x{img.shape[0]}  (skew={angle:.2f}°)")

    cv2.imwrite(CORRECTED + f"{tag}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return img


# ──────────────────────────────────────────────────────────
# 2. Stitching helpers
# ──────────────────────────────────────────────────────────

def try_opencv_stitcher(images, mode=cv2.Stitcher_SCANS):
    """
    Try OpenCV's built-in stitcher (SCANS mode for flat documents).
    Returns result image or None on failure.
    """
    stitcher = cv2.Stitcher_create(mode)
    stitcher.setPanoConfidenceThresh(0.5)
    status, result = stitcher.stitch(images)
    statuses = {
        cv2.Stitcher_OK: "OK",
        cv2.Stitcher_ERR_NEED_MORE_IMGS: "need more images",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "homography fail",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "camera params fail",
    }
    print(f"  Stitcher status: {statuses.get(status, status)}")
    return result if status == cv2.Stitcher_OK else None


def match_strip_translation(img_left, img_right, overlap_frac=0.40):
    """
    Find the translation (tx, ty) of img_right relative to img_left
    by matching a right strip of img_left to a left strip of img_right.
    Returns (tx, ty) or None.
    """
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]

    strip_w = int(min(w_l, w_r) * overlap_frac)
    x_left_start = w_l - strip_w  # strip starts here in img_left

    ref_strip = img_left[:, x_left_start:]
    mov_strip = img_right[:, :strip_w]

    gray_ref = cv2.cvtColor(ref_strip, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(mov_strip, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.01,
                            edgeThreshold=20, sigma=1.6)
    kp1, des1 = sift.detectAndCompute(gray_ref, None)
    kp2, des2 = sift.detectAndCompute(gray_mov, None)
    print(f"  Strip KP: ref={len(kp1)}, mov={len(kp2)}")

    if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
        print("  Too few strip keypoints")
        return None

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=150))
    raw = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.68 * n.distance]
    print(f"  Strip good matches: {len(good)}")

    if len(good) < 6:
        return None

    # Compute translation candidates
    # Each match: ref point at (x_left_start + x1, y1) in img_left coords
    #             mov point at (x2, y2) in img_right coords
    # Translation of img_right origin relative to img_left origin:
    #   tx = (x_left_start + x1) - x2
    #   ty = y1 - y2
    txs, tys = [], []
    for m in good:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        txs.append(x_left_start + x1 - x2)
        tys.append(y1 - y2)

    tx_med = np.median(txs)
    ty_med = np.median(tys)

    # Inliers within ±15 pixels of median
    inliers = np.abs(np.array(txs) - tx_med) < 15
    inliers &= np.abs(np.array(tys) - ty_med) < 15
    n_in = inliers.sum()
    print(f"  Translation: tx={tx_med:.1f}  ty={ty_med:.1f}  inliers={n_in}/{len(good)}")

    if n_in < 4:
        print("  Too few inlier translations")
        return None

    # Refine from inliers
    tx_final = float(np.median(np.array(txs)[inliers]))
    ty_final = float(np.median(np.array(tys)[inliers]))
    return tx_final, ty_final


def place_on_canvas(images, translations_from_first):
    """
    Given images and their translations relative to images[0] origin,
    build a canvas and composite with seam cut.
    """
    # Find canvas bounds
    h0, w0 = images[0].shape[:2]
    all_corners = [(0, 0, w0, h0)]  # (x0, y0, x1, y1)
    for i, (tx, ty) in enumerate(translations_from_first[1:], 1):
        hi, wi = images[i].shape[:2]
        all_corners.append((tx, ty, tx+wi, ty+hi))

    xs = [c[0] for c in all_corners] + [c[2] for c in all_corners]
    ys = [c[1] for c in all_corners] + [c[3] for c in all_corners]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    off_x, off_y = -x_min, -y_min

    cw = x_max - x_min
    ch = y_max - y_min
    print(f"  Canvas: {cw}x{ch}  offset=({off_x},{off_y})")

    canvas = np.zeros((ch, cw, 3), np.uint8)
    mask   = np.zeros((ch, cw), np.uint8)

    for i, (img, (tx, ty, _, _)) in enumerate(zip(images, all_corners)):
        hi, wi = img.shape[:2]
        ox = int(tx) + off_x
        oy = int(ty) + off_y
        # Clip to canvas
        x_start = max(0, ox);     y_start = max(0, oy)
        x_end   = min(cw, ox+wi); y_end   = min(ch, oy+hi)
        src_x0  = x_start - ox;   src_y0  = y_start - oy
        src_x1  = x_end   - ox;   src_y1  = y_end   - oy

        region_canvas = canvas[y_start:y_end, x_start:x_end]
        region_mask   = mask[y_start:y_end, x_start:x_end]
        region_new    = img[src_y0:src_y1, src_x0:src_x1]
        existing      = region_mask > 0

        if existing.any():
            # Seam at midpoint of overlap
            cols_ov = np.where(existing.any(0))[0]
            seam = int(cols_ov.mean()) if len(cols_ov) else region_canvas.shape[1] // 2
            take_new = np.zeros_like(existing)
            take_new[:, seam:] = True
            region_canvas[~existing] = region_new[~existing]
            region_canvas[existing & take_new] = region_new[existing & take_new]
        else:
            region_canvas[:] = region_new

        mask[y_start:y_end, x_start:x_end] = 255

    # Crop to content
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    return canvas[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]


def stitch_row(images, tag):
    """
    Stitch a list of images left→right.
    Tries cv2.Stitcher first, then manual translation matching.
    """
    n = len(images)
    if n == 1:
        return images[0]

    print(f"\n[Stitch] Row {tag}: trying cv2.Stitcher (SCANS mode)...")
    result = try_opencv_stitcher(images)
    if result is not None:
        print("  cv2.Stitcher succeeded!")
        return result

    print("  cv2.Stitcher failed, switching to manual translation matching...")

    # Compute pairwise translations
    translations = [(0, 0)]  # image 0 at origin
    prev_tx, prev_ty = 0, 0
    for i in range(1, n):
        print(f"\n  Matching image {i+1} to image {i}...")
        result_trans = match_strip_translation(images[i-1], images[i])
        if result_trans is None:
            print(f"  Strip matching failed for image {i+1}, using estimate")
            dx_estimate = int(images[i-1].shape[1] * 0.78)
            result_trans = (dx_estimate, 0)
        lx, ly = result_trans
        # Cumulative: position of image i relative to image 0
        tx_abs = prev_tx + lx
        ty_abs = prev_ty + ly
        translations.append((tx_abs, ty_abs))
        prev_tx, prev_ty = tx_abs, ty_abs

    print(f"\n  Building canvas...")
    # Build corners list
    corners = []
    for img, (tx, ty) in zip(images, translations):
        hi, wi = img.shape[:2]
        corners.append((tx, ty, tx+wi, ty+hi))

    return place_on_canvas(images, corners)


# ──────────────────────────────────────────────────────────
# 3. Vertical stitching
# ──────────────────────────────────────────────────────────

def stitch_vertical(rows, tag="final"):
    if len(rows) == 1:
        return rows[0]

    print(f"\n[Vertical stitch] {len(rows)} rows...")
    result = rows[0]
    for i, row in enumerate(rows[1:], 2):
        print(f"\n  Row {i}: trying cv2.Stitcher (SCANS)...")
        combined = try_opencv_stitcher([result, row])
        if combined is not None:
            result = combined
            continue

        print(f"  Stitcher failed, using manual translation matching...")
        trans = match_strip_translation_vertical(result, row)
        if trans is None:
            # Simple vertical stack
            w = min(result.shape[1], row.shape[1])
            result = np.vstack([result[:, :w], row[:, :w]])
        else:
            tx, ty = trans
            corners = [(0, 0, result.shape[1], result.shape[0]),
                       (tx, ty, tx+row.shape[1], ty+row.shape[0])]
            result = place_on_canvas([result, row], corners)
    return result


def match_strip_translation_vertical(img_top, img_bottom, overlap_frac=0.25):
    """Find vertical translation between img_top and img_bottom."""
    h_t, w_t = img_top.shape[:2]
    h_b, w_b = img_bottom.shape[:2]

    strip_h = int(min(h_t, h_b) * overlap_frac)
    ref_strip = img_top[h_t - strip_h:, :]
    mov_strip = img_bottom[:strip_h, :]

    gray_ref = cv2.cvtColor(ref_strip, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(mov_strip, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01)
    kp1, des1 = sift.detectAndCompute(gray_ref, None)
    kp2, des2 = sift.detectAndCompute(gray_mov, None)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return None

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=100))
    raw = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.68 * n.distance]
    if len(good) < 5:
        return None

    y_top_start = h_t - strip_h
    txs, tys = [], []
    for m in good:
        x1, y1 = kp1[m.queryIdx].pt
        x2, y2 = kp2[m.trainIdx].pt
        txs.append(x1 - x2)
        tys.append((y_top_start + y1) - y2)

    tx_med = np.median(txs)
    ty_med = np.median(tys)
    inliers = (np.abs(np.array(txs)-tx_med) < 15) & (np.abs(np.array(tys)-ty_med) < 15)
    if inliers.sum() < 4:
        return None

    tx_final = float(np.median(np.array(txs)[inliers]))
    ty_final = float(np.median(np.array(tys)[inliers]))
    print(f"  Vertical translation: tx={tx_final:.1f}  ty={ty_final:.1f}  inliers={inliers.sum()}")
    return tx_final, ty_final


# ──────────────────────────────────────────────────────────
# 4. Main
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
    print(f"\nSaved row {row_tag}: {out}  ({result.shape[1]}x{result.shape[0]})")
    return result


if __name__ == "__main__":
    row_imgs = {}
    for rtag, files in ROW_FILES.items():
        if not files:
            continue
        r = process_row(rtag, files)
        if r is not None:
            row_imgs[rtag] = r

    rows_in_order = [row_imgs[k] for k in sorted(row_imgs) if k in row_imgs]

    if len(rows_in_order) >= 2:
        print(f"\n{'='*55}\n Final vertical stitch\n{'='*55}")
        final = stitch_vertical(rows_in_order)
        out = OUTPUT_DIR + "final_map.png"
        cv2.imwrite(out, final)
        print(f"\nFinal map: {out}  ({final.shape[1]}x{final.shape[0]})")
    elif rows_in_order:
        out = OUTPUT_DIR + "row_1_result.png"
        cv2.imwrite(out, rows_in_order[0])
        print(f"\nRow 1 only. Saved: {out}  ({rows_in_order[0].shape[1]}x{rows_in_order[0].shape[0]})")

    print("\nDone.")
