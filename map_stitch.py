"""
Map photo stitching script.
Processes smartphone photos of a physical map and stitches them into one image.

Layout:
  Row 1: photos 1,2,3  (IMG_6837, 6838, 6839)
  Row 2: photos 4,5,6  (IMG_6840, 6841, 6842)
  Row 3: photos 7,8,9  (IMG_6843, 6844, 6845)
  Row 4: photos 10,11,12 (IMG_6846, 6847, 6848)  -- 2 grid tall, 3 grid wide

Photos 10-12 are taller (2 grid rows x 3 grid cols).
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

UPLOAD_DIR = "/root/.claude/uploads/1b9fe45d-a299-5f9c-9937-d45d27c12f22/"
OUTPUT_DIR = "/home/user/yurumemo-johokyoku/map_output/"
DEBUG_DIR = OUTPUT_DIR + "debug/"
CORRECTED_DIR = OUTPUT_DIR + "corrected/"
ROWS_DIR = OUTPUT_DIR + "rows/"

for d in [OUTPUT_DIR, DEBUG_DIR, CORRECTED_DIR, ROWS_DIR]:
    os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────
# Image loading
# ──────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")
    print(f"  Loaded {Path(path).name}: {img.shape[1]}x{img.shape[0]}")
    return img


# ──────────────────────────────────────────────
# Perspective correction
# Uses the map outer frame (straight edges) to
# unwarp trapezoidal distortion.
# ──────────────────────────────────────────────

def order_points(pts: np.ndarray) -> np.ndarray:
    """Return 4 points in [TL, TR, BR, BL] order."""
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL
    rect[2] = pts[np.argmax(s)]   # BR
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # TR
    rect[3] = pts[np.argmax(d)]   # BL
    return rect


def find_map_quad(img: np.ndarray, debug_name: str = None):
    """
    Detect the four corners of the map's rectangular frame.
    Returns ordered (TL,TR,BR,BL) array or None if detection fails.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Boost contrast so the frame edge stands out
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Canny edges
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    # Dilate slightly to connect edge fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_d = cv2.dilate(edges, kernel, iterations=1)

    if debug_name:
        cv2.imwrite(DEBUG_DIR + f"{debug_name}_edges.jpg", edges_d)

    # Find contours and pick the largest quad
    contours, _ = cv2.findContours(edges_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area = 0
    best_quad = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < (h * w * 0.1):   # ignore tiny contours
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2)

    if best_quad is not None:
        print(f"  Quad found via contour (area={best_area:.0f})")
        return order_points(best_quad)

    # Fallback: use Hough lines to find the four border lines
    lines = cv2.HoughLinesP(
        edges_d, 1, np.pi / 180,
        threshold=int(min(w, h) * 0.3),
        minLineLength=int(min(w, h) * 0.3),
        maxLineGap=30
    )
    if lines is not None:
        print(f"  Falling back to Hough lines ({len(lines)} found)")
        horizontal, vertical = [], []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10:
                horizontal.append((min(y1, y2) + max(y1, y2)) / 2)
            elif angle > 80:
                vertical.append((min(x1, x2) + max(x1, x2)) / 2)
        if horizontal and vertical:
            top    = min(horizontal)
            bottom = max(horizontal)
            left   = min(vertical)
            right  = max(vertical)
            quad = np.array([[left, top], [right, top],
                              [right, bottom], [left, bottom]], dtype=np.float32)
            print(f"  Quad from Hough: L={left:.0f} R={right:.0f} T={top:.0f} B={bottom:.0f}")
            return order_points(quad)

    print("  WARNING: Could not detect map border — using full image")
    return None


def perspective_correct(img: np.ndarray, src_quad=None, tag: str = "") -> np.ndarray:
    """
    Warp img to a rectangle.
    If src_quad is None, returns img with only a small crop applied.
    """
    h, w = img.shape[:2]

    if src_quad is None:
        # Just remove likely photo margins (camera keeps content centered)
        margin_frac = 0.02
        mx = int(w * margin_frac)
        my = int(h * margin_frac)
        return img[my:h-my, mx:w-mx]

    pts = order_points(src_quad)
    tl, tr, br, bl = pts

    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    dst_w = int(max(width_top, width_bottom))

    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    dst_h = int(max(height_left, height_right))

    dst = np.array([[0, 0], [dst_w - 1, 0],
                    [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (dst_w, dst_h),
                                  flags=cv2.INTER_LANCZOS4)
    print(f"  {tag} warped: {dst_w}x{dst_h}")
    return warped


# ──────────────────────────────────────────────
# Grid-line straightness check & correction
# ──────────────────────────────────────────────

def detect_grid_skew(img: np.ndarray) -> float:
    """
    Returns the dominant skew angle (degrees) of the grid lines.
    Uses Hough transform on a small strip near image edges.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Use the top and bottom horizontal strips
    strip_h = max(50, h // 10)
    strip = np.vstack([gray[:strip_h, :], gray[h-strip_h:, :]])

    edges = cv2.Canny(strip, 30, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720,
                             threshold=50, minLineLength=w // 4, maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 5:     # near-horizontal
            angles.append(a)
    if not angles:
        return 0.0
    return float(np.median(angles))


def correct_rotation(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by -angle_deg to correct skew."""
    if abs(angle_deg) < 0.1:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ──────────────────────────────────────────────
# Feature-based alignment
# ──────────────────────────────────────────────

def match_and_align(img_ref: np.ndarray, img_mov: np.ndarray,
                    overlap_hint: float = 0.25) -> tuple:
    """
    Compute homography to map img_mov onto img_ref's coordinate system.
    overlap_hint: expected fraction of image width that overlaps.
    Returns (H, n_inliers) or (None, 0).
    """
    h_r, w_r = img_ref.shape[:2]
    h_m, w_m = img_mov.shape[:2]

    # Only search in the overlapping strip to speed things up
    ovlp_w = int(w_r * (overlap_hint + 0.1))
    ref_strip = img_ref[:, max(0, w_r - ovlp_w):]
    mov_strip = img_mov[:, :min(w_m, ovlp_w)]

    gray_ref = cv2.cvtColor(ref_strip, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(mov_strip, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.02)
    kp1, d1 = sift.detectAndCompute(gray_ref, None)
    kp2, d2 = sift.detectAndCompute(gray_mov, None)

    if d1 is None or d2 is None or len(kp1) < 10 or len(kp2) < 10:
        print(f"  Too few keypoints: ref={len(kp1) if kp1 else 0}, mov={len(kp2) if kp2 else 0}")
        return None, 0

    # FLANN matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(d1, d2, k=2)

    good = [m for m, n in raw if m.distance < 0.72 * n.distance]
    print(f"  Feature matches (after ratio test): {len(good)}")

    if len(good) < 8:
        return None, 0

    # Translate strip keypoints back to full-image coordinates
    x_offset_ref = max(0, w_r - ovlp_w)
    pts_ref = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_ref[:, 0] += x_offset_ref

    pts_mov = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_mov, pts_ref, cv2.RANSAC, 4.0)
    n_inliers = int(mask.sum()) if mask is not None else 0
    print(f"  Homography inliers: {n_inliers}/{len(good)}")
    return H, n_inliers


# ──────────────────────────────────────────────
# Canvas compositing with seam-cut blending
# ──────────────────────────────────────────────

def warp_onto_canvas(canvas: np.ndarray, canvas_mask: np.ndarray,
                     img: np.ndarray, H: np.ndarray) -> tuple:
    """
    Warp img into canvas coordinates using H, then composite with seam-cut.
    Returns updated (canvas, canvas_mask).
    """
    h_c, w_c = canvas.shape[:2]
    h_i, w_i = img.shape[:2]

    # Warp image onto canvas
    warped = cv2.warpPerspective(img, H, (w_c, h_c),
                                  flags=cv2.INTER_LANCZOS4)
    warped_mask = cv2.warpPerspective(
        np.ones((h_i, w_i), dtype=np.uint8) * 255,
        H, (w_c, h_c), flags=cv2.INTER_NEAREST
    )

    overlap = (canvas_mask > 0) & (warped_mask > 0)

    if overlap.any():
        # Find the seam column (vertical seam at middle of overlap)
        overlap_cols = np.where(overlap.any(axis=0))[0]
        seam_col = int(overlap_cols.mean())

        # Left of seam: keep canvas; right of seam: use warped
        # This avoids text doubling
        seam_mask = np.zeros((h_c, w_c), dtype=bool)
        seam_mask[:, seam_col:] = True

        # For pixels only in warped (no overlap), just take warped
        only_warped = (warped_mask > 0) & (canvas_mask == 0)

        # Apply
        composite = canvas.copy()
        composite[only_warped] = warped[only_warped]
        composite[overlap & seam_mask] = warped[overlap & seam_mask]

        new_mask = canvas_mask.copy()
        new_mask[only_warped] = 255
        return composite, new_mask
    else:
        # No overlap: just add the warped region
        composite = canvas.copy()
        only_warped = warped_mask > 0
        composite[only_warped] = warped[only_warped]
        new_mask = canvas_mask.copy()
        new_mask[only_warped] = 255
        return composite, new_mask


def stitch_row(images: list, tag: str = "row") -> np.ndarray:
    """
    Horizontally stitch a list of images (left→right order).
    Returns the stitched image.
    """
    assert len(images) >= 2, "Need at least 2 images"

    # Start with the leftmost image as the reference
    ref = images[0]
    h_r, w_r = ref.shape[:2]

    # Estimate canvas width: sum of all widths × 0.75 (account for overlaps)
    total_w = int(sum(im.shape[1] for im in images) * 0.80)
    total_h = max(im.shape[0] for im in images)

    # Build canvas large enough
    canvas_w = max(total_w, w_r * len(images))
    canvas_h = total_h + 200  # extra vertical space

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    # Place reference image at (0, 0) on canvas
    canvas[:h_r, :w_r] = ref
    canvas_mask[:h_r, :w_r] = 255

    # Keep track of cumulative transform (starts as identity)
    for i, img in enumerate(images[1:], start=1):
        print(f"\n  Aligning image {i+1} onto canvas...")
        H, n_inliers = match_and_align(canvas[:, :canvas_w], img)

        if H is None or n_inliers < 6:
            print(f"  WARNING: Alignment failed for image {i+1}, using translation estimate")
            # Estimate translation from canvas mask: place just after the current content
            cols_filled = np.where(canvas_mask.any(axis=0))[0]
            x_offset = int(cols_filled.max()) - int(img.shape[1] * 0.25) if len(cols_filled) else 0
            x_offset = max(0, x_offset)
            H = np.array([[1, 0, x_offset], [0, 1, 0], [0, 0, 1]], dtype=np.float64)

        canvas, canvas_mask = warp_onto_canvas(canvas, canvas_mask, img, H)
        print(f"  Canvas filled cols: {np.where(canvas_mask.any(axis=0))[0][[0,-1]]}")

    # Crop canvas to content
    rows = np.where(canvas_mask.any(axis=1))[0]
    cols = np.where(canvas_mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return canvas
    cropped = canvas[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    print(f"\n  Final stitched size: {cropped.shape[1]}x{cropped.shape[0]}")
    return cropped


# ──────────────────────────────────────────────
# Vertical stitching (for stacking rows)
# ──────────────────────────────────────────────

def stitch_vertical(img_top: np.ndarray, img_bottom: np.ndarray) -> np.ndarray:
    """
    Vertically stitch two images (top / bottom).
    Uses feature matching on a horizontal overlap strip.
    """
    h_t, w_t = img_top.shape[:2]
    h_b, w_b = img_bottom.shape[:2]

    # Strip at the bottom of top image and top of bottom image
    strip_h = max(80, min(h_t, h_b) // 5)
    ref_strip = img_top[h_t - strip_h:, :]
    mov_strip = img_bottom[:strip_h, :]

    gray_ref = cv2.cvtColor(ref_strip, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(mov_strip, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    kp1, d1 = sift.detectAndCompute(gray_ref, None)
    kp2, d2 = sift.detectAndCompute(gray_mov, None)

    good_matches = []
    if d1 is not None and d2 is not None and len(kp1) >= 8 and len(kp2) >= 8:
        FLANN_INDEX_KDTREE = 1
        flann = cv2.FlannBasedMatcher(
            dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=100))
        raw = flann.knnMatch(d1, d2, k=2)
        good_matches = [m for m, n in raw if m.distance < 0.72 * n.distance]

    print(f"  Vertical matches: {len(good_matches)}")

    if len(good_matches) >= 8:
        pts_top = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts_top[:, 1] += (h_t - strip_h)  # full image coords
        pts_bot = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        H, mask = cv2.findHomography(pts_bot, pts_top, cv2.RANSAC, 4.0)
        n_inliers = int(mask.sum()) if mask is not None else 0
        print(f"  Vertical homography inliers: {n_inliers}")
    else:
        H = None

    # Fallback: simple vertical stack with overlap trim
    if H is None:
        print("  Vertical alignment failed, using simple stack")
        # Trim bottom image to match top width
        w = min(w_t, w_b)
        return np.vstack([img_top[:, :w], img_bottom[:, :w]])

    # Canvas
    canvas_h = h_t + h_b + 200
    canvas_w = max(w_t, w_b) + 200
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    canvas[:h_t, :w_t] = img_top
    canvas_mask[:h_t, :w_t] = 255

    canvas, canvas_mask = warp_onto_canvas(canvas, canvas_mask, img_bottom, H)

    rows = np.where(canvas_mask.any(axis=1))[0]
    cols = np.where(canvas_mask.any(axis=0))[0]
    return canvas[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]


# ──────────────────────────────────────────────
# Per-image pre-processing pipeline
# ──────────────────────────────────────────────

def preprocess(img: np.ndarray, tag: str) -> np.ndarray:
    """Perspective-correct + rotation-correct a single map photo."""
    print(f"\n[Preprocess] {tag}")
    quad = find_map_quad(img, debug_name=tag)
    corrected = perspective_correct(img, quad, tag=tag)
    skew = detect_grid_skew(corrected)
    print(f"  Detected skew: {skew:.2f}°")
    if abs(skew) > 0.15:
        corrected = correct_rotation(corrected, skew)
    cv2.imwrite(CORRECTED_DIR + f"{tag}.jpg", corrected,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return corrected


# ──────────────────────────────────────────────
# Main: process whichever rows are available
# ──────────────────────────────────────────────

def process_row(file_names: list, row_tag: str) -> np.ndarray | None:
    """Load, preprocess, stitch a row. Returns stitched image or None."""
    paths = [UPLOAD_DIR + f for f in file_names]
    # Check which files exist
    existing = [(p, tag) for p, tag in zip(paths, [f"r{row_tag}_{i+1}" for i in range(len(paths))]) if Path(p).exists()]
    if not existing:
        print(f"\nRow {row_tag}: no files found, skipping.")
        return None
    if len(existing) < len(file_names):
        missing = [p for p, _ in zip(paths, range(len(paths))) if not Path(p).exists()]
        print(f"\nRow {row_tag}: WARNING — {len(existing)}/{len(file_names)} files found.")

    print(f"\n{'='*50}")
    print(f"Processing Row {row_tag} ({len(existing)} images)")
    print(f"{'='*50}")

    images = []
    for path, tag in existing:
        img = load_image(path)
        processed = preprocess(img, tag)
        images.append(processed)

    if len(images) == 1:
        result = images[0]
    else:
        print(f"\n[Stitch] Row {row_tag}")
        result = stitch_row(images, tag=row_tag)

    out_path = ROWS_DIR + f"row_{row_tag}.jpg"
    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nSaved: {out_path}  ({result.shape[1]}x{result.shape[0]})")
    return result


def final_stitch(row_results: list) -> np.ndarray:
    """Vertically stitch all row images."""
    valid = [r for r in row_results if r is not None]
    if len(valid) == 0:
        raise RuntimeError("No rows to stitch")
    if len(valid) == 1:
        return valid[0]

    print(f"\n{'='*50}")
    print(f"Final vertical stitch: {len(valid)} rows")
    print(f"{'='*50}")

    result = valid[0]
    for i, row_img in enumerate(valid[1:], start=2):
        print(f"\nStitching row {i} onto result...")
        result = stitch_vertical(result, row_img)

    return result


# ──────────────────────────────────────────────
# File name registry
# (update when more uploads arrive)
# ──────────────────────────────────────────────

ROW_FILES = {
    "1": ["69a424a3-IMG_6837.jpeg",   # photo 1
          "95841777-IMG_6838.jpeg",   # photo 2
          "d9b6fffb-IMG_6839.jpeg"],  # photo 3
    "2": [],   # photos 4-6 — not yet uploaded
    "3": [],   # photos 7-9 — not yet uploaded
    "4": [],   # photos 10-12 (tall) — not yet uploaded
}


if __name__ == "__main__":
    row_results = {}
    for row_tag, file_names in ROW_FILES.items():
        if not file_names:
            continue
        result = process_row(file_names, row_tag)
        row_results[row_tag] = result

    available_rows = [row_results[k] for k in sorted(row_results) if row_results.get(k) is not None]

    if len(available_rows) >= 2:
        final = final_stitch(available_rows)
        cv2.imwrite(OUTPUT_DIR + "final_map.png", final)
        print(f"\nFinal map saved: {OUTPUT_DIR}final_map.png  ({final.shape[1]}x{final.shape[0]})")
    elif len(available_rows) == 1:
        out = OUTPUT_DIR + "row_1_stitched.png"
        cv2.imwrite(out, available_rows[0])
        print(f"\nOnly row 1 available. Saved: {out}")
    else:
        print("\nNo images processed.")

    print("\nDone.")
