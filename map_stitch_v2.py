"""
Map stitching v2 — robust approach.

Key improvements:
- Better map-content crop using the map's thin outer frame
- Full-image (not strip) SIFT feature matching between adjacent images
- Translation-priority homography (handles minimal distortion maps)
- Seam-cut blending (no text doubling)
"""

import cv2
import numpy as np
import os
from pathlib import Path

UPLOAD_DIR  = "/root/.claude/uploads/1b9fe45d-a299-5f9c-9937-d45d27c12f22/"
OUTPUT_DIR  = "/home/user/yurumemo-johokyoku/map_output/"
CORRECTED   = OUTPUT_DIR + "corrected/"
ROWS_DIR    = OUTPUT_DIR + "rows/"
DEBUG_DIR   = OUTPUT_DIR + "debug/"

for d in [OUTPUT_DIR, CORRECTED, ROWS_DIR, DEBUG_DIR]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────
# Perspective / crop helpers
# ─────────────────────────────────────────────────────────

def order_pts(pts):
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1)
    rect[0] = pts[s.argmin()]   # TL
    rect[2] = pts[s.argmax()]   # BR
    rect[1] = pts[d.argmin()]   # TR
    rect[3] = pts[d.argmax()]   # BL
    return rect


def find_map_rect(img, tag=""):
    """
    Find the rectangular map-content area inside the photo.
    Strategy: detect the thin outer frame of the map using morphology.
    Returns 4 corner points (TL,TR,BR,BL) in the original image, or None.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Find bright (white/light) map region by thresholding
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Morphological close to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    bright_clean = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    # 2) Find the largest connected bright rectangle
    cnts, _ = cv2.findContours(bright_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Sort by area (largest first)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for cnt in cnts[:5]:
        area = cv2.contourArea(cnt)
        if area < h * w * 0.05:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            # Check aspect ratio is reasonable (not tiny sliver)
            rect = cv2.minAreaRect(pts)
            bw, bh = rect[1]
            if min(bw, bh) > 0 and max(bw, bh) / min(bw, bh) < 5:
                print(f"  [{tag}] quad from bright contour, area={area:.0f}")
                return order_pts(pts)

    # 3) Fallback: use the bounding box of all bright pixels
    ys, xs = np.where(bright_clean > 0)
    if len(xs) < 100:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    # Add 1% margin
    mx = int((x1 - x0) * 0.01)
    my = int((y1 - y0) * 0.01)
    x0, x1 = max(0, x0 + mx), min(w, x1 - mx)
    y0, y1 = max(0, y0 + my), min(h, y1 - my)
    print(f"  [{tag}] bbox fallback: ({x0},{y0})-({x1},{y1})")
    return order_pts(np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32))


def warp_to_rect(img, quad, tag=""):
    """Perspective-warp img so that quad becomes a rectangle."""
    pts = order_pts(quad)
    tl, tr, br, bl = pts
    w = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
    h = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
    dst = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LANCZOS4)
    print(f"  [{tag}] warped to {w}x{h}")
    return warped, M


def auto_crop_to_map(img, tag=""):
    """
    Detect & perspective-correct the map area in a photo.
    Returns cropped image.
    """
    quad = find_map_rect(img, tag)
    if quad is not None:
        warped, _ = warp_to_rect(img, quad, tag)
        # Trim small black/dark border that can remain after warp
        warped = trim_dark_border(warped)
    else:
        print(f"  [{tag}] no quad found, using full image")
        warped = img.copy()
        warped = trim_dark_border(warped)
    return warped


def trim_dark_border(img, threshold=15, min_frac=0.03):
    """Remove near-black border rows/cols around image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Per-row and per-column mean
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    min_val = threshold

    r_start = next((i for i, v in enumerate(row_means) if v > min_val), 0)
    r_end   = next((i for i, v in enumerate(reversed(row_means)) if v > min_val), 0)
    c_start = next((i for i, v in enumerate(col_means) if v > min_val), 0)
    c_end   = next((i for i, v in enumerate(reversed(col_means)) if v > min_val), 0)

    r_end = h - r_end
    c_end = w - c_end
    # Don't over-trim (keep at least 90% of image)
    if (r_end - r_start) < h * 0.5 or (c_end - c_start) < w * 0.5:
        return img
    return img[r_start:r_end, c_start:c_end]


def skew_angle(img):
    """Measure skew from near-horizontal lines at top/bottom margins."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sh = max(40, h // 8)
    strip = np.vstack([gray[:sh], gray[-sh:]])
    edges = cv2.Canny(cv2.GaussianBlur(strip, (5,5), 0), 30, 90)
    lines = cv2.HoughLinesP(edges, 1, np.pi/720, 40,
                             minLineLength=w//4, maxLineGap=30)
    if lines is None: return 0.0
    angles = []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        a = np.degrees(np.arctan2(y2-y1, x2-x1))
        if abs(a) < 8: angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def deskew(img):
    """Correct small rotation."""
    angle = skew_angle(img)
    if abs(angle) < 0.2:
        return img
    print(f"  Deskew: {angle:.2f}°")
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
    cos_a, sin_a = abs(M[0,0]), abs(M[0,1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0,2] += (nw - w) / 2
    M[1,2] += (nh - h) / 2
    return cv2.warpAffine(img, M, (nw, nh),
                           flags=cv2.INTER_LANCZOS4,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(255,255,255))


# ─────────────────────────────────────────────────────────
# Feature matching & homography
# ─────────────────────────────────────────────────────────

def match_images(img_a, img_b, max_feat=10000):
    """
    Find homography that maps img_b → img_a coordinate space.
    Uses full images (not strips).
    Returns (H, n_inliers) or (None, 0).
    """
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=max_feat, contrastThreshold=0.015,
                            edgeThreshold=15, sigma=1.6)
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)
    print(f"  KP: a={len(kp_a)}, b={len(kp_b)}")

    if des_a is None or des_b is None or len(kp_a) < 10 or len(kp_b) < 10:
        return None, 0

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=150))
    try:
        raw = flann.knnMatch(des_a, des_b, k=2)
    except cv2.error:
        return None, 0

    good = [m for m,n in raw if m.distance < 0.70 * n.distance]
    print(f"  Good matches: {len(good)}")
    if len(good) < 10:
        return None, 0

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 3.0,
                                   maxIters=5000, confidence=0.995)
    n = int(mask.sum()) if mask is not None else 0
    print(f"  Homography inliers: {n}/{len(good)}")
    return H, n


def translation_only(H):
    """
    Extract pure translation from a homography H.
    If H is close to a translation, return it; otherwise extract translation component.
    """
    tx = H[0, 2]
    ty = H[1, 2]
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)
    return T


def is_valid_H(H, img_shape, max_shear=0.15, max_scale_diff=0.25):
    """
    Quick sanity check on homography.
    Returns True if H looks like a plausible camera motion for a flat map.
    """
    if H is None: return False
    h, w = img_shape[:2]
    # Check corners don't fly off
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    warped  = cv2.perspectiveTransform(corners, H).reshape(4,2)
    # New bounding box should be reasonable
    x_min, y_min = warped.min(axis=0)
    x_max, y_max = warped.max(axis=0)
    new_w = x_max - x_min
    new_h = y_max - y_min
    # Should not stretch more than 50% or compress more than 50%
    if new_w < w * 0.5 or new_w > w * 2.0: return False
    if new_h < h * 0.5 or new_h > h * 2.0: return False
    # Check that the shape is still roughly rectangular (parallelogram tolerance)
    top_w = np.linalg.norm(warped[1] - warped[0])
    bot_w = np.linalg.norm(warped[2] - warped[3])
    if max(top_w,bot_w) / (min(top_w,bot_w)+1) > 1.3: return False
    return True


# ─────────────────────────────────────────────────────────
# Canvas compositing
# ─────────────────────────────────────────────────────────

def warp_onto(canvas, canvas_mask, new_img, H):
    """
    Warp new_img using H onto canvas. Composite with seam-cut.
    Returns (new_canvas, new_mask).
    """
    hc, wc = canvas.shape[:2]
    hn, wn = new_img.shape[:2]

    warped = cv2.warpPerspective(new_img, H, (wc, hc),
                                  flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    warped_mask = cv2.warpPerspective(
        np.ones((hn, wn), np.uint8)*255, H, (wc, hc),
        flags=cv2.INTER_NEAREST)

    only_new  = (warped_mask > 0) & (canvas_mask == 0)
    overlap   = (warped_mask > 0) & (canvas_mask  > 0)

    out = canvas.copy()
    out_mask = canvas_mask.copy()

    # Non-overlapping new region: take new_img
    out[only_new] = warped[only_new]
    out_mask[only_new] = 255

    # Overlap region: seam at the midpoint column (or row for vertical stitch)
    if overlap.any():
        overlap_cols = np.where(overlap.any(axis=0))[0]
        overlap_rows = np.where(overlap.any(axis=1))[0]

        # Determine if this is predominantly horizontal or vertical overlap
        col_span = overlap_cols[-1] - overlap_cols[0] if len(overlap_cols) else 0
        row_span = overlap_rows[-1] - overlap_rows[0] if len(overlap_rows) else 0

        if col_span >= row_span:
            # Vertical seam (horizontal stitch)
            seam = int(overlap_cols.mean())
            # Right of seam: use new image; left: keep canvas
            right_of_seam = np.zeros((hc, wc), bool)
            right_of_seam[:, seam:] = True
            use_new = overlap & right_of_seam
        else:
            # Horizontal seam (vertical stitch)
            seam = int(overlap_rows.mean())
            below_seam = np.zeros((hc, wc), bool)
            below_seam[seam:, :] = True
            use_new = overlap & below_seam

        out[use_new] = warped[use_new]

    return out, out_mask


def build_canvas(images, transforms):
    """
    Place all images on a canvas using their transforms.
    transforms[i] maps image i to canvas coordinates.
    """
    # Find canvas bounds by projecting all corners
    all_corners = []
    for img, H in zip(images, transforms):
        h, w = img.shape[:2]
        c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        tc = cv2.perspectiveTransform(c, H).reshape(4,2)
        all_corners.append(tc)
    all_corners = np.vstack(all_corners)
    x_min = int(np.floor(all_corners[:,0].min()))
    y_min = int(np.floor(all_corners[:,1].min()))
    x_max = int(np.ceil( all_corners[:,0].max()))
    y_max = int(np.ceil( all_corners[:,1].max()))

    # Translation to make all coords positive
    tx = max(0, -x_min)
    ty = max(0, -y_min)
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)

    cw = x_max - x_min + 1
    ch = y_max - y_min + 1
    print(f"  Canvas: {cw}x{ch}  (offset tx={tx}, ty={ty})")

    canvas = np.zeros((ch, cw, 3), np.uint8)
    canvas_mask = np.zeros((ch, cw), np.uint8)

    for i, (img, H) in enumerate(zip(images, transforms)):
        H_adj = T @ H
        canvas, canvas_mask = warp_onto(canvas, canvas_mask, img, H_adj)
        print(f"  Placed image {i+1}")

    # Crop to content
    rows = np.where(canvas_mask.any(1))[0]
    cols = np.where(canvas_mask.any(0))[0]
    if len(rows) and len(cols):
        canvas = canvas[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    return canvas


# ─────────────────────────────────────────────────────────
# Row stitching
# ─────────────────────────────────────────────────────────

def stitch_row_horizontal(images, tag):
    """
    Stitch a list of images left→right.
    Computes pairwise transforms and builds a single canvas.
    """
    n = len(images)
    if n == 1:
        return images[0]

    # Compute cumulative transforms: each image relative to image 0
    # H_chain[i] maps image i → image 0 space
    H_chain = [np.eye(3, dtype=np.float64)]   # image 0 is identity

    ref = images[0]
    for i in range(1, n):
        print(f"\n  Matching image {i+1} → image {i}...")
        H, n_inl = match_images(ref, images[i])

        if H is None or n_inl < 8 or not is_valid_H(H, images[i].shape):
            print(f"  WARNING: bad homography, using translation estimate")
            # Estimate offset: new image placed just to the right with 20% overlap
            prev_w = ref.shape[1]
            dx = int(prev_w * 0.80)
            H = np.array([[1,0,dx],[0,1,0],[0,0,1]], np.float64)

        # If the homography looks very non-affine (large projective component),
        # fall back to affine to avoid explosion
        if abs(H[2,0]) > 1e-3 or abs(H[2,1]) > 1e-3:
            print("  Detected strong projective component — using affine fallback")
            H[2, :] = [0, 0, 1]

        # Chain: H_total_i = H_{i→i-1} × H_total_{i-1}
        H_chain.append(H @ H_chain[-1])
        ref = images[i]   # use current as reference for next pair

    print(f"\n  Building canvas ({n} images)...")
    result = build_canvas(images, H_chain)
    return result


def stitch_rows_vertical(rows, tag="final"):
    """Vertically stack row images."""
    if len(rows) == 1:
        return rows[0]

    result = rows[0]
    for i, row in enumerate(rows[1:], 2):
        print(f"\n  Vertical stitch: adding row {i}...")
        H, n_inl = match_images(result, row)

        if H is None or n_inl < 8 or not is_valid_H(H, row.shape):
            print(f"  WARNING: vertical alignment failed, using simple stack")
            w = min(result.shape[1], row.shape[1])
            result = np.vstack([result[:, :w], row[:, :w]])
            continue

        if abs(H[2,0]) > 1e-3 or abs(H[2,1]) > 1e-3:
            H[2,:] = [0,0,1]

        result = build_canvas([result, row], [np.eye(3), H])

    return result


# ─────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────

def preprocess(path, tag):
    """Load → crop to map → deskew."""
    print(f"\n[Pre] {tag}  ({Path(path).name})")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    print(f"  Loaded: {img.shape[1]}x{img.shape[0]}")
    img = auto_crop_to_map(img, tag)
    img = deskew(img)
    print(f"  Final:  {img.shape[1]}x{img.shape[0]}")
    cv2.imwrite(CORRECTED + f"{tag}.jpg", img,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return img


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

ROW_FILES = {
    "1": ["69a424a3-IMG_6837.jpeg",
          "95841777-IMG_6838.jpeg",
          "d9b6fffb-IMG_6839.jpeg"],
    "2": [],
    "3": [],
    "4": [],
}


def process_row(row_tag, file_names):
    paths = [UPLOAD_DIR + f for f in file_names]
    present = [(p, f"r{row_tag}_{i+1}") for i,(p,_) in enumerate(zip(paths, file_names))
               if Path(p).exists()]
    if not present:
        print(f"\nRow {row_tag}: no files, skipping.")
        return None
    if len(present) < len(file_names):
        print(f"\nRow {row_tag}: {len(present)}/{len(file_names)} files present.")

    print(f"\n{'='*55}")
    print(f" Row {row_tag}  ({len(present)} images)")
    print(f"{'='*55}")

    imgs = [preprocess(p, tag) for p, tag in present]

    print(f"\n[Stitch] Row {row_tag}")
    result = stitch_row_horizontal(imgs, row_tag)

    out = ROWS_DIR + f"row_{row_tag}.png"
    cv2.imwrite(out, result)
    print(f"\nSaved: {out}  ({result.shape[1]}x{result.shape[0]})")
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
        print(f"\n{'='*55}")
        print(f" Final vertical stitch ({len(rows_in_order)} rows)")
        print(f"{'='*55}")
        final = stitch_rows_vertical(rows_in_order)
        out = OUTPUT_DIR + "final_map.png"
        cv2.imwrite(out, final)
        print(f"\nFinal map: {out}  ({final.shape[1]}x{final.shape[0]})")
    elif len(rows_in_order) == 1:
        out = OUTPUT_DIR + "row_1_stitched.png"
        cv2.imwrite(out, rows_in_order[0])
        print(f"\nOnly row 1 available. Saved: {out}  ({rows_in_order[0].shape[1]}x{rows_in_order[0].shape[0]})")

    print("\nDone.")
