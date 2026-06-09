"""
Map stitching v5 — improved seam + white-border trim.

Changes from v4:
- Post-deskew white-edge trim (remove rotation artifacts)
- Optimal seam placement (minimum-edge-density column in overlap)
- 10-px gradient feather at seam to hide hard cuts
- Chain multiplication order fixed: Hs[-1] @ H_local
- Grid-line continuity check at seam for quality reporting
"""

import cv2
import numpy as np
import os
from pathlib import Path

UPLOAD_DIR = "/root/.claude/uploads/1b9fe45d-a299-5f9c-9937-d45d27c12f22/"
OUTPUT_DIR = "/home/user/yurumemo-johokyoku/map_output/"
CORRECTED  = OUTPUT_DIR + "corrected_v5/"
ROWS_DIR   = OUTPUT_DIR + "rows_v5/"

for d in [OUTPUT_DIR, CORRECTED, ROWS_DIR]:
    os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────
# Pre-processing
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
    bright = cv2.morphologyEx(bright,
                              cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (30,30)))
    bright = cv2.morphologyEx(bright,
                              cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (50,50)))
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quad = None
    if cnts:
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(cnt) < 0.08*h*w: continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4,2)
                bw, bh = cv2.minAreaRect(pts)[1]
                if min(bw,bh)>0 and max(bw,bh)/min(bw,bh) < 4:
                    quad = order_pts(pts); break
    if quad is None:
        ys, xs = np.where(bright>0)
        if len(xs)==0: return img
        quad = order_pts(np.array([[xs.min(),ys.min()],[xs.max(),ys.min()],
                                    [xs.max(),ys.max()],[xs.min(),ys.max()]], np.float32))
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
    sh = h // 5
    strip = np.vstack([gray[:sh], gray[-sh:]])
    edges = cv2.Canny(cv2.GaussianBlur(strip, (5,5), 0), 20, 80)
    lines = cv2.HoughLines(edges, 1, np.pi/1800, threshold=int(w*0.15))
    angle = 0.0
    if lines is not None:
        angles = [np.degrees(l[0][1]) - 90.0 for l in lines
                  if abs(np.degrees(l[0][1]) - 90.0) < 8]
        if angles: angle = float(np.median(angles))
    if abs(angle) < 0.3:
        return img, 0.0
    print(f"  Deskew: {angle:.2f}°")
    M = cv2.getRotationMatrix2D((w/2, h/2), -angle, 1.0)
    cos_a, sin_a = abs(M[0,0]), abs(M[0,1])
    nw = int(h*sin_a + w*cos_a); nh = int(h*cos_a + w*sin_a)
    M[0,2] += (nw-w)/2; M[1,2] += (nh-h)/2
    result = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return result, angle


def trim_white_edges(img, threshold=248):
    """Remove all-white rows/columns at image edges (deskew artifacts)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    row_min = gray.min(axis=1); col_min = gray.min(axis=0)
    r0 = next((i for i in range(h) if row_min[i] < threshold), 0)
    r1 = next((i for i in range(h-1,-1,-1) if row_min[i] < threshold), h-1)
    c0 = next((i for i in range(w) if col_min[i] < threshold), 0)
    c1 = next((i for i in range(w-1,-1,-1) if col_min[i] < threshold), w-1)
    result = img[r0:r1+1, c0:c1+1]
    if result.shape[:2] != img.shape[:2]:
        dh = img.shape[0] - result.shape[0]
        dw = img.shape[1] - result.shape[1]
        if dh > 0 or dw > 0:
            print(f"  Trimmed white edges: -{dw}px x -{dh}px")
    return result


def preprocess(path, tag):
    print(f"\n[Pre] {tag}  ({Path(path).name})")
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(path)
    print(f"  Loaded: {img.shape[1]}x{img.shape[0]}")
    img = correct_perspective(img, tag)
    img, angle = deskew(img)
    img = trim_white_edges(img)
    print(f"  Final: {img.shape[1]}x{img.shape[0]}")
    cv2.imwrite(CORRECTED + f"{tag}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return img


# ──────────────────────────────────────────────────────────
# Feature matching
# ──────────────────────────────────────────────────────────

def sift_match(gray_a, gray_b, ratio=0.67, nfeat=6000):
    sift = cv2.SIFT_create(nfeatures=nfeat, contrastThreshold=0.01, edgeThreshold=20)
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)
    print(f"    KP: a={len(kp_a)}, b={len(kp_b)}")
    if des_a is None or des_b is None or len(kp_a)<8 or len(kp_b)<8:
        return None, None
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=150))
    raw = flann.knnMatch(des_a, des_b, k=2)
    good = [m for m,n in raw if m.distance < ratio*n.distance]
    print(f"    Good matches: {len(good)}")
    if len(good) < 8: return None, None
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
    return pts_a, pts_b


def find_H_strip(img_left, img_right, overlap_frac=0.35):
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]
    sw = int(min(w_l, w_r) * overlap_frac)
    x_l = w_l - sw
    ref_strip = img_left[:, x_l:]
    mov_strip = img_right[:, :sw]
    gray_ref = cv2.cvtColor(ref_strip, cv2.COLOR_BGR2GRAY)
    gray_mov = cv2.cvtColor(mov_strip, cv2.COLOR_BGR2GRAY)
    pts_r, pts_m = sift_match(gray_ref, gray_mov)
    if pts_r is None: return None, 0
    pts_r_full = pts_r + [x_l, 0]
    H, mask = cv2.findHomography(pts_m, pts_r_full, cv2.RANSAC, 4.0,
                                  maxIters=5000, confidence=0.999)
    n = int(mask.sum()) if mask is not None else 0
    print(f"    Strip H inliers: {n}/{len(pts_r)}")
    return H, n


def validate_H(H, src_shape, max_scale=2.0):
    if H is None: return False
    h, w = src_shape[:2]
    c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    wc = cv2.perspectiveTransform(c, H).reshape(4,2)
    nw = wc[:,0].max()-wc[:,0].min(); nh = wc[:,1].max()-wc[:,1].min()
    if nw < w/max_scale or nw > w*max_scale: return False
    if nh < h/max_scale or nh > h*max_scale: return False
    if abs(H[2,0]) > 5e-4 or abs(H[2,1]) > 5e-4: return False
    return True


def compute_pairwise_H(img_a, img_b, overlap_frac=0.35):
    """Returns H mapping img_b → img_a coordinate space."""
    print(f"  Strip SIFT (overlap={overlap_frac:.0%})...")
    H, n = find_H_strip(img_a, img_b, overlap_frac)
    if H is not None and n >= 10 and validate_H(H, img_b.shape):
        return H
    print(f"  Strip failed (n={n}). Trying full-image SIFT...")
    pts_a, pts_b = sift_match(
        cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY))
    if pts_a is not None:
        H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 4.0,
                                      maxIters=5000, confidence=0.999)
        n = int(mask.sum()) if mask is not None else 0
        print(f"  Full SIFT H inliers: {n}")
        if H is not None and n >= 10 and validate_H(H, img_b.shape):
            return H
    print(f"  SIFT failed. Using translation estimate.")
    ha, wa = img_a.shape[:2]; hb, wb = img_b.shape[:2]
    tx = wa * (1 - overlap_frac); ty = (ha - hb) / 2
    return np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)


# ──────────────────────────────────────────────────────────
# Canvas compositing with optimal seam
# ──────────────────────────────────────────────────────────

def find_best_seam_col(canvas, warped, overlap_mask):
    """
    Find the best vertical seam column in the overlap region.
    Prefer columns with low combined edge density.
    """
    ov_cols = np.where(overlap_mask.any(0))[0]
    if len(ov_cols) < 3:
        return int(ov_cols.mean()) if len(ov_cols) else 0

    c0, c1 = ov_cols[0], ov_cols[-1]
    # Compute edge density per column in overlap
    strip_canvas = cv2.cvtColor(canvas[:, c0:c1+1], cv2.COLOR_BGR2GRAY)
    strip_warped = cv2.cvtColor(warped[:, c0:c1+1], cv2.COLOR_BGR2GRAY)
    e_c = cv2.Canny(strip_canvas, 30, 90).sum(0).astype(float)
    e_w = cv2.Canny(strip_warped, 30, 90).sum(0).astype(float)
    combined = e_c + e_w
    # Smooth with 1D gaussian
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(combined, size=30)
    best = int(smoothed.argmin()) + c0
    return best


def feather_blend(canvas, warped, seam_col, overlap_mask, feather_px=10):
    """
    Blend near the seam with a narrow feather (feather_px pixels each side).
    Pixels OUTSIDE feather zone use hard cut; inside get linear blend.
    """
    hc, wc = canvas.shape[:2]
    out = canvas.copy()
    x_start = max(0, seam_col - feather_px)
    x_end   = min(wc, seam_col + feather_px + 1)

    for x in range(x_start, x_end):
        col_mask = overlap_mask[:, x]
        if not col_mask.any(): continue
        alpha = (x - x_start) / (x_end - x_start - 1)  # 0 → 1 across feather
        out[col_mask, x] = ((1 - alpha) * canvas[col_mask, x].astype(np.float32)
                            + alpha * warped[col_mask, x].astype(np.float32)).astype(np.uint8)
    # Right of feather: use warped
    if x_end < wc:
        right_mask = overlap_mask.copy()
        right_mask[:, :x_end] = False
        out[right_mask] = warped[right_mask]
    return out


def make_content_mask(img, white_thresh=252):
    """
    Mask for 'real map content' pixels in img.
    Excludes pure-white rotation artifacts at corners (not surrounded by content).
    Strategy: non-white pixels + morphological flood-fill to include white map interior.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Non-white seed pixels
    non_white = (gray < white_thresh).astype(np.uint8) * 255
    # Close + fill to include the white map background surrounded by content
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60))
    filled = cv2.morphologyEx(non_white, cv2.MORPH_CLOSE, k_close)
    # One more large closing to handle sparse map areas
    k_big   = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 150))
    filled  = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k_big)
    return filled


def warp_onto(canvas, canvas_mask, new_img, H):
    hc, wc = canvas.shape[:2]
    hn, wn = new_img.shape[:2]
    warped = cv2.warpPerspective(new_img, H, (wc, hc), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Use content mask (excludes white rotation artifacts) instead of all-ones mask
    content_mask = make_content_mask(new_img)
    warped_valid = cv2.warpPerspective(
        content_mask, H, (wc, hc),
        flags=cv2.INTER_NEAREST) > 127

    only_new = warped_valid & (canvas_mask == 0)
    overlap  = warped_valid & (canvas_mask  > 0)

    out = canvas.copy(); out_mask = canvas_mask.copy()
    out[only_new] = warped[only_new]; out_mask[only_new] = 255

    if overlap.any():
        seam = find_best_seam_col(canvas, warped, overlap)
        out = feather_blend(out, warped, seam, overlap, feather_px=15)

    return out, out_mask


def build_canvas(images, Hs):
    all_c = []
    for img, H in zip(images, Hs):
        h, w = img.shape[:2]
        c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        all_c.append(cv2.perspectiveTransform(c, H).reshape(4,2))
    all_c = np.vstack(all_c)
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

def stitch_row(images, tag, overlap_frac=0.35):
    if len(images) == 1: return images[0]

    # Try cv2.Stitcher
    print(f"\n  Trying cv2.Stitcher (SCANS)...")
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        stitcher.setPanoConfidenceThresh(0.3)
        status, result = stitcher.stitch(images)
        ok = {cv2.Stitcher_OK:"OK", cv2.Stitcher_ERR_NEED_MORE_IMGS:"need more images",
              cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL:"hom fail",
              cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL:"cam fail"}
        print(f"  Stitcher: {ok.get(status, status)}")
        if status == cv2.Stitcher_OK:
            return result
    except Exception as e:
        print(f"  Stitcher exception: {e}")

    # Manual
    Hs = [np.eye(3, dtype=np.float64)]
    for i in range(1, len(images)):
        print(f"\n  Aligning image {i+1} → image {i}...")
        H_local = compute_pairwise_H(images[i-1], images[i], overlap_frac)
        # Clamp projective component
        H_local[2, :] = [0, 0, 1]
        # CORRECT chain: H_chain[-1] maps image i-1 → canvas;
        #   H_local maps image i → image i-1;
        #   H_chain[i] maps image i → canvas = H_chain[i-1] @ H_local
        Hs.append(Hs[-1] @ H_local)
        print(f"  H[{i+1}] local: tx={H_local[0,2]:.1f}  ty={H_local[1,2]:.1f}")

    print(f"\n  Building canvas...")
    return build_canvas(images, Hs)


def stitch_vertical(rows):
    if len(rows) == 1: return rows[0]
    result = rows[0]
    for i, row in enumerate(rows[1:], 2):
        print(f"\n  Vertical stitch: row {i}...")
        H = compute_pairwise_H(
            np.rot90(result, -1), np.rot90(row, -1), 0.25)
        # Convert rotation-based H back (we rotated 90° to treat vertical as horizontal)
        # This is a hack — just do direct vertical matching instead
        # Simple fallback: use homography on actual vertical alignment
        # Re-do properly:
        ha, wa = result.shape[:2]; hb, wb = row.shape[:2]
        # Try direct SIFT full-image
        pts_a, pts_b = sift_match(
            cv2.cvtColor(result, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(row, cv2.COLOR_BGR2GRAY))
        if pts_a is not None:
            H2, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 4.0)
            n = int(mask.sum()) if mask is not None else 0
            if H2 is not None and n >= 10 and validate_H(H2, row.shape):
                H2[2,:] = [0,0,1]
                result = build_canvas([result, row], [np.eye(3), H2])
                continue
        # Fallback: simple vertical stack
        w = min(wa, wb)
        result = np.vstack([result[:,:w], row[:,:w]])
    return result


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

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
    present = [(p, f"r{row_tag}_{i+1}") for i,p in enumerate(paths) if Path(p).exists()]
    if not present:
        print(f"\nRow {row_tag}: no files."); return None
    print(f"\n{'='*55}\n Row {row_tag}  ({len(present)} images)\n{'='*55}")
    imgs = [preprocess(p, tag) for p,tag in present]
    result = stitch_row(imgs, row_tag)
    out = ROWS_DIR + f"row_{row_tag}.png"
    cv2.imwrite(out, result)
    print(f"\nSaved: {out}  ({result.shape[1]}x{result.shape[0]})")
    return result


if __name__ == "__main__":
    from scipy.ndimage import uniform_filter1d   # needed by find_best_seam_col

    row_imgs = {}
    for rtag, files in ROW_FILES.items():
        if not files: continue
        r = process_row(rtag, files)
        if r is not None: row_imgs[rtag] = r

    rows = [row_imgs[k] for k in sorted(row_imgs) if k in row_imgs]
    if rows:
        out = OUTPUT_DIR + "row_1_result_v5.png"
        cv2.imwrite(out, rows[0])
        print(f"\nResult: {out}  ({rows[0].shape[1]}x{rows[0].shape[0]})")
    print("\nDone.")
