"""
map_stitch_v6.py — Complete map stitching for all 12 photos.

Key: NO EXIF rotation. Raw portrait images (4284×5712) are horizontally
adjacent for same-row photos — confirmed empirically: without EXIF rotation
SIFT gives tx≈3927 (horizontal offset) for row 1. Applying EXIF rotation
(→ landscape 5712×4284) makes same-row photos appear vertically adjacent,
breaking horizontal stitching.

Fixes vs map_stitch_final.py:
1. No EXIF rotation
2. H translation validation rejects bad SIFT (prevents OOM canvas)
3. Perspective sanity check: skip if detected region < 55% of image dimension
4. Canvas size guard with scale-down fallback
5. SIFT down-scales large images before feature extraction
"""

import cv2
import numpy as np
import os
from pathlib import Path
from scipy.ndimage import uniform_filter1d

UPLOAD_DIR = "/root/.claude/uploads/1b9fe45d-a299-5f9c-9937-d45d27c12f22/"
OUTPUT_DIR = "/home/user/yurumemo-johokyoku/map_output/"
CORRECTED  = OUTPUT_DIR + "corrected_v6/"
ROWS_DIR   = OUTPUT_DIR + "rows_v6/"

for d in [OUTPUT_DIR, CORRECTED, ROWS_DIR]:
    os.makedirs(d, exist_ok=True)

MAX_ROW_DIM   = 18000
MAX_FINAL_DIM = 25000

ROW_FILES = {
    "1": [("69a424a3-IMG_6837.jpeg", "r1_1"),
          ("95841777-IMG_6838.jpeg", "r1_2"),
          ("d9b6fffb-IMG_6839.jpeg", "r1_3")],
    "2": [("e316d2e3-IMG_6842.jpeg", "r2_1"),
          ("fd9b9afe-IMG_6843.jpeg", "r2_2"),
          ("213fbf53-IMG_6844.jpeg", "r2_3")],
    "3": [("9a0ccada-IMG_6845.jpeg", "r3_1"),
          ("51c8fcf1-IMG_6846.jpeg", "r3_2"),
          ("801c9748-IMG_6847.jpeg", "r3_3")],
    "4": [("c816dae8-IMG_6848.jpeg", "r4_1"),
          ("d2f6ad4d-IMG_6849.jpeg", "r4_2"),
          ("7ae2ed01-IMG_6850.jpeg", "r4_3")],
}


# ───────────────────────────────────────
# Pre-processing
# ───────────────────────────────────────

def order_pts(pts):
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1)
    rect[0] = pts[s.argmin()]; rect[2] = pts[s.argmax()]
    rect[1] = pts[d.argmin()]; rect[3] = pts[d.argmax()]
    return rect


def correct_perspective(img, tag=""):
    """Warp map to rectangle. Returns original if detection is unreliable."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    quad = None

    # Method 1: Hough line frame detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=int(min(w, h)*0.28),
                             minLineLength=int(min(w, h)*0.28),
                             maxLineGap=60)
    if lines is not None:
        h_ys, v_xs = [], []
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if ang < 10 or ang > 170:
                h_ys.append((min(y1,y2)+max(y1,y2))//2)
            elif 80 < ang < 100:
                v_xs.append((min(x1,x2)+max(x1,x2))//2)
        if len(h_ys) >= 2 and len(v_xs) >= 2:
            top = min(h_ys); bot = max(h_ys)
            left = min(v_xs); right = max(v_xs)
            if (bot-top) > h*0.5 and (right-left) > w*0.5:
                quad = order_pts(np.float32([[left,top],[right,top],
                                              [right,bot],[left,bot]]))
                print(f"  [{tag}] Hough quad: ({left},{top})→({right},{bot})")

    # Method 2: bright region contour
    if quad is None:
        _, bright = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (30,30)))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (50,50)))
        cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
                if cv2.contourArea(cnt) < 0.25*h*w: continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                if len(approx) == 4:
                    pts_c = approx.reshape(4, 2)
                    bw, bh = cv2.minAreaRect(pts_c)[1]
                    if min(bw,bh) > 0 and max(bw,bh)/min(bw,bh) < 4:
                        quad = order_pts(pts_c)
                        print(f"  [{tag}] Bright contour quad")
                        break

    # Method 3: bounding box of bright pixels
    if quad is None:
        _, bright = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (20,20)))
        ys, xs = np.where(bright > 0)
        if len(xs) > 0:
            quad = order_pts(np.float32([
                [xs.min(),ys.min()],[xs.max(),ys.min()],
                [xs.max(),ys.max()],[xs.min(),ys.max()]]))
            print(f"  [{tag}] Bright bbox quad")

    if quad is None:
        print(f"  [{tag}] Perspective: no detection, using original {w}×{h}")
        return img

    pts = quad
    wp = int(max(np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[3])))
    hp = int(max(np.linalg.norm(pts[3]-pts[0]), np.linalg.norm(pts[2]-pts[1])))

    # Skip if result < 55% of original dimension
    if wp < w*0.55 or hp < h*0.55:
        print(f"  [{tag}] Perspective skipped: detected {wp}×{hp} < 55% of {w}×{h}")
        return img
    # Skip if output area < 95% of original (quad over-cropped real content)
    if (wp * hp) < 0.95 * (w * h):
        print(f"  [{tag}] Perspective skipped: area {wp*hp} < 95% of {w*h}")
        return img

    dst = np.float32([[0,0],[wp-1,0],[wp-1,hp-1],[0,hp-1]])
    M = cv2.getPerspectiveTransform(pts, dst)
    out = cv2.warpPerspective(img, M, (wp,hp), flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    print(f"  [{tag}] Perspective: {w}×{h} → {wp}×{hp}")
    return out


MAX_DESKEW_DEG = 1.5  # skip deskew above this angle (likely false Hough detection)


def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sh = h // 5
    strip = np.vstack([gray[:sh], gray[-sh:]])
    edges = cv2.Canny(cv2.GaussianBlur(strip,(5,5),0), 20, 80)
    lines = cv2.HoughLines(edges, 1, np.pi/1800, threshold=int(w*0.15))
    angle = 0.0
    if lines is not None:
        angles = [np.degrees(l[0][1])-90 for l in lines
                  if abs(np.degrees(l[0][1])-90) < 8]
        if angles: angle = float(np.median(angles))
    if abs(angle) < 0.3: return img, 0.0
    if abs(angle) > MAX_DESKEW_DEG:
        print(f"  Deskew {angle:.2f}° exceeds cap {MAX_DESKEW_DEG}°, skipping")
        return img, 0.0
    print(f"  Deskew: {angle:.2f}°")
    M = cv2.getRotationMatrix2D((w/2,h/2), -angle, 1.0)
    ca, sa = abs(M[0,0]), abs(M[0,1])
    nw, nh = int(h*sa+w*ca), int(h*ca+w*sa)
    M[0,2] += (nw-w)/2; M[1,2] += (nh-h)/2
    out = cv2.warpAffine(img, M, (nw,nh), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return out, angle


def trim_white_edges(img, thr=248):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rm = gray.min(1); cm = gray.min(0)
    r0 = next((i for i in range(h) if rm[i] < thr), 0)
    r1 = next((i for i in range(h-1,-1,-1) if rm[i] < thr), h-1)
    c0 = next((i for i in range(w) if cm[i] < thr), 0)
    c1 = next((i for i in range(w-1,-1,-1) if cm[i] < thr), w-1)
    result = img[r0:r1+1, c0:c1+1]
    if (r0, r1+1, c0, c1+1) != (0, h, 0, w):
        print(f"  Trim: {w}×{h} → {result.shape[1]}×{result.shape[0]}")
    return result


def preprocess(path, tag):
    print(f"\n[Pre] {tag}  ({Path(path).name})")
    img = cv2.imread(path)   # NO EXIF rotation — raw portrait is correct for stitching
    if img is None: raise FileNotFoundError(path)
    print(f"  Loaded: {img.shape[1]}×{img.shape[0]}")
    img = correct_perspective(img, tag)
    img, _ = deskew(img)
    img = trim_white_edges(img)
    print(f"  Final: {img.shape[1]}×{img.shape[0]}")
    cv2.imwrite(CORRECTED + f"{tag}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return img


# ───────────────────────────────────────
# Feature matching
# ───────────────────────────────────────

def sift_match(gray_a, gray_b, ratio=0.67, nfeat=8000, max_dim=4000):
    """SIFT match. Returns keypoints in ORIGINAL (unscaled) coordinates."""
    ha, wa = gray_a.shape; hb, wb = gray_b.shape
    s = min(1.0, max_dim / max(ha, wa, hb, wb))
    if s < 0.99:
        g_a = cv2.resize(gray_a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        g_b = cv2.resize(gray_b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        print(f"    SIFT scale: {s:.3f}×  ({wa}×{ha}→{g_a.shape[1]}×{g_a.shape[0]})")
    else:
        g_a, g_b = gray_a, gray_b

    sift = cv2.SIFT_create(nfeatures=nfeat, contrastThreshold=0.01, edgeThreshold=20)
    kp_a, des_a = sift.detectAndCompute(g_a, None)
    kp_b, des_b = sift.detectAndCompute(g_b, None)
    print(f"    KP: a={len(kp_a)}, b={len(kp_b)}")
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return None, None

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=150))
    raw = flann.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in raw if m.distance < ratio*n.distance]
    print(f"    Good matches: {len(good)}")
    if len(good) < 8: return None, None

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]) / s
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]) / s
    return pts_a, pts_b


def find_H_strip(img_ref, img_mov, overlap_frac, direction="h"):
    """Match in overlap strip. Returns H mapping img_mov → img_ref coords."""
    hr, wr = img_ref.shape[:2]; hm, wm = img_mov.shape[:2]
    if direction == "h":
        sw = int(min(wr, wm) * overlap_frac)
        xr = wr - sw
        ref_strip = img_ref[:, xr:]
        mov_strip = img_mov[:, :sw]
        ox_r, oy_r = xr, 0; ox_m, oy_m = 0, 0
    else:
        sw = int(min(hr, hm) * overlap_frac)
        yr = hr - sw
        ref_strip = img_ref[yr:, :]
        mov_strip = img_mov[:sw, :]
        ox_r, oy_r = 0, yr; ox_m, oy_m = 0, 0

    gray_r = cv2.cvtColor(ref_strip, cv2.COLOR_BGR2GRAY)
    gray_m = cv2.cvtColor(mov_strip, cv2.COLOR_BGR2GRAY)
    pts_r, pts_m = sift_match(gray_r, gray_m)
    if pts_r is None: return None, 0

    pts_r_full = pts_r + np.float32([ox_r, oy_r])
    pts_m_full = pts_m + np.float32([ox_m, oy_m])
    H, mask = cv2.findHomography(pts_m_full, pts_r_full, cv2.RANSAC, 4.0,
                                  maxIters=5000, confidence=0.999)
    n = int(mask.sum()) if mask is not None else 0
    print(f"    Strip H inliers: {n}")
    return H, n


def validate_H(H, src_shape, ref_shape, direction="h"):
    """Check H has reasonable scale AND translation for stitching direction."""
    if H is None: return False
    hs, ws = src_shape[:2]; hr, wr = ref_shape[:2]

    # Scale sanity
    c = np.float32([[0,0],[ws,0],[ws,hs],[0,hs]]).reshape(-1,1,2)
    wc = cv2.perspectiveTransform(c, H).reshape(4, 2)
    nw = wc[:,0].max()-wc[:,0].min(); nh = wc[:,1].max()-wc[:,1].min()
    if nw < ws*0.5 or nw > ws*2.0: return False
    if nh < hs*0.5 or nh > hs*2.0: return False
    if abs(H[2,0]) > 5e-4 or abs(H[2,1]) > 5e-4: return False

    # Translation direction sanity
    tx, ty = H[0,2], H[1,2]
    if direction == "h":
        # img_mov is to the right: tx should be ~wr*(0.25..1.3), |ty| small
        if tx < wr*0.20 or tx > wr*1.40:
            print(f"    Reject H: tx={tx:.0f} outside [{wr*0.20:.0f}, {wr*1.40:.0f}]")
            return False
        if abs(ty) > hr*0.40:
            print(f"    Reject H: |ty|={abs(ty):.0f} > {hr*0.40:.0f}")
            return False
    else:
        # img_mov is below: ty should be ~hr*(0.25..1.1), |tx| small
        if ty < hr*0.20 or ty > hr*1.10:
            print(f"    Reject H: ty={ty:.0f} outside [{hr*0.20:.0f}, {hr*1.10:.0f}]")
            return False
        if abs(tx) > wr*0.40:
            print(f"    Reject H: |tx|={abs(tx):.0f} > {wr*0.40:.0f}")
            return False
    return True


def compute_H(img_ref, img_mov, overlap_frac=0.30, direction="h"):
    """Compute H mapping img_mov → img_ref coords."""
    print(f"  Strip SIFT ({direction}, {overlap_frac:.0%})...")
    H, n = find_H_strip(img_ref, img_mov, overlap_frac, direction)
    if H is not None and n >= 10 and validate_H(H, img_mov.shape, img_ref.shape, direction):
        return H

    print(f"  Retry wider overlap (45%)...")
    H, n = find_H_strip(img_ref, img_mov, 0.45, direction)
    if H is not None and n >= 10 and validate_H(H, img_mov.shape, img_ref.shape, direction):
        return H

    print(f"  Full-image SIFT...")
    pts_r, pts_m = sift_match(
        cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY))
    if pts_r is not None:
        H, mask = cv2.findHomography(pts_m, pts_r, cv2.RANSAC, 4.0,
                                      maxIters=5000, confidence=0.999)
        n = int(mask.sum()) if mask is not None else 0
        print(f"  Full SIFT inliers: {n}")
        if H is not None and n >= 10 and validate_H(H, img_mov.shape, img_ref.shape, direction):
            return H

    print(f"  Translation fallback.")
    hr, wr = img_ref.shape[:2]; hm, wm = img_mov.shape[:2]
    if direction == "h":
        tx = wr*(1-overlap_frac); ty = (hr-hm)/2
    else:
        tx = (wr-wm)/2; ty = hr*(1-overlap_frac)
    return np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)


# ───────────────────────────────────────
# Canvas compositing
# ───────────────────────────────────────

def make_content_mask(img, white_thresh=252):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nw = (gray < white_thresh).astype(np.uint8) * 255
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (80,80))
    filled = cv2.morphologyEx(nw, cv2.MORPH_CLOSE, k1)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (200,200))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k2)
    return filled


def find_best_seam(canvas, warped, overlap_mask, direction="v"):
    if direction == "v":
        ov_cols = np.where(overlap_mask.any(0))[0]
        if len(ov_cols) < 3: return int(ov_cols.mean()) if len(ov_cols) else 0
        c0, c1 = ov_cols[0], ov_cols[-1]
        ec = cv2.Canny(cv2.cvtColor(canvas[:,c0:c1+1],cv2.COLOR_BGR2GRAY),30,90).sum(0).astype(float)
        ew = cv2.Canny(cv2.cvtColor(warped[:,c0:c1+1],cv2.COLOR_BGR2GRAY),30,90).sum(0).astype(float)
        return int(uniform_filter1d(ec+ew, 40).argmin()) + c0
    else:
        ov_rows = np.where(overlap_mask.any(1))[0]
        if len(ov_rows) < 3: return int(ov_rows.mean()) if len(ov_rows) else 0
        r0, r1 = ov_rows[0], ov_rows[-1]
        ec = cv2.Canny(cv2.cvtColor(canvas[r0:r1+1,:],cv2.COLOR_BGR2GRAY),30,90).sum(1).astype(float)
        ew = cv2.Canny(cv2.cvtColor(warped[r0:r1+1,:],cv2.COLOR_BGR2GRAY),30,90).sum(1).astype(float)
        return int(uniform_filter1d(ec+ew, 40).argmin()) + r0


def feather(canvas, warped, seam, overlap_mask, px=12, direction="v"):
    hc, wc = canvas.shape[:2]
    out = canvas.copy()
    if direction == "v":
        x0 = max(0, seam-px); x1 = min(wc, seam+px+1)
        for x in range(x0, x1):
            col = overlap_mask[:,x]
            if not col.any(): continue
            a = (x-x0)/(x1-x0-1) if x1 > x0+1 else 0.5
            out[col,x] = ((1-a)*canvas[col,x].astype(np.float32)
                          + a*warped[col,x].astype(np.float32)).astype(np.uint8)
        rm = overlap_mask.copy(); rm[:,:x1] = False
        out[rm] = warped[rm]
    else:
        y0 = max(0, seam-px); y1 = min(hc, seam+px+1)
        for y in range(y0, y1):
            row = overlap_mask[y,:]
            if not row.any(): continue
            a = (y-y0)/(y1-y0-1) if y1 > y0+1 else 0.5
            out[y,row] = ((1-a)*canvas[y,row].astype(np.float32)
                          + a*warped[y,row].astype(np.float32)).astype(np.uint8)
        bm = overlap_mask.copy(); bm[:y1,:] = False
        out[bm] = warped[bm]
    return out


def warp_onto(canvas, canvas_mask, new_img, H, seam_dir="v"):
    hc, wc = canvas.shape[:2]
    warped = cv2.warpPerspective(new_img, H, (wc,hc), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cm = make_content_mask(new_img)
    warped_valid = cv2.warpPerspective(cm, H, (wc,hc), flags=cv2.INTER_NEAREST) > 127

    only_new = warped_valid & (canvas_mask == 0)
    overlap  = warped_valid & (canvas_mask  > 0)

    out = canvas.copy(); om = canvas_mask.copy()
    out[only_new] = warped[only_new]; om[only_new] = 255

    if overlap.any():
        seam = find_best_seam(canvas, warped, overlap, seam_dir)
        out = feather(out, warped, seam, overlap, px=12, direction=seam_dir)

    return out, om


def estimate_canvas_wh(images, Hs):
    all_c = []
    for img, H in zip(images, Hs):
        h, w = img.shape[:2]
        c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        all_c.append(cv2.perspectiveTransform(c, H).reshape(4,2))
    pts = np.vstack(all_c)
    return (int(np.ceil(pts[:,0].max()) - np.floor(pts[:,0].min())),
            int(np.ceil(pts[:,1].max()) - np.floor(pts[:,1].min())))


def scale_H(H, s):
    S  = np.array([[s,0,0],[0,s,0],[0,0,1]], np.float64)
    Si = np.array([[1/s,0,0],[0,1/s,0],[0,0,1]], np.float64)
    return S @ H @ Si


def build_canvas(images, Hs, seam_dir="v", max_dim=18000):
    est_w, est_h = estimate_canvas_wh(images, Hs)
    print(f"  Estimated canvas: {est_w}×{est_h}")

    s = 1.0
    if est_w > max_dim or est_h > max_dim:
        s = max_dim / max(est_w, est_h)
        print(f"  Scaling to {s:.3f}× to fit within {max_dim}px")
        images = [cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                  for img in images]
        Hs = [scale_H(H, s) for H in Hs]
        est_w, est_h = estimate_canvas_wh(images, Hs)

    all_c = []
    for img, H in zip(images, Hs):
        h, w = img.shape[:2]
        c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        all_c.append(cv2.perspectiveTransform(c, H).reshape(4,2))
    pts = np.vstack(all_c)
    x_min = int(np.floor(pts[:,0].min())); x_max = int(np.ceil(pts[:,0].max()))
    y_min = int(np.floor(pts[:,1].min())); y_max = int(np.ceil(pts[:,1].max()))
    tx, ty = max(0, -x_min), max(0, -y_min)
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)
    cw, ch = x_max-x_min+1, y_max-y_min+1
    print(f"  Building canvas: {cw}×{ch}  scale={s:.3f}")

    canvas = np.zeros((ch, cw, 3), np.uint8)
    canvas_mask = np.zeros((ch, cw), np.uint8)
    for i, (img, H) in enumerate(zip(images, Hs)):
        canvas, canvas_mask = warp_onto(canvas, canvas_mask, img, T @ H, seam_dir)
        print(f"  Placed image {i+1}/{len(images)}")

    rows = np.where(canvas_mask.any(1))[0]
    cols = np.where(canvas_mask.any(0))[0]
    if len(rows) and len(cols):
        canvas = canvas[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    return canvas


# ───────────────────────────────────────
# Row and final stitching
# ───────────────────────────────────────

def stitch_row(images, tag, overlap_frac=0.35):
    if len(images) == 1: return images[0]
    Hs = [np.eye(3, dtype=np.float64)]
    for i in range(1, len(images)):
        print(f"\n  Aligning {tag}:{i+1}→{i}...")
        H = compute_H(images[i-1], images[i], overlap_frac, "h")
        H[2,:] = [0, 0, 1]
        Hs.append(Hs[-1] @ H)
        print(f"  H local: tx={H[0,2]:.0f}  ty={H[1,2]:.0f}")
    return build_canvas(images, Hs, seam_dir="v", max_dim=MAX_ROW_DIM)


def stitch_vertical(row_imgs, overlap_frac=0.20):
    """
    Vertically stitch row images using TRANSLATION-ONLY homography.
    Scale/rotation from SIFT is discarded to prevent compounding distortion
    across rows (row 3→2 H had 0.75x scale that cascaded to 0.62x at row 4).
    """
    if len(row_imgs) == 1: return row_imgs[0]
    Hs = [np.eye(3, dtype=np.float64)]
    for i in range(1, len(row_imgs)):
        print(f"\n  Vertical align row {i+1}→{i}...")
        H_full = compute_H(row_imgs[i-1], row_imgs[i], overlap_frac, "v")
        tx, ty = H_full[0, 2], H_full[1, 2]
        H_trans = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)
        Hs.append(Hs[-1] @ H_trans)
        print(f"  Translation: tx={tx:.0f}  ty={ty:.0f}")
    return build_canvas(row_imgs, Hs, seam_dir="h", max_dim=MAX_FINAL_DIM)


# ───────────────────────────────────────
# Main
# ───────────────────────────────────────

if __name__ == "__main__":
    row_results = {}

    for rtag, entries in ROW_FILES.items():
        present = [(UPLOAD_DIR+f, tag)
                   for f, tag in entries if Path(UPLOAD_DIR+f).exists()]
        if not present:
            print(f"\nRow {rtag}: no files, skipping."); continue

        print(f"\n{'='*60}\n Row {rtag}  ({len(present)} images)\n{'='*60}")
        imgs = [preprocess(path, tag) for path, tag in present]

        print(f"\n[Stitch] Row {rtag}...")
        result = stitch_row(imgs, f"r{rtag}")
        out_path = ROWS_DIR + f"row_{rtag}.png"
        cv2.imwrite(out_path, result)
        print(f"\nRow {rtag} saved: {out_path}  ({result.shape[1]}×{result.shape[0]})")
        row_results[rtag] = result

    rows_ordered = [row_results[k] for k in sorted(row_results) if k in row_results]
    if len(rows_ordered) >= 2:
        print(f"\n{'='*60}\n Final vertical stitch ({len(rows_ordered)} rows)\n{'='*60}")
        final = stitch_vertical(rows_ordered)
        out_path = OUTPUT_DIR + "final_map.png"
        cv2.imwrite(out_path, final)
        print(f"\nFINAL MAP: {out_path}  ({final.shape[1]}×{final.shape[0]})")
    elif rows_ordered:
        cv2.imwrite(OUTPUT_DIR+"final_map.png", rows_ordered[0])
        print(f"\nOnly 1 row: saved as final_map.png")

    print("\nDone.")
