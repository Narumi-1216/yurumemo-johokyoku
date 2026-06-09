"""
Map stitching — FINAL version.

Critical fix: all images have EXIF Orientation=6 (rotate 90° CW for correct display).
After rotation: 4284x5712 → 5712x4284 (landscape).

Layout:
  Row 1:  photos  1,  2,  3  (IMG_6837-6839)  — each 1 grid cell
  Row 2:  photos  4,  5,  6  (IMG_6842-6844)  — each 1 grid cell
  Row 3:  photos  7,  8,  9  (IMG_6845-6847)  — each 1 grid cell
  Row 4:  photos 10, 11, 12  (IMG_6848-6850)  — each 2×1 grid cells (taller)
"""

import cv2
import numpy as np
import os
from pathlib import Path
import piexif

UPLOAD_DIR = "/root/.claude/uploads/1b9fe45d-a299-5f9c-9937-d45d27c12f22/"
OUTPUT_DIR = "/home/user/yurumemo-johokyoku/map_output/"
CORRECTED  = OUTPUT_DIR + "corrected_final/"
ROWS_DIR   = OUTPUT_DIR + "rows_final/"

for d in [OUTPUT_DIR, CORRECTED, ROWS_DIR]:
    os.makedirs(d, exist_ok=True)


# ──────────────────────────────────────────────────────────
# EXIF-aware image loading
# ──────────────────────────────────────────────────────────

EXIF_ROTATE = {
    3: cv2.ROTATE_180,
    6: cv2.ROTATE_90_CLOCKWISE,
    8: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

def load_with_exif(path):
    """Load image and apply EXIF rotation."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    try:
        exif = piexif.load(path)
        orientation = exif['0th'].get(piexif.ImageIFD.Orientation, 1)
        if orientation in EXIF_ROTATE:
            img = cv2.rotate(img, EXIF_ROTATE[orientation])
            print(f"  EXIF orient={orientation} → rotated")
    except Exception:
        pass
    return img


# ──────────────────────────────────────────────────────────
# Perspective correction
# ──────────────────────────────────────────────────────────

def order_pts(pts):
    pts = pts.astype(np.float32)
    rect = np.zeros((4, 2), np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1)
    rect[0] = pts[s.argmin()]; rect[2] = pts[s.argmax()]
    rect[1] = pts[d.argmin()]; rect[3] = pts[d.argmax()]
    return rect


def correct_perspective(img, tag=""):
    """
    Find the map content rectangle (bright area) and warp to a clean rectangle.
    Uses dark-frame detection (Canny + Hough) as primary, bright-region as fallback.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Method 1: find the rectangular map frame via Hough lines ──
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=int(min(w,h)*0.35),
                             minLineLength=int(min(w,h)*0.35),
                             maxLineGap=40)
    quad = None
    if lines is not None:
        h_lines, v_lines = [], []
        for line in lines:
            x1,y1,x2,y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if angle < 12 or angle > 168:
                h_lines.append((min(y1,y2)+max(y1,y2))//2)
            elif 78 < angle < 102:
                v_lines.append((min(x1,x2)+max(x1,x2))//2)
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            top    = min(h_lines); bottom = max(h_lines)
            left   = min(v_lines); right  = max(v_lines)
            # Sanity: map should cover >50% of the image
            if (bottom-top) > h*0.5 and (right-left) > w*0.5:
                quad = order_pts(np.float32([[left,top],[right,top],
                                              [right,bottom],[left,bottom]]))

    # ── Method 2: bright-region fallback ──
    if quad is None:
        _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT,(30,30)))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT,(50,50)))
        cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
                if cv2.contourArea(cnt) < 0.08*h*w: continue
                peri = cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                if len(approx)==4:
                    pts = approx.reshape(4,2)
                    bw,bh = cv2.minAreaRect(pts)[1]
                    if min(bw,bh)>0 and max(bw,bh)/min(bw,bh)<4:
                        quad = order_pts(pts); break
        if quad is None:
            ys, xs = np.where(bright>0)
            if len(xs)==0: return img
            quad = order_pts(np.float32([[xs.min(),ys.min()],[xs.max(),ys.min()],
                                          [xs.max(),ys.max()],[xs.min(),ys.max()]]))

    pts = quad
    wp = int(max(np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[3])))
    hp = int(max(np.linalg.norm(pts[3]-pts[0]), np.linalg.norm(pts[2]-pts[1])))
    dst = np.float32([[0,0],[wp-1,0],[wp-1,hp-1],[0,hp-1]])
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (wp, hp), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    print(f"  [{tag}] perspectived: {wp}x{hp}")
    return warped


def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sh = h // 5
    strip = np.vstack([gray[:sh], gray[-sh:]])
    edges = cv2.Canny(cv2.GaussianBlur(strip,(5,5),0), 20, 80)
    lines = cv2.HoughLines(edges, 1, np.pi/1800, threshold=int(w*0.15))
    angle = 0.0
    if lines is not None:
        angles = [np.degrees(l[0][1])-90.0 for l in lines
                  if abs(np.degrees(l[0][1])-90.0)<8]
        if angles: angle = float(np.median(angles))
    if abs(angle) < 0.3: return img, 0.0
    print(f"  Deskew: {angle:.2f}°")
    M = cv2.getRotationMatrix2D((w/2,h/2),-angle,1.0)
    cos_a,sin_a = abs(M[0,0]),abs(M[0,1])
    nw,nh = int(h*sin_a+w*cos_a), int(h*cos_a+w*sin_a)
    M[0,2]+=(nw-w)/2; M[1,2]+=(nh-h)/2
    out = cv2.warpAffine(img,M,(nw,nh),flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
    return out, angle


def trim_white_edges(img, thr=248):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    row_min = gray.min(1); col_min = gray.min(0)
    r0 = next((i for i in range(h) if row_min[i]<thr), 0)
    r1 = next((i for i in range(h-1,-1,-1) if row_min[i]<thr), h-1)
    c0 = next((i for i in range(w) if col_min[i]<thr), 0)
    c1 = next((i for i in range(w-1,-1,-1) if col_min[i]<thr), w-1)
    result = img[r0:r1+1, c0:c1+1]
    if result.shape[:2] != img.shape[:2]:
        print(f"  Trimmed: {img.shape[1]}x{img.shape[0]} → {result.shape[1]}x{result.shape[0]}")
    return result


def preprocess(path, tag):
    print(f"\n[Pre] {tag}  ({Path(path).name})")
    img = load_with_exif(path)
    print(f"  Loaded (after EXIF): {img.shape[1]}x{img.shape[0]}")
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
    print(f"    Matches: {len(good)}")
    if len(good)<8: return None, None
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
    return pts_a, pts_b


def find_H_strip(img_left, img_right, overlap_frac, direction="h"):
    """
    Match in overlap strip.
    direction='h': horizontal stitch (right edge of left img, left edge of right img)
    direction='v': vertical stitch   (bottom edge of top img, top edge of bottom img)
    """
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]
    if direction == "h":
        sw = int(min(w_l, w_r) * overlap_frac)
        x_l = w_l - sw
        ref_strip = img_left[:, x_l:]
        mov_strip = img_right[:, :sw]
        ox_r, oy_r = x_l, 0
        ox_m, oy_m = 0,   0
    else:
        sw = int(min(h_l, h_r) * overlap_frac)
        y_l = h_l - sw
        ref_strip = img_left[y_l:, :]
        mov_strip = img_right[:sw, :]
        ox_r, oy_r = 0, y_l
        ox_m, oy_m = 0, 0

    gray_r = cv2.cvtColor(ref_strip, cv2.COLOR_BGR2GRAY)
    gray_m = cv2.cvtColor(mov_strip, cv2.COLOR_BGR2GRAY)
    pts_r, pts_m = sift_match(gray_r, gray_m)
    if pts_r is None: return None, 0

    pts_r += [ox_r, oy_r]
    H, mask = cv2.findHomography(pts_m + [ox_m, oy_m], pts_r,
                                  cv2.RANSAC, 4.0,
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


def compute_H(img_a, img_b, overlap_frac=0.30, direction="h"):
    """Compute H mapping img_b → img_a coordinate space."""
    print(f"  Strip SIFT ({direction}, {overlap_frac:.0%})...")
    H, n = find_H_strip(img_a, img_b, overlap_frac, direction)
    if H is not None and n >= 10 and validate_H(H, img_b.shape):
        return H

    # Try wider overlap
    print(f"  Trying wider overlap (45%)...")
    H, n = find_H_strip(img_a, img_b, 0.45, direction)
    if H is not None and n >= 10 and validate_H(H, img_b.shape):
        return H

    print(f"  Trying full-image SIFT...")
    pts_a, pts_b = sift_match(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY))
    if pts_a is not None:
        H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 4.0,
                                      maxIters=5000, confidence=0.999)
        n = int(mask.sum()) if mask is not None else 0
        print(f"  Full SIFT inliers: {n}")
        if H is not None and n >= 10 and validate_H(H, img_b.shape):
            return H

    print(f"  Using translation estimate.")
    ha, wa = img_a.shape[:2]; hb, wb = img_b.shape[:2]
    if direction == "h":
        tx = wa*(1-overlap_frac); ty = (ha-hb)/2
    else:
        tx = (wa-wb)/2;            ty = ha*(1-overlap_frac)
    return np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)


# ──────────────────────────────────────────────────────────
# Canvas compositing
# ──────────────────────────────────────────────────────────

def make_content_mask(img, white_thresh=252):
    """Mask for real map content (excludes pure-white rotation artifacts)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    non_white = (gray < white_thresh).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    filled = cv2.morphologyEx(non_white, cv2.MORPH_CLOSE, k)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 200))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, k2)
    return filled


def find_best_seam(canvas, warped, overlap_mask, direction="v"):
    """
    Find the seam position with minimum combined edge density.
    direction='v': vertical seam (for horizontal stitch)
    direction='h': horizontal seam (for vertical stitch)
    """
    from scipy.ndimage import uniform_filter1d
    if direction == "v":
        ov_cols = np.where(overlap_mask.any(0))[0]
        if len(ov_cols) < 3: return int(ov_cols.mean()) if len(ov_cols) else 0
        c0, c1 = ov_cols[0], ov_cols[-1]
        e_c = cv2.Canny(cv2.cvtColor(canvas[:,c0:c1+1],cv2.COLOR_BGR2GRAY),30,90).sum(0).astype(float)
        e_w = cv2.Canny(cv2.cvtColor(warped[:,c0:c1+1],cv2.COLOR_BGR2GRAY),30,90).sum(0).astype(float)
        smoothed = uniform_filter1d(e_c+e_w, size=40)
        return int(smoothed.argmin()) + c0
    else:
        ov_rows = np.where(overlap_mask.any(1))[0]
        if len(ov_rows) < 3: return int(ov_rows.mean()) if len(ov_rows) else 0
        r0, r1 = ov_rows[0], ov_rows[-1]
        e_c = cv2.Canny(cv2.cvtColor(canvas[r0:r1+1,:],cv2.COLOR_BGR2GRAY),30,90).sum(1).astype(float)
        e_w = cv2.Canny(cv2.cvtColor(warped[r0:r1+1,:],cv2.COLOR_BGR2GRAY),30,90).sum(1).astype(float)
        smoothed = uniform_filter1d(e_c+e_w, size=40)
        return int(smoothed.argmin()) + r0


def feather(canvas, warped, seam, overlap_mask, px=12, direction="v"):
    """Feather-blend near the seam."""
    hc, wc = canvas.shape[:2]
    out = canvas.copy()
    if direction == "v":
        x0 = max(0, seam-px); x1 = min(wc, seam+px+1)
        for x in range(x0, x1):
            col = overlap_mask[:,x]
            if not col.any(): continue
            alpha = (x-x0)/(x1-x0-1) if x1>x0+1 else 0.5
            out[col,x] = ((1-alpha)*canvas[col,x].astype(np.float32)
                          + alpha*warped[col,x].astype(np.float32)).astype(np.uint8)
        right_of_seam = overlap_mask.copy(); right_of_seam[:,:x1]=False
        out[right_of_seam] = warped[right_of_seam]
    else:
        y0 = max(0, seam-px); y1 = min(hc, seam+px+1)
        for y in range(y0, y1):
            row = overlap_mask[y,:]
            if not row.any(): continue
            alpha = (y-y0)/(y1-y0-1) if y1>y0+1 else 0.5
            out[y,row] = ((1-alpha)*canvas[y,row].astype(np.float32)
                          + alpha*warped[y,row].astype(np.float32)).astype(np.uint8)
        below_seam = overlap_mask.copy(); below_seam[:y1,:]=False
        out[below_seam] = warped[below_seam]
    return out


def warp_onto(canvas, canvas_mask, new_img, H, seam_dir="v"):
    hc, wc = canvas.shape[:2]
    hn, wn = new_img.shape[:2]
    warped = cv2.warpPerspective(new_img, H, (wc,hc), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    content_mask = make_content_mask(new_img)
    warped_valid = cv2.warpPerspective(
        content_mask, H, (wc,hc), flags=cv2.INTER_NEAREST) > 127

    only_new = warped_valid & (canvas_mask==0)
    overlap  = warped_valid & (canvas_mask >0)

    out = canvas.copy(); out_mask = canvas_mask.copy()
    out[only_new] = warped[only_new]; out_mask[only_new] = 255

    if overlap.any():
        seam = find_best_seam(canvas, warped, overlap, seam_dir)
        out = feather(out, warped, seam, overlap, px=12, direction=seam_dir)

    return out, out_mask


def build_canvas(images, Hs, seam_dir="v"):
    all_c = []
    for img, H in zip(images, Hs):
        h,w = img.shape[:2]
        c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        all_c.append(cv2.perspectiveTransform(c,H).reshape(4,2))
    all_c = np.vstack(all_c)
    x_min = int(np.floor(all_c[:,0].min())); x_max = int(np.ceil(all_c[:,0].max()))
    y_min = int(np.floor(all_c[:,1].min())); y_max = int(np.ceil(all_c[:,1].max()))
    tx, ty = max(0,-x_min), max(0,-y_min)
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)
    cw, ch = x_max-x_min+1, y_max-y_min+1
    print(f"  Canvas: {cw}x{ch}")
    canvas = np.zeros((ch,cw,3), np.uint8)
    canvas_mask = np.zeros((ch,cw), np.uint8)
    for i, (img,H) in enumerate(zip(images,Hs)):
        H_adj = T @ H
        canvas, canvas_mask = warp_onto(canvas, canvas_mask, img, H_adj, seam_dir)
        print(f"  Placed image {i+1}")
    rows = np.where(canvas_mask.any(1))[0]; cols = np.where(canvas_mask.any(0))[0]
    if len(rows) and len(cols):
        canvas = canvas[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    return canvas


# ──────────────────────────────────────────────────────────
# Row stitching
# ──────────────────────────────────────────────────────────

def stitch_row(images, tag, overlap_frac=0.30):
    """Horizontally stitch images (left → right)."""
    if len(images) == 1: return images[0]

    Hs = [np.eye(3, dtype=np.float64)]
    for i in range(1, len(images)):
        print(f"\n  Aligning {tag} image {i+1} → {i}...")
        H_local = compute_H(images[i-1], images[i], overlap_frac, "h")
        H_local[2,:] = [0,0,1]
        # Correct chain: H_chain[-1] @ H_local
        Hs.append(Hs[-1] @ H_local)
        print(f"  H local: tx={H_local[0,2]:.0f}  ty={H_local[1,2]:.0f}")

    return build_canvas(images, Hs, seam_dir="v")


def stitch_vertical(rows, overlap_frac=0.15):
    """Vertically stitch row images (top → bottom)."""
    if len(rows) == 1: return rows[0]

    Hs = [np.eye(3, dtype=np.float64)]
    ref = rows[0]
    for i in range(1, len(rows)):
        print(f"\n  Vertical align row {i+1} → {i}...")
        H_local = compute_H(ref, rows[i], overlap_frac, "v")
        H_local[2,:] = [0,0,1]
        Hs.append(Hs[-1] @ H_local)
        print(f"  H local: tx={H_local[0,2]:.0f}  ty={H_local[1,2]:.0f}")
        ref = rows[i]

    return build_canvas(rows, Hs, seam_dir="h")


# ──────────────────────────────────────────────────────────
# File registry
# ──────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scipy.ndimage import uniform_filter1d

    row_results = {}

    for rtag, entries in ROW_FILES.items():
        present = [(UPLOAD_DIR+f, tag) for f,tag in entries if Path(UPLOAD_DIR+f).exists()]
        if not present:
            print(f"\nRow {rtag}: no files, skipping.")
            continue

        print(f"\n{'='*60}\n Row {rtag}  ({len(present)} images)\n{'='*60}")
        imgs = [preprocess(path, tag) for path,tag in present]

        print(f"\n[Stitch] Row {rtag}...")
        result = stitch_row(imgs, rtag)
        out = ROWS_DIR + f"row_{rtag}.png"
        cv2.imwrite(out, result)
        print(f"\nRow {rtag} saved: {out}  ({result.shape[1]}x{result.shape[0]})")
        row_results[rtag] = result

    # Final vertical stitch
    rows_in_order = [row_results[k] for k in sorted(row_results) if k in row_results]
    if len(rows_in_order) >= 2:
        print(f"\n{'='*60}\n Final vertical stitch ({len(rows_in_order)} rows)\n{'='*60}")
        final = stitch_vertical(rows_in_order)
        out = OUTPUT_DIR + "final_map.png"
        cv2.imwrite(out, final)
        print(f"\nFINAL MAP: {out}  ({final.shape[1]}x{final.shape[0]})")
    elif rows_in_order:
        out = OUTPUT_DIR + "final_map.png"
        cv2.imwrite(out, rows_in_order[0])
        print(f"\nOnly 1 row. Saved: {out}")

    print("\nDone.")
