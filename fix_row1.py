"""
Re-preprocess r1_2 with deskew capped at 1.5° (was 3.2° → white corner explosion).
Re-stitch row 1 and update the final map.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from scipy.ndimage import uniform_filter1d

UPLOAD_DIR = "/root/.claude/uploads/1b9fe45d-a299-5f9c-9937-d45d27c12f22/"
OUTPUT_DIR = "/home/user/yurumemo-johokyoku/map_output/"
ROWS_DIR   = OUTPUT_DIR + "rows_v6/"
CORR_DIR   = OUTPUT_DIR + "corrected_v6/"

MAX_DESKEW_DEG = 1.5   # cap: above this, skip deskew (likely false detection)


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
    quad = None
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                             threshold=int(min(w,h)*0.28),
                             minLineLength=int(min(w,h)*0.28),
                             maxLineGap=60)
    if lines is not None:
        h_ys, v_xs = [], []
        for ln in lines:
            x1,y1,x2,y2 = ln[0]
            ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if ang < 10 or ang > 170:  h_ys.append((min(y1,y2)+max(y1,y2))//2)
            elif 80 < ang < 100:       v_xs.append((min(x1,x2)+max(x1,x2))//2)
        if len(h_ys)>=2 and len(v_xs)>=2:
            top=min(h_ys); bot=max(h_ys); left=min(v_xs); right=max(v_xs)
            if (bot-top)>h*0.5 and (right-left)>w*0.5:
                quad = order_pts(np.float32([[left,top],[right,top],[right,bot],[left,bot]]))
    if quad is None:
        _, bright = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT,(30,30)))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT,(50,50)))
        cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
                if cv2.contourArea(cnt) < 0.25*h*w: continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
                if len(approx)==4:
                    pts_c = approx.reshape(4,2)
                    bw,bh = cv2.minAreaRect(pts_c)[1]
                    if min(bw,bh)>0 and max(bw,bh)/min(bw,bh)<4:
                        quad = order_pts(pts_c); break
        if quad is None:
            _, bright = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
            bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_RECT,(20,20)))
            ys,xs = np.where(bright>0)
            if len(xs)>0:
                quad = order_pts(np.float32([[xs.min(),ys.min()],[xs.max(),ys.min()],
                                              [xs.max(),ys.max()],[xs.min(),ys.max()]]))
    if quad is None:
        return img
    pts = quad
    wp = int(max(np.linalg.norm(pts[1]-pts[0]), np.linalg.norm(pts[2]-pts[3])))
    hp = int(max(np.linalg.norm(pts[3]-pts[0]), np.linalg.norm(pts[2]-pts[1])))
    if wp < w*0.55 or hp < h*0.55:
        print(f"  [{tag}] Perspective skipped: {wp}×{hp} < 55% of {w}×{h}")
        return img
    # Skip if output area < 95% of original (quad cropped real content)
    if (wp * hp) < 0.95 * (w * h):
        print(f"  [{tag}] Perspective skipped: area {wp*hp} < 95% of {w*h}")
        return img
    dst = np.float32([[0,0],[wp-1,0],[wp-1,hp-1],[0,hp-1]])
    M = cv2.getPerspectiveTransform(pts, dst)
    out = cv2.warpPerspective(img, M, (wp,hp), flags=cv2.INTER_LANCZOS4,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    print(f"  [{tag}] Perspective: {w}×{h} → {wp}×{hp}")
    return out


def deskew(img, max_deg=MAX_DESKEW_DEG):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sh = h//5
    strip = np.vstack([gray[:sh], gray[-sh:]])
    edges = cv2.Canny(cv2.GaussianBlur(strip,(5,5),0), 20, 80)
    lines = cv2.HoughLines(edges, 1, np.pi/1800, threshold=int(w*0.15))
    angle = 0.0
    if lines is not None:
        angles = [np.degrees(l[0][1])-90 for l in lines if abs(np.degrees(l[0][1])-90)<8]
        if angles: angle = float(np.median(angles))
    if abs(angle) < 0.3:
        return img, 0.0
    if abs(angle) > max_deg:
        print(f"  Deskew {angle:.2f}° exceeds cap {max_deg}°, skipping")
        return img, 0.0
    print(f"  Deskew: {angle:.2f}°")
    M = cv2.getRotationMatrix2D((w/2,h/2), -angle, 1.0)
    ca, sa = abs(M[0,0]), abs(M[0,1])
    nw, nh = int(h*sa+w*ca), int(h*ca+w*sa)
    M[0,2]+=(nw-w)/2; M[1,2]+=(nh-h)/2
    out = cv2.warpAffine(img, M, (nw,nh), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return out, angle


def trim_white_edges(img, thr=248):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = gray.shape
    rm = gray.min(1); cm = gray.min(0)
    r0 = next((i for i in range(h) if rm[i]<thr), 0)
    r1 = next((i for i in range(h-1,-1,-1) if rm[i]<thr), h-1)
    c0 = next((i for i in range(w) if cm[i]<thr), 0)
    c1 = next((i for i in range(w-1,-1,-1) if cm[i]<thr), w-1)
    result = img[r0:r1+1, c0:c1+1]
    if (r0, r1+1, c0, c1+1) != (0,h,0,w):
        print(f"  Trim: {w}×{h} → {result.shape[1]}×{result.shape[0]}")
    return result


def preprocess(path, tag):
    print(f"\n[Pre] {tag}")
    img = cv2.imread(path)
    if img is None: raise FileNotFoundError(path)
    print(f"  Loaded: {img.shape[1]}×{img.shape[0]}")
    img = correct_perspective(img, tag)
    img, _ = deskew(img)
    img = trim_white_edges(img)
    print(f"  Final: {img.shape[1]}×{img.shape[0]}")
    cv2.imwrite(CORR_DIR + f"{tag}_fixed.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 96])
    return img


def sift_match(gray_a, gray_b, ratio=0.67, nfeat=8000, max_dim=4000):
    ha, wa = gray_a.shape; hb, wb = gray_b.shape
    s = min(1.0, max_dim/max(ha,wa,hb,wb))
    if s<0.99:
        g_a = cv2.resize(gray_a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        g_b = cv2.resize(gray_b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        print(f"    SIFT scale: {s:.3f}")
    else:
        g_a, g_b = gray_a, gray_b
    sift = cv2.SIFT_create(nfeatures=nfeat, contrastThreshold=0.01, edgeThreshold=20)
    kp_a, des_a = sift.detectAndCompute(g_a, None)
    kp_b, des_b = sift.detectAndCompute(g_b, None)
    print(f"    KP: a={len(kp_a)}, b={len(kp_b)}")
    if des_a is None or des_b is None or len(kp_a)<8 or len(kp_b)<8: return None,None
    flann = cv2.FlannBasedMatcher(dict(algorithm=1,trees=5), dict(checks=150))
    raw = flann.knnMatch(des_a, des_b, k=2)
    good = [m for m,n in raw if m.distance<ratio*n.distance]
    print(f"    Matches: {len(good)}")
    if len(good)<8: return None,None
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])/s
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])/s
    return pts_a, pts_b


def find_H_strip_h(img_ref, img_mov, overlap_frac):
    hr, wr = img_ref.shape[:2]; hm, wm = img_mov.shape[:2]
    sw = int(min(wr,wm)*overlap_frac)
    xr = wr-sw
    ref_strip = img_ref[:, xr:]
    mov_strip = img_mov[:, :sw]
    gray_r = cv2.cvtColor(ref_strip, cv2.COLOR_BGR2GRAY)
    gray_m = cv2.cvtColor(mov_strip, cv2.COLOR_BGR2GRAY)
    pts_r, pts_m = sift_match(gray_r, gray_m)
    if pts_r is None: return None, 0
    pts_r_full = pts_r + np.float32([xr, 0])
    H, mask = cv2.findHomography(pts_m, pts_r_full, cv2.RANSAC, 4.0,
                                  maxIters=5000, confidence=0.999)
    n = int(mask.sum()) if mask is not None else 0
    return H, n


def validate_H_h(H, src_shape, ref_shape):
    if H is None: return False
    hs,ws = src_shape[:2]; hr,wr = ref_shape[:2]
    c = np.float32([[0,0],[ws,0],[ws,hs],[0,hs]]).reshape(-1,1,2)
    wc = cv2.perspectiveTransform(c, H).reshape(4,2)
    nw = wc[:,0].max()-wc[:,0].min(); nh = wc[:,1].max()-wc[:,1].min()
    if nw<ws*0.5 or nw>ws*2.0 or nh<hs*0.5 or nh>hs*2.0: return False
    if abs(H[2,0])>5e-4 or abs(H[2,1])>5e-4: return False
    tx,ty = H[0,2], H[1,2]
    if tx<wr*0.20 or tx>wr*1.40 or abs(ty)>hr*0.40: return False
    return True


def compute_H_h(img_ref, img_mov, overlap_frac=0.35):
    for frac in [overlap_frac, 0.50, 0.60]:
        print(f"  Strip SIFT (h, {frac:.0%})...")
        H, n = find_H_strip_h(img_ref, img_mov, frac)
        if H is not None and n>=10 and validate_H_h(H, img_mov.shape, img_ref.shape):
            print(f"  OK: {n} inliers  tx={H[0,2]:.0f}  ty={H[1,2]:.0f}")
            return H
        print(f"  {n} inliers — insufficient")
    print(f"  Full-image SIFT...")
    pts_r,pts_m = sift_match(cv2.cvtColor(img_ref,cv2.COLOR_BGR2GRAY),
                              cv2.cvtColor(img_mov,cv2.COLOR_BGR2GRAY))
    if pts_r is not None:
        H, mask = cv2.findHomography(pts_m,pts_r,cv2.RANSAC,4.0,maxIters=5000,confidence=0.999)
        n = int(mask.sum()) if mask is not None else 0
        print(f"  Full SIFT inliers: {n}")
        if H is not None and n>=10 and validate_H_h(H,img_mov.shape,img_ref.shape):
            return H
    print(f"  Translation fallback.")
    hr,wr = img_ref.shape[:2]; hm,wm = img_mov.shape[:2]
    tx = wr*(1-overlap_frac); ty = (hr-hm)/2
    return np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)


def make_content_mask(img, white_thresh=252):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nw = (gray<white_thresh).astype(np.uint8)*255
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT,(80,80))
    m = cv2.morphologyEx(nw, cv2.MORPH_CLOSE, k1)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT,(200,200))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)
    return m


def find_best_seam_col(canvas, warped, overlap_mask):
    ov_cols = np.where(overlap_mask.any(0))[0]
    if len(ov_cols)<3: return int(ov_cols.mean()) if len(ov_cols) else 0
    c0,c1 = ov_cols[0],ov_cols[-1]
    ec = cv2.Canny(cv2.cvtColor(canvas[:,c0:c1+1],cv2.COLOR_BGR2GRAY),30,90).sum(0).astype(float)
    ew = cv2.Canny(cv2.cvtColor(warped[:,c0:c1+1],cv2.COLOR_BGR2GRAY),30,90).sum(0).astype(float)
    return int(uniform_filter1d(ec+ew,40).argmin()) + c0


def feather_v(canvas, warped, seam, overlap_mask, px=12):
    hc,wc = canvas.shape[:2]; out = canvas.copy()
    x0=max(0,seam-px); x1=min(wc,seam+px+1)
    for x in range(x0,x1):
        col = overlap_mask[:,x]
        if not col.any(): continue
        a = (x-x0)/(x1-x0-1) if x1>x0+1 else 0.5
        out[col,x] = ((1-a)*canvas[col,x].astype(np.float32)+a*warped[col,x].astype(np.float32)).astype(np.uint8)
    rm = overlap_mask.copy(); rm[:,:x1]=False
    out[rm]=warped[rm]
    return out


def warp_onto(canvas, canvas_mask, new_img, H):
    hc,wc = canvas.shape[:2]
    warped = cv2.warpPerspective(new_img, H, (wc,hc), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cm = make_content_mask(new_img)
    warped_valid = cv2.warpPerspective(cm, H, (wc,hc), flags=cv2.INTER_NEAREST)>127
    only_new = warped_valid & (canvas_mask==0)
    overlap  = warped_valid & (canvas_mask >0)
    out = canvas.copy(); om = canvas_mask.copy()
    out[only_new]=warped[only_new]; om[only_new]=255
    if overlap.any():
        seam = find_best_seam_col(canvas, warped, overlap)
        out = feather_v(out, warped, seam, overlap, px=12)
    return out, om


def stitch_row1(imgs):
    Hs = [np.eye(3, dtype=np.float64)]
    for i in range(1, len(imgs)):
        print(f"\n  Aligning r1:{i+1}→{i}...")
        H = compute_H_h(imgs[i-1], imgs[i], overlap_frac=0.35)
        H[2,:] = [0,0,1]
        Hs.append(Hs[-1] @ H)
        print(f"  H local: tx={H[0,2]:.0f}  ty={H[1,2]:.0f}")
    # Build canvas
    all_c = []
    for img,H in zip(imgs,Hs):
        h,w = img.shape[:2]
        c = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        all_c.append(cv2.perspectiveTransform(c,H).reshape(4,2))
    pts = np.vstack(all_c)
    x_min=int(np.floor(pts[:,0].min())); x_max=int(np.ceil(pts[:,0].max()))
    y_min=int(np.floor(pts[:,1].min())); y_max=int(np.ceil(pts[:,1].max()))
    tx,ty = max(0,-x_min), max(0,-y_min)
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], np.float64)
    cw,ch = x_max-x_min+1, y_max-y_min+1
    print(f"\n  Canvas: {cw}×{ch}")
    canvas = np.zeros((ch,cw,3), np.uint8)
    cm = np.zeros((ch,cw), np.uint8)
    for i,(img,H) in enumerate(zip(imgs,Hs)):
        canvas, cm = warp_onto(canvas, cm, img, T@H)
        print(f"  Placed {i+1}/3")
    rows=np.where(cm.any(1))[0]; cols=np.where(cm.any(0))[0]
    if len(rows) and len(cols):
        canvas = canvas[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    return canvas


if __name__ == "__main__":
    # Re-preprocess all row 1 images with capped deskew
    row1_files = [
        ("69a424a3-IMG_6837.jpeg", "r1_1"),
        ("95841777-IMG_6838.jpeg", "r1_2"),
        ("d9b6fffb-IMG_6839.jpeg", "r1_3"),
    ]
    imgs = [preprocess(UPLOAD_DIR+f, tag) for f, tag in row1_files]

    print(f"\n[Stitch] Row 1 (fixed)...")
    row1 = stitch_row1(imgs)

    out = ROWS_DIR + "row_1.png"
    cv2.imwrite(out, row1)
    print(f"\nRow 1 fixed: {out}  ({row1.shape[1]}×{row1.shape[0]})")

    # Quick preview
    s = 800/row1.shape[1]
    p = cv2.resize(row1, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(OUTPUT_DIR + "rows_v6/row_1_preview.jpg", p, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"Preview: {p.shape[1]}×{p.shape[0]}")
