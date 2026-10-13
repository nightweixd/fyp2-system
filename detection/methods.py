"""Tile alignment and grout-width analysis using YOLO oriented boxes."""
import cv2
import numpy as np
import os
import datetime
import math
_model = None

def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        # Load the trained OBB model.
        path = os.environ.get("YOLO_MODEL_PATH", "runs/obb/train4/weights/best.pt")
        _model = YOLO(path)
    return _model

# Inspection thresholds
# Centroid deviation is in mm; skeleton deviation is in degrees.
CENTROID_TOL_PX  = 1  # Max allowed centroid deviation (mm)
SKELETON_TOL_DEG = 2.0   # Angular tolerance in degrees (angular deviation from 0°/90°)
EDGE_DEV_TOL_MM  = 1.5   # Max allowed edge deviation (mm)
WIDTH_MIN_MM     = 6.0
WIDTH_MAX_MM     = 20.0
WIDTH_STD_TOL_MM = 2.0

ROW_THRESHOLD = 80.0
COL_THRESHOLD = 80.0

BORDER_MARGIN  = 0
MIN_AREA_RATIO = 0.7
BOTTOM_CUTOFF_RATIO = 0.85  # Ignore tiles whose centroid is in the bottom 15% of frame (robot zone)
TILE_SIZE_MM   = 300
MANUAL_MM_PX   = 2.4

RESULTS_DIR = "static/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def _ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def save_result(img, prefix):
    p = f"{RESULTS_DIR}/{prefix}_{_ts()}.jpg"
    cv2.imwrite(p, img)
    return p

def filter_tiles_obb(obbs, shape, border=BORDER_MARGIN, ratio=MIN_AREA_RATIO):
    """
    Filters Oriented Bounding Boxes (4 corners) to remove false positives.
    """
    h, w = shape[:2]
    if len(obbs) == 0:
        return np.array([])

    out = []
    areas = [cv2.contourArea(pts) for pts in obbs]
    avg_a = np.mean(areas) if areas else 0

    for pts, a in zip(obbs, areas):
        # Reject boxes with off-screen corners.
        if any(p[0] <= border or p[1] <= border or p[0] >= w-border or p[1] >= h-border for p in pts):
            continue

        # Bottom cutoff: skip tiles whose centroid is in the robot zone
        cy_pts = float(np.mean(pts[:, 1]))
        if cy_pts >= h * BOTTOM_CUTOFF_RATIO:
            continue

        # Area check
        if a < ratio * avg_a:
            continue

        # Aspect ratio check using minAreaRect
        rect = cv2.minAreaRect(pts)
        rw, rh = rect[1]
        if rh == 0 or rw == 0: continue
        aspect = max(rw, rh) / min(rw, rh)

        # Allow aspect ratios from 1.2 to 3.5 to handle perspective
        # distortion at oblique camera angles on site.
        if aspect < 1.2 or aspect > 3.5:
            continue

        # Area Check (Filter out the floor trap and robot chassis)
        # Calculate the bounding box area relative to the whole image
        image_area = w * h

        if a < (image_area * 0.02):
            # Reject small detections such as floor traps.
            continue

        if a > (image_area * 0.25):
            # Reject large detections around the robot chassis.
            continue

        out.append(pts)

    print(f" Filtered: {len(obbs)} → {len(out)} OBB tiles")
    return np.array(out)

def get_scale_obb(obbs):
    if MANUAL_MM_PX > 0:
        return MANUAL_MM_PX
    if len(obbs) == 0:
        return 1.0

    # Calculate average width/height from the rotated rectangles
    dims = []
    for pts in obbs:
        rect = cv2.minAreaRect(pts)
        dims.extend([rect[1][0], rect[1][1]])

    avg = np.mean(dims)
    return TILE_SIZE_MM / avg if avg else 1.0

def skeletonize(binary):
    skel = np.zeros(binary.shape, np.uint8)
    el = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    img = binary.copy()
    while True:
        op = cv2.morphologyEx(img, cv2.MORPH_OPEN, el)
        temp = cv2.subtract(img, op)
        er = cv2.erode(img, el)
        skel = cv2.bitwise_or(skel, temp)
        img = er.copy()
        if cv2.countNonZero(img)==0: break
    return skel


# METHOD 1 — CENTROID (OBB with Angled Grid Support)

def run_centroid(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, {"error":"Cannot load image"}

    model = get_model()
    res = model(image_path, verbose=False)

    if not hasattr(res[0], 'obb') or res[0].obb is None:
        return None, {"error": "Model must be a YOLOv8 OBB model. Standard boxes detected."}

    obbs = res[0].obb.xyxyxyxy.cpu().numpy()
    obbs = filter_tiles_obb(obbs, img.shape)

    if len(obbs) < 2: return None, {"error":"Need ≥2 valid tiles"}

    scale = get_scale_obb(obbs)

    # Calculate centroid as the average of the 4 OBB corners
    cxys = [(int(np.mean(pts[:,0])), int(np.mean(pts[:,1]))) for pts in obbs]

    # LINE OF BEST FIT MATH
    def fit_and_measure(points):
        if len(points) < 2:
            return (points[0][0], points[0][1]), (1.0, 0.0), [0.0]

        pts = np.array(points, dtype=np.float32)
        # Calculate the mathematical line of best fit (Linear Regression)
        [vx, vy, x0, y0] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

        devs = []
        for px, py in points:
            # Measure perpendicular distance from the centroid to the angled line
            d = abs((px - x0) * vy - (py - y0) * vx)
            devs.append(d)
        return (x0, y0), (vx, vy), devs

    # Row analysis
    cxys_y_sorted = sorted(cxys, key=lambda c:c[1])
    rows, cur_r = [], [cxys_y_sorted[0]]
    for i in range(1, len(cxys_y_sorted)):
        if abs(cxys_y_sorted[i][1] - cxys_y_sorted[i-1][1]) < ROW_THRESHOLD:
            cur_r.append(cxys_y_sorted[i])
        else:
            rows.append(cur_r); cur_r=[cxys_y_sorted[i]]
    rows.append(cur_r)

    row_lines = [fit_and_measure(r) for r in rows]

    # Column analysis
    cxys_x_sorted = sorted(cxys, key=lambda c:c[0])
    cols, cur_c = [], [cxys_x_sorted[0]]
    for i in range(1, len(cxys_x_sorted)):
        if abs(cxys_x_sorted[i][0] - cxys_x_sorted[i-1][0]) < COL_THRESHOLD:
            cur_c.append(cxys_x_sorted[i])
        else:
            cols.append(cur_c); cur_c=[cxys_x_sorted[i]]
    cols.append(cur_c)

    col_lines = [fit_and_measure(c) for c in cols]

    tile_recs=[]
    for i, pts in enumerate(obbs):
        cx, cy = cxys[i] if i<len(cxys) else (0,0)

        # Find exactly which row and column this specific tile belongs to
        rd = 0.0
        for r_idx, r in enumerate(rows):
            if (cx, cy) in r:
                pt_idx = r.index((cx, cy))
                rd = row_lines[r_idx][2][pt_idx]
                break

        cd = 0.0
        for c_idx, c in enumerate(cols):
            if (cx, cy) in c:
                pt_idx = c.index((cx, cy))
                cd = col_lines[c_idx][2][pt_idx]
                break

        # Calculate average deviation from the true physical grid angles
        td = (rd + cd) / 2
        td_mm = round(td * scale, 2)
        ok = td < CENTROID_TOL_PX
        col = (0, 255, 0) if ok else (0, 0, 255)

        # Draw OBB Polygon
        cv2.polylines(img, [pts.astype(np.int32)], True, col, 2)

        # Draw the exact center dot for visual confirmation
        cv2.circle(img, (int(cx), int(cy)), 3, col, -1)

        tx, ty = int(np.min(pts[:,0])), int(np.min(pts[:,1]))
        label = f"T{i+1}: {td_mm}mm"
        cv2.putText(img, label, (tx+4, ty+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        x, y, w, h = cv2.boundingRect(pts.astype(np.float32))
        tile_recs.append({"tile_id":i+1, "bbox":[x, y, x+w, y+h],
                           "alignment_deviation":td_mm, "width":None,
                           "align_ok":ok, "width_ok":None,
                           "status":"PASS" if ok else "FAIL"})

    # DRAW ANGLED ALIGNMENT LINES
    mult = max(img.shape[0], img.shape[1])
    for (x0, y0), (vx, vy), devs in row_lines:
        if len(devs) > 1:
            avg_dev = np.mean(devs)
            col_line = (0,255,0) if avg_dev < CENTROID_TOL_PX else (0,0,255)
            # Extend the mathematical line all the way across the screen
            pt1 = (int(x0 - vx * mult), int(y0 - vy * mult))
            pt2 = (int(x0 + vx * mult), int(y0 + vy * mult))
            cv2.line(img, pt1, pt2, col_line, 2)

    for (x0, y0), (vx, vy), devs in col_lines:
        if len(devs) > 1:
            avg_dev = np.mean(devs)
            col_line = (0,255,0) if avg_dev < CENTROID_TOL_PX else (0,0,255)
            pt1 = (int(x0 - vx * mult), int(y0 - vy * mult))
            pt2 = (int(x0 + vx * mult), int(y0 + vy * mult))
            cv2.line(img, pt1, pt2, col_line, 2)

    all_devs = [r["alignment_deviation"] for r in tile_recs]
    avg_mm = round(float(np.mean(all_devs)), 2) if all_devs else 0.0
    any_fail = any(r["status"] == "FAIL" for r in tile_recs)
    status = "FAIL" if any_fail else "PASS"

    summary = f"Tiles:{len(obbs)} | Avg Dev:{avg_mm}mm | {status}"
    cv2.putText(img, summary, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, summary, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

    out = save_result(img, "centroid")
    return out, {"method":"Geometric Centroid", "status":status,
                "deviation":avg_mm, "width":None, "tiles":len(obbs),
                "tile_results":tile_recs}


# METHOD 2 — SKELETON (OBB)
# Alignment now uses angular deviation (degrees) from
# separate H/V regression lines, matching thesis Section 3.5.2

def run_skeleton(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, {"error": "Cannot load image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    model = get_model()
    res   = model(image_path, verbose=False)

    if not hasattr(res[0], 'obb') or res[0].obb is None:
        return None, {"error": "Model must be a YOLOv8 OBB model."}

    obbs  = res[0].obb.xyxyxyxy.cpu().numpy()
    obbs  = filter_tiles_obb(obbs, img.shape)
    if len(obbs) == 0: return None, {"error": "No valid tiles"}

    scale = get_scale_obb(obbs)

    # Build grout-only mask (hollow out tile interiors)
    all_x = obbs[:, :, 0].flatten()
    all_y = obbs[:, :, 1].flatten()
    mnx   = max(0, int(np.min(all_x)));  mny = max(0, int(np.min(all_y)))
    mxx   = min(w, int(np.max(all_x)));  mxy = min(h, int(np.max(all_y)))
    mask  = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (mnx, mny), (mxx, mxy), 255, -1)

    for pts in obbs:
        cx, cy     = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        shrunk_pts = []
        for p in pts:
            vec    = [cx - p[0], cy - p[1]]
            length = np.hypot(vec[0], vec[1])
            if length > 0:
                nx = p[0] + (vec[0] / length) * 45
                ny = p[1] + (vec[1] / length) * 45
                shrunk_pts.append([nx, ny])
            else:
                shrunk_pts.append(p)
        cv2.fillPoly(mask, [np.array(shrunk_pts, dtype=np.int32)], 0)

    # Binary segmentation & skeletonization
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 8)
    if cv2.countNonZero(binary) > binary.size * 0.5:
        binary = cv2.bitwise_not(binary)
    binary = cv2.bitwise_and(binary, mask)

    skel = skeletonize(binary)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Global width stats for summary
    sy, sx = np.where(skel > 0)
    all_w  = dist[sy, sx] * 2.0

    # Split skeleton into H and V grout lines
    hk     = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    vk     = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    skel_h = cv2.morphologyEx(skel, cv2.MORPH_OPEN, hk)
    skel_v = cv2.morphologyEx(skel, cv2.MORPH_OPEN, vk)

    # Compute tile centroids for row/col grouping
    cxys = [(int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))) for pts in obbs]

    # Group tiles into ROWS
    cxys_y_sorted = sorted(enumerate(cxys), key=lambda x: x[1][1])
    rows, cur_r = [], [cxys_y_sorted[0]]
    for i in range(1, len(cxys_y_sorted)):
        if abs(cxys_y_sorted[i][1][1] - cxys_y_sorted[i-1][1][1]) < ROW_THRESHOLD:
            cur_r.append(cxys_y_sorted[i])
        else:
            rows.append(cur_r); cur_r = [cxys_y_sorted[i]]
    rows.append(cur_r)

    row_angle_devs = {}
    for row in rows:
        idxs  = [item[0] for item in row]
        ys    = [item[1][1] for item in row]
        avg_y = int(np.mean(ys))

        band_y1 = max(0, avg_y - int(ROW_THRESHOLD // 2))
        band_y2 = min(h, avg_y + int(ROW_THRESHOLD // 2))
        band    = skel_h[band_y1:band_y2, :]
        hy, hx_px = np.where(band > 0)

        if len(hx_px) >= 10:
            hpts     = np.column_stack((hx_px.astype(np.float32),
                                        hy.astype(np.float32)))
            lp       = cv2.fitLine(hpts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx2, vy2 = float(lp[0]), float(lp[1])
            angle    = abs(math.degrees(math.atan2(vy2, vx2)))
            angle    = min(angle, 180.0 - angle)
            for idx in idxs:
                row_angle_devs[idx] = angle
        else:
            for idx in idxs:
                row_angle_devs[idx] = 0.0

    # Group tiles into COLUMNS
    cxys_x_sorted = sorted(enumerate(cxys), key=lambda x: x[1][0])
    cols, cur_c = [], [cxys_x_sorted[0]]
    for i in range(1, len(cxys_x_sorted)):
        if abs(cxys_x_sorted[i][1][0] - cxys_x_sorted[i-1][1][0]) < COL_THRESHOLD:
            cur_c.append(cxys_x_sorted[i])
        else:
            cols.append(cur_c); cur_c = [cxys_x_sorted[i]]
    cols.append(cur_c)

    col_angle_devs = {}
    for col_grp in cols:
        idxs  = [item[0] for item in col_grp]
        xs    = [item[1][0] for item in col_grp]
        avg_x = int(np.mean(xs))

        band_x1 = max(0, avg_x - int(COL_THRESHOLD // 2))
        band_x2 = min(w, avg_x + int(COL_THRESHOLD // 2))
        band    = skel_v[:, band_x1:band_x2]
        vy_px, vx_px = np.where(band > 0)

        if len(vx_px) >= 10:
            vpts     = np.column_stack((vx_px.astype(np.float32),
                                        vy_px.astype(np.float32)))
            lp       = cv2.fitLine(vpts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx2, vy2 = float(lp[0]), float(lp[1])
            angle    = abs(math.degrees(math.atan2(vy2, vx2)))
            angle    = min(angle, 180.0 - angle)
            for idx in idxs:
                col_angle_devs[idx] = abs(90.0 - angle)
        else:
            for idx in idxs:
                col_angle_devs[idx] = 0.0

    # Per-tile analysis
    tile_recs = []

    for i, pts in enumerate(obbs):
        bx, by, bw, bh = cv2.boundingRect(pts.astype(np.float32))
        x1, y1 = bx, by
        x2, y2 = bx + bw, by + bh

        # WIDTH — distance transform at skeleton pixels in tile ROI
        roi_s  = skel[y1:y2, x1:x2]
        roi_d  = dist[y1:y2, x1:x2]
        ry, rx = np.where(roi_s > 0)
        w_roi  = float(np.mean(roi_d[ry, rx] * 2.0)) * scale if len(ry) > 0 else 0.0

        # ALIGNMENT — from full-width/height band regression
        r_dev = row_angle_devs.get(i, 0.0)
        c_dev = col_angle_devs.get(i, 0.0)
        d_deg = round((r_dev + c_dev) / 2, 2)

        a_ok = d_deg < SKELETON_TOL_DEG
        w_ok = WIDTH_MIN_MM <= w_roi <= WIDTH_MAX_MM

        # SEPARATE FAIL REASON
        if a_ok and w_ok:
            status_tile = "PASS"
            col = (0, 255, 0)       # green
        elif not a_ok and not w_ok:
            status_tile = "FAIL-BOTH"
            col = (0, 0, 255)       # red
        elif not a_ok:
            status_tile = "FAIL-ALIGN"
            col = (0, 0, 255)       # red
        else:
            status_tile = "FAIL-WIDTH"
            col = (0, 165, 255)     # orange — width fail is visually distinct

        cv2.polylines(img, [pts.astype(np.int32)], True, col, 2)
        tx, ty = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
        wl     = f"{w_roi:.1f}mm" if w_roi > 0 else "?"
        cv2.putText(img, f"T{i+1}",
                    (tx+4, ty+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        cv2.putText(img, f"W:{wl} D:{d_deg}deg",
                    (tx+4, ty+38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(img, status_tile,
                    (tx+4, ty+54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        tile_recs.append({
            "tile_id":             i + 1,
            "bbox":                [x1, y1, x2, y2],
            "alignment_deviation": d_deg,
            "width":               round(w_roi, 2),
            "align_ok":            a_ok,
            "width_ok":            w_ok,
            "status":              status_tile
        })

    # Skeleton pixel overlay
    for py2, px2 in zip(sy, sx):
        col = (0, 255, 0) if dist[py2, px2] * 2 * scale >= WIDTH_MIN_MM else (0, 165, 255)
        cv2.circle(img, (px2, py2), 1, col, -1)

    # Summary — show separate align and width results
    all_devs = [r["alignment_deviation"] for r in tile_recs]
    avg_dev  = round(float(np.mean(all_devs)), 2) if all_devs else 0.0
    avg_wid  = round(float(np.mean(all_w)) * scale, 2) if len(all_w) else 0.0
    std_wid  = round(float(np.std(all_w))  * scale, 2) if len(all_w) else 0.0

    align_fail = any(not r["align_ok"] for r in tile_recs)
    width_fail = any(r["width_ok"] == False for r in tile_recs)

    if align_fail and width_fail:
        status = "FAIL-BOTH"
    elif align_fail:
        status = "FAIL-ALIGN"
    elif width_fail:
        status = "FAIL-WIDTH"
    else:
        status = "PASS"

    summary = f"Tiles:{len(obbs)} | W:{avg_wid}mm+/-{std_wid}mm | D:{avg_dev}deg | {status}"
    cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    out = save_result(img, "skeleton")
    return out, {"method":        "Skeletonisation",
                 "status":        status,
                 "deviation_mm":  avg_dev,
                 "degree":        avg_dev,
                 "tiles":         len(obbs),
                 "tile_results":  tile_recs}


# METHOD 3 — EDGE (OBB)

def run_edge(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, {"error": "Cannot load image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]

    model = get_model()
    res   = model(image_path, verbose=False)

    if not hasattr(res[0], 'obb') or res[0].obb is None:
        return None, {"error": "Model must be a YOLOv8 OBB model."}

    obbs  = res[0].obb.xyxyxyxy.cpu().numpy()
    obbs  = filter_tiles_obb(obbs, img.shape)
    if len(obbs) == 0: return None, {"error": "No tiles detected"}

    scale = get_scale_obb(obbs)

    # Build grout-only mask
    # Start BLACK (grout exposed), paint tile interiors WHITE, then
    # invert so only the grout channel pixels remain visible.
    # A small inward shrink (8 px) avoids tile-surface false edges.
    tile_mask = np.zeros((h, w), dtype=np.uint8)   # all black = grout exposed
    for pts in obbs:
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        shrunk = []
        for p in pts:
            vec    = [cx - p[0], cy - p[1]]
            length = np.hypot(vec[0], vec[1])
            if length > 0:
                shrunk.append([p[0] + (vec[0] / length) * 8,
                                p[1] + (vec[1] / length) * 8])
            else:
                shrunk.append(list(p))
        cv2.fillPoly(tile_mask, [np.array(shrunk, dtype=np.int32)], 255)

    # grout_mask=255 where grout is, 0 where tiles are
    grout_mask = cv2.bitwise_not(tile_mask)

    # Pre-process & Canny on grout region only
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    masked = cv2.bitwise_and(blur, blur, mask=grout_mask)
    edges  = cv2.Canny(masked, 50, 150)

    # Separate horizontal and vertical grout edges
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    ve = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vk)   # edges of vertical grout lines
    he = cv2.morphologyEx(edges, cv2.MORPH_OPEN, hk)   # edges of horizontal grout lines

    # Caliper width measurement
    # For each scanline through a grout strip, find the leftmost and
    # rightmost (or topmost / bottommost) edge pixel and measure the gap.
    # This is the actual joint width — not line-fit scatter.
    def caliper_width_mm(edge_strip, vertical_grout):
        """
        edge_strip   : binary edge image cropped to a tile's bounding box
        vertical_grout : True  → scan rows for Horizontal gaps
                         False → scan cols for Vertical gaps
        Returns mean width in mm, or 0.0 if no valid measurements.
        """
        widths = []

        # Define realistic grout gap limits in pixels based on the image scale
        # Reject anything larger than 40mm (it's measuring across the tile)
        # Reject anything smaller than 2mm (it's just noise/thick edges)
        MAX_PX_GAP = 40.0 / scale if scale > 0 else 50
        MIN_PX_GAP = 2.0 / scale if scale > 0 else 2

        if vertical_grout:
            for row in range(edge_strip.shape[0]):
                cols = np.where(edge_strip[row] > 0)[0]
                # Measure distance between *adjacent* edges in this row
                for k in range(len(cols) - 1):
                    gap = float(cols[k+1] - cols[k])
                    if MIN_PX_GAP <= gap <= MAX_PX_GAP:
                        widths.append(gap)
        else:
            for col in range(edge_strip.shape[1]):
                rows = np.where(edge_strip[:, col] > 0)[0]
                # Measure distance between *adjacent* edges in this column
                for k in range(len(rows) - 1):
                    gap = float(rows[k+1] - rows[k])
                    if MIN_PX_GAP <= gap <= MAX_PX_GAP:
                        widths.append(gap)

        if not widths:
            return 0.0, 0.0

        arr = np.array(widths)
        return float(np.mean(arr)) * scale, float(np.std(arr)) * scale

    # Per-tile analysis
    tile_recs = []
    all_widths_mm = []

    for i, pts in enumerate(obbs):
        bx, by, bw, bh = cv2.boundingRect(pts.astype(np.float32))
        x1 = max(0, bx - 5);  y1 = max(0, by - 5)
        x2 = min(w, bx+bw+5); y2 = min(h, by+bh+5)

        tile_widths = []

        # Horizontal grout lines around this tile (use he strip)
        for e_img, vg in [(ve, True), (he, False)]:
            strip = e_img[y1:y2, x1:x2]
            if cv2.countNonZero(strip) < 10:
                continue
            w_mean, w_std = caliper_width_mm(strip, vg)
            if w_mean > 0:
                tile_widths.append(w_mean)
                # Overlay detected edge pixels in cyan for visual feedback
                ey_px, ex_px = np.where(strip > 0)
                for (epy, epx) in zip(ey_px, ex_px):
                    img[epy + y1, epx + x1] = (255, 255, 0)   # cyan

        w_tile     = float(np.mean(tile_widths)) if tile_widths else 0.0
        w_ok       = WIDTH_MIN_MM <= w_tile <= WIDTH_MAX_MM
        if w_tile > 0:
            all_widths_mm.append(w_tile)

        # Width is what edge method measures; alignment is out-of-scope
        # (thesis Section 3.5.3e notes it "will not be effective for alignment")
        # so we only classify on width here.
        if w_tile == 0.0:
            status_tile = "NO-DATA"
            col = (128, 128, 128)   # grey — no grout edges found in ROI
        elif w_ok:
            status_tile = "PASS"
            col = (0, 255, 0)       # green
        else:
            status_tile = "FAIL-WIDTH"
            col = (0, 165, 255)     # orange (width fail, same as Method 2)

        cv2.polylines(img, [pts.astype(np.int32)], True, col, 2)
        tx, ty = int(np.min(pts[:, 0])), int(np.min(pts[:, 1]))
        cv2.putText(img, f"T{i+1}",
                    (tx+4, ty+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        wl = f"{w_tile:.1f}mm" if w_tile > 0 else "?"
        cv2.putText(img, f"W:{wl}",
                    (tx+4, ty+38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(img, status_tile,
                    (tx+4, ty+54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        tile_recs.append({
            "tile_id":             i + 1,
            "bbox":                [bx, by, bx+bw, by+bh],
            "alignment_deviation": None,   # not measured by edge method
            "width":               round(w_tile, 2),
            "align_ok":            None,
            "width_ok":            w_ok if w_tile > 0 else None,
            "status":              status_tile
        })

    # Summary
    valid_widths = [r["width"] for r in tile_recs if r["width"] and r["width"] > 0]
    avg_wid = round(float(np.mean(valid_widths)), 2)  if valid_widths else 0.0
    std_wid = round(float(np.std(valid_widths)),  2)  if valid_widths else 0.0

    width_fail = any(r["status"] == "FAIL-WIDTH" for r in tile_recs)
    status     = "FAIL" if width_fail else "PASS"

    summary = f"Tiles:{len(obbs)} | W:{avg_wid}mm+/-{std_wid}mm | {status}"
    cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

    out = save_result(img, "edge")
    return out, {"method":      "Edge Detection",
                 "status":      status,
                 "deviation":   None,
                 "width":       avg_wid,
                 "width_std":   std_wid,
                 "tiles":       len(obbs),
                 "tile_results": tile_recs}
