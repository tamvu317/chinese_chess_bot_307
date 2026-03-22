import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import math
import subprocess
import os
import json
from ultralytics import YOLO

# =============================================================
# 1. CẤU HÌNH HỆ THỐNG
# =============================================================
MODEL_PATH = "best (2).pt"
CONF_THRESHOLD = 0.6
CAMERA_INDEX = 1
THINK_TIME_MS = 2000

ROBOT_SIDE = 'b'

FEN_MAP = {
    "tuongdo": "K", "sido": "A", "tinhdo": "B", "mado": "N", "xedo": "R", "phaodo": "C", "totdo": "P",
    "tuongden": "k", "siden": "a", "tinhden": "b", "maden": "n", "xeden": "r", "phaoden": "c", "totden": "p"
}

ROW_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
REAL_WIDTH_A1_A9 = 230.0
REAL_HEIGHT_A1_J1 = 234.0
OFFSET_X = 115.0
OFFSET_Y = 300.0

# Tham số xử lý ảnh
LENGTH_VAL = 100
THRESH_VAL = 135
MAX_THICK = 28
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 60
HOUGH_MIN_LINE_LEN = 120
HOUGH_MAX_LINE_GAP = 20
SNAP_RADIUS = 12
USE_HOMOGRAPHY_DEFAULT = 0
GRID_LINE_THICK_DEFAULT = 2
GRID_POINT_RADIUS_DEFAULT = 4

# =============================================================
# 2. KHỞI TẠO PIKAFISH ENGINE
# =============================================================
print("⏳ Đang khởi động Pikafish...")
try:
    engine = subprocess.Popen(
        'pikafish.exe',
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print("✅ Pikafish OK!\n")
except FileNotFoundError:
    print("❌ Không tìm thấy pikafish.exe!")
    os._exit(1)


def get_best_move_from_pikafish(fen_string, think_time_ms=2000):
    engine.stdin.write(f"position fen {fen_string}\n")
    engine.stdin.write(f"go movetime {think_time_ms}\n")
    engine.stdin.flush()
    while True:
        line = engine.stdout.readline().strip()
        if line.startswith("bestmove"):
            best_move = line.split(" ")[1]
            return best_move if best_move != "(none)" else None


def ucci_to_grid_label(ucci_str):
    file_char = ucci_str[0].lower()
    rank_char = ucci_str[1]
    col = str(ord(file_char) - ord('a') + 1)
    row = chr(ord('A') + (9 - int(rank_char)))
    return f"{row}{col}"


def generate_fen_and_mapping(mapped_points, yolo_pieces):
    board = [["" for _ in range(9)] for _ in range(10)]
    grid_to_piece = {}

    for p in yolo_pieces:
        cx, cy, cls_name = p[0], p[1], p[3]
        best_dist = float('inf')
        best_row, best_col = -1, -1
        best_label = ""

        for pt in mapped_points:
            dist = math.hypot(cx - pt['px'], cy - pt['py'])
            if dist < best_dist:
                best_dist = dist
                best_row = ord(pt['label'][0]) - ord('A')
                best_col = int(pt['label'][1:]) - 1
                best_label = pt['label']

        if best_row != -1 and best_col != -1 and best_dist < 25:
            board[best_row][best_col] = FEN_MAP.get(cls_name, "")
            grid_to_piece[best_label] = cls_name

    fen_rows = []
    for row in board:
        empty_count = 0
        row_str = ""
        for cell in row:
            if cell == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += cell
        if empty_count > 0:
            row_str += str(empty_count)
        fen_rows.append(row_str)

    fen_string = "/".join(fen_rows) + f" {ROBOT_SIDE} - - 0 1"
    return fen_string, grid_to_piece


def cluster_1d(values, n_clusters):
    if len(values) < n_clusters:
        return None
    sorted_vals = sorted(values)
    vmin, vmax = sorted_vals[0], sorted_vals[-1]
    centers = [vmin + i * (vmax - vmin) / (n_clusters - 1) for i in range(n_clusters)]
    for _ in range(30):
        clusters = [[] for _ in range(n_clusters)]
        for v in sorted_vals:
            idx = min(range(n_clusters), key=lambda i: abs(v - centers[i]))
            clusters[idx].append(v)
        new_centers = [sum(c) / len(c) if c else centers[i] for i, c in enumerate(clusters)]
        if all(abs(a - b) < 0.5 for a, b in zip(centers, new_centers)):
            break
        centers = new_centers
    return sorted(centers)


def line_from_segment(seg):
    x1, y1, x2, y2 = seg
    a = float(y1 - y2)
    b = float(x2 - x1)
    c = float(x1 * y2 - x2 * y1)
    norm = math.hypot(a, b)
    if norm < 1e-6:
        return None
    return (a / norm, b / norm, c / norm)


def split_lines_by_orientation(lines):
    if len(lines) < 6:
        return None, None

    vectors = []
    for line in lines:
        a, b, _ = line
        angle = math.atan2(-a, b)  # hướng của đoạn thẳng
        vectors.append(np.array([math.cos(2.0 * angle), math.sin(2.0 * angle)], dtype=np.float32))

    centers = [vectors[0], vectors[len(vectors) // 2]]
    labels = [0] * len(vectors)
    for _ in range(15):
        for i, v in enumerate(vectors):
            d0 = np.linalg.norm(v - centers[0])
            d1 = np.linalg.norm(v - centers[1])
            labels[i] = 0 if d0 <= d1 else 1

        new_centers = []
        for k in (0, 1):
            members = [vectors[i] for i, lb in enumerate(labels) if lb == k]
            if members:
                m = np.mean(members, axis=0)
                m_norm = np.linalg.norm(m)
                new_centers.append(m / m_norm if m_norm > 1e-6 else centers[k])
            else:
                new_centers.append(centers[k])

        if all(np.linalg.norm(new_centers[k] - centers[k]) < 1e-3 for k in (0, 1)):
            break
        centers = new_centers

    group_0 = [lines[i] for i, lb in enumerate(labels) if lb == 0]
    group_1 = [lines[i] for i, lb in enumerate(labels) if lb == 1]
    if not group_0 or not group_1:
        return None, None
    return group_0, group_1


def cluster_line_family(lines, n_clusters, center_xy):
    if len(lines) < n_clusters:
        return None

    cx, cy = center_xy
    offsets = [ln[0] * cx + ln[1] * cy + ln[2] for ln in lines]
    centers = cluster_1d(offsets, n_clusters)
    if not centers:
        return None

    grouped = [[] for _ in range(n_clusters)]
    for idx, off in enumerate(offsets):
        cidx = min(range(n_clusters), key=lambda i: abs(off - centers[i]))
        grouped[cidx].append(lines[idx])

    merged = []
    for i, members in enumerate(grouped):
        if not members:
            continue
        arr = np.array(members, dtype=np.float32)
        mean_line = arr.mean(axis=0)
        a, b, c = float(mean_line[0]), float(mean_line[1]), float(mean_line[2])
        norm = math.hypot(a, b)
        if norm < 1e-6:
            continue
        a, b, c = a / norm, b / norm, c / norm
        if a * cx + b * cy + c > 0:
            a, b, c = -a, -b, -c
        merged.append((a, b, c, centers[i]))

    merged.sort(key=lambda x: x[3])
    return [(a, b, c) for a, b, c, _ in merged]


def intersect_lines(line_1, line_2):
    a1, b1, c1 = line_1
    a2, b2, c2 = line_2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return (float(x), float(y))


def draw_infinite_line(img, line, color, thickness=1):
    h, w = img.shape[:2]
    a, b, c = line

    points = []
    if abs(b) > 1e-6:
        y0 = (-c - a * 0.0) / b
        y1 = (-c - a * (w - 1.0)) / b
        if 0 <= y0 < h:
            points.append((0, int(round(y0))))
        if 0 <= y1 < h:
            points.append((w - 1, int(round(y1))))

    if abs(a) > 1e-6:
        x0 = (-c - b * 0.0) / a
        x1 = (-c - b * (h - 1.0)) / a
        if 0 <= x0 < w:
            points.append((int(round(x0)), 0))
        if 0 <= x1 < w:
            points.append((int(round(x1)), h - 1))

    if len(points) >= 2:
        p1 = points[0]
        p2 = points[1]
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def line_y_at_x(line, x):
    a, b, c = line
    if abs(b) < 1e-6:
        return None
    return float((-a * x - c) / b)


def line_x_at_y(line, y):
    a, b, c = line
    if abs(a) < 1e-6:
        return None
    return float((-b * y - c) / a)


def sort_lines_by_axis(lines, axis, w, h):
    if axis == 'y':
        x_mid = w / 2.0
        keyed = []
        for ln in lines:
            yv = line_y_at_x(ln, x_mid)
            if yv is None:
                yv = h / 2.0
            keyed.append((yv, ln))
        keyed.sort(key=lambda x: x[0])
        return [ln for _, ln in keyed]

    y_mid = h / 2.0
    keyed = []
    for ln in lines:
        xv = line_x_at_y(ln, y_mid)
        if xv is None:
            xv = w / 2.0
        keyed.append((xv, ln))
    keyed.sort(key=lambda x: x[0])
    return [ln for _, ln in keyed]


def snap_points_to_features(gray, points, radius=12):
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=6)
    if corners is None:
        return points

    candidates = [(float(p[0][0]), float(p[0][1])) for p in corners]
    snapped = []
    for px, py in points:
        best = None
        best_d = float('inf')
        for cx, cy in candidates:
            d = math.hypot(px - cx, py - cy)
            if d < best_d and d <= radius:
                best_d = d
                best = (cx, cy)
        if best is not None:
            snapped.append((best[0], best[1]))
        else:
            snapped.append((px, py))
    return snapped


def detect_board_points_with_hough(frame, params=None):
    h, w = frame.shape[:2]
    params = params or {}
    canny_low = int(params.get("canny_low", CANNY_LOW))
    canny_high = int(params.get("canny_high", CANNY_HIGH))
    hough_threshold = int(params.get("hough_threshold", HOUGH_THRESHOLD))
    hough_min_line_len = int(params.get("hough_min_line_len", HOUGH_MIN_LINE_LEN))
    hough_max_line_gap = int(params.get("hough_max_line_gap", HOUGH_MAX_LINE_GAP))
    snap_radius = int(params.get("snap_radius", SNAP_RADIUS))
    use_homography = int(params.get("use_homography", USE_HOMOGRAPHY_DEFAULT))

    canny_high = max(canny_low + 1, canny_high)
    hough_min_line_len = max(20, hough_min_line_len)
    hough_max_line_gap = max(1, hough_max_line_gap)
    snap_radius = max(2, snap_radius)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_line_len,
        maxLineGap=hough_max_line_gap,
    )
    if raw_lines is None:
        return None, edges, None

    norm_lines = []
    for line in raw_lines:
        x1, y1, x2, y2 = line[0]
        if math.hypot(x2 - x1, y2 - y1) < 40:
            continue
        coeff = line_from_segment((x1, y1, x2, y2))
        if coeff is not None:
            norm_lines.append(coeff)

    if len(norm_lines) < 12:
        return None, edges, None

    family_a, family_b = split_lines_by_orientation(norm_lines)
    if family_a is None or family_b is None:
        return None, edges, None

    center_xy = (w / 2.0, h / 2.0)
    best_grid = None
    for rows_count, cols_count in ((10, 9), (9, 10)):
        row_lines = cluster_line_family(family_a, rows_count, center_xy)
        col_lines = cluster_line_family(family_b, cols_count, center_xy)
        if row_lines is None or col_lines is None:
            continue

        raw_intersections = []
        for rl in row_lines:
            for cl in col_lines:
                p = intersect_lines(rl, cl)
                if p is None:
                    continue
                x, y = p
                if -40 <= x <= w + 40 and -40 <= y <= h + 40:
                    raw_intersections.append((x, y))

        if len(raw_intersections) < 60:
            continue

        score = len(raw_intersections)
        if best_grid is None or score > best_grid[0]:
            best_grid = (score, row_lines, col_lines, rows_count, cols_count)

    if best_grid is None:
        return None, edges, None

    _, row_lines, col_lines, rows_count, cols_count = best_grid

    # Bảo đảm nhóm 10 là hàng và nhóm 9 là cột theo trục ảnh.
    if rows_count != 10 or cols_count != 9:
        row_lines, col_lines = col_lines, row_lines

    row_lines = sort_lines_by_axis(row_lines, axis='y', w=w, h=h)
    col_lines = sort_lines_by_axis(col_lines, axis='x', w=w, h=h)

    tl = intersect_lines(row_lines[0], col_lines[0])
    tr = intersect_lines(row_lines[0], col_lines[-1])
    br = intersect_lines(row_lines[-1], col_lines[-1])
    bl = intersect_lines(row_lines[-1], col_lines[0])
    if not tl or not tr or not br or not bl:
        return None, edges, None

    corners = np.array([tl, tr, br, bl], dtype=np.float32)

    projected_points = []
    if use_homography:
        cell = 80.0
        dst_corners = np.array(
            [
                [0.0, 0.0],
                [8.0 * cell, 0.0],
                [8.0 * cell, 9.0 * cell],
                [0.0, 9.0 * cell],
            ],
            dtype=np.float32,
        )

        h_mat = cv2.getPerspectiveTransform(corners, dst_corners)
        if abs(np.linalg.det(h_mat)) < 1e-6:
            return None, edges, None

        h_inv = np.linalg.inv(h_mat)
        for r_idx in range(10):
            for c_idx in range(9):
                flat_pt = np.array([[[c_idx * cell, r_idx * cell]]], dtype=np.float32)
                src_pt = cv2.perspectiveTransform(flat_pt, h_inv)[0][0]
                px = float(src_pt[0])
                py = float(src_pt[1])
                projected_points.append((px, py))
    else:
        for r_idx in range(10):
            for c_idx in range(9):
                p = intersect_lines(row_lines[r_idx], col_lines[c_idx])
                if p is None:
                    return None, edges, None
                projected_points.append((float(p[0]), float(p[1])))

    snapped_points = snap_points_to_features(gray, projected_points, radius=snap_radius)

    mapped_points = []
    idx = 0
    for r_idx in range(10):
        for c_idx in range(9):
            px, py = snapped_points[idx]
            idx += 1
            label = f"{ROW_LABELS[r_idx]}{c_idx + 1}"
            mapped_points.append({'label': label, 'px': int(round(px)), 'py': int(round(py))})

    debug_data = {
        "projected_points": projected_points,
        "snapped_points": snapped_points,
        "corners": corners,
        "use_homography": use_homography,
        "row_lines": row_lines,
        "col_lines": col_lines,
    }
    return mapped_points, edges, debug_data


# =============================================================
# 3. HÀM XỬ LÝ FRAME: YOLO + GRID + PIKAFISH
# =============================================================
def process_frame(frame, detect_params=None):
    """Nhận 1 frame 640x640, trả về (annotated_frame, kết quả dict) hoặc None nếu lỗi."""
    annotated = frame.copy()

    # --- YOLO nhận diện quân cờ ---
    results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
    yolo_pieces = []
    pieces_list = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        r = int(min(x2 - x1, y2 - y1) / 2)
        cls_name = model.names[int(box.cls[0])]
        yolo_pieces.append((cx, cy, r, cls_name, int(x1), int(y1), int(x2), int(y2)))
        pieces_list.append((cx, cy, r))
        cv2.putText(annotated, cls_name, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    print(f"\n    Phát hiện {len(yolo_pieces)} quân cờ")
    for p in yolo_pieces:
        print(f"      - {p[3]} tại pixel ({p[0]}, {p[1]})")

    # --- Quét lưới bàn cờ bằng Canny -> Hough -> giao điểm -> Homography ---
    mapped_points, edge_img, debug_data = detect_board_points_with_hough(frame, detect_params)
    if mapped_points is None:
        print("    Không xác định được lưới từ Hough/Homography!")
        return annotated, None

    print(f"    Hough/Homography suy ra {len(mapped_points)} điểm lưới")

    pixel_A1 = next((pt['px'], pt['py']) for pt in mapped_points if pt['label'] == 'A1')
    pixel_A9 = next((pt['px'], pt['py']) for pt in mapped_points if pt['label'] == 'A9')
    pixel_J1 = next((pt['px'], pt['py']) for pt in mapped_points if pt['label'] == 'J1')

    # Tính tọa độ robot
    if pixel_A1 and pixel_A9 and pixel_J1:
        dx = abs(pixel_A9[0] - pixel_A1[0])
        dy = abs(pixel_J1[1] - pixel_A1[1])
        if dx > 0 and dy > 0:
            mapped_points.sort(key=lambda k: k['label'])
            for pt in mapped_points:
                pt['rob_x'] = round(((pt['px'] - pixel_A1[0]) * (REAL_WIDTH_A1_A9 / dx)) - OFFSET_X, 1)
                pt['rob_y'] = round(((pixel_A1[1] - pt['py']) * (REAL_HEIGHT_A1_J1 / dy)) - OFFSET_Y, 1)

                is_piece = any(math.hypot(pt['px'] - cx, pt['py'] - cy) < 15
                               for cx, cy, _ in pieces_list)
                cv2.circle(annotated, (pt['px'], pt['py']), 3, (0, 0, 255), -1)
                color = (0, 255, 0) if is_piece else (0, 255, 255)
                cv2.putText(annotated, pt['label'], (pt['px'] - 12, pt['py'] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                if pt['label'] == 'A1':
                    cv2.circle(annotated, (pt['px'], pt['py']), 8, (255, 0, 0), -1)

    edge_preview = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
    edge_preview = cv2.resize(edge_preview, (220, 220))
    annotated[8:228, 8:228] = edge_preview
    cv2.putText(annotated, "Canny", (12, 244), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    show_debug_overlay = bool((detect_params or {}).get("show_debug_overlay", 1))
    grid_line_thick = int((detect_params or {}).get("grid_line_thick", GRID_LINE_THICK_DEFAULT))
    grid_point_radius = int((detect_params or {}).get("grid_point_radius", GRID_POINT_RADIUS_DEFAULT))
    grid_line_thick = max(1, min(10, grid_line_thick))
    grid_point_radius = max(1, min(12, grid_point_radius))

    if show_debug_overlay and debug_data:
        for ln in debug_data.get("row_lines", []):
            draw_infinite_line(annotated, ln, (0, 220, 255), thickness=grid_line_thick)
        for ln in debug_data.get("col_lines", []):
            draw_infinite_line(annotated, ln, (255, 220, 0), thickness=grid_line_thick)

        for px, py in debug_data["projected_points"]:
            cv2.circle(annotated, (int(round(px)), int(round(py))), grid_point_radius, (255, 120, 0), -1)
        for px, py in debug_data["snapped_points"]:
            cv2.circle(annotated, (int(round(px)), int(round(py))), grid_point_radius, (0, 255, 0), -1)

        for x, y in debug_data["corners"]:
            cv2.circle(annotated, (int(round(x)), int(round(y))), 6, (255, 0, 255), 2)

        mode_txt = "H:ON" if debug_data.get("use_homography", 0) else "H:OFF"
        cv2.putText(annotated, f"Cam: du cam | Luc: snapped | {mode_txt}", (250, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    print(f"    Lưới: {len(mapped_points)} giao điểm")

    # --- FEN + Pikafish ---
    current_fen, grid_to_piece = generate_fen_and_mapping(mapped_points, yolo_pieces)
    print(f"\n     FEN: {current_fen}")

    print("    Pikafish đang tính...")
    best_move = get_best_move_from_pikafish(current_fen, THINK_TIME_MS)

    if best_move:
        start_label = ucci_to_grid_label(best_move[:2])
        end_label = ucci_to_grid_label(best_move[2:4])
        piece_name = grid_to_piece.get(start_label, "?")

        print(f"    Nước đi: {piece_name.upper()} {start_label} → {end_label} (UCI: {best_move})")

        start_xy = end_xy = None
        start_pixel = end_pixel = None
        for pt in mapped_points:
            if pt['label'] == start_label:
                start_xy = (pt.get('rob_x'), pt.get('rob_y'))
                start_pixel = (pt['px'], pt['py'])
            if pt['label'] == end_label:
                end_xy = (pt.get('rob_x'), pt.get('rob_y'))
                end_pixel = (pt['px'], pt['py'])

        if start_xy and end_xy and start_xy[0] is not None:
            print(f"    BỐC ({start_label}): X={start_xy[0]} mm, Y={start_xy[1]} mm")
            print(f"    ĐẶT ({end_label}): X={end_xy[0]} mm, Y={end_xy[1]} mm")

        if start_pixel and end_pixel:
            cv2.arrowedLine(annotated, start_pixel, end_pixel, (0, 0, 255), 3, tipLength=0.15)
            cv2.putText(annotated, piece_name, (start_pixel[0] - 10, start_pixel[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        print("    Hết cờ / chiếu bí!")

    # --- Xuất file ---
    csv_path = "grid_coordinates.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Label,Row,Col,Robot_X_mm,Robot_Y_mm,Pixel_X,Pixel_Y,Has_Piece,Piece_Name\n")
        for pt in mapped_points:
            pn = grid_to_piece.get(pt['label'], "")
            f.write(f"{pt['label']},{pt['label'][0]},{pt['label'][1:]},{pt.get('rob_x',0)},{pt.get('rob_y',0)},{pt['px']},{pt['py']},{1 if pn else 0},{pn}\n")

    robot_command = {
        "fen": current_fen,
        "robot_side": ROBOT_SIDE,
        "move_uci": best_move,
        "action": None
    }
    if best_move:
        start_pt = next((pt for pt in mapped_points if pt['label'] == ucci_to_grid_label(best_move[:2])), None)
        end_pt = next((pt for pt in mapped_points if pt['label'] == ucci_to_grid_label(best_move[2:4])), None)
        target_piece = grid_to_piece.get(ucci_to_grid_label(best_move[2:4]), "")
        robot_command["action"] = {
            "pick": {
                "label": start_pt['label'] if start_pt else "",
                "piece": grid_to_piece.get(start_pt['label'], "") if start_pt else "",
                "x_mm": start_pt.get('rob_x', 0) if start_pt else 0,
                "y_mm": start_pt.get('rob_y', 0) if start_pt else 0,
            },
            "place": {
                "label": end_pt['label'] if end_pt else "",
                "x_mm": end_pt.get('rob_x', 0) if end_pt else 0,
                "y_mm": end_pt.get('rob_y', 0) if end_pt else 0,
            },
            "capture": target_piece if target_piece else None,
        }

    with open("robot_command.json", "w", encoding="utf-8") as f:
        json.dump(robot_command, f, ensure_ascii=False, indent=2)

    print("    Đã xuất grid_coordinates.csv + robot_command.json")

    return annotated, {
        "fen": current_fen,
        "best_move": best_move,
        "grid_to_piece": grid_to_piece,
        "mapped_points": mapped_points,
    }


# =============================================================
# 4. KHỞI TẠO YOLO + CAMERA
# =============================================================
print(" Đang tải mô hình YOLO...")
model = YOLO(MODEL_PATH)

print(f" Đang mở Camera (index={CAMERA_INDEX})...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f" Không mở được Camera số {CAMERA_INDEX}!")
    engine.stdin.write("quit\n")
    engine.stdin.flush()
    os._exit(1)


def nothing(_):
    pass


def create_detection_control_panel():
    cv2.namedWindow("Grid Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grid Control", 450, 260)
    cv2.createTrackbar("Canny Low", "Grid Control", CANNY_LOW, 255, nothing)
    cv2.createTrackbar("Canny High", "Grid Control", CANNY_HIGH, 255, nothing)
    cv2.createTrackbar("Hough Thresh", "Grid Control", HOUGH_THRESHOLD, 200, nothing)
    cv2.createTrackbar("Min Line Len", "Grid Control", HOUGH_MIN_LINE_LEN, 400, nothing)
    cv2.createTrackbar("Max Line Gap", "Grid Control", HOUGH_MAX_LINE_GAP, 120, nothing)
    cv2.createTrackbar("Snap Radius", "Grid Control", SNAP_RADIUS, 30, nothing)
    cv2.createTrackbar("Use Homography", "Grid Control", USE_HOMOGRAPHY_DEFAULT, 1, nothing)
    cv2.createTrackbar("Grid Line Thick", "Grid Control", GRID_LINE_THICK_DEFAULT, 10, nothing)
    cv2.createTrackbar("Point Radius", "Grid Control", GRID_POINT_RADIUS_DEFAULT, 12, nothing)
    cv2.createTrackbar("Show Debug", "Grid Control", 1, 1, nothing)


def get_detection_params_from_ui():
    canny_low = cv2.getTrackbarPos("Canny Low", "Grid Control")
    canny_high = cv2.getTrackbarPos("Canny High", "Grid Control")
    return {
        "canny_low": canny_low,
        "canny_high": max(canny_low + 1, canny_high),
        "hough_threshold": max(10, cv2.getTrackbarPos("Hough Thresh", "Grid Control")),
        "hough_min_line_len": max(20, cv2.getTrackbarPos("Min Line Len", "Grid Control")),
        "hough_max_line_gap": max(1, cv2.getTrackbarPos("Max Line Gap", "Grid Control")),
        "snap_radius": max(2, cv2.getTrackbarPos("Snap Radius", "Grid Control")),
        "use_homography": cv2.getTrackbarPos("Use Homography", "Grid Control"),
        "grid_line_thick": max(1, cv2.getTrackbarPos("Grid Line Thick", "Grid Control")),
        "grid_point_radius": max(1, cv2.getTrackbarPos("Point Radius", "Grid Control")),
        "show_debug_overlay": cv2.getTrackbarPos("Show Debug", "Grid Control"),
    }


create_detection_control_panel()

print("\n" + "="*60)
print("  HỆ THỐNG ROBOT CỜ TƯỚNG TỰ ĐỘNG")
print("    SPACE = Chụp & Xử lý  |  Q = Thoát")
print("    Dùng cửa sổ Grid Control để tinh chỉnh Canny/Hough/Snap")
print("="*60 + "\n")

# =============================================================
# 5. VÒNG LẶP CHÍNH - CAMERA LIVE
# =============================================================
while True:
    ret, frame_raw = cap.read()
    if not ret:
        print(" Mất tín hiệu Camera!")
        break

    # Bước 1: Tự động cắt viền đen (Iriun camera letterbox)
    gray_raw = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
    _, mask_raw = cv2.threshold(gray_raw, 20, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask_raw)
    if coords is not None:
        rx, ry, rw, rh = cv2.boundingRect(coords)
        frame_raw = frame_raw[ry:ry+rh, rx:rx+rw]

    # Bước 2: Crop giữa thành vuông
    h, w = frame_raw.shape[:2]
    if w > h:
        x_start = (w - h) // 2
        frame_raw = frame_raw[:, x_start:x_start + h]
    elif h > w:
        y_start = (h - w) // 2
        frame_raw = frame_raw[y_start:y_start + w, :]
    frame = cv2.resize(frame_raw, (640, 640))

    cv2.imshow("CAMERA LIVE - Nhan SPACE de xu ly", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        print()
        print(" ĐÃ CHỤP! Đang xử lý...")

        detect_params = get_detection_params_from_ui()
        print(
            "    Param: "
            f"Canny=({detect_params['canny_low']},{detect_params['canny_high']}), "
            f"HoughT={detect_params['hough_threshold']}, "
            f"MinLen={detect_params['hough_min_line_len']}, "
            f"Gap={detect_params['hough_max_line_gap']}, "
            f"Snap={detect_params['snap_radius']}, "
            f"Homography={detect_params['use_homography']}, "
            f"LineThick={detect_params['grid_line_thick']}, "
            f"PointR={detect_params['grid_point_radius']}"
        )

        annotated, result = process_frame(frame, detect_params)

        if result:
            cv2.imshow("KET QUA NHAN DIEN", annotated)
            print()
        else:
            print(" Xử lý thất bại. Thử chụp lại!")
            print()

cap.release()
cv2.destroyAllWindows()
engine.stdin.write("quit\n")
engine.stdin.flush()
  