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
            dist = math.sqrt((cx - pt['px'])**2 + (cy - pt['py'])**2)
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


# =============================================================
# 3. HÀM XỬ LÝ FRAME: YOLO + GRID + PIKAFISH
# =============================================================
def process_frame(frame):
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

    # --- Quét lưới bàn cờ ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary_global = cv2.threshold(gray, THRESH_VAL, 255, cv2.THRESH_BINARY_INV)
    binary_adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 5)
    binary = cv2.bitwise_or(binary_global, binary_adapt)

    scale = max(5, 120 - LENGTH_VAL)
    h_size = int(frame.shape[1] / scale)
    v_size = int(frame.shape[0] / scale)

    mask_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1)))
    mask_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size)))

    for cnt in cv2.findContours(mask_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.boundingRect(cnt)[3] > MAX_THICK * 2:
            cv2.drawContours(mask_h, [cnt], -1, 0, -1)
    for cnt in cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.boundingRect(cnt)[2] > MAX_THICK * 2:
            cv2.drawContours(mask_v, [cnt], -1, 0, -1)

    mask_joints = cv2.bitwise_and(
        cv2.dilate(mask_h, np.ones((3, 3)), iterations=4),
        cv2.dilate(mask_v, np.ones((3, 3)), iterations=4)
    )

    detected_points = []
    for cnt in cv2.findContours(mask_joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) > 0:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                detected_points.append((cx, cy))

    for px, py, _ in pieces_list:
        detected_points.append((px, py))

    unique_points = []
    for cx, cy in detected_points:
        if not any(abs(cx - ux) < 8 and abs(cy - uy) < 8 for ux, uy in unique_points):
            unique_points.append((cx, cy))

    print(f"    {len(unique_points)} điểm thô phát hiện")

    # --- Grid fitting ---
    mapped_points = []
    pixel_A1 = pixel_A9 = pixel_J1 = None

    if len(unique_points) < 15:
        print("    Không đủ điểm để khớp lưới!")
        return annotated, None

    col_centers = cluster_1d([p[0] for p in unique_points], 9)
    row_centers = cluster_1d([p[1] for p in unique_points], 10)

    if not col_centers or not row_centers:
        print("    Không phân cụm được lưới!")
        return annotated, None

    avg_step_x = (col_centers[-1] - col_centers[0]) / 8.0
    avg_step_y = (row_centers[-1] - row_centers[0]) / 9.0
    snap_threshold = min(avg_step_x, avg_step_y) * 0.35

    for r_idx in range(10):
        for c_idx in range(9):
            grid_x = col_centers[c_idx]
            grid_y = row_centers[r_idx]

            best_dist = float('inf')
            best_x, best_y = grid_x, grid_y
            for px, py in unique_points:
                dist = math.sqrt((grid_x - px)**2 + (grid_y - py)**2)
                if dist < best_dist and dist < snap_threshold:
                    best_dist = dist
                    best_x, best_y = px, py

            label_str = f"{ROW_LABELS[r_idx]}{c_idx + 1}"
            mapped_points.append({'label': label_str, 'px': int(best_x), 'py': int(best_y)})
            if label_str == 'A1': pixel_A1 = (int(best_x), int(best_y))
            if label_str == 'A9': pixel_A9 = (int(best_x), int(best_y))
            if label_str == 'J1': pixel_J1 = (int(best_x), int(best_y))

    # Tính tọa độ robot
    if pixel_A1 and pixel_A9 and pixel_J1:
        dx = abs(pixel_A9[0] - pixel_A1[0])
        dy = abs(pixel_J1[1] - pixel_A1[1])
        if dx > 0 and dy > 0:
            mapped_points.sort(key=lambda k: k['label'])
            for pt in mapped_points:
                pt['rob_x'] = round(((pt['px'] - pixel_A1[0]) * (REAL_WIDTH_A1_A9 / dx)) - OFFSET_X, 1)
                pt['rob_y'] = round(((pixel_A1[1] - pt['py']) * (REAL_HEIGHT_A1_J1 / dy)) - OFFSET_Y, 1)

                is_piece = any(math.sqrt((pt['px'] - cx)**2 + (pt['py'] - cy)**2) < 15
                               for cx, cy, _ in pieces_list)
                cv2.circle(annotated, (pt['px'], pt['py']), 3, (0, 0, 255), -1)
                color = (0, 255, 0) if is_piece else (0, 255, 255)
                cv2.putText(annotated, pt['label'], (pt['px'] - 12, pt['py'] + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                if pt['label'] == 'A1':
                    cv2.circle(annotated, (pt['px'], pt['py']), 8, (255, 0, 0), -1)

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

print("\n" + "="*60)
print("  HỆ THỐNG ROBOT CỜ TƯỚNG TỰ ĐỘNG")
print("    SPACE = Chụp & Xử lý  |  Q = Thoát")
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
        print("\n" + ""*25)
        print(" ĐÃ CHỤP! Đang xử lý...")

        annotated, result = process_frame(frame)

        if result:
            cv2.imshow("KET QUA NHAN DIEN", annotated)
            print(""*25)
        else:
            print(" Xử lý thất bại. Thử chụp lại!")
            print(""*25)

cap.release()
cv2.destroyAllWindows()
engine.stdin.write("quit\n")
engine.stdin.flush()
  