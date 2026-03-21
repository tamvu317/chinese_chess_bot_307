"""
CCR3 - Chinese Chess Robot
Module: Software - Vision - pikafish engine
Nhận diện bàn cờ và quân cờ bằng YOLO, chuyển đổi sang mã FEN, gọi Pikafish tính nước đi, rồi dịch ngược lại thành tọa độ XY cho robot thực thi.
Input: Ảnh chụp bàn cờ từ camera
Output: Nước đi tốt nhất của Pikafish và tọa độ XY để robot gắp quân
"""
import sys
sys.stdout.reconfigure(encoding='utf-8') # Tránh lỗi hiển thị Emoji trên Windows

import cv2
import numpy as np
import math
import subprocess
import os
import json
import time
import importlib
from ultralytics import YOLO

# =============================================================
# 1. CẤU HÌNH HỆ THỐNG
# =============================================================
MODEL_PATH = "best (2).pt"                
CONF_THRESHOLD = 0.7                      
CAMERA_INDEX = 1                          
THINK_TIME_MS = 2000                      
ROBOT_COMMAND_PATH = "robot_command.json"
REFERENCE_GRID_JSON_PATH = "board_reference_points.json"
REFERENCE_GRID_CSV_PATH = "board_reference_points.csv"
LIVE_GRID_CSV_PATH = "grid_coordinates.csv"

# Bật/tắt gửi trực tiếp lệnh sang Arduino qua Serial.
ENABLE_ARDUINO_SERIAL = True
ARDUINO_PORT = "COM6"
ARDUINO_BAUDRATE = 9600
ARDUINO_RESPONSE_TIMEOUT_SEC = 3.0

# Tham số tinh chỉnh lưới/neighbor để dễ chỉnh ở một nơi
PIECE_TO_GRID_MAX_DIST_PX = 30
GRID_NEIGHBOR_TOLERANCE_PX = 15
GRID_NEIGHBOR_MIN_SAME_ROW = 5
GRID_NEIGHBOR_MIN_SAME_COL = 4
GRID_AXIS_MIN_NEIGHBORS = 5
GRID_BOUNDARY_TOLERANCE_PX = 10
PIECE_CENTER_OVERRIDE_MAX_DIST_PX = 24
PIECE_OCCLUSION_BBOX_MARGIN_PX = 4

# Hệ số lọc bóng quanh quân cờ: giảm ăn nhầm giao điểm lân cận
SHADOW_RADIUS_SCALE = 0.55
SHADOW_RADIUS_BUFFER_WEIGHT = 0.5
SHADOW_RADIUS_MIN_PX = 6

# Tọa độ khay bỏ quân bị ăn (chỉnh theo vị trí thực tế của robot)
CAPTURE_BIN_POINT = {
    "label": "CAPTURE_BIN",
    "x_mm": -180.0,
    "y_mm": -320.0,
}

# Mặc định Robot cầm quân Đen (Pikafish sẽ tính nước đi cho phe Đen)
ROBOT_SIDE = 'b' # 'w' cho quân trắng, 'b' cho quân đen 

# TỪ ĐIỂN DỊCH TÊN YOLO SANG KÝ HIỆU FEN QUỐC TẾ
FEN_MAP = {
    "tuongdo": "K", "sido": "A", "tinhdo": "B", "mado": "N", "xedo": "R", "phaodo": "C", "totdo": "P",
    "tuongden": "k", "siden": "a", "tinhden": "b", "maden": "n", "xeden": "r", "phaoden": "c", "totden": "p"
}

# =============================================================
# 2. KHỞI TẠO PIKAFISH ENGINE VÀ BẮT MẠCH THẮNG THUA
# =============================================================
print(" Đang khởi động não bộ Pikafish...")
try:
    engine = subprocess.Popen(
        'pikafish.exe', 
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    print(" Đã kết nối thành công với Pikafish!\n")
except FileNotFoundError:
    print(" LỖI: Không tìm thấy file pikafish.exe.")
    os._exit(1)

def get_best_move_from_pikafish(fen_string, think_time_ms):
    engine.stdin.write(f"position fen {fen_string}\n")
    engine.stdin.write(f"go movetime {think_time_ms}\n")
    engine.stdin.flush()
    
    game_over_msg = None
    
    while True:
        line = engine.stdout.readline().strip()
        
        # --- BỘ LỌC TÍN HIỆU THẮNG/THUA ---
        parts = line.split()
        if "score" in parts and "mate" in parts:
            mate_idx = parts.index("mate")
            if mate_idx + 1 < len(parts):
                mate_val = parts[mate_idx + 1]
                if mate_val == "1":
                    game_over_msg = " CHIẾU TƯỚNG HẾT CỜ! NƯỚC ĐI NÀY SẼ KẾT LIỄU BẠN (MÁY THẮNG)!"
                elif mate_val == "-1" or mate_val == "0":
                    game_over_msg = " MÁY TÍNH ĐANG BỊ CHIẾU BÍ VÀ SẮP THUA!"

        if line.startswith("bestmove"):
            best_move = parts[1] if len(parts) > 1 else ""
            
            # Nếu máy trả về none tức là nó không còn nước đi hợp lệ nào
            if best_move == "(none)" or best_move == "0000":
                return None, " CHÚC MỪNG! BẠN ĐÃ CHIẾU BÍ VÀ GIÀNH CHIẾN THẮNG TRƯỚC AI!"
            
            return best_move, game_over_msg

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

        if best_row != -1 and best_col != -1 and best_dist < PIECE_TO_GRID_MAX_DIST_PX:
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


def build_point_payload(label, xy, piece_name=None):
    if not xy or xy[0] is None or xy[1] is None:
        return None

    point_data = {
        "label": label,
        "x_mm": xy[0],
        "y_mm": xy[1],
    }
    if piece_name:
        point_data["piece"] = piece_name
    return point_data


def write_robot_command(command_data):
    temp_path = f"{ROBOT_COMMAND_PATH}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(command_data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, ROBOT_COMMAND_PATH)


def _read_serial_line(ser, timeout_sec):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if line:
            return line
    return None


def build_arduino_command_text(command_data):
    mode = command_data.get("mode", "")
    steps = command_data.get("steps", [])

    if mode == "mode_1_move":
        if not steps:
            return None, "Khong co step cho mode_1_move"

        move_step = steps[0]
        from_pt = move_step.get("from", {})
        to_pt = move_step.get("to", {})
        cmd = f"move {from_pt.get('x_mm', 0):.1f} {from_pt.get('y_mm', 0):.1f} {to_pt.get('x_mm', 0):.1f} {to_pt.get('y_mm', 0):.1f}"
        return cmd, None

    if mode == "mode_2_capture":
        if len(steps) < 2:
            return None, "Khong du step cho mode_2_capture"

        enemy_pt = steps[0].get("from", {})
        my_pt = steps[1].get("from", {})
        cmd = f"capture {enemy_pt.get('x_mm', 0):.1f} {enemy_pt.get('y_mm', 0):.1f} {my_pt.get('x_mm', 0):.1f} {my_pt.get('y_mm', 0):.1f}"
        return cmd, None

    return None, f"Mode khong ho tro: {mode}"


def send_command_to_arduino(command_data):
    if not ENABLE_ARDUINO_SERIAL:
        return False, "Serial disabled"

    try:
        serial_module = importlib.import_module("serial")
    except ModuleNotFoundError:
        return False, "Chua cai pyserial. Cai bang: pip install pyserial"

    command_text, err_msg = build_arduino_command_text(command_data)
    if err_msg:
        return False, err_msg

    try:
        with serial_module.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE, timeout=0.2, write_timeout=1) as ser:
            time.sleep(2.0)

            serial_line = f"{command_text}\\n"
            ser.write(serial_line.encode("ascii"))
            response = _read_serial_line(ser, ARDUINO_RESPONSE_TIMEOUT_SEC)

            if response:
                return True, f"Da gui lenh '{command_text}' | Phan hoi: {response}"
            return True, f"Da gui lenh '{command_text}'"
    except Exception as exc:
        return False, f"Loi serial: {exc}"


def save_grid_csv(csv_path, mapped_points, grid_to_piece=None):
    grid_to_piece = grid_to_piece or {}
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Label,Row,Col,Robot_X_mm,Robot_Y_mm,Pixel_X,Pixel_Y,Has_Piece,Piece_Name,Point_Source\n")
        for pt in sorted(mapped_points, key=lambda p: p['label']):
            label = pt['label']
            piece_name = grid_to_piece.get(label, "")
            point_source = pt.get("point_source", "detected")
            f.write(
                f"{label},{label[0]},{label[1:]},{pt.get('rob_x', 0)},{pt.get('rob_y', 0)},{pt['px']},{pt['py']},{1 if piece_name else 0},{piece_name},{point_source}\n"
            )


def save_reference_grid(mapped_points):
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "points": [
            {
                "label": pt["label"],
                "px": int(pt["px"]),
                "py": int(pt["py"]),
                "rob_x": float(pt.get("rob_x", 0.0)),
                "rob_y": float(pt.get("rob_y", 0.0)),
                "point_source": "reference",
            }
            for pt in sorted(mapped_points, key=lambda p: p["label"])
        ],
    }

    temp_path = f"{REFERENCE_GRID_JSON_PATH}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, REFERENCE_GRID_JSON_PATH)

    save_grid_csv(REFERENCE_GRID_CSV_PATH, payload["points"], grid_to_piece={})


def load_reference_grid():
    if not os.path.exists(REFERENCE_GRID_JSON_PATH):
        return None

    try:
        with open(REFERENCE_GRID_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = data.get("points", [])
        if len(points) < 90:
            return None

        cleaned = []
        for pt in points:
            cleaned.append(
                {
                    "label": str(pt["label"]),
                    "px": int(pt["px"]),
                    "py": int(pt["py"]),
                    "rob_x": float(pt.get("rob_x", 0.0)),
                    "rob_y": float(pt.get("rob_y", 0.0)),
                    "point_source": "reference",
                }
            )
        return sorted(cleaned, key=lambda p: p["label"])
    except Exception as exc:
        print(f" CẢNH BÁO: Không đọc được mốc bàn cờ chuẩn: {exc}")
        return None


def detect_grid_points(frame, pieces_list, length_val, thresh_val, min_thick, max_thick, shadow_buf):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    scale = max(5, 120 - length_val)
    h_size = int(frame.shape[1] / scale)
    v_size = int(frame.shape[0] / scale)

    mask_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1)))
    mask_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size)))

    for cnt in cv2.findContours(mask_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if not (min_thick <= cv2.boundingRect(cnt)[3] <= max_thick):
            cv2.drawContours(mask_h, [cnt], -1, 0, -1)
    for cnt in cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if not (min_thick <= cv2.boundingRect(cnt)[2] <= max_thick):
            cv2.drawContours(mask_v, [cnt], -1, 0, -1)

    mask_joints = cv2.bitwise_and(
        cv2.dilate(mask_h, np.ones((3, 3)), iterations=5),
        cv2.dilate(mask_v, np.ones((3, 3)), iterations=5),
    )

    raw_points = []
    for cnt in cv2.findContours(mask_joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(cnt) <= 0:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue

        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

        is_shadow = any(
            math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            < max(SHADOW_RADIUS_MIN_PX, pr * SHADOW_RADIUS_SCALE + shadow_buf * SHADOW_RADIUS_BUFFER_WEIGHT)
            for px, py, pr in pieces_list
        )
        if not is_shadow:
            raw_points.append((cx, cy))

    raw_points.extend([(px, py) for px, py, _ in pieces_list])

    final_points = []
    for cx, cy in raw_points:
        same_row = sum(1 for _, ty in raw_points if abs(cy - ty) <= GRID_NEIGHBOR_TOLERANCE_PX)
        same_col = sum(1 for tx, _ in raw_points if abs(cx - tx) <= GRID_NEIGHBOR_TOLERANCE_PX)
        if same_row >= GRID_NEIGHBOR_MIN_SAME_ROW and same_col >= GRID_NEIGHBOR_MIN_SAME_COL:
            final_points.append((cx, cy))

    if len(final_points) <= 10:
        return []

    xs = [p[0] for p in final_points]
    ys = [p[1] for p in final_points]
    valid_xs = [x for x in xs if sum(1 for tx in xs if abs(tx - x) <= GRID_NEIGHBOR_TOLERANCE_PX) >= GRID_AXIS_MIN_NEIGHBORS]
    valid_ys = [y for y in ys if sum(1 for ty in ys if abs(ty - y) <= GRID_NEIGHBOR_TOLERANCE_PX) >= GRID_AXIS_MIN_NEIGHBORS]

    if not valid_xs or not valid_ys:
        return []

    min_x, max_x = min(valid_xs), max(valid_xs)
    min_y, max_y = min(valid_ys), max(valid_ys)
    width, height = max_x - min_x, max_y - min_y
    if width <= 0 or height <= 0:
        return []

    step_x, step_y = width / 8.0, height / 9.0
    river_top = min_y + 4.2 * step_y
    river_bottom = min_y + 4.8 * step_y

    mapped_points = []
    pixel_a1 = pixel_a9 = pixel_j1 = None

    for cx, cy in final_points:
        if (
            cx < min_x - GRID_BOUNDARY_TOLERANCE_PX
            or cx > max_x + GRID_BOUNDARY_TOLERANCE_PX
            or cy < min_y - GRID_BOUNDARY_TOLERANCE_PX
            or cy > max_y + GRID_BOUNDARY_TOLERANCE_PX
        ):
            continue
        if river_top < cy < river_bottom:
            continue

        c_idx = max(0, min(int(round((cx - min_x) / step_x)), 8))
        r_idx = max(0, min(int(round((cy - min_y) / step_y)), 9))
        label_str = f"{ROW_LABELS[r_idx]}{c_idx + 1}"

        if any(pt["label"] == label_str for pt in mapped_points):
            continue

        mapped_points.append({"label": label_str, "px": cx, "py": cy, "point_source": "detected"})
        if label_str == "A1":
            pixel_a1 = (cx, cy)
        elif label_str == "A9":
            pixel_a9 = (cx, cy)
        elif label_str == "J1":
            pixel_j1 = (cx, cy)

    if not (pixel_a1 and pixel_a9 and pixel_j1):
        return mapped_points

    dx = abs(pixel_a9[0] - pixel_a1[0])
    dy = abs(pixel_j1[1] - pixel_a1[1])
    if dx <= 0 or dy <= 0:
        return mapped_points

    mapped_points.sort(key=lambda k: k["label"])
    for pt in mapped_points:
        pt["rob_x"] = -round(((pt["px"] - pixel_a1[0]) * (REAL_WIDTH_A1_A9 / dx)) - OFFSET_X, 1)
        pt["rob_y"] = round(((pixel_a1[1] - pt["py"]) * (REAL_HEIGHT_A1_J1 / dy)) - OFFSET_Y, 1)

    return mapped_points


def _find_covering_piece_center(ref_pt, yolo_pieces):
    rx, ry = ref_pt["px"], ref_pt["py"]
    best = None
    best_dist = float("inf")

    for piece in yolo_pieces:
        cx, cy, r, _cls_name, x1, y1, x2, y2 = piece
        margin = max(PIECE_OCCLUSION_BBOX_MARGIN_PX, int(r * 0.15))
        if x1 - margin <= rx <= x2 + margin and y1 - margin <= ry <= y2 + margin:
            dist = math.sqrt((cx - rx) ** 2 + (cy - ry) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = (int(cx), int(cy))

    return best


def merge_with_reference(current_points, reference_points, pieces_list, yolo_pieces):
    if not reference_points:
        return current_points

    current_map = {pt["label"]: pt for pt in current_points}
    merged_points = []

    for ref_pt in sorted(reference_points, key=lambda p: p["label"]):
        label = ref_pt["label"]
        # Nếu mốc chuẩn nằm trong vùng bbox quân cờ, coi là bị che và dùng tâm quân.
        covering_piece = _find_covering_piece_center(ref_pt, yolo_pieces)
        if covering_piece:
            merged_points.append(
                {
                    "label": label,
                    "px": int(covering_piece[0]),
                    "py": int(covering_piece[1]),
                    "rob_x": float(ref_pt.get("rob_x", 0.0)),
                    "rob_y": float(ref_pt.get("rob_y", 0.0)),
                    "point_source": "piece_center",
                }
            )
            continue

        # Theo mode mới: luôn ưu tiên mốc chuẩn cho mọi điểm nhìn thấy.
        # Chỉ khi giao điểm bị che (không còn trong current_map) mới thay tạm bằng tâm quân.
        if label in current_map:
            merged_points.append(
                {
                    "label": label,
                    "px": int(ref_pt["px"]),
                    "py": int(ref_pt["py"]),
                    "rob_x": float(ref_pt.get("rob_x", 0.0)),
                    "rob_y": float(ref_pt.get("rob_y", 0.0)),
                    "point_source": "reference",
                }
            )
            continue

        nearest_piece = None
        best_dist = float("inf")
        for px, py, _ in pieces_list:
            dist = math.sqrt((px - ref_pt["px"]) ** 2 + (py - ref_pt["py"]) ** 2)
            if dist < best_dist:
                best_dist = dist
                nearest_piece = (px, py)

        if nearest_piece and best_dist <= max(PIECE_CENTER_OVERRIDE_MAX_DIST_PX, 36):
            merged_points.append(
                {
                    "label": label,
                    "px": int(nearest_piece[0]),
                    "py": int(nearest_piece[1]),
                    "rob_x": float(ref_pt.get("rob_x", 0.0)),
                    "rob_y": float(ref_pt.get("rob_y", 0.0)),
                    "point_source": "piece_center",
                }
            )
        else:
            merged_points.append(
                {
                    "label": label,
                    "px": int(ref_pt["px"]),
                    "py": int(ref_pt["py"]),
                    "rob_x": float(ref_pt.get("rob_x", 0.0)),
                    "rob_y": float(ref_pt.get("rob_y", 0.0)),
                    "point_source": "reference",
                }
            )

    return merged_points


# =============================================================
# 3. KHỞI TẠO MẮT THẦN (YOLO & CAMERA)
# =============================================================
print(" Đang tải mô hình Mắt Thần (YOLO)...")
model = YOLO(MODEL_PATH)

print("🎥 Đang kết nối Camera...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f" KHÔNG MỞ ĐƯỢC CAMERA SỐ {CAMERA_INDEX}!")
    os._exit(1)

# =============================================================
# 4. CẤU HÌNH THANH TRƯỢT OPENCV
# =============================================================
def nothing(x): pass
cv2.namedWindow("Control Panel")
cv2.resizeWindow("Control Panel", 450, 450) # Tăng chiều cao cửa sổ để chứa thêm nút
cv2.createTrackbar("Line Length", "Control Panel", 100, 100, nothing)
cv2.createTrackbar("Threshold", "Control Panel", 135, 255, nothing)
cv2.createTrackbar("Min Thickness", "Control Panel", 1, 20, nothing)
cv2.createTrackbar("Max Thickness", "Control Panel", 20, 50, nothing)
cv2.createTrackbar("Shadow Buffer", "Control Panel", 8, 30, nothing)
cv2.createTrackbar("Show Text (0/1)", "Control Panel", 1, 1, nothing)

# --- THÊM 4 THANH TRƯỢT MỚI DÙNG ĐỂ CẮT LỀ (CROP) ---
cv2.createTrackbar("Crop Left", "Control Panel", 86, 300, nothing)
cv2.createTrackbar("Crop Right", "Control Panel", 206, 300, nothing)
cv2.createTrackbar("Crop Top", "Control Panel", 61, 300, nothing)
cv2.createTrackbar("Crop Bottom", "Control Panel", 221, 300, nothing)

ROW_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
REAL_WIDTH_A1_A9 = 238.0
REAL_HEIGHT_A1_J1 = 240.0
OFFSET_X = 119
OFFSET_Y = 200

print("\n" + "="*60)
print("  HỆ THỐNG ROBOT CỜ TƯỚNG TỰ ĐỘNG  ")
print(" - Kéo các thanh 'Crop' để gọt bỏ phần nền thừa bên ngoài.")
print(" - Nhấn phím C khi bàn trống để lưu mốc 90 điểm chuẩn.")
print(" - Nhấn phím SPACE để Máy chụp ảnh và phân tích nước đi.")
print("="*60 + "\n")

# Chống lặp nước đi: sau khi robot đi, chỉ tính tiếp khi phát hiện đối thủ đã đi.
awaiting_opponent_move = False
fen_before_robot_move = None
fen_after_robot_move = None
reference_points = load_reference_grid()
if reference_points:
    print(f" Đã nạp mốc chuẩn: {len(reference_points)} điểm từ {REFERENCE_GRID_JSON_PATH}")
else:
    print(" Chưa có mốc chuẩn. Hãy bấm phím C khi bàn trống để lưu tọa độ chuẩn.")

# =============================================================
# 5. VÒNG LẶP CHÍNH 
# =============================================================
while True:
    ret, frame_raw = cap.read()
    if not ret: break

    # Ép khung hình thành hình vuông 640x640 (Dữ liệu gốc)
    h, w = frame_raw.shape[:2]
    size = min(h, w)
    x = (w - size) // 2
    y = (h - size) // 2
    frame_square = cv2.resize(frame_raw[y:y+size, x:x+size], (480, 480))

    # Đọc thông số cắt lề từ thanh trượt
    c_left = cv2.getTrackbarPos("Crop Left", "Control Panel")
    c_right = cv2.getTrackbarPos("Crop Right", "Control Panel")
    c_top = cv2.getTrackbarPos("Crop Top", "Control Panel")
    c_bottom = cv2.getTrackbarPos("Crop Bottom", "Control Panel")

    # Giới hạn an toàn (Tránh người dùng kéo quá tay làm văng phần mềm)
    if c_left + c_right >= 630: c_right = 630 - c_left
    if c_top + c_bottom >= 630: c_bottom = 630 - c_top

    # Áp dụng thuật toán Cắt Lề và Phóng to (Zoom) lại thành 640x640
    frame_cropped = frame_square[c_top:640-c_bottom, c_left:640-c_right]
    frame = cv2.resize(frame_cropped, (640, 640))

    cv2.imshow("LIVE CAMERA - Nhan C de luu ban trong, SPACE de tinh nuoc", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    
    elif key in (ord(' '), ord('c')):
        is_calibration_capture = key == ord('c')
        print("\n" + ""*25)
        if is_calibration_capture:
            print(" ĐÃ CHỤP BÀN TRỐNG! ĐANG LƯU MỐC CHUẨN...")
        else:
            print(" ĐÃ CHỤP ẢNH! ĐANG XỬ LÝ LỌC NHIỄU...")
        
        length_val = max(1, cv2.getTrackbarPos("Line Length", "Control Panel"))
        thresh_val = cv2.getTrackbarPos("Threshold", "Control Panel")
        min_thick = cv2.getTrackbarPos("Min Thickness", "Control Panel")
        max_thick = max(min_thick + 1, cv2.getTrackbarPos("Max Thickness", "Control Panel"))
        shadow_buf = cv2.getTrackbarPos("Shadow Buffer", "Control Panel")
        show_text = cv2.getTrackbarPos("Show Text (0/1)", "Control Panel")

        annotated_frame = frame.copy()

        # BƯỚC 1: AI YOLO
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
            
            if show_text == 1:
                cv2.putText(annotated_frame, cls_name, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # BƯỚC 2: QUÉT LƯỚI TỌA ĐỘ + GHÉP MỐC CHUẨN
        current_points = detect_grid_points(frame, pieces_list, length_val, thresh_val, min_thick, max_thick, shadow_buf)
        mapped_points = merge_with_reference(current_points, reference_points, pieces_list, yolo_pieces)

        for pt in mapped_points:
            source = pt.get("point_source", "detected")
            color = (0, 255, 255)
            if source == "piece_center":
                color = (0, 165, 255)
            elif source == "reference":
                color = (255, 200, 0)

            cv2.circle(annotated_frame, (pt['px'], pt['py']), 3, color, -1)
            if show_text == 1 and 'rob_x' in pt and 'rob_y' in pt:
                cv2.putText(
                    annotated_frame,
                    f"{pt['label']}[{int(pt['rob_x'])},{int(pt['rob_y'])}]",
                    (pt['px'] - 35, pt['py'] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                )
            if pt['label'] == 'A1':
                cv2.circle(annotated_frame, (pt['px'], pt['py']), 8, (255, 0, 0), -1)

        if is_calibration_capture:
            if len(current_points) < 90:
                print(f" Lỗi: chỉ phát hiện {len(current_points)} điểm. Hãy để bàn trống rõ hơn rồi bấm C lại.")
                cv2.imshow("KẾT QUẢ QUÉT AI (SNAPSHOT)", annotated_frame)
                continue

            save_reference_grid(current_points)
            reference_points = load_reference_grid()
            print(f" Đã lưu mốc chuẩn vào {REFERENCE_GRID_JSON_PATH} và {REFERENCE_GRID_CSV_PATH}.")
            cv2.imshow("KẾT QUẢ QUÉT AI (SNAPSHOT)", annotated_frame)
            continue

        cv2.imshow("KẾT QUẢ QUÉT AI (SNAPSHOT)", annotated_frame)

        # BƯỚC 3: XỬ LÝ FEN VÀ GỌI PIKAFISH
        if not mapped_points:
            print(" Lỗi: Không bắt được lưới bàn cờ. Vui lòng kiểm tra ánh sáng!")
            continue

        current_fen, grid_to_piece = generate_fen_and_mapping(mapped_points, yolo_pieces)
        save_grid_csv(LIVE_GRID_CSV_PATH, mapped_points, grid_to_piece)
        print(f" MÃ FEN HIỆN TẠI TỪ CAMERA:\n ---> {current_fen} <---")

        if awaiting_opponent_move:
            if fen_after_robot_move is None:
                if current_fen == fen_before_robot_move:
                    print(" Chưa thấy bàn cờ đổi sau lệnh robot. Bỏ qua lần chụp này.")
                    continue

                fen_after_robot_move = current_fen
                print(" Đã ghi nhận trạng thái sau khi robot đi. Đang chờ đối thủ đi.")
                continue

            if current_fen == fen_after_robot_move:
                print(" Chưa thấy đối thủ đi. Bỏ qua lần chụp này.")
                continue

            print(" Đã phát hiện bàn cờ đổi do đối thủ. Tiếp tục tính nước đi cho robot.")
            awaiting_opponent_move = False
            fen_before_robot_move = None
            fen_after_robot_move = None
        
        print(" PIKAFISH ĐANG TÍNH TOÁN NƯỚC ĐI CHO ĐEN...")
        
        # --- LẤY KẾT QUẢ VÀ TRẠNG THÁI TỪ PIKAFISH ---
        best_move, game_over_msg = get_best_move_from_pikafish(current_fen, THINK_TIME_MS)
        
        if best_move:
            start_ucci = best_move[:2]
            end_ucci = best_move[2:4]
            
            start_label = ucci_to_grid_label(start_ucci)
            end_label = ucci_to_grid_label(end_ucci)
            piece_name = grid_to_piece.get(start_label, "Một quân cờ")
            captured_piece_name = grid_to_piece.get(end_label)
            
            print(f" PIKAFISH QUYẾT ĐỊNH: Quân {piece_name.upper()} di chuyển từ {start_label} đến {end_label}")
            
            start_xy = end_xy = None
            for pt in mapped_points:
                if pt['label'] == start_label:
                    start_xy = (pt.get('rob_x'), pt.get('rob_y'))
                if pt['label'] == end_label:
                    end_xy = (pt.get('rob_x'), pt.get('rob_y'))
            
            if start_xy and end_xy and start_xy[0] is not None:
                start_point = build_point_payload(start_label, start_xy, piece_name)
                end_point = build_point_payload(end_label, end_xy, captured_piece_name)
                command_id = f"cmd_{int(time.time() * 1000)}"

                robot_command = {
                    "command_id": command_id,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "fen": current_fen,
                    "robot_side": ROBOT_SIDE,
                    "move_uci": best_move,
                    "mode": "mode_1_move",
                    "steps": [
                        {
                            "step": "move_piece",
                            "from": start_point,
                            "to": end_point,
                        }
                    ],
                }

                if captured_piece_name:
                    robot_command["mode"] = "mode_2_capture"
                    robot_command["steps"] = [
                        {
                            "step": "remove_captured_piece",
                            "from": build_point_payload(end_label, end_xy, captured_piece_name),
                            "to": CAPTURE_BIN_POINT,
                        },
                        {
                            "step": "move_piece",
                            "from": build_point_payload(start_label, start_xy, piece_name),
                            "to": build_point_payload(end_label, end_xy),
                        },
                    ]

                    print(f"\n CHẾ ĐỘ 2 (ĂN QUÂN):")
                    print(f"    BƯỚC 1 - Gắp quân bị ăn tại ({end_label}): X= {end_xy[0]} mm, Y= {end_xy[1]} mm")
                    print(f"             Đưa về khay ăn quân: X= {CAPTURE_BIN_POINT['x_mm']} mm, Y= {CAPTURE_BIN_POINT['y_mm']} mm")
                    print(f"    BƯỚC 2 - Bốc quân mình tại ({start_label}): X= {start_xy[0]} mm, Y= {start_xy[1]} mm")
                    print(f"             Đặt vào ({end_label}): X= {end_xy[0]} mm, Y= {end_xy[1]} mm")
                else:
                    print(f"\n CHẾ ĐỘ 1 (DI CHUYỂN THƯỜNG):")
                    print(f"    BỐC quân tại ({start_label}): X= {start_xy[0]} mm, Y= {start_xy[1]} mm")
                    print(f"    ĐẶT quân tại ({end_label}): X= {end_xy[0]} mm, Y= {end_xy[1]} mm")

                write_robot_command(robot_command)
                print(f" Đã ghi lệnh robot vào file: {ROBOT_COMMAND_PATH}")

                sent_ok = True
                if ENABLE_ARDUINO_SERIAL:
                    sent_ok, sent_msg = send_command_to_arduino(robot_command)
                    if sent_ok:
                        print(f" Đã gửi lệnh sang Arduino: {sent_msg}")
                    else:
                        print(f" CẢNH BÁO SERIAL: {sent_msg}")

                if sent_ok:
                    awaiting_opponent_move = True
                    fen_before_robot_move = current_fen
                    fen_after_robot_move = None
                    print(" Đã khóa lượt robot. Chờ đối thủ đi rồi mới tính tiếp.")
                else:
                    print(" Chưa khóa lượt robot vì gửi lệnh thất bại. Bạn có thể chụp lại để thử lại.")
            else:
                print(" CẢNH BÁO: Không lấy được tọa độ XY do khuất lưới (A1, A9, J1)!")
                
            # IN THÔNG BÁO CHIẾU BÍ CỦA MÁY (NẾU CÓ)
            if game_over_msg:
                print("\n" + ""*20)
                print(f" {game_over_msg}")
                print(""*20)
                
        else:
            # TRƯỜNG HỢP BẠN ĐÃ CHIẾU BÍ MÁY (MÁY TRẢ VỀ NONE)
            print("\n" + ""*20)
            print(f" {game_over_msg}")
            print(""*20)
        
        print(""*25 + "\n")

cap.release()
cv2.destroyAllWindows()