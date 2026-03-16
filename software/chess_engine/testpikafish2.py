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
from ultralytics import YOLO

# =============================================================
# 1. CẤU HÌNH HỆ THỐNG
# =============================================================
MODEL_PATH = "best (2).pt"                
CONF_THRESHOLD = 0.7                      
CAMERA_INDEX = 1                          
THINK_TIME_MS = 2000                      

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


# =============================================================
# 3. KHỞI TẠO MẮT THẦN (YOLO & CAMERA)
# =============================================================
print(" Đang tải mô hình Mắt Thần (YOLO)...")
model = YOLO(MODEL_PATH)

print(" Đang kết nối Camera...")
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
cv2.createTrackbar("Max Thickness", "Control Panel", 28, 50, nothing)
cv2.createTrackbar("Shadow Buffer", "Control Panel", 10, 30, nothing)
cv2.createTrackbar("Show Text (0/1)", "Control Panel", 1, 1, nothing)

# --- THÊM 4 THANH TRƯỢT MỚI DÙNG ĐỂ CẮT LỀ (CROP) ---
cv2.createTrackbar("Crop Left", "Control Panel", 0, 300, nothing)
cv2.createTrackbar("Crop Right", "Control Panel", 0, 300, nothing)
cv2.createTrackbar("Crop Top", "Control Panel", 0, 300, nothing)
cv2.createTrackbar("Crop Bottom", "Control Panel", 0, 300, nothing)

ROW_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
REAL_WIDTH_A1_A9 = 230.0
REAL_HEIGHT_A1_J1 = 234.0
OFFSET_X = 115.0
OFFSET_Y = 300.0

print("\n" + "="*60)
print("  HỆ THỐNG ROBOT CỜ TƯỚNG TỰ ĐỘNG  ")
print(" - Kéo các thanh 'Crop' để gọt bỏ phần nền thừa bên ngoài.")
print(" - Nhấn phím SPACE để Máy chụp ảnh và phân tích nước đi.")
print("="*60 + "\n")

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
    frame_square = cv2.resize(frame_raw[y:y+size, x:x+size], (640, 640))

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

    cv2.imshow("LIVE CAMERA - Nhan SPACE de may tinh nuoc di", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    
    elif key == ord(' '):  
        print("\n" + ""*25)
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

        # BƯỚC 2: QUÉT LƯỚI TỌA ĐỘ
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        scale = max(5, 120 - length_val)
        h_size, v_size = int(frame.shape[1] / scale), int(frame.shape[0] / scale)

        mask_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1)))
        mask_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size)))

        for cnt in cv2.findContours(mask_h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if not (min_thick <= cv2.boundingRect(cnt)[3] <= max_thick): cv2.drawContours(mask_h, [cnt], -1, 0, -1)
        for cnt in cv2.findContours(mask_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if not (min_thick <= cv2.boundingRect(cnt)[2] <= max_thick): cv2.drawContours(mask_v, [cnt], -1, 0, -1)

        mask_joints = cv2.bitwise_and(cv2.dilate(mask_h, np.ones((3, 3)), iterations=5),
                          cv2.dilate(mask_v, np.ones((3, 3)), iterations=5))

        raw_points = []
        for cnt in cv2.findContours(mask_joints, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            if cv2.contourArea(cnt) > 0:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    is_shadow = any(math.sqrt((cx - px)**2 + (cy - py)**2) < (pr + shadow_buf) for px, py, pr in pieces_list)
                    if not is_shadow: raw_points.append((cx, cy))

        raw_points.extend([(px, py) for px, py, _ in pieces_list])

        final_points = []
        for cx, cy in raw_points:
            if sum(1 for tx, ty in raw_points if abs(cy - ty) <= 15) >= 5 and sum(1 for tx, ty in raw_points if abs(cx - tx) <= 15) >= 4:
                final_points.append((cx, cy))

        mapped_points = []
        pixel_A1 = pixel_A9 = pixel_J1 = None

        if len(final_points) > 10:
            xs, ys = [p[0] for p in final_points], [p[1] for p in final_points]

            valid_xs = [x for x in xs if sum(1 for tx in xs if abs(tx - x) <= 15) >= 5]
            valid_ys = [y for y in ys if sum(1 for ty in ys if abs(ty - y) <= 15) >= 5]

            if valid_xs and valid_ys:
                min_x, max_x = min(valid_xs), max(valid_xs)
                min_y, max_y = min(valid_ys), max(valid_ys)
                width, height = max_x - min_x, max_y - min_y

                if width > 0 and height > 0:
                    step_x, step_y = width / 8.0, height / 9.0
                    river_top = min_y + 4.2 * step_y
                    river_bottom = min_y + 4.8 * step_y

                    for cx, cy in final_points:
                        if cx < min_x - 15 or cx > max_x + 15 or cy < min_y - 15 or cy > max_y + 15:
                            continue
                        if river_top < cy < river_bottom:
                            continue

                        c_idx, r_idx = max(0, min(int(round((cx - min_x)/step_x)), 8)), max(0, min(int(round((cy - min_y)/step_y)), 9))
                        label_str = f"{ROW_LABELS[r_idx]}{c_idx + 1}"

                        if not any(pt['label'] == label_str for pt in mapped_points):
                            mapped_points.append({'label': label_str, 'px': cx, 'py': cy})
                            if label_str == 'A1': pixel_A1 = (cx, cy)
                            if label_str == 'A9': pixel_A9 = (cx, cy)
                            if label_str == 'J1': pixel_J1 = (cx, cy)

                    if pixel_A1 and pixel_A9 and pixel_J1:
                        dx, dy = abs(pixel_A9[0] - pixel_A1[0]), abs(pixel_J1[1] - pixel_A1[1])
                        if dx > 0 and dy > 0:
                            mapped_points.sort(key=lambda k: k['label'])
                            for pt in mapped_points:
                                pt['rob_x'] = round(((pt['px'] - pixel_A1[0]) * (REAL_WIDTH_A1_A9 / dx)) - OFFSET_X, 1)
                                pt['rob_y'] = round(((pixel_A1[1] - pt['py']) * (REAL_HEIGHT_A1_J1 / dy)) - OFFSET_Y, 1)

                                is_piece = any(math.sqrt((pt['px']-c_x)**2 + (pt['py']-c_y)**2) < 5 for c_x, c_y, _ in pieces_list)
                                cv2.circle(annotated_frame, (pt['px'], pt['py']), 3, (0, 0, 255), -1)
                                if show_text == 1:
                                    cv2.putText(annotated_frame, f"[{int(pt['rob_x'])},{int(pt['rob_y'])}]", (pt['px']-20, pt['py']+20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0) if is_piece else (0, 255, 255), 1)
                                if pt['label'] == 'A1': cv2.circle(annotated_frame, (pt['px'], pt['py']), 8, (255, 0, 0), -1)

        cv2.imshow("KẾT QUẢ QUÉT AI (SNAPSHOT)", annotated_frame)

        # BƯỚC 3: XỬ LÝ FEN VÀ GỌI PIKAFISH
        if not mapped_points:
            print(" Lỗi: Không bắt được lưới bàn cờ. Vui lòng kiểm tra ánh sáng!")
            continue

        current_fen, grid_to_piece = generate_fen_and_mapping(mapped_points, yolo_pieces)
        print(f" MÃ FEN HIỆN TẠI TỪ CAMERA:\n ---> {current_fen} <---")
        
        print(" PIKAFISH ĐANG TÍNH TOÁN NƯỚC ĐI CHO ĐEN...")
        
        # --- LẤY KẾT QUẢ VÀ TRẠNG THÁI TỪ PIKAFISH ---
        best_move, game_over_msg = get_best_move_from_pikafish(current_fen, THINK_TIME_MS)
        
        if best_move:
            start_ucci = best_move[:2]
            end_ucci = best_move[2:4]
            
            start_label = ucci_to_grid_label(start_ucci)
            end_label = ucci_to_grid_label(end_ucci)
            piece_name = grid_to_piece.get(start_label, "Một quân cờ")
            
            print(f" PIKAFISH QUYẾT ĐỊNH: Quân {piece_name.upper()} di chuyển từ {start_label} đến {end_label}")
            
            start_xy = end_xy = None
            for pt in mapped_points:
                if pt['label'] == start_label:
                    start_xy = (pt.get('rob_x'), pt.get('rob_y'))
                if pt['label'] == end_label:
                    end_xy = (pt.get('rob_x'), pt.get('rob_y'))
            
            if start_xy and end_xy and start_xy[0] is not None:
                print(f"\n TỌA ĐỘ ĐỂ ROBOT THỰC THI GẮP QUÂN ĐEN:")
                print(f"    BỐC quân tại ({start_label}): X= {start_xy[0]} mm, Y= {start_xy[1]} mm")
                print(f"    ĐẶT quân tại ({end_label}): X= {end_xy[0]} mm, Y= {end_xy[1]} mm")
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
