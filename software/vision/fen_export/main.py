import subprocess
import sys
import os

# === CẤU HÌNH ===
# Đặt đường dẫn tới file pikafish.exe tại đây
PIKAFISH_PATH = os.path.join(os.path.dirname(__file__), "pikafish.exe")
THINK_TIME_MS = 2000  # Thời gian suy nghĩ (ms)


def parse_fen_board(fen):
    """Phân tích FEN thành mảng bàn cờ 10x9."""
    parts = fen.split()
    rows = parts[0].split("/")
    board = []
    for row in rows:
        board_row = []
        for ch in row:
            if ch.isdigit():
                board_row.extend(["."] * int(ch))
            else:
                board_row.append(ch)
        board.append(board_row)
    return board, parts


def board_to_fen(board, parts, swap_turn=True):
    """Chuyển mảng bàn cờ trở lại chuỗi FEN."""
    rows = []
    for board_row in board:
        fen_row = ""
        empty = 0
        for cell in board_row:
            if cell == ".":
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += cell
        if empty > 0:
            fen_row += str(empty)
        rows.append(fen_row)

    new_parts = list(parts)
    new_parts[0] = "/".join(rows)

    if swap_turn:
        new_parts[1] = "b" if parts[1] == "w" else "w"

    # Tăng số nước đi
    if len(new_parts) > 5:
        move_num = int(new_parts[5])
        if parts[1] == "b":
            new_parts[5] = str(move_num + 1)

    return " ".join(new_parts)


def apply_move(fen, move_uci):
    """Áp dụng nước đi UCI (vd: 'h2e2') lên FEN, trả về FEN mới."""
    board, parts = parse_fen_board(fen)

    src_file = ord(move_uci[0]) - ord("a")
    src_rank = int(move_uci[1])
    dst_file = ord(move_uci[2]) - ord("a")
    dst_rank = int(move_uci[3])

    src_row = 9 - src_rank
    src_col = src_file
    dst_row = 9 - dst_rank
    dst_col = dst_file

    piece = board[src_row][src_col]
    board[src_row][src_col] = "."
    board[dst_row][dst_col] = piece

    return board_to_fen(board, parts)


def is_valid_move(board, move_uci, turn):
    """Kiểm tra nước đi có hợp lệ theo luật cờ tướng không."""
    src_col = ord(move_uci[0]) - ord("a")
    src_rank = int(move_uci[1])
    dst_col = ord(move_uci[2]) - ord("a")
    dst_rank = int(move_uci[3])

    sr = 9 - src_rank
    sc = src_col
    dr = 9 - dst_rank
    dc = dst_col

    # Kiểm tra tọa độ trong bàn cờ
    if not (0 <= sr <= 9 and 0 <= sc <= 8 and 0 <= dr <= 9 and 0 <= dc <= 8):
        return False, "Tọa độ nằm ngoài bàn cờ!"

    piece = board[sr][sc]
    target = board[dr][dc]

    # Không được ăn quân cùng phe
    if target != ".":
        if turn == "w" and target.isupper():
            return False, f"Không thể ăn quân cùng phe '{target}' tại {move_uci[2:4].upper()}!"
        if turn == "b" and target.islower():
            return False, f"Không thể ăn quân cùng phe '{target}' tại {move_uci[2:4].upper()}!"

    p = piece.upper()
    row_diff = dr - sr
    col_diff = dc - sc

    PIECE_NAMES = {
        "R": "Xe", "N": "Mã", "B": "Tượng", "A": "Sĩ",
        "K": "Tướng", "C": "Pháo", "P": "Tốt/Chốt"
    }
    piece_name = PIECE_NAMES.get(p, p)

    # === XE (R) - đi thẳng ngang/dọc, không nhảy qua quân ===
    if p == "R":
        if sr != dr and sc != dc:
            return False, f"{piece_name} chỉ đi thẳng ngang hoặc dọc!"
        # Kiểm tra không có quân cản giữa đường
        if sr == dr:  # ngang
            step = 1 if dc > sc else -1
            for c in range(sc + step, dc, step):
                if board[sr][c] != ".":
                    return False, f"{piece_name} bị cản bởi quân tại đường đi!"
        else:  # dọc
            step = 1 if dr > sr else -1
            for r in range(sr + step, dr, step):
                if board[r][sc] != ".":
                    return False, f"{piece_name} bị cản bởi quân tại đường đi!"
        return True, ""

    # === MÃ (N) - đi chữ Nhật, kiểm tra cản chân ===
    if p == "N":
        valid_jumps = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                       (1, -2), (1, 2), (2, -1), (2, 1)]
        if (row_diff, col_diff) not in valid_jumps:
            return False, f"{piece_name} phải đi hình chữ Nhật (L)!"
        # Kiểm tra cản chân mã
        if abs(row_diff) == 2:
            block_r = sr + (1 if row_diff > 0 else -1)
            if board[block_r][sc] != ".":
                return False, f"{piece_name} bị cản chân (ô giữa có quân)!"
        else:
            block_c = sc + (1 if col_diff > 0 else -1)
            if board[sr][block_c] != ".":
                return False, f"{piece_name} bị cản chân (ô giữa có quân)!"
        return True, ""

    # === TƯỢNG (B) - đi chéo 2 ô, không qua sông, kiểm tra cản mắt ===
    if p == "B":
        if abs(row_diff) != 2 or abs(col_diff) != 2:
            return False, f"{piece_name} phải đi chéo đúng 2 ô!"
        # Kiểm tra cản mắt tượng
        mid_r = (sr + dr) // 2
        mid_c = (sc + dc) // 2
        if board[mid_r][mid_c] != ".":
            return False, f"{piece_name} bị cản mắt (ô chéo giữa có quân)!"
        # Không được qua sông
        if turn == "w" and dr < 5:
            return False, f"{piece_name} Đỏ không được qua sông (hàng 5-9)!"
        if turn == "b" and dr > 4:
            return False, f"{piece_name} Đen không được qua sông (hàng 0-4)!"
        return True, ""

    # === SĨ (A) - đi chéo 1 ô, chỉ trong cung ===
    if p == "A":
        if abs(row_diff) != 1 or abs(col_diff) != 1:
            return False, f"{piece_name} phải đi chéo đúng 1 ô!"
        # Phải ở trong cung (cột d-f = 3-5)
        if dc < 3 or dc > 5:
            return False, f"{piece_name} phải ở trong cung (cột d-f)!"
        if turn == "w" and (dr < 7 or dr > 9):
            return False, f"{piece_name} Đỏ phải ở trong cung (hàng 0-2)!"
        if turn == "b" and (dr < 0 or dr > 2):
            return False, f"{piece_name} Đen phải ở trong cung (hàng 7-9)!"
        return True, ""

    # === TƯỚNG (K) - đi 1 ô ngang/dọc, chỉ trong cung ===
    if p == "K":
        if abs(row_diff) + abs(col_diff) != 1:
            return False, f"{piece_name} chỉ đi 1 ô ngang hoặc dọc!"
        if dc < 3 or dc > 5:
            return False, f"{piece_name} phải ở trong cung (cột d-f)!"
        if turn == "w" and (dr < 7 or dr > 9):
            return False, f"{piece_name} Đỏ phải ở trong cung (hàng 0-2)!"
        if turn == "b" and (dr < 0 or dr > 2):
            return False, f"{piece_name} Đen phải ở trong cung (hàng 7-9)!"
        return True, ""

    # === PHÁO (C) - đi thẳng như Xe, ăn phải nhảy qua đúng 1 quân ===
    if p == "C":
        if sr != dr and sc != dc:
            return False, f"{piece_name} chỉ đi thẳng ngang hoặc dọc!"
        # Đếm quân giữa đường
        count = 0
        if sr == dr:  # ngang
            step = 1 if dc > sc else -1
            for c in range(sc + step, dc, step):
                if board[sr][c] != ".":
                    count += 1
        else:  # dọc
            step = 1 if dr > sr else -1
            for r in range(sr + step, dr, step):
                if board[r][sc] != ".":
                    count += 1
        if target == ".":
            if count != 0:
                return False, f"{piece_name} di chuyển (không ăn) không được có quân cản!"
        else:
            if count != 1:
                return False, f"{piece_name} ăn quân phải nhảy qua đúng 1 quân (hiện có {count})!"
        return True, ""

    # === TỐT/CHỐT (P) - tiến 1, qua sông được đi ngang ===
    if p == "P":
        if abs(row_diff) + abs(col_diff) != 1:
            return False, f"{piece_name} chỉ đi 1 ô!"
        if turn == "w":
            # Đỏ đi lên (row giảm)
            if row_diff > 0:
                return False, f"{piece_name} Đỏ không được đi lùi!"
            if sr > 4 and col_diff != 0:
                return False, f"{piece_name} Đỏ chưa qua sông, chỉ được tiến thẳng!"
        else:
            # Đen đi xuống (row tăng)
            if row_diff < 0:
                return False, f"{piece_name} Đen không được đi lùi!"
            if sr < 5 and col_diff != 0:
                return False, f"{piece_name} Đen chưa qua sông, chỉ được tiến thẳng!"
        return True, ""

    return True, ""


def get_best_move(fen, engine_path, think_time_ms):
    """Gửi FEN tới Pikafish và nhận nước đi tốt nhất qua UCI."""
    proc = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def send(cmd):
        proc.stdin.write(cmd + "\n")
        proc.stdin.flush()

    def read_until(keyword):
        lines = []
        while True:
            line = proc.stdout.readline().strip()
            lines.append(line)
            if keyword in line:
                break
        return lines

    send("uci")
    read_until("uciok")

    send("isready")
    read_until("readyok")

    send(f"position fen {fen}")
    send(f"go movetime {think_time_ms}")

    lines = read_until("bestmove")
    best_line = [l for l in lines if l.startswith("bestmove")][-1]
    best_move = best_line.split()[1]

    # Kiểm tra có chiếu bí / hòa không từ info score
    mate_info = None
    for line in lines:
        if "score mate" in line:
            mate_info = line

    send("quit")
    proc.wait(timeout=5)

    return best_move, mate_info


def main():
    if not os.path.isfile(PIKAFISH_PATH):
        print(f"[LỖI] Không tìm thấy Pikafish tại: {PIKAFISH_PATH}")
        print("Hãy tải Pikafish từ: https://github.com/official-pikafish/Pikafish/releases")
        print("Đặt file pikafish.exe vào cùng thư mục với main.py")
        sys.exit(1)

    print("=== PIKAFISH - Tính nước đi Cờ Tướng ===")
    print(f"Engine: {PIKAFISH_PATH}")
    print(f"Thời gian suy nghĩ: {THINK_TIME_MS}ms")
    print()

    # FEN mặc định: vị trí khởi đầu cờ tướng
    default_fen = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
    current_fen = default_fen

    print("Hướng dẫn:")
    print("  - Nhập 'fen <chuỗi FEN>' để đặt bàn cờ mới")
    print("  - Nhập nước đi của Đỏ (vd: H7E7) rồi Pikafish sẽ đáp lại")
    print("  - Nhập 'auto' để Pikafish tự đi cho cả Đỏ")
    print("  - Nhập 'show' để xem FEN hiện tại")
    print("  - Nhập 'reset' để quay về vị trí khởi đầu")
    print("  - Nhập 'q' để thoát")
    print()

    print(f"FEN hiện tại: {current_fen}")
    print()

    while True:
        turn = current_fen.split()[1]
        turn_name = "Đỏ (w)" if turn == "w" else "Đen (b)"
        user_input = input(f"[{turn_name}] Nhập nước đi hoặc lệnh:\n> ").strip()

        if not user_input:
            continue

        if user_input.lower() == "q":
            print("Tạm biệt!")
            break

        if user_input.lower() == "show":
            print(f"FEN hiện tại: {current_fen}")
            print()
            continue

        if user_input.lower() == "reset":
            current_fen = default_fen
            print(f"Đã reset! FEN: {current_fen}")
            print()
            continue

        if user_input.lower().startswith("fen "):
            current_fen = user_input[4:].strip()
            print(f"Đã đặt FEN: {current_fen}")
            print()
            continue

        if user_input.lower() == "auto":
            # Pikafish tự đi cho bên hiện tại
            try:
                print(f"\nPikafish đang suy nghĩ...")
                best_move, mate_info = get_best_move(current_fen, PIKAFISH_PATH, THINK_TIME_MS)
                if best_move == "(none)" or best_move == "0000":
                    loser = "Đỏ" if current_fen.split()[1] == "w" else "Đen"
                    print(f"\n=== HẾT CỜ! {loser} thua (bị chiếu bí / hết nước đi)! ===")
                    break
                print(f"Pikafish đi: {best_move}")
                current_fen = apply_move(current_fen, best_move)
                print(f"FEN mới    : {current_fen}")
            except FileNotFoundError:
                print("[LỖI] Không thể chạy Pikafish. Kiểm tra lại đường dẫn.")
            except Exception as e:
                print(f"[LỖI] {e}")
            print()
            continue

        # Xử lý nước đi của người chơi (vd: H7E7, h7e7)
        move = user_input.lower().replace(" ", "")
        if len(move) == 4 and move[0].isalpha() and move[1].isdigit() and move[2].isalpha() and move[3].isdigit():
            try:
                # Kiểm tra quân tại vị trí xuất phát có đúng phe không
                board, parts = parse_fen_board(current_fen)
                src_col = ord(move[0]) - ord("a")
                src_row = 9 - int(move[1])
                piece = board[src_row][src_col]
                turn = current_fen.split()[1]

                if piece == ".":
                    print(f"[LỖI] Ô {move[:2].upper()} không có quân nào!")
                    print()
                    continue

                if turn == "w" and piece.islower():
                    print(f"[LỖI] Ô {move[:2].upper()} là quân Đen '{piece}'. Lượt Đỏ phải đi quân Đỏ (chữ HOA)!")
                    print()
                    continue

                if turn == "b" and piece.isupper():
                    print(f"[LỖI] Ô {move[:2].upper()} là quân Đỏ '{piece}'. Lượt Đen phải đi quân Đen (chữ thường)!")
                    print()
                    continue

                # Kiểm tra nước đi có hợp lệ theo luật cờ tướng
                valid, reason = is_valid_move(board, move, turn)
                if not valid:
                    print(f"[SAI NƯỚC ĐI] {reason}")
                    print()
                    continue

                # Áp dụng nước đi của người chơi
                print(f"\nBạn đi: {move} (quân '{piece}')")
                current_fen = apply_move(current_fen, move)
                print(f"FEN sau nước đi: {current_fen}")

                # Pikafish đáp lại (Đen)
                print(f"Pikafish đang suy nghĩ...")
                best_move, mate_info = get_best_move(current_fen, PIKAFISH_PATH, THINK_TIME_MS)
                if best_move == "(none)" or best_move == "0000":
                    winner = "Đỏ" if current_fen.split()[1] == "b" else "Đen"
                    print(f"\n=== HẾT CỜ! {winner} thắng! Đối phương bị chiếu bí / hết nước đi! ===")
                    break
                print(f"Pikafish đáp: {best_move}")
                current_fen = apply_move(current_fen, best_move)
                print(f"FEN mới     : {current_fen}")
                if mate_info:
                    print(f"[Thông tin] {mate_info}")
            except FileNotFoundError:
                print("[LỖI] Không thể chạy Pikafish. Kiểm tra lại đường dẫn.")
            except Exception as e:
                print(f"[LỖI] {e}")
        else:
            print("[LỖI] Không hiểu lệnh. Nhập nước đi (vd: H7E7) hoặc lệnh (show/reset/auto/q)")

        print()


if __name__ == "__main__":
    main()
