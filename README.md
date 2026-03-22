# CCR3 - Chinese Chess Robot 🤖♟️

> Robot chơi cờ tướng tự động sử dụng Computer Vision, AI Engine và cánh tay SCARA – Dự án của nhóm Lab307.

## Tổng quan

**Chinese Chess Robot (CCR3)** là hệ thống robot có khả năng:
- 📷 Nhận diện bàn cờ tướng và các quân cờ qua camera (YOLO)
- 🧠 Tính toán nước đi tối ưu bằng engine Pikafish
- 🦾 Điều khiển cánh tay robot SCARA để gắp và di chuyển quân cờ
- 🎮 Chơi đối kháng với người chơi trong thời gian thực

## Kiến trúc hệ thống

```
Camera  ──►  Vision Module  ──►  FEN Export  ──►  Chess Engine (Pikafish)
                                                         │
                                                         ▼
Player  ◄──  Robot Arm (SCARA)  ◄──  Motion Planning  ◄──  Best Move
```

## Cấu trúc thư mục

```
CCR3-ChineseChessRobot/
├── docs/              # Tài liệu dự án
├── hardware/          # Thiết kế phần cứng (CAD, PCB, BOM)
├── firmware/          # Code Arduino/MCU (PlatformIO)
├── software/          # Code chính (Vision, Control, Engine, UI)
├── simulation/        # Mô phỏng (URDF, MATLAB, Gazebo)
├── datasets/          # Dataset huấn luyện
├── tests/             # Unit tests & Integration tests
└── scripts/           # Scripts tiện ích (setup, run)
```

## Yêu cầu hệ thống

### Phần cứng
- Raspberry Pi 4B (hoặc PC)
- Arduino Mega 2560
- Camera (USB hoặc Pi Camera)
- Cánh tay SCARA (custom)
- Stepper motors + drivers

### Phần mềm
- Python 3.9+
- OpenCV
- YOLOv8 (Ultralytics)
- Pikafish chess engine
- PlatformIO (firmware)

## Cài đặt nhanh

```bash
# Clone repository
git clone https://github.com/AvQ1301/chinese_chees_bot_307.git
cd chinese_chees_bot_307

# Chạy script cài đặt
chmod +x scripts/setup.sh
./scripts/setup.sh

# Chạy hệ thống
./scripts/run.sh
```

## Quản lý gói với `uv`

[`uv`](https://github.com/astral-sh/uv) là công cụ quản lý gói Python cực nhanh (thay thế cho `pip` và `venv`), được khuyến nghị cho dự án này.

### Cài đặt `uv`

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Hoặc qua `pip`:**
```bash
pip install uv
```

Sau khi cài đặt, khởi động lại terminal và kiểm tra:
```bash
uv --version
```

### Sử dụng `uv` cho dự án

**1. Tạo môi trường ảo:**
```bash
uv venv
```
> Tạo thư mục `.venv/` trong thư mục hiện tại.

**2. Kích hoạt môi trường ảo:**
```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**3. Cài đặt dependencies:**
```bash
uv pip install -r requirements.txt
```

**4. Thêm gói mới:**
```bash
uv pip install <tên-gói>
```

**5. Xuất danh sách dependencies hiện tại:**
```bash
uv pip freeze > requirements.txt
```

---

## 🪟 Cài đặt WSL (dành cho Windows)

> [!IMPORTANT]
> Tất cả thành viên dùng Windows **bắt buộc** cài WSL 2 để đảm bảo môi trường phát triển đồng nhất với Linux.

### Bước 1 – Bật WSL

Mở **PowerShell** (Run as Administrator) và chạy:

```powershell
wsl --install
```

Lệnh này sẽ tự động:
- Bật tính năng Virtual Machine Platform & WSL
- Cài đặt **Ubuntu** (distro mặc định)
- Khởi động lại máy nếu cần

> Nếu muốn chọn distro khác: `wsl --install -d <DistroName>` (ví dụ: `Ubuntu-22.04`)

### Bước 2 – Kiểm tra phiên bản WSL

```powershell
wsl --version
wsl --list --verbose
```

Đảm bảo distro đang chạy ở **VERSION 2**.

### Bước 3 – Vào môi trường WSL

Mở **Windows Terminal** hoặc tìm kiếm **Ubuntu** trong Start Menu. Lần đầu sẽ yêu cầu tạo username/password Linux.

### Bước 4 – Clone và cài đặt dự án trong WSL

Sau khi vào terminal Ubuntu, thực hiện các bước như trên Linux:

```bash
# Cài git nếu chưa có
sudo apt update && sudo apt install -y git curl

# Cài uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone repo
git clone https://github.com/AvQ1301/chinese_chees_bot_307.git
cd chinese_chees_bot_307

# Tạo môi trường ảo và cài dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

> [!TIP]
> Nên lưu code trong filesystem của WSL (`~/...`) thay vì `/mnt/c/...` để tránh vấn đề hiệu năng I/O.

---

## 🤖 Training trên Google Colab

Dành cho các thành viên muốn huấn luyện model YOLO mà không có GPU mạnh.

### Chuẩn bị

1. Truy cập [Google Colab](https://colab.research.google.com/) và đăng nhập bằng tài khoản Google.
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Mount Google Drive để lưu model và dataset:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Clone repo và cài đặt

```python
# Clone repository
!git clone https://github.com/AvQ1301/chinese_chees_bot_307.git
%cd chinese_chees_bot_307

# Cài dependencies
!pip install -r requirements.txt
```

### Huấn luyện YOLOv8

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")  # hoặc yolov8s.pt, yolov8m.pt

# Train
model.train(
    data="datasets/data.yaml",   # file cấu hình dataset
    epochs=100,
    imgsz=640,
    project="/content/drive/MyDrive/CCR3/runs",  # lưu vào Drive
    name="chess_piece_detection"
)
```

### Lưu và tải model về

```python
# Sau khi train xong, model best.pt sẽ nằm tại:
# /content/drive/MyDrive/CCR3/runs/chess_piece_detection/weights/best.pt

# Tải về máy cục bộ
from google.colab import files
files.download('/content/drive/MyDrive/CCR3/runs/chess_piece_detection/weights/best.pt')
```

> [!TIP]
> Dùng **Colab Pro** hoặc lên lịch train để tránh bị ngắt kết nối khi train lâu.

---

## 🖥️ Training trên Remote Server (Lab Case PC)

Lab có một máy tính case với GPU để train model. Kết nối qua **SSH**.

### Thông tin kết nối

| Thông số | Giá trị |
|----------|---------|
| Host | `<IP_LAB_SERVER>` *(hỏi trưởng nhóm)* |
| Port | `22` (mặc định) |
| Username | `<username>` *(được cấp)* |

> [!CAUTION]
> Không chia sẻ thông tin đăng nhập server lên GitHub hoặc chat nhóm.

### Kết nối SSH

```bash
# Kết nối lần đầu
ssh <username>@<IP_LAB_SERVER>

# Tùy chọn: cấu hình SSH key để không cần nhập mật khẩu
ssh-keygen -t ed25519 -C "ccr3-lab307"
ssh-copy-id <username>@<IP_LAB_SERVER>
```

### Cài đặt môi trường trên server (lần đầu)

```bash
# Clone repo
git clone https://github.com/AvQ1301/chinese_chees_bot_307.git
cd chinese_chees_bot_307

# Cài uv và tạo môi trường
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Chạy training trong background (không bị ngắt khi đóng SSH)

Dùng `tmux` để giữ session train dù mất kết nối:

```bash
# Tạo session mới
tmux new -s training

# Chạy lệnh train bên trong session
python software/vision/train.py

# Detach khỏi session (giữ nguyên tiến trình): Ctrl+B, sau đó nhấn D

# Lần sau SSH vào, attach lại session để xem log
tmux attach -t training
```

### Chuyển file sau khi train xong

```bash
# Từ máy cá nhân – tải model về local
scp <username>@<IP_LAB_SERVER>:~/chinese_chees_bot_307/runs/weights/best.pt ./software/vision/models/
```

---

## Thành viên nhóm

| Tên | Vai trò |
|-----|---------|
| TBD | Vision / CV |
| TBD | Control / Kinematics |
| TBD | Hardware / Mechanical |
| TBD | Firmware / Electronics |

## License

Dự án này được phân phối theo giấy phép [MIT](LICENSE).

## Liên hệ

- **Lab307** – Hanoi University of Science and Technology (HUST)
- GitHub: [AvQ1301](https://github.com/AvQ1301)
CCR3 - Chinese Chess Robot project by Lab307 team. A robotic system that plays Chinese Chess (Xiangqi) using computer vision, AI engine, and SCARA robot arm.
