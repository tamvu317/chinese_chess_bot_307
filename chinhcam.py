import cv2

# 1. Đọc ảnh nguồn (ảnh có góc nhìn phối cảnh và nền gỗ)
# Thay 'image_1.png' bằng đường dẫn thực tế đến tệp ảnh của bạn
img = cv2.VideoCapture(1)

if img is None:
    print("Không thể mở tệp ảnh. Vui lòng kiểm tra lại đường dẫn.")
else:
    # 2. Định nghĩa khu vực cắt (crop)
    # Vì ảnh gốc đã ở góc nhìn chim bay (trên cao xuống), ta chỉ cần xác định
    # 4 tọa độ (xmin, ymin, xmax, ymax) bao quanh bàn cờ.
    # Tọa độ này được ước lượng từ ảnh ví dụ của bạn. Bạn CẦN TÍNH TOÁN LẠI
    # để khớp chính xác với ảnh thực tế.
    ymin, ymax = 50, 950   # Tọa độ y (chiều cao), cắt từ trên xuống dưới
    xmin, xmax = 210, 790  # Tọa độ x (chiều rộng), cắt từ trái sang phải

    # 3. Thực hiện cắt ảnh
    cropped_img = img[ymin:ymax, xmin:xmax]

    # 4. Hiển thị và lưu kết quả
    cv2.imshow('Ảnh Cắt (Chỉ Bàn Cờ, Không Co Giãn)', cropped_img)
    # Lệnh sau sẽ lưu ảnh kết quả thành file mới
    # cv2.imwrite('ban_co_da_cat.png', cropped_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()