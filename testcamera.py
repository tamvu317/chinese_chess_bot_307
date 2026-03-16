import cv2

cap = cv2.VideoCapture(0)  # Thử index 0, nếu không được thì đổi sang 1, 2...

if not cap.isOpened():
    print("Không mở được camera!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    size = min(h, w)
    x = (w - size) // 2
    y = (h - size) // 2
    frame = frame[y:y+size, x:x+size]
    frame = cv2.resize(frame, (640, 640))
    cv2.imshow("Iriun Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()