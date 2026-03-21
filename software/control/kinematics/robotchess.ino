#include <Arduino.h>
#include <math.h>

// ====== Tham so tay may ======
float L1 = 290.0;
float L2 = 180.0;

// ĐÃ CẬP NHẬT THEO HỆ TRỤC MỚI (X_mới = Y_cũ, Y_mới = -X_cũ)
float graveX = -200; // (cũ: 300)
float graveY = -300; // (cũ: 200)

// ====== Goc ban dau ======
float initialTheta1 = 58.6;
float initialTheta2 = 179.5;

// ĐÃ CẬP NHẬT THEO HỆ TRỤC MỚI
float initialX = 94.7; // (cũ: 56)
float initialY = -56; // (cũ: 94.7)

// ====== CNC Shield V3 ======
#define X_STEP_PIN 4
#define X_DIR_PIN 7
#define Y_STEP_PIN 2
#define Y_DIR_PIN 5
#define Z_STEP_PIN 3
#define Z_DIR_PIN 6
#define RELAY_PIN 12

// ====== Step motor ======
const float stepsPerDegree_Z = 8.89;
const float stepsPerDegree_Theta1 = 11.11;
const float stepsPerDegree_Theta2 = 8.89;

float currentTheta1 = initialTheta1;
float currentTheta2 = initialTheta2;

// ====== Quay motor ======
void rotateMotor(int stepPin, int dirPin, float angle, float stepsPerDegree) {
  int steps = round(abs(angle) * stepsPerDegree);
  bool direction = (angle > 0) ? HIGH : LOW;

  digitalWrite(dirPin, direction);
  delay(10);

  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(1000);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(1000);
  }
}

// ====== ĐỘNG HỌC NGƯỢC ======
bool inverseKinematics(float inputX, float inputY, float &theta1, float &theta2) {
  // --- THÊM PHÉP XOAY TRỤC NGƯỢC KIM ĐỒNG HỒ 90 ĐỘ TẠI ĐÂY ---
  float x = -inputY;
  float y = inputX;
  // -----------------------------------------------------------

  float d = sqrt(x*x + y*y);

  if (d < abs(L1 - L2) || d > (L1 + L2)) {
    Serial.println("Loi: Toa do nam ngoai vung hoat dong!");
    return false;
  }

  float cos_theta2 = (d*d - L1*L1 - L2*L2) / (2.0 * L1 * L2);
  // Giới hạn giá trị cos để tránh lỗi tập xác định (NaN) của sqrt
  cos_theta2 = constrain(cos_theta2, -1.0, 1.0);
  float sin_theta2 = sqrt(1.0 - cos_theta2 * cos_theta2);

  theta2 = atan2(sin_theta2, cos_theta2) * 180.0 / PI;

  float alpha = atan2(y, x) * 180.0 / PI;
  float beta = atan2(L2 * sin(theta2 * PI / 180.0),
                     L1 + L2 * cos(theta2 * PI / 180.0)) * 180.0 / PI;

  theta1 = alpha - beta;

  return true;
}

void pick() {
  rotateMotor(Z_STEP_PIN, Z_DIR_PIN, 160, stepsPerDegree_Z);
  delay(1000);
  rotateMotor(Z_STEP_PIN, Z_DIR_PIN, -160, stepsPerDegree_Z);
}

void place() {
  rotateMotor(Z_STEP_PIN, Z_DIR_PIN, 160, stepsPerDegree_Z);
  delay(500);
  digitalWrite(RELAY_PIN, HIGH);
  delay(1000);
  rotateMotor(Z_STEP_PIN, Z_DIR_PIN, -160, stepsPerDegree_Z);
  digitalWrite(RELAY_PIN, LOW);
}

// ====== Di chuyen toi diem ======
void moveToXY(float x, float y) {
  float theta1, theta2;

  if (!inverseKinematics(x, y, theta1, theta2)) return;

  float deltaTheta1 = theta1 - currentTheta1;
  float deltaTheta2 = theta2 - currentTheta2;

  // Điểm bình thường: Chân 4,7 (X) quay trước, 2,5 (Y) quay sau
  rotateMotor(X_STEP_PIN, X_DIR_PIN, deltaTheta1, stepsPerDegree_Theta1);
  delay(500);

  rotateMotor(Y_STEP_PIN, Y_DIR_PIN, deltaTheta2, stepsPerDegree_Theta2);
  delay(500);

  currentTheta1 = theta1;
  currentTheta2 = theta2;
}

// ====== Di chuyen ve HOME ======
void goHome() {
  float theta1, theta2;

  if (!inverseKinematics(initialX, initialY, theta1, theta2)) return;

  float deltaTheta1 = theta1 - currentTheta1;
  float deltaTheta2 = theta2 - currentTheta2;

  // ĐÃ ĐỔI THỨ TỰ: Chân 2,5 (Y) quay trước, 4,7 (X) quay sau
  rotateMotor(Y_STEP_PIN, Y_DIR_PIN, deltaTheta2, stepsPerDegree_Theta2);
  delay(500);

  rotateMotor(X_STEP_PIN, X_DIR_PIN, deltaTheta1, stepsPerDegree_Theta1);
  delay(500);

  currentTheta1 = theta1;
  currentTheta2 = theta2;
}

void movePiece(float x1, float y1, float x2, float y2) {
  Serial.println("=== MOVE ===");

  moveToXY(x1, y1);
  pick();

  moveToXY(x2, y2);
  place();

  // Gọi hàm goHome thay vì moveToXY
  goHome();
}

void capturePiece(float enemyX, float enemyY, float myX, float myY) {
  Serial.println("=== CAPTURE ===");

  // 1. Gắp quân địch
  moveToXY(enemyX, enemyY);
  pick();

  // 2. Mang ra ngoài
  moveToXY(graveX, graveY);
  place();

  // 3. Lấy quân mình
  moveToXY(myX, myY);
  pick();

  // 4. Đặt vào vị trí vừa ăn (CHÍNH LÀ enemyX, enemyY)
  moveToXY(enemyX, enemyY);
  place();

  // 5. Về home bằng hàm goHome
  goHome();
}


// ====== Setup ======
void setup() {
  Serial.begin(9600);

  pinMode(X_STEP_PIN, OUTPUT);
  pinMode(X_DIR_PIN, OUTPUT);
  pinMode(Y_STEP_PIN, OUTPUT);
  pinMode(Y_DIR_PIN, OUTPUT);
  pinMode(Z_STEP_PIN, OUTPUT);
  pinMode(Z_DIR_PIN, OUTPUT);
  pinMode(RELAY_PIN, OUTPUT);

  Serial.println("Da nap xong code! Nhap: coord x y");
}

// ====== Loop ======
void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    String cmdLower = cmd;
    cmdLower.toLowerCase();

    if (cmdLower.startsWith("move")) {
      int s1 = cmdLower.indexOf(' ');
      int s2 = cmdLower.indexOf(' ', s1 + 1);
      int s3 = cmdLower.indexOf(' ', s2 + 1);
      int s4 = cmdLower.indexOf(' ', s3 + 1);

      if (s1 != -1 && s2 != -1 && s3 != -1 && s4 != -1) {
        float x1 = cmd.substring(s1 + 1, s2).toFloat();
        float y1 = cmd.substring(s2 + 1, s3).toFloat();
        float x2 = cmd.substring(s3 + 1, s4).toFloat();
        float y2 = cmd.substring(s4 + 1).toFloat();

        movePiece(x1, y1, x2, y2);
      } else {
        Serial.println("Loi: Lenh move can 4 toa do (VD: move 200 0 300 0)");
      }
    } 
    else if (cmdLower.startsWith("capture")) {
      int s1 = cmdLower.indexOf(' ');
      int s2 = cmdLower.indexOf(' ', s1 + 1);
      int s3 = cmdLower.indexOf(' ', s2 + 1);
      int s4 = cmdLower.indexOf(' ', s3 + 1);

      if (s1 != -1 && s2 != -1 && s3 != -1 && s4 != -1) {
        float ex = cmd.substring(s1 + 1, s2).toFloat();
        float ey = cmd.substring(s2 + 1, s3).toFloat();
        float mx = cmd.substring(s3 + 1, s4).toFloat();
        float my = cmd.substring(s4 + 1).toFloat();

        capturePiece(ex, ey, mx, my);
      } else {
        Serial.println("Loi: Lenh capture can 4 toa do!");
      }
    }
    else {
      Serial.println("Lenh khong hop le!");
    }
  }
}