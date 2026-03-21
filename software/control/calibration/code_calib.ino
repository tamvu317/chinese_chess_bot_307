#include <Arduino.h>
#include <math.h>

// ====== Tham so tay may ======
float L1 = 290.0;
float L2 = 180.0;

// ====== Goc ban dau cua 2 khop ======
float initialTheta1 = 0.0;    // Khop 1 ban dau o 0°
float initialTheta2 = 180.0;  // Khop 2 ban dau o 180°
float initialX = 110.0;
float initialY = 0.0; // Sửa lại logic: 290 + 180*cos(180) = 110, y = 0

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

// ====== Ham quay motor ======
void rotateMotor(int stepPin, int dirPin, float angle, float stepsPerDegree) {
  if (abs(angle) < 0.01) return; // Không quay nếu góc quá nhỏ

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
  
  Serial.print("Quay: ");
  Serial.print(angle, 2);
  Serial.print(" deg (");
  Serial.print(steps);
  Serial.println(" steps)");
}

// ====== DONG HOC THUAN ======
void forwardKinematics(float theta1, float theta2, float &x, float &y) {
  float t1 = theta1 * PI / 180.0;
  float t2 = theta2 * PI / 180.0;
  float t2_absolute = t1 + t2;
  
  x = L1 * cos(t1) + L2 * cos(t2_absolute);
  y = L1 * sin(t1) + L2 * sin(t2_absolute);
}

// ====== DONG HOC NGUOC ======
bool inverseKinematics(float x, float y, float &theta1, float &theta2) {
  float d = sqrt(x*x + y*y);
  
  if (d < abs(L1 - L2) || d > (L1 + L2)) {
    Serial.println("Loi: Toa do nam ngoai vung lam viec!");
    return false;
  }
  
  float cos_theta2 = (x*x + y*y - L1*L1 - L2*L2) / (2.0 * L1 * L2);
  cos_theta2 = constrain(cos_theta2, -1.0, 1.0); // Dam bao gia tri trong khoang [-1, 1]
  
  float theta2_rad = acos(cos_theta2); // Dung khuyu tay (elbow up/down tuy vao dau)
  theta2 = theta2_rad * 180.0 / PI;
  
  float alpha = atan2(y, x);
  float beta = atan2(L2 * sin(theta2_rad), L1 + L2 * cos(theta2_rad));
  theta1 = (alpha - beta) * 180.0 / PI;
  
  return true;
}

void printStatus() {
  Serial.print("Trang thai: T1=");
  Serial.print(currentTheta1, 2);
  Serial.print(", T2(rel)=");
  Serial.print(currentTheta2, 2);
  float curX, curY;
  forwardKinematics(currentTheta1, currentTheta2, curX, curY);
  Serial.print(" -> X="); Serial.print(curX, 1);
  Serial.print(", Y="); Serial.println(curY, 1);
}

void setup() {
  Serial.begin(9600);
  pinMode(X_STEP_PIN, OUTPUT); pinMode(X_DIR_PIN, OUTPUT);
  pinMode(Y_STEP_PIN, OUTPUT); pinMode(Y_DIR_PIN, OUTPUT);
  pinMode(Z_STEP_PIN, OUTPUT); pinMode(Z_DIR_PIN, OUTPUT);
  pinMode(RELAY_PIN, OUTPUT);

  Serial.println("ROBOT ARM READY. Commands: 'home', 'angle T1 T2', 'coord X Y'");
  printStatus();
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.length() == 0) return;

    if (cmd == "home") {
      rotateMotor(X_STEP_PIN, X_DIR_PIN, initialTheta1 - currentTheta1, stepsPerDegree_Theta1);
      rotateMotor(Y_STEP_PIN, Y_DIR_PIN, initialTheta2 - currentTheta2, stepsPerDegree_Theta2);
      currentTheta1 = initialTheta1;
      currentTheta2 = initialTheta2;
      printStatus();
    } 
    else {
      int spacePos = cmd.indexOf(' ');
      if (spacePos == -1) return;

      String mode = cmd.substring(0, spacePos);
      String params = cmd.substring(spacePos + 1);
      int space2 = params.indexOf(' ');

      if (mode == "angle" && space2 != -1) {
        float t1 = params.substring(0, space2).toFloat();
        float t2 = params.substring(space2 + 1).toFloat();
        
        rotateMotor(X_STEP_PIN, X_DIR_PIN, t1 - currentTheta1, stepsPerDegree_Theta1);
        rotateMotor(Y_STEP_PIN, Y_DIR_PIN, t2 - currentTheta2, stepsPerDegree_Theta2);
        
        currentTheta1 = t1;
        currentTheta2 = t2;
        printStatus();
      } 
      else if (mode == "coord" && space2 != -1) {
        float targetX = params.substring(0, space2).toFloat();
        float targetY = params.substring(space2 + 1).toFloat();
        float t1, t2;

        if (inverseKinematics(targetX, targetY, t1, t2)) {
          rotateMotor(X_STEP_PIN, X_DIR_PIN, t1 - currentTheta1, stepsPerDegree_Theta1);
          rotateMotor(Y_STEP_PIN, Y_DIR_PIN, t2 - currentTheta2, stepsPerDegree_Theta2);
          currentTheta1 = t1;
          currentTheta2 = t2;
          printStatus();
        }
      } else {
        Serial.println("Lenh khong hop le!");
      }
    }
  }
}