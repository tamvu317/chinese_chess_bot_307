#define STEP_PIN 3
#define DIR_PIN  6
#define EN_PIN   8

const float STEP_PER_DEG = 3200.0 / 360.0;

void setup() {
  Serial.begin(9600);

  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);

  digitalWrite(EN_PIN, LOW);

  Serial.println("Nhap goc (vd: 10 hoac -10):");
}

void loop() {
  if (Serial.available()) {

    float angle = Serial.parseFloat();
    if (angle == 0) return;

    bool dir = (angle > 0);

    long steps = abs(angle) * STEP_PER_DEG;

    digitalWrite(DIR_PIN, dir);

    delayMicroseconds(50);   // rất quan trọng

    for (long i = 0; i < steps; i++) {
      digitalWrite(STEP_PIN, HIGH);
      delayMicroseconds(1000);
      digitalWrite(STEP_PIN, LOW);
      delayMicroseconds(1000);
    }

    Serial.print("Da quay: ");
    Serial.print(angle);
    Serial.println(" do");
  }
}