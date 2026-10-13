#include <Arduino.h>
#include <SoftwareSerial.h>
#include <Wire.h>
#include <QMC5883LCompass.h>
#include "CytronMotorDriver.h"

// Bluetooth
String inString;
SoftwareSerial bt(18, 19);
int maxRows = 3; // Default size
int maxCols = 3; // Default size
// Motors
// RIGHT Motor (motor wiring)
CytronMD motor1(PWM_PWM, 8, 9);
// LEFT Motor (motor wiring)
CytronMD motor2(PWM_PWM, 10, 11);

// Compass
QMC5883LCompass compass;
int referenceHeading = 0;
int currentTargetHeading = 0;

// Autonomous Variables
bool autonomousMode = false;
int gridRow = 0;
int gridCol = 0;

// State Machine
enum AutoState { AUTO_IDLE, AUTO_CAPTURE, AUTO_FORWARD, AUTO_NEXT_ROW };
AutoState autoState = AUTO_IDLE;

// Tuning Angles
int angleR1 = 95; // Right turn to face sideways
int angleR2 = 190; // Right turn to face down
int angleL1 = 100; // Left turn to face sideways
int angleL2 = 10; // Left turn to face up

void setup() {
  Wire.setSDA(16);
  Wire.setSCL(17);
  Wire.begin();

  Serial.begin(9600);
  bt.begin(9600);

  // COMPASS INITIALIZATION
  compass.init();
  compass.setCalibrationOffsets(-211.00, -92.00, 196.00);
  compass.setCalibrationScales(0.74, 0.72, 3.62);
  compass.setSmoothing(10, true);
  delay(1500);

  // SET STARTING DIRECTION AS REFERENCE (0°)
  Serial.println("Setting reference heading...");
  int sum = 0;
  for (int i = 0; i < 10; i++) {
    compass.read();
    int h = compass.getAzimuth();
    if (h < 0) h += 360;
    sum += h;
    delay(150);
  }
  referenceHeading = sum / 10;
  currentTargetHeading = 0;

  Serial.println("=======================================");
  Serial.println("   MOTION 2350 PRO - ROBOT READY");
  Serial.println("=======================================");
  bt.println("ROBOT_READY");
}

void loop() {
  // Listen for Bluetooth Commands
  if (bt.available() > 0) {
    inString = bt.readStringUntil('\r');
    inString.trim();

    if (inString == "b0" || inString == "bAUTO_STOP") {
      stopAutonomousMode();
    }
    // DYNAMIC GRID PARSER
    else if (inString.startsWith("bAUTO_START")) {
      if (autonomousMode == true) {
        Serial.println("Robot is already running! Ignoring duplicate start command.");
      }
      else {
        int firstPipe = inString.indexOf('|');
        int secondPipe = inString.lastIndexOf('|');

        if (firstPipe > 0 && secondPipe > firstPipe) {
          maxRows = inString.substring(firstPipe + 1, secondPipe).toInt();
          maxCols = inString.substring(secondPipe + 1).toInt();
        } else {
          maxRows = 3;
          maxCols = 3;
        }

        Serial.print("Starting dynamic grid: ");
        Serial.print(maxRows);
        Serial.print("x");
        Serial.println(maxCols);

        startAutonomousMode();
      }

      delay(100);
      while(bt.available() > 0) {
        bt.read();
      }
    }
    else if (inString == "bCALIBRATE") {
      calibrateCompass();
    }

    else if (inString.startsWith("bANGLES")) {
      int p1 = inString.indexOf('|');
      int p2 = inString.indexOf('|', p1 + 1);
      int p3 = inString.indexOf('|', p2 + 1);
      int p4 = inString.indexOf('|', p3 + 1);

      if (p1 > 0 && p2 > 0 && p3 > 0 && p4 > 0) {
        angleR1 = inString.substring(p1 + 1, p2).toInt();
        angleR2 = inString.substring(p2 + 1, p3).toInt();
        angleL1 = inString.substring(p3 + 1, p4).toInt();
        angleL2 = inString.substring(p4 + 1).toInt();

        Serial.print("Angles Synced! R1:"); Serial.print(angleR1);
        Serial.print(" R2:"); Serial.print(angleR2);
        Serial.print(" L1:"); Serial.print(angleL1);
        Serial.print(" L2:"); Serial.println(angleL2);
      }
    }

    // SLOWED DOWN MANUAL DRIVING
    else if (autonomousMode == false) {
      if (inString == "b1") {
        Serial.println("Manual: FORWARD");
        motor1.setSpeed(100);
        motor2.setSpeed(100);
      }
      else if (inString == "b2") {
        Serial.println("Manual: REVERSE");
        motor1.setSpeed(-100);
        motor2.setSpeed(-100);
      }
      else if (inString == "b4") {
        Serial.println("Manual: LEFT");
        motor1.setSpeed(-100);
        motor2.setSpeed(100);
      }
      else if (inString == "b3") {
        Serial.println("Manual: RIGHT");
        motor1.setSpeed(100);
        motor2.setSpeed(-100);
      }
    }
  }

  // Run the actual robot behavior
  if (autonomousMode == true) {
    runAutonomousMode();
  }
}

// MAIN AUTONOMOUS LOGIC
void startAutonomousMode() {
  autonomousMode = true;
  gridRow = 0;
  gridCol = 0;

  int sum = 0;
  for (int i = 0; i < 10; i++) {
    compass.read();
    int h = compass.getAzimuth();
    if (h < 0) h += 360;
    sum += h;
    delay(100);
  }
  referenceHeading = sum / 10;
  currentTargetHeading = 0;

  autoState = AUTO_CAPTURE;
  bt.println("AUTO_STARTED");
}

void stopAutonomousMode() {
  autonomousMode = false;
  robotStop();
  autoState = AUTO_IDLE;
  bt.println("AUTO_STOPPED");
}

void runAutonomousMode() {
  switch (autoState) {
    case AUTO_IDLE:
      break;

    case AUTO_CAPTURE:
      robotStop();
      delay(2500);

      bt.print("CAPTURE_NOW|R");
      bt.print(gridRow);
      bt.print("|C");
      bt.println(gridCol);

      delay(2000);

      bool endOfLane;
      if (gridCol % 2 == 0) {
          endOfLane = (gridRow >= (maxRows - 1));
      } else {
          endOfLane = (gridRow <= 0);
      }

      if (endOfLane) {
          if (gridCol >= (maxCols - 1)) {
              stopAutonomousMode();
              bt.println("GRID_COMPLETE");
          } else {
              autoState = AUTO_NEXT_ROW;
          }
      } else {
          autoState = AUTO_FORWARD;
      }
      break;

    case AUTO_FORWARD:
      // SLOW SPRINT: 120 Speed, 2200 Milliseconds
      driveStraightSmooth(currentTargetHeading, 120, 2200);
      if (gridCol % 2 == 0) gridRow++;
      else gridRow--;

      autoState = AUTO_CAPTURE;
      break;

    case AUTO_NEXT_ROW:
      Serial.println(" End of lane. Executing Empirical Snake Turn...");

      if (gridCol % 2 == 0) {
        // EVEN COLUMNS (Driving UP)

        // Force the robot to declare its current position as exactly 0°
        forceHeadingSync(0);

        currentTargetHeading = angleR1;
        turnSnake(currentTargetHeading, 'R');
        delay(1000);

        Serial.println(" Stepping to next column");
        driveStraightSmooth(currentTargetHeading, 120, 2000);
        delay(1000);

        currentTargetHeading = angleR2;
        turnSnake(currentTargetHeading, 'R');

        currentTargetHeading = 180;

      } else {
        // ODD COLUMNS (Driving DOWN)

        // Force the robot to declare its current position as exactly 180°
        forceHeadingSync(180);

        currentTargetHeading = angleL1;
        turnSnake(currentTargetHeading, 'L');
        delay(1000);

        Serial.println("⬅ Stepping to next column");
        driveStraightSmooth(currentTargetHeading, 120, 2000);
        delay(1000);

        currentTargetHeading = angleL2;
        turnSnake(currentTargetHeading, 'L');

        currentTargetHeading = 0;
      }

      delay(1000);
      gridCol++;
      autoState = AUTO_CAPTURE;
      break;
  }
}

// COMPASS & MOVEMENT HELPERS

int getRelativeHeading() {
  compass.read();
  int absolute = compass.getAzimuth();
  if (absolute < 0) absolute += 360;

  int relative = absolute - referenceHeading;
  while (relative < 0) relative += 360;
  while (relative >= 360) relative -= 360;
  return relative;
}

void robotStop() {
  motor1.setSpeed(0);
  motor2.setSpeed(0);
}

// Forces the robot to recalibrate to a known physical heading
void forceHeadingSync(int physicalHeading) {
  robotStop();
  delay(1000); // Let motors power down and magnetic field completely collapse

  int sum = 0;
  for (int i = 0; i < 10; i++) {
    compass.read();
    int h = compass.getAzimuth();
    if (h < 0) h += 360;
    sum += h;
    delay(50);
  }
  int rawAverage = sum / 10;

  // Shift the global reference so getRelativeHeading() perfectly matches our physical location
  referenceHeading = rawAverage - physicalHeading;
  while (referenceHeading < 0) referenceHeading += 360;
  while (referenceHeading >= 360) referenceHeading -= 360;

  Serial.print(" FORCED SYNC! Robot is declaring its current physical position as: ");
  Serial.print(physicalHeading);
  Serial.println("°");
}

void calibrateCompass() {
  bt.println("CAL_START|Taring Compass to 0°...");

  int sum = 0;
  for (int i = 0; i < 10; i++) {
    compass.read();
    int h = compass.getAzimuth();
    if (h < 0) h += 360;
    sum += h;
    delay(100);
  }
  referenceHeading = sum / 10;
  currentTargetHeading = 0;

  bt.print("TELEMETRY|");
  bt.print(compass.getX()); bt.print("|"); bt.print(compass.getY()); bt.print("|");
  bt.print(0); bt.print("|"); bt.println(0);

  bt.println("CAL_DONE|Robot is now facing 0°");
}

void driveStraightSmooth(int targetRelativeHeading, int maxSpeed, unsigned long durationMillis) {
  unsigned long startTime = millis();
  int rampDuration = 400;
  int minMotorSpeed = 70;

  int rightMotorOffset = 0;
  int leftMotorOffset = 4;

  Serial.println("Continuous Drive to next tile...");

  while (millis() - startTime < durationMillis) {
    unsigned long elapsed = millis() - startTime;
    int currentSpeed = maxSpeed;

    if (elapsed < rampDuration) {
      currentSpeed = map(elapsed, 0, rampDuration, minMotorSpeed, maxSpeed);
    } else if (elapsed > durationMillis - rampDuration) {
      unsigned long timeLeft = durationMillis - elapsed;
      currentSpeed = map(timeLeft, 0, rampDuration, minMotorSpeed, maxSpeed);
    }

    int rightSpeed = currentSpeed + rightMotorOffset;
    int leftSpeed = currentSpeed + leftMotorOffset;

    motor1.setSpeed(constrain(rightSpeed, 0, 255));
    motor2.setSpeed(constrain(leftSpeed, 0, 255));

    delay(40);
  }

  robotStop();
  delay(600); // Lets chassis physically settle completely
  Serial.println("Arrived at Grid Tile. Settle complete.");
}

void turnSmooth(int targetRelativeDegree) {
  int baseTurnPower = 150;

  Serial.print("Pure Math Turn to: ");
  Serial.println(targetRelativeDegree);

  while (true) {
    robotStop();
    delay(300);

    int current = getRelativeHeading();
    int error = targetRelativeDegree - current;

    while (error < -180) error += 360;
    while (error > 180) error -= 360;

    bt.print("TELEMETRY|");
    bt.print(compass.getX()); bt.print("|"); bt.print(compass.getY()); bt.print("|");
    bt.print(current); bt.print("|"); bt.println(targetRelativeDegree);

    if (abs(error) <= 3) {
      Serial.println("Perfectly aligned!");
      break;
    }

    int currentPower;
    int pulseDuration;

    if (abs(error) > 45) {
      currentPower = baseTurnPower;
      pulseDuration = 180;
    } else if (abs(error) > 10) {
      currentPower = baseTurnPower - 10;
      pulseDuration = 80;
    } else {
      currentPower = 95;
      pulseDuration = 20;
    }

    if (error > 0) {
      motor1.setSpeed(-currentPower);
      motor2.setSpeed(currentPower);
    } else {
      motor1.setSpeed(currentPower);
      motor2.setSpeed(-currentPower);
    }

    delay(pulseDuration);
  }

  robotStop();
  delay(400);
}

// THE "JERK BIT-BY-BIT" STUTTER TURN
void turnSnake(int targetRelativeDegree, char turnDirection) {
  int baseTurnPower = 140;

  Serial.print("Stutter Snake Turn ");
  Serial.print(turnDirection);
  Serial.print(" to: ");
  Serial.println(targetRelativeDegree);

  while (true) {
    robotStop();
    delay(300);

    int current = getRelativeHeading();
    int error = targetRelativeDegree - current;

    while (error < -180) error += 360;
    while (error > 180) error -= 360;

    bt.print("TELEMETRY|");
    bt.print(compass.getX()); bt.print("|"); bt.print(compass.getY()); bt.print("|");
    bt.print(current); bt.print("|"); bt.println(targetRelativeDegree);

    if (abs(error) <= 3) {
      Serial.println("Perfectly aligned!");
      break;
    }

    int currentPower;
    int pulseDuration;

    if (abs(error) > 45) {
      currentPower = baseTurnPower;
      pulseDuration = 160;
    } else if (abs(error) > 15) {
      currentPower = baseTurnPower - 20;
      pulseDuration = 80;
    } else {
      currentPower = 95;
      pulseDuration = 30;
    }

    if (turnDirection == 'R') {
      motor1.setSpeed(-currentPower);
      motor2.setSpeed(currentPower);
    } else {
      motor1.setSpeed(currentPower);
      motor2.setSpeed(-currentPower);
    }

    delay(pulseDuration);
  }

  robotStop();
  delay(500);
}
