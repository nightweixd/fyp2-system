# TileVision

My FYP2 project for inspecting floor tiles with a camera and a Bluetooth-controlled robot.

The Flask dashboard shows the camera feed, runs tile analysis, and keeps inspection history in SQLite. The robot captures images across a grid, which can be combined into a floor map.

The detection code uses YOLO oriented bounding boxes, then compares three methods:

- Centroid: measures tile alignment from fitted rows and columns.
- Skeleton: measures grout width and angular alignment.
- Edge: measures grout width from detected edges.

## Run locally

Use Python 3.10 or later. From the project folder:

```sh
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

The activation command above is for Windows. On Linux or macOS, use `source .venv/bin/activate`. Supply the trained YOLO OBB weights separately. The default path is `runs/obb/train4/weights/best.pt`; `YOLO_MODEL_PATH` can point to another location.

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
$env:YOLO_MODEL_PATH = "C:\models\tile-obb.pt"
$env:BLUETOOTH_COM_PORT = "COM4"
$env:CAMERA_SOURCE = "0"
python app.py
```

Open `http://localhost:5000`. `CAMERA_SOURCE` accepts a webcam index or video-stream URL. It defaults to camera index 2, with a fallback to index 0. Set `FLASK_SECRET_KEY` if a persistent session key is needed; otherwise the server generates one when it starts.

For an image folder:

```sh
python batch_detect.py --input ./images --method centroid --output ./results
```

## Robot firmware

The Arduino sketch is in `sketch_motion2350_full_code__1_/`. It uses SoftwareSerial, QMC5883LCompass, and CytronMotorDriver. Configure the board, motor pins, compass calibration, turn angles, and movement timings for the robot before running it.

## Repository contents

This repository contains the source code. Images, datasets, model weights, inspection databases, and generated results are excluded. The app creates its output folders and database on startup.

The inspection thresholds and camera scale are configured in `detection/methods.py`. They need calibration for the test setup. Skeleton alignment is measured in degrees; the original dashboard and history do not consistently label that unit. This is a local project prototype, and hardware operation and detection accuracy need to be checked with the trained model and robot.
