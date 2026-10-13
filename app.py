"""Flask dashboard, inspection history, and Bluetooth robot control."""
import os, sys, json, datetime, threading, time, secrets
import cv2, numpy as np
import serial
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import sqlite3

sys.path.insert(0, os.path.dirname(__file__))
from detection.methods import run_centroid, run_skeleton, run_edge

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Bluetooth Settings
BLUETOOTH_COM_PORT = os.environ.get("BLUETOOTH_COM_PORT", "COM4")
BAUD_RATE = 9600
bt_serial = None

CAPTURE_DIR = "static/captures"
RESULTS_DIR = "static/results"
FLOORPLAN_DIR = "static/floorplans"
DB_PATH = "database/inspections.db"
for d in [CAPTURE_DIR, RESULTS_DIR, FLOORPLAN_DIR, "database"]:
    os.makedirs(d, exist_ok=True)

camera = None
cam_lock = threading.Lock()
CAMERA_URL = os.environ.get("CAMERA_SOURCE", "2")
if CAMERA_URL.isdigit():
    CAMERA_URL = int(CAMERA_URL)
cam_ok = False

robot_state = {
    "running": False,
    "method": "centroid",
    "interval": 5,
    "count": 0,
    "last_cmd": None,
    "grid_position": {"row": 0, "col": 0}
}

floor_plan = {
    "images": [],
    "stitched": None,
    "grid": {},
    "rows": 0,
    "cols": 0
}

# Camera Functions
def open_camera(url=None):
    global camera, cam_ok, CAMERA_URL
    if url: CAMERA_URL = url
    with cam_lock:
        if camera and camera.isOpened(): camera.release()
        camera = cv2.VideoCapture(CAMERA_URL)
        if not camera.isOpened(): camera = cv2.VideoCapture(0)
        cam_ok = camera.isOpened()
    return cam_ok

def read_frame():
    global camera, cam_ok
    if camera is None: open_camera()
    with cam_lock:
        if camera and camera.isOpened():
            ret, frame = camera.read()
            cam_ok = ret
            return ret, frame
    return False, None

def gen_frames():
    while True:
        ret, frame = read_frame()
        if not ret:
            time.sleep(0.1)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

# Database Functions
def init_db():
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS inspections(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, image_path TEXT, result_path TEXT, method TEXT, status TEXT,
        deviation_mm REAL, width_mm REAL, width_std REAL, tiles_count INTEGER,
        source TEXT, tile_results TEXT, grid_row INTEGER, grid_col INTEGER
    )""")
    c.commit(); c.close()

def db_insert(d):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("""INSERT INTO inspections
        (timestamp,image_path,result_path,method,status,
         deviation_mm,width_mm,width_std,tiles_count,source,tile_results,
         grid_row,grid_col)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (datetime.datetime.now().isoformat(),
         d.get("image_path", ""), d.get("result_path", ""),
         d.get("method", ""), d.get("status", ""),
         d.get("deviation"), d.get("width"),
         d.get("width_std"), d.get("tiles", 0),
         d.get("source", "manual"),
         json.dumps(d.get("tile_results", [])),
         d.get("grid_row", 0), d.get("grid_col", 0)))
    conn.commit()
    lid = cur.lastrowid
    conn.close()
    return lid

def db_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""SELECT id,timestamp,method,status,deviation_mm,
        width_mm,width_std,tiles_count,result_path,source,tile_results,
        grid_row,grid_col FROM inspections ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
    conn.close()
    keys = ["id", "timestamp", "method", "status", "deviation_mm",
            "width_mm", "width_std", "tiles_count", "result_path", "source",
            "tile_results", "grid_row", "grid_col"]
    out = []
    for r in rows:
        row = dict(zip(keys, r))
        try: row["tile_results"] = json.loads(row["tile_results"] or "[]")
        except: row["tile_results"] = []
        out.append(row)
    return out

def db_stats():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    pass_ = conn.execute("SELECT COUNT(*) FROM inspections WHERE status='PASS'").fetchone()[0]
    fail = conn.execute("SELECT COUNT(*) FROM inspections WHERE status='FAIL'").fetchone()[0]
    conn.close()
    return {"total": total, "pass": pass_, "fail": fail}

# Image Stitching Functions
def stitch_images(image_list):
    if len(image_list) < 2: return None
    images = [cv2.imread(p) for p in image_list if cv2.imread(p) is not None]
    if len(images) < 2: return None
    try:
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, stitched = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"{FLOORPLAN_DIR}/stitched_{ts}.jpg"
            cv2.imwrite(path, stitched)
            return path
    except Exception as e: print(e)
    return None

def create_grid_floorplan(grid_dict, rows=3, cols=3):
    """
    Build a grid map with row zero at the bottom, matching robot coordinates.
    """
    if not grid_dict:
        print(" No grid data available to generate map.")
        return None

    # CONFIGURATION
    fixed_rows = rows
    fixed_cols = cols

    # Define directories (ensure these match the application paths)
    FLOORPLAN_DIR = "static/floorplans"
    if not os.path.exists(FLOORPLAN_DIR):
        os.makedirs(FLOORPLAN_DIR)

    # INITIALIZE DIMENSIONS
    sample_img = None
    # Look for any valid image in the grid to determine tile size
    for r_idx in grid_dict:
        for c_idx in grid_dict[r_idx]:
            p = grid_dict[r_idx][c_idx]
            if p and os.path.exists(p):
                sample_img = cv2.imread(p)
                if sample_img is not None:
                    break
        if sample_img is not None:
            break

    if sample_img is None:
        print(" Could not load any images from grid_dict.")
        return None

    h, w = sample_img.shape[:2]
    # Missing captures remain black.
    canvas = np.zeros((h * fixed_rows, w * fixed_cols, 3), dtype=np.uint8)

    # FILL THE GRID (BOTTOM-TO-TOP)
    for r in range(fixed_rows):
        # INVERT Y-AXIS:
        # r=0 (bottom row) -> canvas_row = 2
        # r=1 (middle row) -> canvas_row = 1
        # r=2 (top row)    -> canvas_row = 0
        canvas_row = (fixed_rows - 1) - r

        for c in range(fixed_cols):
            # Column 0 is Left, Column 2 is Right
            if r in grid_dict and c in grid_dict[r]:
                img_path = grid_dict[r][c]

                if img_path and os.path.exists(img_path):
                    tile_img = cv2.imread(img_path)
                    if tile_img is not None:
                        # Resize to match first tile if needed
                        if tile_img.shape[:2] != (h, w):
                            tile_img = cv2.resize(tile_img, (w, h))

                        # Place at the correct canvas position
                        y_start = canvas_row * h
                        x_start = c * w
                        canvas[y_start:y_start+h, x_start:x_start+w] = tile_img

    # SAVE THE FINAL MAP
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{FLOORPLAN_DIR}/grid_{ts}.jpg"
    cv2.imwrite(out_path, canvas)
    print(f"Generated {rows}x{cols} grid map: {out_path}")
    return out_path

# Detection Wrapper
def do_detect(img_path, method, source, grid_row=None, grid_col=None):
    res = None
    out_path = None

    # Pass only 1 argument and unpack the returned tuple
    if method == "centroid":
        out_path, res = run_centroid(img_path)
    elif method == "skeleton":
        out_path, res = run_skeleton(img_path)
    elif method == "edge":
        out_path, res = run_edge(img_path)

# Safely handle unsupported methods or complete failures
    if res is None:
        return None, {"error": f"Unsupported method '{method}' or detection failed."}

    # Safely handle handled errors from the detection scripts
    if "error" in res:
        return None, {"error": res["error"]}

    # Add the tracking metadata to the JSON result
    res["image_path"] = img_path
    res["result_path"] = out_path
    res["method"] = method
    res["source"] = source
    if grid_row is not None: res["grid_row"] = grid_row
    if grid_col is not None: res["grid_col"] = grid_col

    db_insert(res)
    socketio.emit("stats_update", db_stats())
    socketio.emit("new_result", res)
    return out_path, res


# BLUETOOTH LISTENER
def bluetooth_listener_loop():
    global bt_serial, robot_state, floor_plan

    while True:
        # CHECK CONNECTION
        if bt_serial is None or not bt_serial.is_open:
            try:
                print(f" Attempting Bluetooth connection on {BLUETOOTH_COM_PORT}...")
                bt_serial = serial.Serial(BLUETOOTH_COM_PORT, BAUD_RATE, timeout=1)
                print(" Bluetooth Connected!")
                socketio.emit("bluetooth_status", {"connected": True})
            except Exception as e:
                print(f" BT Connection Failed: {e}. Retrying in 3s...")
                socketio.emit("bluetooth_status", {"connected": False})
                time.sleep(3)
                continue

        # LISTEN FOR MESSAGES
        try:
            if bt_serial.in_waiting > 0:
                line = bt_serial.readline().decode('utf-8', errors='ignore').strip()
                print(f" Arduino: {line}")


                if line.startswith("TELEMETRY|"):
                    try:
                        parts = line.split("|")
                        # Structure: TELEMETRY | X | Y | Heading | Target
                        telemetry_data = {
                            "x": parts[1],
                            "y": parts[2],
                            "heading": parts[3],
                            "target": parts[4]
                        }
                        # Send live to the web dashboard
                        socketio.emit('update_telemetry', telemetry_data)
                    except Exception as e:
                        print(f"Telemetry Parse Error: {e}")
                    continue # Skip the rest of the loop for telemetry lines
                    # HANDLE COMPASS CALIBRATION MESSAGES

                if line.startswith("CAL_START"):
                    # Extract message if present
                    parts = line.split("|")
                    msg = parts[1] if len(parts) > 1 else "Calibrating compass..."
                    socketio.emit("calibration_status", {"status": "started", "message": msg})
                    print(f" Calibration started: {msg}")

                elif line.startswith("CAL_DONE"):
                    parts = line.split("|")
                    msg = parts[1] if len(parts) > 1 else "Calibration complete"
                    socketio.emit("calibration_status", {"status": "done", "message": msg})
                    print(f" Calibration complete: {msg}")


                elif line == "AUTO_STARTED":
                    if not robot_state["running"]:
                        robot_state["running"] = True
                        socketio.emit("robot_status", {"running": True, "count": robot_state["count"]})
                        print(" Robot officially started! (Duplicates suppressed)")

                elif line == "AUTO_STOPPED" or line == "GRID_COMPLETE":
                    robot_state["running"] = False
                    socketio.emit("robot_status", {"running": False, "count": robot_state["count"]})
                    if line == "GRID_COMPLETE":
                        socketio.emit("grid_complete", {"message": "Grid scan complete."})

                # HANDLE CAPTURE COMMAND
                elif line.startswith("CAPTURE_NOW"):
                    # Parse grid position from format: CAPTURE_NOW|R0|C1
                    parts = line.split("|")
                    row, col = 0, 0

                    for part in parts:
                        if part.startswith("R"):
                            try: row = int(part[1:])
                            except: pass
                        if part.startswith("C"):
                            try: col = int(part[1:])
                            except: pass

                    print(f" Capture Request: Row {row}, Col {col}")

                    # Take Photo
                    ret, frame = read_frame()
                    if ret:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"robot_R{row}_C{col}_{ts}.jpg"
                        img_path = f"{CAPTURE_DIR}/{filename}"
                        cv2.imwrite(img_path, frame)

                        # Save to grid dictionary
                        if row not in floor_plan["grid"]:
                            floor_plan["grid"][row] = {}
                        floor_plan["grid"][row][col] = img_path
                        floor_plan["images"].append(img_path)

                        robot_state["count"] += 1
                        robot_state["grid_position"] = {"row": row, "col": col}

                        socketio.emit("new_capture", {
                            "path": img_path,
                            "filename": filename,
                            "row": row,
                            "col": col,
                            "status": "READY"
                        })
                        print(f" Saved to Grid: Row {row}, Col {col}")
                    else:
                        print(" Camera failed to capture")

        except Exception as e:
            print(f" Bluetooth lost connection: {e}")
            try:
                bt_serial.close()
            except:
                pass
            bt_serial = None # Force the Connect Section to run again
            time.sleep(2)

        time.sleep(0.05)

# Routes
@app.route("/")
def index(): return render_template("dashboard.html")

@app.route("/video_feed")
def video_feed(): return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files: return jsonify({"error":"No file"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error":"No file"}), 400
    method = request.form.get("method", "centroid")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{CAPTURE_DIR}/upload_{ts}.jpg"
    file.save(path)
    rp, resp = do_detect(path, method, source="upload")
    if rp is None: return jsonify(resp), 422
    return jsonify(resp)

@app.route("/api/capture", methods=["POST"])
def capture():
    ret, frame = read_frame()
    if not ret: return jsonify({"error": "Camera not available"}), 500
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{CAPTURE_DIR}/capture_{ts}.jpg"
    cv2.imwrite(path, frame)
    return jsonify({"path": path, "timestamp": ts})

@app.route("/api/capture_and_detect", methods=["POST"])
def capture_and_detect():
    method = (request.json or {}).get("method", "centroid")
    ret, frame = read_frame()
    if not ret: return jsonify({"error": "Camera not available"}), 500
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{CAPTURE_DIR}/manual_{ts}.jpg"
    cv2.imwrite(path, frame)
    rp, resp = do_detect(path, method, "manual")
    if rp is None: return jsonify(resp), 422
    return jsonify(resp)

ACTIVE_ROWS = 3
ACTIVE_COLS = 3

@app.route("/api/robot/auto_start", methods=["POST"])
def auto_start():
    global bt_serial
    data = request.json or {}

    # Grab Grid Size
    rows = data.get("rows", 3)
    cols = data.get("cols", 3)

    # Grab Angles
    r1 = data.get("r1", 98)
    r2 = data.get("r2", 195)
    l1 = data.get("l1", 105)
    l2 = data.get("l2", 15)

    if bt_serial and bt_serial.is_open:
        try:
            # First, send the angles to update the robot's RAM
            angle_cmd = f"bANGLES|{r1}|{r2}|{l1}|{l2}\r"
            bt_serial.write(angle_cmd.encode('utf-8'))

            # Give the Arduino a tiny fraction of a second to process the angles
            time.sleep(0.2)

            # Then, immediately trigger the start command!
            start_cmd = f"bAUTO_START|{rows}|{cols}\r"
            bt_serial.write(start_cmd.encode('utf-8'))

            floor_plan.update({"rows": int(rows), "cols": int(cols), "grid": {}, "images": [], "stitched": None})
            robot_state["count"] = 0

            return jsonify({"ok": True, "msg": "Grid scan started."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Bluetooth not connected."}), 500

@app.route("/api/robot/auto_stop", methods=["POST"])
def auto_stop():
    global bt_serial
    if bt_serial and bt_serial.is_open:
        try:
            bt_serial.write(b"bAUTO_STOP\r")
            return jsonify({"ok": True, "msg": "Auto Mode Stopped!"})
        except Exception as e: return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Bluetooth not connected."}), 500

@app.route("/api/robot/calibrate_compass", methods=["POST"])
def calibrate_compass():
    global bt_serial
    if bt_serial and bt_serial.is_open:
        try:
            bt_serial.write(b"bCALIBRATE\r")
            return jsonify({"ok": True, "msg": "Tare command sent. Zeroing compass..."})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Bluetooth not connected."}), 500

@app.route('/api/detect', methods=['POST'])
def manual_detect():
    data = request.get_json(silent=True) or {}
    raw_path = data.get("image_path")
    method = data.get("method", "centroid")
    if not raw_path:
        return jsonify({"error": "No image selected"}), 400

    # Run the detection
    result_path, analysis_data = do_detect(raw_path, method, "manual")
    if result_path is None:
        return jsonify(analysis_data), 422

    # Update the Grid Dictionary with the Analyzed Image
    # Replace the matching capture with its analysed result.
    for r in floor_plan["grid"]:
        for c in floor_plan["grid"][r]:
            if floor_plan["grid"][r][c] == raw_path:
                # Swap the raw photo for the one with detection boxes/lines
                floor_plan["grid"][r][c] = result_path
                break

    return jsonify(analysis_data)

@app.route("/api/robot/set_angles", methods=["POST"])
def set_angles():
    global bt_serial
    data = request.json or {}
    r1 = data.get("r1", 98)
    r2 = data.get("r2", 195)
    l1 = data.get("l1", 105)
    l2 = data.get("l2", 15)

    if bt_serial and bt_serial.is_open:
        try:
            # Build the string: bANGLES|98|195|105|15
            command = f"bANGLES|{r1}|{r2}|{l1}|{l2}\r"
            bt_serial.write(command.encode('utf-8'))
            return jsonify({"ok": True, "msg": "Angles successfully synced to Robot!"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Bluetooth not connected."}), 500

@app.route("/api/floorplan/stitch", methods=["POST"])
def stitch_floorplan():
    rows = sqlite3.connect(DB_PATH).execute("SELECT image_path FROM inspections WHERE source='robot' ORDER BY id DESC LIMIT 20").fetchall()
    paths = [r[0] for r in rows if os.path.exists(r[0])]
    if len(paths) < 2: return jsonify({"error": "Need at least 2 images"}), 400
    res = stitch_images(paths[::-1])
    if res: return jsonify({"ok": True, "path": res})
    return jsonify({"error": "Stitching failed"}), 500

@app.route("/api/floorplan/grid", methods=["POST"])
def create_grid_plan():
    if not floor_plan["grid"]: return jsonify({"error": "No grid data available"}), 400
    res = create_grid_floorplan(floor_plan["grid"], floor_plan["rows"], floor_plan["cols"])
    if res: return jsonify({"ok": True, "path": res})
    return jsonify({"error": "Grid plan creation failed"}), 500

@app.route("/api/floorplan/generate_analysis_grid", methods=["POST"])
def generate_analysis_grid():
    if not floor_plan["grid"]:
        return jsonify({"error": "No grid data available. Run the robot first."}), 400

    # Grab the specific method you clicked on the dashboard (e.g. skeleton)
    req_data = request.get_json(silent=True) or {}
    current_method = req_data.get("method", "centroid")

    # Dynamically figure out grid size (Automatically supports 3x3, 4x4, 5x5...)
    max_r = max(floor_plan["grid"].keys())
    max_c = max(max(floor_plan["grid"][r].keys()) for r in floor_plan["grid"] if floor_plan["grid"][r])

    analyzed_results_grid = {}

    # Process every photo using the selected AI
    for r in range(max_r + 1):
        analyzed_results_grid[r] = {}
        for c in range(max_c + 1):
            if r in floor_plan["grid"] and c in floor_plan["grid"][r]:
                raw_path = floor_plan["grid"][r][c]

                print(f" Processing Tile R{r}|C{c} with {current_method}...")
                res_path, resp_data = do_detect(raw_path, current_method, source="robot", grid_row=r, grid_col=c)

                # If analysis passes, use the result. If it fails, fallback to the raw photo.
                if res_path:
                    analyzed_results_grid[r][c] = res_path
                else:
                    analyzed_results_grid[r][c] = raw_path

    # Generate the final combined map!
    analysis_map_path = create_grid_floorplan(analyzed_results_grid, max_r + 1, max_c + 1)

    if analysis_map_path:
        return jsonify({"ok": True, "path": analysis_map_path})
    return jsonify({"error": "Failed to create analysis grid"}), 500

@app.route("/api/history")
def history(): return jsonify(db_history(int(request.args.get("limit", 50))))

@app.route("/api/floorplan/analysis_grid", methods=["POST"])
def create_analysis_grid():
    """
    Build a map of analysed captures, leaving other slots black.
    """
    if not floor_plan["grid"]:
        return jsonify({"error": "No data in grid"}), 400

    # Create a temporary dict specifically for the Analysis Map
    analysis_only_grid = {}
    rows = floor_plan["rows"] or max(floor_plan["grid"]) + 1
    cols = floor_plan["cols"] or max(c for row in floor_plan["grid"].values() for c in row) + 1

    for r in range(rows):
        analysis_only_grid[r] = {}
        for c in range(cols):
            # Check if this coordinate exists in the capture grid
            if r in floor_plan["grid"] and c in floor_plan["grid"][r]:
                path = floor_plan["grid"][r][c]
                # Only include it if it's an analyzed result (located in static/results)
                if "static/results" in path or "analyzed" in path:
                    analysis_only_grid[r][c] = path
                else:
                    analysis_only_grid[r][c] = None # Leave empty

    # Build the map from grid coordinates
    res = create_grid_floorplan(analysis_only_grid, rows, cols)

    if res: return jsonify({"ok": True, "path": res})
    return jsonify({"error": "Failed to generate analysis grid"}), 500

@app.route("/api/stats")
def stats(): return jsonify(db_stats())

@app.route("/api/set_camera", methods=["POST"])
def set_camera(): return jsonify({"ok": open_camera(request.json.get("url", ""))})

@app.route("/api/clear_history", methods=["POST"])
def clear_history():
    sqlite3.connect(DB_PATH).execute("DELETE FROM inspections").connection.commit()
    return jsonify({"ok": True})

@socketio.on("connect")
def on_connect():
    emit("stats_update", db_stats())
    emit("robot_status", {"running": robot_state["running"], "count": robot_state["count"]})

if __name__ == "__main__":
    init_db()
    open_camera()
    threading.Thread(target=bluetooth_listener_loop, daemon=True).start()
    print("=" * 60)
    print("  TileVision — http://localhost:5000")
    print("=" * 60)
    socketio.run(app, debug=False, host="127.0.0.1", port=5000, allow_unsafe_werkzeug=True)
