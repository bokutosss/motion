"""Phone-connected motion detection system.

Watches a camera (a phone running IP Webcam / DroidCam, or a local webcam),
detects motion with OpenCV, and serves a mobile-friendly dashboard so you can
check the live feed, alerts and snapshots from your phone's browser.

Run:  python app.py --source http://PHONE_IP:8080/video --password yourpass
"""
import argparse
import datetime as dt
import hmac
from io import BytesIO
import os
import socket
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory
import qrcode

# --------------------------------------------------------------------------
# Configuration (command-line flags, with environment-variable fallbacks)
# --------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Phone-connected motion detector")
parser.add_argument("--source", default=os.environ.get("CAMERA_SOURCE", "0"),
                    help="Camera index (0) or stream URL, e.g. http://192.168.1.50:8080/video")
parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
parser.add_argument("--cert", default=os.environ.get("TLS_CERT", ""),
                    help="TLS certificate file (required with --key for phone-browser camera access)")
parser.add_argument("--key", default=os.environ.get("TLS_KEY", ""),
                    help="TLS private key file")
parser.add_argument("--min-area", type=int, default=int(os.environ.get("MIN_AREA", "1500")),
                    help="Smallest moving blob (pixels) that counts as motion. Higher = less sensitive.")
parser.add_argument("--confirm-frames", type=int, default=int(os.environ.get("CONFIRM_FRAMES", "3")),
                    help="Consecutive motion frames required before recording an event.")
parser.add_argument("--cooldown", type=float, default=float(os.environ.get("COOLDOWN", "10")),
                    help="Seconds of stillness before a new motion event can start.")
parser.add_argument("--user", default=os.environ.get("ACCESS_USER", "admin"))
parser.add_argument("--password", default=os.environ.get("ACCESS_PASSWORD", ""),
                    help="Dashboard password (strongly recommended).")
parser.add_argument("--telegram-token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""))
parser.add_argument("--telegram-chat", default=os.environ.get("TELEGRAM_CHAT_ID", ""))
args = parser.parse_args() if __name__ == "__main__" else parser.parse_args([])
if bool(args.cert) != bool(args.key):
    parser.error("--cert and --key must be provided together")
if args.confirm_frames < 1:
    parser.error("--confirm-frames must be at least 1")

SOURCE = int(args.source) if str(args.source).isdigit() else args.source
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024


# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------
class State:
    def __init__(self):
        self.armed = True
        self.motion = False
        self.last_motion = None      # ISO string
        self.camera_ok = False
        self.fps = 0.0
        self.jpeg = None
        self.phone_mode = False
        self.phone_frame = None
        self.phone_frame_at = 0.0
        self.frame_cond = threading.Condition()
        self.object_model = "starting"
        self.primary_object = "none"
        self.body_parts = []
        self.events = deque(maxlen=50)   # newest first


state = State()


# --------------------------------------------------------------------------
# Object recognition
# --------------------------------------------------------------------------
OBJECT_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
OBJECT_MODEL_URLS = {
    "objects": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov_int8.onnx",
    "pose": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar_int8bq.onnx",
}
POSE_NAMES = (
    "nose", "left eye inner", "left eye", "left eye outer", "right eye inner",
    "right eye", "right eye outer", "left ear", "right ear", "mouth left",
    "mouth right", "left shoulder", "right shoulder", "left elbow", "right elbow",
    "left wrist", "right wrist", "left pinky", "right pinky", "left index",
    "right index", "left thumb", "right thumb", "left hip", "right hip", "left knee",
    "right knee", "left ankle", "right ankle", "left heel", "right heel",
    "left foot", "right foot",
)
POSE_CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31),
    (28, 30), (30, 32), (0, 7), (0, 8),
)
BODY_PART_IDS = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)


def download_model_file(path, url):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    response = requests.get(url, stream=True, timeout=(10, 120))
    response.raise_for_status()
    temporary_path = path + ".part"
    with open(temporary_path, "wb") as model_file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                model_file.write(chunk)
    os.replace(temporary_path, path)


class ObjectDetector:
    def __init__(self):
        configured_dir = os.environ.get("MODEL_CACHE_DIR")
        model_dir = (os.path.join(BASE_DIR, configured_dir) if configured_dir and not os.path.isabs(configured_dir)
                     else configured_dir or os.path.join(os.path.expanduser("~"), ".cache", "motion"))
        os.makedirs(model_dir, exist_ok=True)
        self.object_model_path = os.path.join(model_dir, "object_detection_yolox.onnx")
        self.pose_model_path = os.path.join(model_dir, "pose_estimation_mediapipe.onnx")
        download_model_file(self.object_model_path, OBJECT_MODEL_URLS["objects"])
        download_model_file(self.pose_model_path, OBJECT_MODEL_URLS["pose"])
        self.net = cv2.dnn.readNet(self.object_model_path)
        self.pose_net = cv2.dnn.readNet(self.pose_model_path)
        for net in (self.net, self.pose_net):
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.grids, self.strides = [], []
        for stride in (8, 16, 32):
            side = 640 // stride
            grid_x, grid_y = np.meshgrid(np.arange(side), np.arange(side))
            self.grids.append(np.stack((grid_x, grid_y), axis=2).reshape(-1, 2))
            self.strides.append(np.full((side * side, 1), stride, dtype=np.float32))
        self.grids = np.concatenate(self.grids).astype(np.float32)
        self.strides = np.concatenate(self.strides)

    def detect(self, frame):
        height, width = frame.shape[:2]
        scale = min(640 / width, 640 / height)
        resized_width, resized_height = int(width * scale), int(height * scale)
        resized = cv2.resize(frame, (resized_width, resized_height)).astype(np.float32)
        input_frame = np.full((640, 640, 3), 114, dtype=np.float32)
        input_frame[:resized_height, :resized_width] = resized
        blob = np.transpose(input_frame, (2, 0, 1))[None]
        self.net.setInput(blob)
        output = self.net.forward()[0].copy()
        output[:, :2] = (output[:, :2] + self.grids) * self.strides
        output[:, 2:4] = np.exp(np.clip(output[:, 2:4], -10, 10)) * self.strides
        scores = output[:, 4:5] * output[:, 5:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]
        detections = []
        boxes = np.column_stack((output[:, 0] - output[:, 2] / 2,
                                 output[:, 1] - output[:, 3] / 2,
                                 output[:, 2], output[:, 3]))
        kept_indices = []
        for class_id in np.unique(class_ids[confidences >= 0.35]):
            class_indices = np.flatnonzero((class_ids == class_id) & (confidences >= 0.35))
            keep = cv2.dnn.NMSBoxes(boxes[class_indices].tolist(),
                                    confidences[class_indices].tolist(), 0.35, 0.45)
            kept_indices.extend(class_indices[np.asarray(keep).reshape(-1)].tolist())
        for index in sorted(kept_indices, key=lambda item: confidences[item], reverse=True)[:12]:
            x, y, box_width, box_height = boxes[index]
            x1 = max(0, min(width - 1, int(x / scale)))
            y1 = max(0, min(height - 1, int(y / scale)))
            x2 = max(0, min(width - 1, int((x + box_width) / scale)))
            y2 = max(0, min(height - 1, int((y + box_height) / scale)))
            if x2 > x1 and y2 > y1:
                detections.append({
                    "label": OBJECT_CLASSES[class_ids[index]],
                    "confidence": float(confidences[index]),
                    "box": (x1, y1, x2 - x1, y2 - y1),
                })
        return detections

    def detect_pose(self, frame, box):
        x, y, width, height = box
        side = max(32, int(max(width, height) * 1.25))
        left = int(x + width / 2 - side / 2)
        top = int(y + height / 2 - side / 2)
        crop = np.zeros((side, side, 3), dtype=np.uint8)
        x1, y1 = max(0, left), max(0, top)
        x2, y2 = min(frame.shape[1], left + side), min(frame.shape[0], top + side)
        crop[y1 - top:y2 - top, x1 - left:x2 - left] = frame[y1:y2, x1:x2]
        pose_input = cv2.resize(crop, (256, 256))
        pose_input = cv2.cvtColor(pose_input, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        self.pose_net.setInput(pose_input[None])
        landmarks, confidence, _, _, _ = self.pose_net.forward(
            self.pose_net.getUnconnectedOutLayersNames())
        if float(confidence[0][0]) < 0.5:
            return []
        landmarks = landmarks[0].reshape(-1, 5)
        visibility = 1 / (1 + np.exp(-np.clip(landmarks[:, 3], -30, 30)))
        points = []
        center_x, center_y = left + side / 2, top + side / 2
        for index, name in enumerate(POSE_NAMES):
            px = int((landmarks[index, 0] - 128) * side / 256 + center_x)
            py = int((landmarks[index, 1] - 128) * side / 256 + center_y)
            points.append({"name": name, "x": px, "y": py,
                           "visibility": float(visibility[index])})
        return points


def match_moving_objects(motion_boxes, detections):
    matches = {}
    for index, (x, y, width, height) in enumerate(motion_boxes):
        motion_area = max(1, width * height)
        best, best_score = None, 0.15
        for detection in detections:
            dx, dy, dw, dh = detection["box"]
            intersection_width = max(0, min(x + width, dx + dw) - max(x, dx))
            intersection_height = max(0, min(y + height, dy + dh) - max(y, dy))
            intersection = intersection_width * intersection_height
            overlap = intersection / max(1, min(motion_area, dw * dh))
            if overlap >= best_score:
                best, best_score = detection, overlap
        if best is not None:
            matches[index] = best
    return matches


def select_primary_object(detections, motion_boxes, frame_shape):
    if not detections:
        return None
    moving_ids = {id(detection) for detection in match_moving_objects(motion_boxes, detections).values()}
    frame_area = max(1, frame_shape[0] * frame_shape[1])
    return max(detections, key=lambda detection: (
        detection["confidence"] + (0.5 if id(detection) in moving_ids else 0)
        + 0.1 * (detection["box"][2] * detection["box"][3] / frame_area)))


def body_part_names(points, threshold=0.6):
    return [points[index]["name"] for index in BODY_PART_IDS
            if index < len(points) and points[index]["visibility"] >= threshold]


def draw_pose(frame, points):
    visible = {index: point for index, point in enumerate(points)
               if point["visibility"] >= 0.6
               and 0 <= point["x"] < frame.shape[1]
               and 0 <= point["y"] < frame.shape[0]}
    for first, second in POSE_CONNECTIONS:
        if first in visible and second in visible:
            cv2.line(frame, (visible[first]["x"], visible[first]["y"]),
                     (visible[second]["x"], visible[second]["y"]), (255, 220, 0), 2)
    for index, point in visible.items():
        cv2.circle(frame, (point["x"], point["y"]), 3, (0, 80, 255), -1)
        if index in BODY_PART_IDS:
            cv2.putText(frame, point["name"], (point["x"] + 4, point["y"] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)


def extract_motion_boxes(mask, min_area):
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.dilate(mask, None, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours
            if cv2.contourArea(contour) >= min_area]


def confirmed_motion(history, moving, required_frames):
    history.append(bool(moving))
    return len(history) >= required_frames and all(history)


# --------------------------------------------------------------------------
# Alerts
# --------------------------------------------------------------------------
def send_telegram(path, caption):
    if not (args.telegram_token and args.telegram_chat):
        return
    try:
        with open(path, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{args.telegram_token}/sendPhoto",
                data={"chat_id": args.telegram_chat, "caption": caption},
                files={"photo": f},
                timeout=15,
            )
    except Exception as exc:  # never let an alert failure stop detection
        print(f"[telegram] failed: {exc}")


def record_event(frame, objects, body_parts):
    now = dt.datetime.now()
    name = now.strftime("motion_%Y%m%d_%H%M%S.jpg")
    path = os.path.join(SNAP_DIR, name)
    cv2.imwrite(path, frame)
    event = {"time": now.strftime("%Y-%m-%d %H:%M:%S"), "file": name,
             "objects": objects, "body_parts": body_parts}
    state.events.appendleft(event)
    state.last_motion = event["time"]
    print(f"[motion] {event['time']} ({', '.join(objects) or 'unidentified'}; "
          f"body: {', '.join(body_parts) or 'none'})")
    caption = f"Motion at {event['time']}: {', '.join(objects) or 'unidentified'}"
    if body_parts:
        caption += f"; visible body parts: {', '.join(body_parts)}"
    threading.Thread(target=send_telegram,
                     args=(path, caption),
                     daemon=True).start()


# --------------------------------------------------------------------------
# Detector thread
# --------------------------------------------------------------------------
def detector_loop():
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)
    try:
        object_detector = ObjectDetector()
        state.object_model = "ready"
        print("[objects] 80-class detector and 33-point pose model ready")
    except Exception as exc:
        object_detector = None
        state.object_model = "unavailable"
        print(f"[objects] detector unavailable: {exc}")
    cap = None
    warmup = 0
    phone_mode_active = False
    last_phone_frame_at = 0.0
    last_seen = 0.0
    last_object_detection_at = 0.0
    object_detections = []
    pose_points = []
    primary_detection = None
    motion_history = deque(maxlen=args.confirm_frames)
    frames, t0 = 0, time.time()

    while True:
        if state.phone_mode:
            if not phone_mode_active:
                warmup = 30
                phone_mode_active = True
            if cap is not None:
                cap.release()
                cap = None
            with state.frame_cond:
                if (state.phone_frame is None
                        or state.phone_frame_at == last_phone_frame_at
                        or time.monotonic() - state.phone_frame_at > 2):
                    if (state.phone_frame is None
                            or time.monotonic() - state.phone_frame_at > 2):
                        state.camera_ok = False
                        state.motion = False
                    state.frame_cond.wait(timeout=1)
                    continue
                frame = state.phone_frame.copy()
                last_phone_frame_at = state.phone_frame_at
            state.camera_ok = True
        else:
            if phone_mode_active:
                warmup = 30
                phone_mode_active = False
            if SOURCE == "phone":
                state.camera_ok = False
                state.motion = False
                time.sleep(1)
                continue
            if cap is None or not cap.isOpened():
                state.camera_ok = False
                cap = cv2.VideoCapture(SOURCE)
                if not cap.isOpened():
                    print("[camera] cannot open source, retrying in 3s")
                    time.sleep(3)
                    continue
                warmup = 30            # let the background model settle

            ok, frame = cap.read()
            if not ok:
                print("[camera] lost frame, reconnecting")
                state.camera_ok = False
                cap.release()
                cap = None
                time.sleep(2)
                continue
            state.camera_ok = True

        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))

        mask = bg.apply(cv2.GaussianBlur(frame, (5, 5), 0))
        boxes = []
        if warmup > 0:
            warmup -= 1
        else:
            boxes = extract_motion_boxes(mask, args.min_area)

        now = time.time()
        moving = confirmed_motion(motion_history, bool(boxes) and state.armed,
                                 args.confirm_frames)
        if object_detector and now - last_object_detection_at >= 0.75:
            try:
                object_detections = object_detector.detect(frame)
                primary_detection = select_primary_object(object_detections, boxes, frame.shape)
                pose_points = (object_detector.detect_pose(frame, primary_detection["box"])
                               if primary_detection and primary_detection["label"] == "person"
                               else [])
                state.primary_object = (primary_detection["label"]
                                        if primary_detection else "none")
                state.body_parts = body_part_names(pose_points)
                last_object_detection_at = now
            except Exception as exc:
                object_detector = None
                object_detections = []
                pose_points = []
                primary_detection = None
                state.primary_object = "none"
                state.body_parts = []
                state.object_model = "unavailable"
                print(f"[objects] inference failed: {exc}")
        object_matches = match_moving_objects(boxes, object_detections) if moving else {}
        object_names = sorted({detection["label"] for detection in object_matches.values()})

        shown = frame.copy()
        for detection in object_detections:
            x, y, box_width, box_height = detection["box"]
            focused = detection is primary_detection
            color = (0, 215, 255) if focused else (255, 180, 0)
            thickness = 3 if focused else 1
            cv2.rectangle(shown, (x, y), (x + box_width, y + box_height), color, thickness)
            prefix = "FOCUS: " if focused else ""
            text = f"{prefix}{detection['label']} {detection['confidence']:.0%}"
            cv2.putText(shown, text, (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        if primary_detection and primary_detection["label"] == "person":
            draw_pose(shown, pose_points)
        for index, (x, y, bw, bh) in enumerate(boxes):
            if index not in object_matches:
                cv2.rectangle(shown, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
        label = "MOTION" if state.motion else (
            "VERIFYING" if state.armed and any(motion_history)
            else ("ARMED" if state.armed else "PAUSED"))
        color = (0, 0, 255) if state.motion else (255, 255, 255)
        cv2.putText(shown, f"{label}  {dt.datetime.now():%H:%M:%S}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if moving:
            last_seen = now
            if not state.motion:
                state.motion = True
                record_event(shown, object_names or [state.primary_object], state.body_parts)
        elif state.motion and (now - last_seen > args.cooldown or not state.armed):
            state.motion = False

        ok, buf = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with state.frame_cond:
                state.jpeg = buf.tobytes()
                state.frame_cond.notify_all()

        frames += 1
        if now - t0 >= 2:
            state.fps = frames / (now - t0)
            frames, t0 = 0, now


# --------------------------------------------------------------------------
# Web server
# --------------------------------------------------------------------------
@app.before_request
def require_login():
    if request.endpoint == "health_check":
        return None
    if not args.password:
        return None
    auth = request.authorization
    ok = (auth is not None
          and hmac.compare_digest(auth.username.encode(), args.user.encode())
          and hmac.compare_digest((auth.password or "").encode(), args.password.encode()))
    if not ok:
        return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="Motion System"'})
    return None


@app.route("/healthz")
def health_check():
    return "ok", 200


def mjpeg():
    last = None
    while True:
        with state.frame_cond:
            state.frame_cond.wait(timeout=5)
            jpeg = state.jpeg
        if jpeg is None or jpeg is last:
            continue
        last = jpeg
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


@app.route("/video")
def video():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(motion=state.motion, armed=state.armed, camera_ok=state.camera_ok,
                   fps=round(state.fps, 1), last_motion=state.last_motion,
                   event_count=len(state.events), phone_mode=state.phone_mode,
                   object_model=state.object_model, primary_object=state.primary_object,
                   body_parts=state.body_parts)


@app.route("/api/camera/connect", methods=["POST"])
def api_camera_connect():
    with state.frame_cond:
        state.phone_mode = True
        state.phone_frame = None
        state.phone_frame_at = 0.0
        state.motion = False
        state.camera_ok = False
        state.primary_object = "none"
        state.body_parts = []
        state.frame_cond.notify_all()
    return jsonify(connected=True)


@app.route("/api/camera/disconnect", methods=["POST"])
def api_camera_disconnect():
    with state.frame_cond:
        state.phone_mode = False
        state.phone_frame = None
        state.phone_frame_at = 0.0
        state.motion = False
        state.camera_ok = False
        state.primary_object = "none"
        state.body_parts = []
        state.frame_cond.notify_all()
    return jsonify(connected=False)


@app.route("/api/camera/frame", methods=["POST"])
def api_camera_frame():
    if not state.phone_mode:
        return jsonify(error="Connect the phone camera before sending frames."), 409
    data = request.get_data(cache=False)
    if not data:
        return jsonify(error="No image data received."), 400
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="The uploaded frame is not a valid image."), 400
    height, width = frame.shape[:2]
    if width > 1920 or height > 1080:
        return jsonify(error="Camera frames must be at most 1920x1080."), 413
    with state.frame_cond:
        state.phone_frame = frame
        state.phone_frame_at = time.monotonic()
        state.frame_cond.notify_all()
    return jsonify(received=True)


@app.route("/api/events")
def api_events():
    return jsonify(list(state.events))


@app.route("/api/arm", methods=["POST"])
def api_arm():
    data = request.get_json(silent=True) or {}
    state.armed = bool(data.get("armed", not state.armed))
    if not state.armed:
        state.motion = False
    return jsonify(armed=state.armed)


@app.route("/snapshots/<path:name>")
def snapshot(name):
    if not name.endswith(".jpg"):
        abort(404)
    return send_from_directory(SNAP_DIR, name)


@app.route("/")
def index():
    return Response(DASHBOARD, mimetype="text/html")


@app.route("/qr.png")
def dashboard_qr():
    url = public_url()
    image = qrcode.make(url)
    output = BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    return send_file(output, mimetype="image/png", max_age=0)


DASHBOARD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Motion System</title>
<style>
  :root { --bg:#0f1115; --card:#181b22; --text:#e8eaed; --muted:#8b93a1; --ok:#2ecc71; --alert:#ff4d4f; --warn:#f5a623; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font-family: system-ui, -apple-system, sans-serif; }
  main { max-width: 720px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 18px; margin: 4px 0 12px; }
  .card { background: var(--card); border-radius: 14px; padding: 12px; margin-bottom: 14px; }
  .video { width:100%; border-radius: 10px; display:block; background:#000; min-height: 180px; }
  .row { display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; }
  .badge { padding: 6px 12px; border-radius: 999px; font-weight:600; font-size: 14px; background:#2a2f3a; }
  .badge.armed { background: rgba(46,204,113,.18); color: var(--ok); }
  .badge.motion { background: rgba(255,77,79,.2); color: var(--alert); animation: pulse 1s infinite; }
  .badge.paused { background: rgba(245,166,35,.18); color: var(--warn); }
  .badge.offline { background: #3a2a2a; color: var(--alert); }
  @keyframes pulse { 50% { opacity:.5; } }
  button { background:#2b6ef2; color:#fff; border:0; border-radius:10px; padding:10px 16px; font-size:15px; font-weight:600; }
  button.secondary { background:#2a2f3a; }
  .meta { color: var(--muted); font-size: 13px; margin-top: 8px; }
  .events { display:grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr)); gap:10px; }
  .events figure { margin:0; }
  .events img { width:100%; border-radius:8px; display:block; }
  .events figcaption { font-size:12px; color:var(--muted); margin-top:4px; }
  .empty { color: var(--muted); font-size: 14px; }
  .camera-note { color: var(--muted); font-size: 13px; margin-top: 8px; }
  .qr { width:180px; height:180px; background:#fff; border-radius:8px; padding:8px; }
</style>
</head>
<body>
<main>
  <h1>Motion System</h1>

  <div class="card">
    <div class="row">
      <div>
        <strong>Open on your phone</strong>
        <div class="camera-note">Scan this code while connected to the same Wi-Fi.</div>
      </div>
      <img class="qr" src="/qr.png" alt="QR code to open Motion System on your phone">
    </div>
    <div class="camera-note">Connect a webcam below. Motion is detected automatically and snapshots appear in Recent motion. Camera access requires HTTPS or localhost.</div>
  </div>

  <div class="card">
    <img class="video" src="/video" alt="Live camera">
    <div class="row" style="margin-top:12px">
      <span id="badge" class="badge">Connecting...</span>
      <div class="row">
        <button id="phoneBtn" class="secondary">Connect webcam</button>
        <button id="armBtn" class="secondary">Pause</button>
      </div>
    </div>
    <div class="meta" id="meta"></div>
    <div class="camera-note" id="cameraNote">Webcam access requires permission and a secure page (HTTPS or localhost).</div>
    <video id="phoneVideo" autoplay playsinline muted hidden></video>
  </div>

  <div class="card">
    <div class="row"><strong>Recent motion</strong></div>
    <div id="events" class="events" style="margin-top:10px"></div>
  </div>
</main>

<script>
let armed = true, wasMotion = false, lastCount = -1;
const badge = document.getElementById('badge');
const meta = document.getElementById('meta');
const armBtn = document.getElementById('armBtn');
const phoneBtn = document.getElementById('phoneBtn');
const phoneVideo = document.getElementById('phoneVideo');
const cameraNote = document.getElementById('cameraNote');
const eventsEl = document.getElementById('events');
let phoneStream = null, sendingFrames = false;

async function refresh() {
  try {
    const s = await (await fetch('/api/status')).json();
    armed = s.armed;
    if (!s.camera_ok) { badge.className = 'badge offline'; badge.textContent = 'Camera offline'; }
    else if (!s.armed) { badge.className = 'badge paused'; badge.textContent = 'Paused'; }
    else if (s.motion) { badge.className = 'badge motion'; badge.textContent = 'MOTION DETECTED'; }
    else { badge.className = 'badge armed'; badge.textContent = 'Armed - all quiet'; }
    armBtn.textContent = s.armed ? 'Pause' : 'Resume';
    if (!phoneStream && !s.phone_mode) phoneBtn.textContent = 'Connect webcam';
        const bodyParts = (s.body_parts || []).join(', ');
        meta.textContent = 'Last motion: ' + (s.last_motion || 'none yet') + '  |  ' + s.fps +
            ' fps  |  Model: ' + s.object_model + '  |  Focus: ' + (s.primary_object || 'none') +
            (bodyParts ? '  |  Body: ' + bodyParts : '');

    if (s.motion && !wasMotion) {
      if (navigator.vibrate) navigator.vibrate([250, 120, 250]);
      document.title = '! Motion detected';
    } else if (!s.motion) { document.title = 'Motion System'; }
    wasMotion = s.motion;

    if (s.event_count !== lastCount || s.motion) { lastCount = s.event_count; loadEvents(); }
  } catch (e) {
    badge.className = 'badge offline'; badge.textContent = 'Server unreachable';
  }
}

async function sendPhoneFrames() {
  if (sendingFrames) return;
  sendingFrames = true;
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  try {
    while (phoneStream) {
      if (phoneVideo.readyState >= 2) {
        const scale = Math.min(1, 640 / phoneVideo.videoWidth);
        canvas.width = Math.round(phoneVideo.videoWidth * scale);
        canvas.height = Math.round(phoneVideo.videoHeight * scale);
        context.drawImage(phoneVideo, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.75));
        if (blob) {
          const response = await fetch('/api/camera/frame', { method: 'POST', body: blob });
          if (!response.ok) throw new Error('Server rejected a camera frame (' + response.status + ').');
        }
      }
      await new Promise(resolve => setTimeout(resolve, 200));
    }
  } catch (e) {
    cameraNote.textContent = e.message;
    await stopPhoneCamera();
  } finally {
    sendingFrames = false;
  }
}

async function stopPhoneCamera() {
  if (phoneStream) phoneStream.getTracks().forEach(track => track.stop());
  phoneStream = null;
  phoneVideo.srcObject = null;
  phoneBtn.textContent = 'Connect webcam';
  try { await fetch('/api/camera/disconnect', { method: 'POST' }); }
  catch (e) { cameraNote.textContent = 'Could not notify the server that the camera disconnected.'; }
}

phoneBtn.onclick = async () => {
  phoneBtn.disabled = true;
  try {
    if (phoneStream) {
      await stopPhoneCamera();
      cameraNote.textContent = 'Phone camera disconnected.';
      return;
    }
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Phone camera access requires HTTPS (or localhost) and a supported browser.');
    }
    const connected = await fetch('/api/camera/connect', { method: 'POST' });
    if (!connected.ok) throw new Error('Could not connect the phone camera to the server.');
    phoneStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 } },
      audio: false
    });
    phoneVideo.srcObject = phoneStream;
    await phoneVideo.play();
    phoneBtn.textContent = 'Disconnect webcam';
    cameraNote.textContent = 'Webcam connected. Motion detection is active.';
    sendPhoneFrames();
  } catch (e) {
    await stopPhoneCamera();
    cameraNote.textContent = e.message || 'Could not access the phone camera.';
  } finally {
    phoneBtn.disabled = false;
  }
};

async function loadEvents() {
  const list = await (await fetch('/api/events')).json();
  if (!list.length) { eventsEl.innerHTML = '<div class="empty">No motion recorded yet.</div>'; return; }
    eventsEl.innerHTML = list.map(e => {
        const objects = (e.objects || []).join(', ') || 'unidentified object';
        const bodyParts = (e.body_parts || []).join(', ');
        return '<figure><a href="/snapshots/' + e.file + '" target="_blank"><img loading="lazy" src="/snapshots/' + e.file +
            '"></a><figcaption>' + e.time + ' - ' + objects +
            (bodyParts ? ' - ' + bodyParts : '') + '</figcaption></figure>';
    }).join('');
}

armBtn.onclick = async () => {
  await fetch('/api/arm', { method: 'POST', headers: {'Content-Type':'application/json'},
                            body: JSON.stringify({ armed: !armed }) });
  refresh();
};

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def public_url():
    external_url = os.environ.get("RENDER_EXTERNAL_URL")
    if external_url:
        return external_url.rstrip("/")
    scheme = "https" if args.cert else "http"
    return f"{scheme}://{local_ip()}:{args.port}"


if __name__ == "__main__":
    threading.Thread(target=detector_loop, daemon=True).start()
    url = public_url()
    print("\n  Motion system running.")
    print(f"  Open this on your phone (same Wi-Fi): {url}")
    if not args.password:
        print("  WARNING: no --password set, anyone on your network can view the camera.")
    print()
    ssl_context = (args.cert, args.key) if args.cert else None
    app.run(host="0.0.0.0", port=args.port, threaded=True, ssl_context=ssl_context)