"""Camera-first runtime used by the Render/Gunicorn entry point.

Keep camera capture and MJPEG output responsive while the optional object models
are downloaded and initialized in the background. The original runtime waited
for model startup before consuming any browser-uploaded frames, leaving the
camera marked offline and the video panel black during that time.
"""
import threading
import time
from collections import deque

import cv2

import app as motion


def _load_detector(result):
    try:
        result["detector"] = motion.ObjectDetector()
        motion.state.object_model = "ready"
        print("[objects] detector ready")
    except Exception as exc:
        result["failed"] = True
        motion.state.object_model = "unavailable"
        print(f"[objects] detector unavailable: {exc}")


def detector_loop():
    state = motion.state
    bg = cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=40, detectShadows=True)
    detector_result = {}
    state.object_model = "initializing"
    threading.Thread(target=_load_detector, args=(detector_result,), daemon=True).start()

    cap = None
    phone_mode_active = False
    last_phone_frame_at = 0.0
    last_seen = 0.0
    last_detection_at = 0.0
    detections = []
    pose_points = []
    primary = None
    history = deque(maxlen=motion.args.confirm_frames)
    warmup = 0
    frames, interval_start = 0, time.time()

    while True:
        if state.phone_mode:
            if not phone_mode_active:
                warmup = 30
                phone_mode_active = True
            if cap is not None:
                cap.release()
                cap = None
            with state.frame_cond:
                stale = (state.phone_frame is None or
                         time.monotonic() - state.phone_frame_at > 3)
                if stale or state.phone_frame_at == last_phone_frame_at:
                    if stale:
                        state.camera_ok = False
                        state.motion = False
                        state.primary_object = "none"
                        state.body_parts = []
                    state.frame_cond.wait(timeout=0.5)
                    continue
                frame = state.phone_frame.copy()
                last_phone_frame_at = state.phone_frame_at
            state.camera_ok = True
        else:
            if phone_mode_active:
                warmup = 30
                phone_mode_active = False
            if motion.SOURCE == "phone":
                state.camera_ok = False
                state.motion = False
                time.sleep(0.5)
                continue
            if cap is None or not cap.isOpened():
                state.camera_ok = False
                cap = cv2.VideoCapture(motion.SOURCE)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    time.sleep(2)
                    continue
                warmup = 30
            ok, frame = cap.read()
            if not ok:
                state.camera_ok = False
                cap.release()
                cap = None
                time.sleep(1)
                continue
            state.camera_ok = True

        height, width = frame.shape[:2]
        if width > 640:
            frame = cv2.resize(frame, (640, max(1, int(height * 640 / width))))

        mask = bg.apply(cv2.GaussianBlur(frame, (5, 5), 0))
        boxes = []
        if warmup:
            warmup -= 1
        else:
            boxes = motion.extract_motion_boxes(mask, motion.args.min_area)

        now = time.time()
        moving = motion.confirmed_motion(
            history, bool(boxes) and state.armed, motion.args.confirm_frames)
        detector = detector_result.get("detector")
        if detector and now - last_detection_at >= 0.75:
            try:
                detections = detector.detect(frame)
                primary = motion.select_primary_object(detections, boxes, frame.shape)
                pose_points = (detector.detect_pose(frame, primary["box"])
                               if primary and primary["label"] == "person" else [])
                state.primary_object = primary["label"] if primary else "none"
                state.body_parts = motion.body_part_names(pose_points)
                last_detection_at = now
            except Exception as exc:
                detector_result["failed"] = True
                detector_result.pop("detector", None)
                detections, pose_points, primary = [], [], None
                state.primary_object = "none"
                state.body_parts = []
                state.object_model = "unavailable"
                print(f"[objects] inference failed: {exc}")

        matches = motion.match_moving_objects(boxes, detections) if moving else {}
        object_names = sorted({item["label"] for item in matches.values()})
        shown = frame.copy()
        for item in detections:
            x, y, box_width, box_height = item["box"]
            focused = item is primary
            color = (0, 215, 255) if focused else (255, 180, 0)
            cv2.rectangle(shown, (x, y), (x + box_width, y + box_height),
                          color, 3 if focused else 1)
            text = ("FOCUS: " if focused else "") + f"{item['label']} {item['confidence']:.0%}"
            cv2.putText(shown, text, (x, max(18, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if primary and primary["label"] == "person":
            motion.draw_pose(shown, pose_points)
        for index, (x, y, box_width, box_height) in enumerate(boxes):
            if index not in matches:
                cv2.rectangle(shown, (x, y), (x + box_width, y + box_height),
                              (0, 255, 0), 1)

        if moving:
            last_seen = now
            if not state.motion:
                state.motion = True
                motion.record_event(
                    shown, object_names or [state.primary_object], state.body_parts)
        elif state.motion and (now - last_seen > motion.args.cooldown or not state.armed):
            state.motion = False

        label = "MOTION" if state.motion else (
            "VERIFYING" if state.armed and any(history)
            else ("ARMED" if state.armed else "PAUSED"))
        color = (0, 0, 255) if state.motion else (255, 255, 255)
        cv2.putText(shown, f"{label}  {time.strftime('%H:%M:%S')}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        ok, encoded = cv2.imencode(
            ".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with state.frame_cond:
                state.jpeg = encoded.tobytes()
                state.frame_cond.notify_all()

        frames += 1
        if now - interval_start >= 2:
            state.fps = frames / (now - interval_start)
            frames, interval_start = 0, now
