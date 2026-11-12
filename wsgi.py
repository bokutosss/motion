import threading

import app as motion
from app import app
from camera_runtime import detector_loop
from flask import request


# Scale both portrait and landscape video before upload. The server accepts
# frames up to 1920x1080, so portrait frames must be constrained by height too.
old_frame_scaling = """        const scale = Math.min(1, 640 / phoneVideo.videoWidth);
        canvas.width = Math.round(phoneVideo.videoWidth * scale);
        canvas.height = Math.round(phoneVideo.videoHeight * scale);"""
new_frame_scaling = """        const videoWidth = phoneVideo.videoWidth;
        const videoHeight = phoneVideo.videoHeight;
        if (!videoWidth || !videoHeight) {
          await new Promise(resolve => setTimeout(resolve, 200));
          continue;
        }
        const scale = Math.min(1, 640 / videoWidth, 1080 / videoHeight);
        canvas.width = Math.max(1, Math.round(videoWidth * scale));
        canvas.height = Math.max(1, Math.round(videoHeight * scale));"""
if old_frame_scaling not in motion.DASHBOARD:
    raise RuntimeError("Could not apply phone-camera frame sizing fix")
motion.DASHBOARD = motion.DASHBOARD.replace(old_frame_scaling, new_frame_scaling, 1)


@app.after_request
def mark_phone_camera_online(response):
    # Reflect a successfully received browser-camera frame immediately, even
    # while the detector loop is busy processing a previous frame.
    if request.endpoint == "api_camera_frame" and response.status_code == 200:
        with motion.state.frame_cond:
            if motion.state.phone_mode:
                motion.state.camera_ok = True
    return response


threading.Thread(target=detector_loop, daemon=True).start()
