import threading

import app as motion
from app import app
from camera_runtime import detector_loop


# Keep portrait phone frames valid before upload so the camera stays online.
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
        canvas.width = Math.round(videoWidth * scale);
        canvas.height = Math.round(videoHeight * scale);"""
if old_frame_scaling not in motion.DASHBOARD:
    raise RuntimeError("Could not apply phone-camera frame sizing fix")
motion.DASHBOARD = motion.DASHBOARD.replace(old_frame_scaling, new_frame_scaling, 1)

threading.Thread(target=detector_loop, daemon=True).start()
