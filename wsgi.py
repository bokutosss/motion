"""WSGI entry point for the Render/Gunicorn deployment."""
import threading

import app as motion
from camera_runtime import detector_loop

# The frame endpoint caps uploads at 1920x1080. Scaling only by width makes
# portrait phone video exceed that height (for example 640x1138), causing
# uploads to be rejected and the camera to appear offline.
motion.DASHBOARD = motion.DASHBOARD.replace(
    "Math.min(1, 640 / phoneVideo.videoWidth)",
    "Math.min(1, 640 / phoneVideo.videoWidth, 1080 / phoneVideo.videoHeight)",
)

app = motion.app
threading.Thread(target=detector_loop, daemon=True).start()
