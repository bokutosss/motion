"""WSGI entry point for the Render/Gunicorn deployment."""
import threading

import app as motion
from camera_runtime import detector_loop

# The frame endpoint caps uploads at 1920x1080. Scale portrait phone video by
# both dimensions so frames are not rejected and the camera marked offline.
motion.DASHBOARD = motion.DASHBOARD.replace(
    "Math.min(1, 640 / phoneVideo.videoWidth)",
    "Math.min(1, 640 / phoneVideo.videoWidth, 1080 / phoneVideo.videoHeight)",
)

# Keep the server-processed MJPEG feed as the default output, but show the
# browser's live camera preview immediately when a phone/webcam is connected.
# This makes it clear that camera capture works even if server processing or
# MJPEG delivery is delayed, and the user still sees the server feed again
# after disconnecting.
motion.DASHBOARD = motion.DASHBOARD.replace(
    '<img class="video" src="/video" alt="Live camera">',
    '<img id="serverVideo" class="video" src="/video" alt="Processed camera output">\n'
    '    <video id="livePreview" class="video" autoplay playsinline muted hidden></video>',
)
motion.DASHBOARD = motion.DASHBOARD.replace(
    "const phoneVideo = document.getElementById('phoneVideo');",
    "const phoneVideo = document.getElementById('phoneVideo');\n"
    "const livePreview = document.getElementById('livePreview');\n"
    "const serverVideo = document.getElementById('serverVideo');",
)
motion.DASHBOARD = motion.DASHBOARD.replace(
    "phoneVideo.srcObject = phoneStream;\n    await phoneVideo.play();",
    "phoneVideo.srcObject = phoneStream;\n"
    "    livePreview.srcObject = phoneStream;\n"
    "    serverVideo.hidden = true;\n"
    "    livePreview.hidden = false;\n"
    "    await phoneVideo.play();\n"
    "    await livePreview.play();",
)
motion.DASHBOARD = motion.DASHBOARD.replace(
    "phoneVideo.srcObject = null;\n  phoneBtn.textContent = 'Connect webcam';",
    "phoneVideo.srcObject = null;\n"
    "  livePreview.pause();\n"
    "  livePreview.srcObject = null;\n"
    "  livePreview.hidden = true;\n"
    "  serverVideo.hidden = false;\n"
    "  phoneBtn.textContent = 'Connect webcam';",
)

app = motion.app
threading.Thread(target=detector_loop, daemon=True).start()
