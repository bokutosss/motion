import threading

from app import app
from camera_runtime import detector_loop

threading.Thread(target=detector_loop, daemon=True).start()
