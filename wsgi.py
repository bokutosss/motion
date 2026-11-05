import threading

from app import app, detector_loop

threading.Thread(target=detector_loop, daemon=True).start()
