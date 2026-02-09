import sys
import os

# Add project root to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main import Kiosk

if __name__ == "__main__":
    kiosk = Kiosk()
    try:
        kiosk.start()
    except KeyboardInterrupt:
        kiosk.stop()
