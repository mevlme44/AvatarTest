import cv2
import threading
import queue
import time
import numpy as np

class Avatar:
    def __init__(self, idle_video_path="assets/idle_loop.mp4", talking_video_path="assets/talking_loop.mp4", display_width=800, display_height=600):
        self.idle_video_path = idle_video_path
        self.talking_video_path = talking_video_path
        self.display_width = display_width
        self.display_height = display_height
        
        self.cap_idle = None
        self.cap_talking = None
        
        self.is_running = False
        self.is_talking = False
        
        # Audio queue for lip sync processing (placeholder)
        self.audio_queue = queue.Queue()
        
        # Create a window
        self.window_name = "Avatar Kiosk"
        
        # Initialize videos
        self._load_videos()

    def _load_videos(self):
        # We assume videos exist or we create dummy frames
        pass # Actual loading in run loop or init

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._render_loop)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join()
        cv2.destroyAllWindows()

    def set_talking(self, talking):
        self.is_talking = talking

    def push_audio(self, audio_chunk):
        """
        Push audio chunk for lip sync processing.
        """
        self.audio_queue.put(audio_chunk)

    def _render_loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        
        # Placeholder for actual video playback
        # We will cycle through frames
        
        frame_idx = 0
        while self.is_running:
            # Determine which frame to show
            if self.is_talking:
                # Show talking animation (or generate via MuseTalk)
                # For now, just a placeholder color change or text
                frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                frame[:] = (0, 255, 0) # Green for talking
                cv2.putText(frame, "TALKING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                # Show idle animation
                frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                frame[:] = (0, 0, 255) # Red for idle
                cv2.putText(frame, "IDLE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow(self.window_name, frame)
            
            key = cv2.waitKey(30) # ~30fps
            if key == 27: # ESC
                self.is_running = False
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test block
    avatar = Avatar()
    avatar.start()
    time.sleep(2)
    avatar.set_talking(True)
    time.sleep(2)
    avatar.set_talking(False)
    time.sleep(2)
    avatar.stop()
