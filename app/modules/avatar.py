import cv2
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import os
import torch
from app.modules.musetalk_wrapper import MuseTalkAvatar

class Avatar:
    def __init__(self, 
                 video_path="assets/vlad_speaking.mp4", 
                 avatar_id="vlad_speaking", 
                 display_width=512, 
                 display_height=512):
        
        self.video_path = video_path
        self.avatar_id = avatar_id
        self.display_width = display_width
        self.display_height = display_height
        
        # Audio processing queue (input from TTS)
        self.audio_queue = queue.Queue()
        # Playback queue (output to screen/speaker)
        self.playback_queue = queue.Queue()
        
        self.is_running = False
        self.is_talking = False
        self.window_name = "Avatar Kiosk"
        self.avatar_fps = int(os.getenv("AVATAR_FPS", "25"))
        self.musetalk_batch_size = int(os.getenv("MUSE_BATCH_SIZE", "4"))
        
        # Initialize MuseTalk
        # We assume MuseTalk folder structure is present
        print("Initializing MuseTalk Avatar...")
        try:
             self.musetalk = MuseTalkAvatar(
                 avatar_id=self.avatar_id,
                 video_path=self.video_path,
                 bbox_shift=0,
                 batch_size=self.musetalk_batch_size,
                 preparation=False, # Set to True manually if needed first time
                 version="v15",
                 device="cuda"
             )
        except Exception as e:
             print(f"Failed to initialize MuseTalk: {e}")
             self.musetalk = None

    def start(self):
        self.is_running = True
        
        # Thread for running inference
        self.proc_thread = threading.Thread(target=self._processing_loop)
        self.proc_thread.start()
        
        # Thread for rendering/playback
        self.render_thread = threading.Thread(target=self._render_loop)
        self.render_thread.start()

    def stop(self):
        self.is_running = False
        if hasattr(self, 'proc_thread') and self.proc_thread.is_alive():
            self.proc_thread.join()
        if hasattr(self, 'render_thread') and self.render_thread.is_alive():
            self.render_thread.join()
        cv2.destroyAllWindows()

    def set_talking(self, talking):
        # This is mainly for external status, 
        # actual talking state is derived from queue activity
        self.is_talking = talking
        # If we start talking, pause recorder immediately to prevent self-listening
        if hasattr(self, 'on_mute_toggle'):
             # We reuse the mute callback logic to pause/resume recorder
             # But we need to distinguish between user mute and system mute (talking)
             # Ideally main.py should handle this via on_speech_start/stop
             pass

    def push_audio(self, audio_chunk, sr):
        """
        Push audio chunk for lip sync processing.
        audio_chunk: numpy array or torch tensor
        sr: sample rate
        """
        if isinstance(audio_chunk, torch.Tensor):
            audio_chunk = audio_chunk.cpu().numpy()
            
        self.audio_queue.put((audio_chunk, sr))

    def _processing_loop(self):
        print("Avatar processing loop started.")
        while self.is_running:
            try:
                audio_chunk, sr = self.audio_queue.get(timeout=0.1)
                
                if self.musetalk:
                    # Run inference
                    # MuseTalk expects 16k usually, we might need to resample if sr != 16000
                    # For now assuming 16k or handling inside wrapper?
                    # MuseTalkWrapper saves to wav which librosa reads as 16k usually if not specified?
                    # Wait, sf.write writes with provided sr. AudioProcessor reads with librosa(sr=16000).
                    # So resampling is handled by librosa load inside AudioProcessor.
                    
                    frames = self.musetalk.infer_audio_chunk(audio_chunk, sr=sr, fps=self.avatar_fps)
                    self.playback_queue.put((audio_chunk, sr, frames))
                else:
                    # Fallback: no frames, just pass audio
                    self.playback_queue.put((audio_chunk, sr, []))
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in avatar processing: {e}")

    def _render_loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Use initial video dimensions or defaults if not set properly
        width = self.display_width
        height = self.display_height
        
        # If we have frame info from MuseTalk, respect aspect ratio
        if self.musetalk and hasattr(self.musetalk, 'frame_list_cycle') and len(self.musetalk.frame_list_cycle) > 0:
             sample_frame = self.musetalk.frame_list_cycle[0]
             h, w = sample_frame.shape[:2]
             width, height = w, h
             
        cv2.resizeWindow(self.window_name, width, height)
        
        print("Avatar render loop started.")
        print("Controls: 'm' to toggle mute/unmute microphone, 'ESC' to exit.")
        
        # Default idle frame
        idle_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(idle_frame, "IDLE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.musetalk and hasattr(self.musetalk, 'frame_list_cycle') and len(self.musetalk.frame_list_cycle) > 0:
             # Use first frame of video as idle
             idle_frame = self.musetalk.frame_list_cycle[0]

        is_muted = False

        while self.is_running:
            try:
                # Check keyboard input first for responsiveness
                key = cv2.waitKey(1) & 0xFF
                if key == 27: # ESC
                    self.is_running = False
                    break
                elif key == ord('m'):
                    is_muted = not is_muted
                    print(f"Microphone {'MUTED' if is_muted else 'UNMUTED'}")
                    # Notify external listener (Kiosk) if callback is set
                    if hasattr(self, 'on_mute_toggle'):
                        self.on_mute_toggle(is_muted)

                # Wait for next playback item
                try:
                    audio_chunk, sr, frames = self.playback_queue.get(timeout=0.05)
                except queue.Empty:
                    # Show idle frame
                    display_frame = idle_frame.copy()
                    if is_muted:
                        cv2.putText(display_frame, "MIC MUTED", (20, height - 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    cv2.imshow(self.window_name, display_frame)
                    continue

                # Play audio
                sd.play(audio_chunk, sr)
                
                # Show frames
                if frames:
                    frame_duration = 1.0 / float(max(1, self.avatar_fps))
                    start_time = time.time()
                    
                    for i, frame in enumerate(frames):
                        display_frame = frame.copy()
                        if is_muted:
                            cv2.putText(display_frame, "MIC MUTED", (20, height - 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        cv2.imshow(self.window_name, display_frame)
                        
                        # Handle keys during playback too
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            self.is_running = False
                            break
                        elif key == ord('m'):
                            is_muted = not is_muted
                            print(f"Microphone {'MUTED' if is_muted else 'UNMUTED'}")
                            if hasattr(self, 'on_mute_toggle'):
                                self.on_mute_toggle(is_muted)
                        
                        # Sync logic
                        elapsed = time.time() - start_time
                        expected = (i + 1) * frame_duration
                        sleep_time = expected - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        
                        if not self.is_running: break
                else:
                    # No frames (fallback), just wait for audio duration
                    duration = len(audio_chunk) / sr
                    start_wait = time.time()
                    while time.time() - start_wait < duration:
                        # Keep UI responsive
                        key = cv2.waitKey(10) & 0xFF
                        if key == 27:
                            self.is_running = False
                            break
                        elif key == ord('m'):
                            is_muted = not is_muted
                            print(f"Microphone {'MUTED' if is_muted else 'UNMUTED'}")
                            if hasattr(self, 'on_mute_toggle'):
                                self.on_mute_toggle(is_muted)
                        
                        # Update mute display if needed
                        if is_muted:
                            display_frame = idle_frame.copy()
                            cv2.putText(display_frame, "MIC MUTED", (20, height - 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.imshow(self.window_name, display_frame)
                        else:
                             # If we were showing mute text, clear it (show idle)
                             # But we are in a loop where we don't redraw unless changed
                             # Actually we should redraw idle frame if we just unmuted?
                             # Or just keep showing current window content (idle frame usually)
                             # Let's just redraw idle frame to be safe/responsive
                             cv2.imshow(self.window_name, idle_frame)
                             
                    # time.sleep(duration) # Replaced by loop
                
                sd.wait() # Ensure audio finished before next chunk? 
                
            except Exception as e:
                print(f"Error in render loop: {e}")
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test block
    avatar = Avatar()
    avatar.start()
    time.sleep(10)
    avatar.stop()
