import sounddevice as sd
import numpy as np
import threading
import queue
import time

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_size=512, vad_threshold=0.5):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad_threshold = vad_threshold
        
        self.is_recording = False
        self.is_paused = False
        # Keep queue bounded so callback never grows memory unbounded.
        self.audio_queue = queue.Queue(maxsize=256)
        self.speech_buffer = []
        
        # Audio input device index (default system input)
        self.device = None
        
        self.vad = None # VAD instance
        self.stream = None
        self.overflow_warnings = 0
        
    def set_vad(self, vad_instance):
        self.vad = vad_instance

    def start_recording(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.is_paused = False
        self.stop_event = threading.Event()
        
        def audio_callback(indata, frames, time, status):
            if status:
                if status.input_overflow:
                    self.overflow_warnings += 1
                    if self.overflow_warnings % 20 == 1:
                        print("Warning: audio input overflow detected; dropping old chunks.")
                else:
                    print(status)

            # Drop input while paused (muted/system speaking) to avoid stale backlog.
            if self.is_paused:
                return

            try:
                self.audio_queue.put_nowait(indata.copy())
            except queue.Full:
                # Drop oldest chunk and keep newest one to limit latency.
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.audio_queue.put_nowait(indata.copy())
                except queue.Full:
                    # If callback is still behind, just skip this chunk.
                    pass
            
        self.stream = sd.InputStream(samplerate=self.sample_rate, 
                                     blocksize=self.chunk_size, 
                                     channels=1, 
                                     dtype='float32',
                                     callback=audio_callback)
        self.stream.start()
        print("Recording started...")

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("Recording stopped.")

    def pause(self):
        self.is_paused = True
        # print("Recording paused (VAD ignored).")

    def resume(self):
        self.is_paused = False
        # Clear queue to avoid processing old audio
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        # print("Recording resumed.")

    def get_audio_chunk(self):
        try:
            return self.audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def process_audio_stream(self, on_speech_start=None, on_speech_end=None):
        """
        Generator that yields detected speech segments as numpy arrays.
        """
        speech_frames = []
        is_speech_active = False
        silence_frames_count = 0
        min_speech_frames = 10 # Minimum speech duration (frames)
        max_silence_frames = 20 # Max silence duration to consider end of speech (frames)
        
        while self.is_recording:
            chunk = self.get_audio_chunk()
            if chunk is None:
                continue
            
            # If paused, consume chunks but do not process VAD
            if self.is_paused:
                # Reset state if paused
                speech_frames = []
                is_speech_active = False
                silence_frames_count = 0
                continue
                
            chunk = chunk.flatten()
            
            # VAD check
            is_speech = False
            if self.vad:
                is_speech, _ = self.vad.process_chunk(chunk)
            
            if is_speech:
                if not is_speech_active:
                    is_speech_active = True
                    if on_speech_start:
                        on_speech_start()
                    speech_frames = []
                
                speech_frames.append(chunk)
                silence_frames_count = 0
            
            elif is_speech_active:
                # Silence detected during speech
                speech_frames.append(chunk) # Include silence at end
                silence_frames_count += 1
                
                if silence_frames_count > max_silence_frames:
                    is_speech_active = False
                    if len(speech_frames) > min_speech_frames:
                        full_audio = np.concatenate(speech_frames)
                        if on_speech_end:
                            on_speech_end()
                        yield full_audio
                    speech_frames = []
                    silence_frames_count = 0
