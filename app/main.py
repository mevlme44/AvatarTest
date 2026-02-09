import threading
import queue
import time
import sys
import os
import numpy as np
import torch
from app.modules.vad import VAD
from app.modules.stt import STT
from app.modules.llm import LLM
from app.modules.tts import StreamTTS
from app.modules.avatar import Avatar
from app.utils.audio import AudioRecorder

class Kiosk:
    def __init__(self):
        # Initialize modules
        self.vad = VAD(threshold=0.5)
        self.recorder = AudioRecorder(sample_rate=16000)
        self.recorder.set_vad(self.vad)
        
        # Initialize STT
        self.stt = STT(model_size="large-v3-turbo", device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize LLM
        model_path = "models/llm/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        if not os.path.exists(model_path):
             print(f"Warning: LLM model not found at {model_path}. Using mock LLM.")
             self.llm = None
        else:
             self.llm = LLM(model_path=model_path, n_gpu_layers=-1)
             
        # Initialize TTS (Stream)
        self.tts = StreamTTS(device="cpu") 
        
        # Initialize Avatar
        self.avatar = Avatar()
        
        # Link callbacks
        self.tts.set_callbacks(on_start=self._on_speech_start, on_stop=self._on_speech_stop)
        
        self.is_running = False
        self.processing_thread = None

    def _on_speech_start(self):
        self.recorder.pause()
        self.avatar.set_talking(True)

    def _on_speech_stop(self):
        self.avatar.set_talking(False)
        self.recorder.resume()

    def start(self):
        self.is_running = True
        
        # Start Avatar UI
        self.avatar.start()
        
        # Start Audio Recorder
        self.recorder.start_recording()
        
        # Start Processing Loop
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        print("Kiosk started. Listening...")
        
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.is_running = False
        self.recorder.stop_recording()
        self.avatar.stop()
        if self.processing_thread:
            self.processing_thread.join()
        if hasattr(self.tts, 'stop'):
            self.tts.stop()
        print("Kiosk stopped.")

    def _processing_loop(self):
        """
        Main loop:
        1. Wait for speech segment
        2. Transcribe
        3. Stream to LLM -> TTS -> Avatar
        """
        # Generator loop blocks here waiting for audio
        for audio_segment in self.recorder.process_audio_stream():
            if not self.is_running:
                break
                
            print(f"Speech detected ({len(audio_segment)/16000:.2f}s). Transcribing...")
            
            # 1. STT
            text, _, _ = self.stt.transcribe(audio_segment)
            if not text.strip():
                continue
            print(f"User: {text}")
            
            # 2. LLM Stream
            prompt = f"User said: {text}. Respond briefly in Russian."
            if self.llm:
                response_generator = self.llm.generate_response(prompt, max_tokens=100, stream=True)
                print("Avatar: ", end="")
                
                def printing_generator(gen):
                    for chunk in gen:
                        print(chunk, end="", flush=True)
                        yield chunk
                    print()
                
                # 3. TTS Stream (runs in background thread via queue)
                # Note: recorder is paused via callback when TTS starts playing
                self.tts.process_stream(printing_generator(response_generator))
                
            else:
                response_text = "Извините, я пока не могу ответить."
                print(f"Avatar: {response_text}")
                
                # Simulate stream for mock
                def mock_gen():
                    yield response_text
                
                self.tts.process_stream(mock_gen())

if __name__ == "__main__":
    kiosk = Kiosk()
    kiosk.start()
