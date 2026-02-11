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
        if torch.cuda.is_available():
            # Allow tensor cores where possible to increase GPU throughput.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        print(f"CUDA availability: {torch.cuda.is_available()}")
        stt_model_size = os.getenv("STT_MODEL_SIZE", "large-v3-turbo")
        stt_beam_size = int(os.getenv("STT_BEAM_SIZE", "1"))
        # Keep llama on CPU by default (can be overridden via env).
        llm_gpu_layers = int(os.getenv("LLM_GPU_LAYERS", "0"))
        llm_threads = int(os.getenv("LLM_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
        llm_batch = int(os.getenv("LLM_BATCH", "512"))
        self.stt_beam_size = max(1, stt_beam_size)

        # Initialize modules
        self.vad = VAD(threshold=0.5)
        self.recorder = AudioRecorder(sample_rate=16000)
        self.recorder.set_vad(self.vad)
        
        # Initialize STT
        self.stt = STT(model_size=stt_model_size, device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize LLM
        model_path = "models/llm/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
        if not os.path.exists(model_path):
             print(f"Warning: LLM model not found at {model_path}. Using mock LLM.")
             self.llm = None
        else:
             self.llm = LLM(
                 model_path=model_path,
                 n_gpu_layers=llm_gpu_layers,
                 n_threads=llm_threads,
                 n_batch=llm_batch,
                 force_gpu=bool(int(os.getenv("LLM_FORCE_GPU", "0")))
             )
             
        # Initialize TTS (Stream)
        self.tts = StreamTTS(device="cuda" if torch.cuda.is_available() else "cpu") 
        
        # Initialize Avatar
        self.avatar = Avatar()
        
        # Link callbacks
        self.tts.set_callbacks(
            on_start=self._on_speech_start, 
            on_stop=self._on_speech_stop,
            on_audio_chunk=self.avatar.push_audio
        )
        
        self.is_running = False
        self.processing_thread = None
        self.user_muted = False
        self.system_pause_active = False

    def _on_speech_start(self):
        print("System: Avatar started talking -> Pausing recorder")
        self.system_pause_active = True
        self._apply_recorder_state()
        self.avatar.set_talking(True)

    def _on_speech_stop(self):
        print("System: Avatar stopped talking -> Resuming recorder")
        self.avatar.set_talking(False)
        self.system_pause_active = False
        self._apply_recorder_state()

    def _on_mute_toggle(self, is_muted):
        self.user_muted = is_muted
        self._apply_recorder_state()

    def _apply_recorder_state(self):
        # Recorder should run only when neither user mute nor system pause is active.
        if self.user_muted or self.system_pause_active:
            self.recorder.pause()
        else:
            self.recorder.resume()

    def start(self):
        self.is_running = True
        
        # Start Avatar UI
        self.avatar.start()
        # Set callback for mute toggle
        self.avatar.on_mute_toggle = self._on_mute_toggle
        
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
            text, _, _ = self.stt.transcribe(audio_segment, beam_size=self.stt_beam_size)
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

def download_dwpose():
    import requests
    url = "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth"
    save_path = "models/dwpose/dw-ll_ucoco_384.pth"
    
    if os.path.exists(save_path):
        return
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Downloading DWPose model to {save_path}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(save_path, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
    print("Download complete.")

if __name__ == "__main__":
    kiosk = Kiosk()
    kiosk.start()
