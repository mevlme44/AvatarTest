import torch
import sounddevice as sd
import time
import numpy as np
import threading
import queue
import re

class TTS:
    def __init__(self, model_id='v3_1_ru', device='cpu', sample_rate=48000):
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.model_id = model_id
        
        print(f"Loading TTS model {model_id}...")
        try:
            self.model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                         model='silero_tts',
                                         language='ru',
                                         speaker=model_id)
            self.model.to(self.device)
            print("TTS loaded.")
        except Exception as e:
            print(f"Failed to load TTS model: {e}")
            self.model = None

    def generate(self, text, speaker='aidar', put_accent=True, put_yo=True):
        if not self.model:
            return torch.zeros(1), self.sample_rate
            
        start_time = time.time()
        if not text.strip():
            return torch.zeros(1), self.sample_rate
            
        audio = self.model.apply_tts(text=text,
                                     speaker=speaker,
                                     sample_rate=self.sample_rate,
                                     put_accent=put_accent,
                                     put_yo=put_yo)
        
        elapsed = time.time() - start_time
        return audio, self.sample_rate

    def play(self, audio, sample_rate=48000):
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
            
        sd.play(audio, sample_rate)
        sd.wait()

class StreamTTS(TTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer = ""
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.play_thread = threading.Thread(target=self._play_loop)
        self.play_thread.start()
        self.is_playing = False
        self.on_start_talking = None
        self.on_stop_talking = None

    def set_callbacks(self, on_start, on_stop):
        self.on_start_talking = on_start
        self.on_stop_talking = on_stop

    def process_stream(self, text_stream):
        """
        Process a stream of text chunks.
        """
        for chunk in text_stream:
            self.buffer += chunk
            # Check for sentence end (simple heuristic)
            if any(punct in chunk for punct in ['.', '!', '?', '\n']):
                # Find the last punctuation index to split safely
                # This is a simplification; robust sentence splitting is harder
                if self.buffer.endswith('.') or self.buffer.endswith('!') or self.buffer.endswith('?'):
                    to_synthesize = self.buffer
                    self.buffer = ""
                    self._synthesize_and_queue(to_synthesize)
        
        # Process remaining buffer
        if self.buffer:
            self._synthesize_and_queue(self.buffer)
            self.buffer = ""
            
        # Wait for queue to empty if we want to block until done
        # But main loop needs to continue? No, main loop should wait for audio to finish before listening again?
        # Usually yes, to avoid self-listening.
        while not self.audio_queue.empty() or self.is_playing:
            time.sleep(0.1)

    def _synthesize_and_queue(self, text):
        if not text.strip():
            return
        # print(f"Synthesizing: {text}")
        audio, sr = self.generate(text)
        self.audio_queue.put((audio, sr))

    def _play_loop(self):
        while not self.stop_event.is_set():
            try:
                audio, sr = self.audio_queue.get(timeout=0.1)
                
                self.is_playing = True
                if self.on_start_talking:
                    self.on_start_talking()
                    
                self.play(audio, sr)
                
                self.is_playing = False
                # Check if queue is empty to trigger stop callback potentially?
                # Ideally we only stop talking when queue is empty AND current playback finished.
                # Here we toggle per sentence which might be jerky for avatar.
                # Better: check queue size.
                if self.audio_queue.empty():
                     if self.on_stop_talking:
                         self.on_stop_talking()
                         
            except queue.Empty:
                continue

    def stop(self):
        self.stop_event.set()
        self.play_thread.join()
