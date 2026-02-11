import torch
import sounddevice as sd
import time
import numpy as np
import threading
import queue
import re

class TTS:
    def __init__(self, model_id='v3_1_ru', device='cuda', sample_rate=48000):
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

        # Silero TTS limitation: max 930 characters?
        # Actually input overflow usually means text is too long for model buffer
        # Let's truncate or split if needed, but here simple truncation for safety
        if len(text) > 800:
            print(f"Warning: TTS text too long ({len(text)}), truncating to 800 chars.")
            text = text[:800]
            
        try:
            audio = self.model.apply_tts(text=text,
                                        speaker=speaker,
                                        sample_rate=self.sample_rate,
                                        put_accent=put_accent,
                                        put_yo=put_yo)
        except ValueError as e:
            print(f"TTS Error: {e}")
            return torch.zeros(1), self.sample_rate
        
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

    def set_callbacks(self, on_start=None, on_stop=None, on_audio_chunk=None):
        self.on_start_talking = on_start
        self.on_stop_talking = on_stop
        self.on_audio_chunk = on_audio_chunk

    def process_stream(self, text_stream):
        """
        Process a stream of text chunks.
        """
        started = False
        try:
            for chunk in text_stream:
                if not started:
                    started = True
                    if self.on_start_talking:
                        self.on_start_talking()

                self.buffer += chunk
                
                # Force flush if buffer gets too long (prevent model input overflow)
                if len(self.buffer) > 250:
                     # Find nearest space to split safely if possible
                     last_space = self.buffer.rfind(' ')
                     if last_space > 0:
                         to_synthesize = self.buffer[:last_space]
                         self.buffer = self.buffer[last_space+1:]
                         self._synthesize_and_queue(to_synthesize)
                     else:
                         # No space, just flush all
                         self._synthesize_and_queue(self.buffer)
                         self.buffer = ""
                     continue

                # Check if there is any sentence terminator in the buffer
                if re.search(r'[.!?\n]', self.buffer):
                    # Split buffer into sentences, keeping delimiters
                    parts = re.split(r'([.!?\n]+)', self.buffer)
                    
                    # Reconstruct completed sentences
                    sentences = []
                    for i in range(0, len(parts) - 1, 2):
                        text_part = parts[i]
                        delim_part = parts[i+1]
                        full_sent = text_part + delim_part
                        sentences.append(full_sent)
                    
                    # The last part is an incomplete remainder
                    remainder = parts[-1]
                    
                    for sent in sentences:
                        if sent.strip():
                            self._synthesize_and_queue(sent)
                    
                    self.buffer = remainder
            
            # Process remaining buffer
            if self.buffer and self.buffer.strip():
                self._synthesize_and_queue(self.buffer)
                self.buffer = ""
                
            # Block until all queued audio was played to avoid recorder resume race.
            while not self.audio_queue.empty() or self.is_playing:
                time.sleep(0.05)
        finally:
            if started and self.on_stop_talking:
                self.on_stop_talking()

    def _synthesize_and_queue(self, text):
        text = text.strip()
        if not text:
            return
        
        # Check if text contains at least one alphanumeric character
        # Silero might fail on just punctuation
        if not any(c.isalnum() for c in text):
            return

        # print(f"Synthesizing: {text}")
        audio, sr = self.generate(text)
        
        # Check if audio is valid and has significant duration
        # If audio is too short (e.g. < 0.1s), MuseTalk might crash
        if isinstance(audio, torch.Tensor):
            if audio.numel() < 1600: # < 0.1s at 16k (assuming resampled later) or 48k? 
                # TTS returns 48k usually. 48000 * 0.1 = 4800 samples.
                # If zero or very short, skip
                if audio.numel() == 0:
                    return
            else:
                pass # valid
        
        self.audio_queue.put((audio, sr))

    def _play_loop(self):
        while not self.stop_event.is_set():
            try:
                audio, sr = self.audio_queue.get(timeout=0.1)
                
                self.is_playing = True

                if self.on_audio_chunk:
                    # Pass audio to external handler (e.g., Avatar)
                    self.on_audio_chunk(audio, sr)
                else:
                    # Default playback
                    self.play(audio, sr)
                
                self.is_playing = False
                         
            except queue.Empty:
                continue

    def stop(self):
        self.stop_event.set()
        self.play_thread.join()
