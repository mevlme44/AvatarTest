import torch
import os
import time
from faster_whisper import WhisperModel

class STT:
    def __init__(self, model_size="large-v3", device="cuda", compute_type="float16", model_path=None):
        self.device = device
        self.compute_type = compute_type
        
        # Determine optimal compute type for current device
        if device == "cpu" or not torch.cuda.is_available():
            self.compute_type = "int8"
            self.device = "cpu"
            print("Running on CPU, switching compute_type to 'int8'...")
        elif compute_type == "float16": 
             pass # Try float16 first if requested on GPU 
             
        # Load model from local path if provided, otherwise download from HF automatically
        try:
            if model_path and os.path.exists(model_path):
                print(f"Loading STT model from local path: {model_path}")
                self.model = WhisperModel(model_path, device=self.device, compute_type=self.compute_type)
            else:
                print(f"Loading STT model '{model_size}' (downloading if needed)...")
                self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        except ValueError as e:
            if "requested compute type" in str(e).lower() or "not supported" in str(e).lower():
                print(f"Compute type '{self.compute_type}' not supported on this device. Retrying with 'int8'...")
                self.compute_type = "int8"
                if model_path and os.path.exists(model_path):
                     self.model = WhisperModel(model_path, device=self.device, compute_type=self.compute_type)
                else:
                     self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
            else:
                raise e

    def transcribe(self, audio_data, language="ru", beam_size=1):
        """
        Transcribe audio data (numpy array or file path)
        Returns:
            text (str): Transcribed text
            segments (list): Raw segments
            info (dict): Transcription info
        """
        start_time = time.time()
        segments, info = self.model.transcribe(audio_data, language=language, beam_size=beam_size)
        
        full_text = ""
        seg_list = []
        for segment in segments:
            full_text += segment.text
            seg_list.append(segment)
            
        elapsed_time = time.time() - start_time
        print(f"Transcription took {elapsed_time:.3f}s")
        
        return full_text.strip(), seg_list, info

if __name__ == "__main__":
    # Test block
    stt = STT(model_size="tiny", device="cpu", compute_type="int8")
    print("STT initialized.")
