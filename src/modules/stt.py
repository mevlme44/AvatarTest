import os
import time
from faster_whisper import WhisperModel

class STT:
    def __init__(self, model_size="large-v3", device="cuda", compute_type="float16", model_path=None):
        self.device = device
        self.compute_type = compute_type
        
        # Load model from local path if provided, otherwise download from HF automatically
        if model_path and os.path.exists(model_path):
            print(f"Loading STT model from local path: {model_path}")
            self.model = WhisperModel(model_path, device=device, compute_type=compute_type)
        else:
            print(f"Loading STT model '{model_size}' (downloading if needed)...")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_data, language="ru"):
        """
        Transcribe audio data (numpy array or file path)
        Returns:
            text (str): Transcribed text
            segments (list): Raw segments
            info (dict): Transcription info
        """
        start_time = time.time()
        segments, info = self.model.transcribe(audio_data, language=language, beam_size=5)
        
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
