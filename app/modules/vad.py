import torch
import numpy as np

class VAD:
    def __init__(self, model_name='silero_vad', threshold=0.5, sample_rate=16000):
        self.sample_rate = sample_rate
        self.threshold = threshold
        # Load model using torch.hub
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model=model_name,
                                           force_reload=True,
                                           trust_repo=True)
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        
        self.vad_iterator = self.VADIterator(self.model)
        self.speaking = False

    def reset(self):
        self.vad_iterator.reset_states()
        self.speaking = False

    def process_chunk(self, audio_chunk):
        """
        Process a chunk of audio (numpy array, float32)
        Returns:
            is_speech (bool): True if speech is detected in this chunk
            speech_prob (float): Probability of speech
        """
        # Convert to torch tensor
        if isinstance(audio_chunk, np.ndarray):
             audio_chunk = torch.from_numpy(audio_chunk)
        
        # Ensure correct shape (1, N) or (N,)
        if len(audio_chunk.shape) == 1:
             audio_chunk = audio_chunk.unsqueeze(0)

        # Process using the iterator (stateful)
        # Note: Silero VAD iterator returns a dict if speech started/ended
        # For simple probability, we use model directly or iterator logic.
        
        # Using model directly for probability per chunk (requires correct chunk size)
        # Typically Silero expects 512, 1024, 1536 samples for 16k rate.
        
        with torch.no_grad():
             speech_prob = self.model(audio_chunk, self.sample_rate).item()
        
        is_speech = speech_prob > self.threshold
        return is_speech, speech_prob

    def is_speech(self, audio_chunk):
        is_speech, _ = self.process_chunk(audio_chunk)
        return is_speech
