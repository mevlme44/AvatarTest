from llama_cpp import Llama
import os

class LLM:
    def __init__(self, model_path=None, context_size=2048, n_gpu_layers=-1, n_threads=None):
        """
        Initialize LLM with llama-cpp-python
        Args:
            model_path (str): Path to GGUF model file
            context_size (int): Max context window
            n_gpu_layers (int): Number of layers to offload to GPU (-1 for all)
            n_threads (int): Number of CPU threads (None for default)
        """
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        print(f"Loading LLM from {model_path}...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False # Suppress output
        )
        print("LLM loaded.")

    def generate_response(self, prompt, system_prompt=None, max_tokens=256, temperature=0.7, top_p=0.9, stop=None, stream=False):
        """
        Generate response from prompt.
        Args:
            prompt (str): User input
            system_prompt (str): Optional system instruction
            max_tokens (int): Max tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling probability
            stop (list): List of stop sequences
            stream (bool): Whether to return a generator for streaming tokens
        Returns:
            str or generator: Generated text
        """
        
        full_prompt = prompt
        if system_prompt:
            # Llama-3 specific format
            full_prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            # Simple format or raw prompt
            pass # Use as is if no system prompt logic needed, or apply default template

        # Default stop tokens for Llama-3
        if stop is None:
            stop = ["<|eot_id|>", "<|end_of_text|>"]

        output = self.llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            echo=False,
            stream=stream
        )

        if stream:
            return self._stream_generator(output)
        else:
            return output['choices'][0]['text']

    def _stream_generator(self, stream_output):
        for chunk in stream_output:
            delta = chunk['choices'][0]['text']
            yield delta

if __name__ == "__main__":
    # Test block
    # Note: Requires a valid model path
    pass
