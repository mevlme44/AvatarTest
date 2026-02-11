import llama_cpp
from llama_cpp import Llama
import os

class LLM:
    def __init__(self, model_path=None, context_size=2048, n_gpu_layers=-1, n_threads=None, n_batch=512, force_gpu=False):
        """
        Initialize LLM with llama-cpp-python
        Args:
            model_path (str): Path to GGUF model file
            context_size (int): Max context window
            n_gpu_layers (int): Number of layers to offload to GPU (-1 for all)
            n_threads (int): Number of CPU threads (None for default)
            n_batch (int): Prompt processing batch size (higher shifts more work to GPU)
            force_gpu (bool): Raise error if GPU offload is unavailable
        """
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        gpu_supported = False
        supports_gpu_fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        if callable(supports_gpu_fn):
            try:
                gpu_supported = bool(supports_gpu_fn())
            except Exception:
                gpu_supported = False

        if force_gpu and n_gpu_layers != 0 and not gpu_supported:
            raise RuntimeError(
                "Текущая сборка llama-cpp-python без GPU offload. "
                "Переустановите пакет с CUDA backend."
            )

        print(f"Loading LLM from {model_path}...")
        print(f"LLM GPU offload supported: {gpu_supported}; n_gpu_layers={n_gpu_layers}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=n_batch,
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
            # Apply default Llama-3 chat template for single turn
            full_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        # Default stop tokens for Llama-3
        if stop is None:
            stop = ["<|eot_id|>", "<|end_of_text|>", "\n\n"] # Aggressive stop on double newline

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
