import os
from huggingface_hub import snapshot_download

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def download_model(repo_id, local_dir, allow_patterns=None, ignore_patterns=None):
    print(f"Downloading {repo_id} to {local_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            resume_download=True
        )
        print(f"Successfully downloaded {repo_id}")
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")

# 1. VAD (Silero) - usually loaded via torch.hub, but we can download the onnx model
# We will handle Silero VAD download inside the code using torch.hub or separate download if needed.
# For now, let's download the ONNX model for local usage if possible, or skip and let the VAD module handle it.
print("Skipping VAD download (will be handled by silero-vad module or torch.hub cache)")

# 2. STT (Faster-Whisper)
# faster-whisper downloads models automatically to a cache, but we can specify a local path.
# We will download 'deepdml/faster-whisper-large-v3-turbo-ct2' or similar.
# Official large-v3 is 'Systran/faster-whisper-large-v3'.
# 'deepdml/faster-whisper-large-v3-turbo-ct2' seems to be a valid CT2 conversion.
stt_model_id = "deepdml/faster-whisper-large-v3-turbo-ct2"
stt_dir = os.path.join(MODELS_DIR, "stt", "faster-whisper-large-v3-turbo")
download_model(stt_model_id, stt_dir)

# 3. LLM (Llama-3 8B GGUF)
# We need a GGUF version for llama-cpp-python.
# Using a quantised version of Llama-3-8B-Instruct or Saiga.
# Saiga is better for Russian.
llm_repo_id = "IlyaGusev/saiga_llama3_8b_gguf" # This might not be the exact repo for GGUF
# Let's search for a reliable GGUF. 'databricks/dbrx-instruct' ... no.
# 'Bartowski/Meta-Llama-3-8B-Instruct-GGUF' is reliable for base.
# 'IlyaGusev/saiga_llama3_8b' usually has GGUF files or links to them.
# Let's try 'IlyaGusev/saiga_llama3_8b_gguf' if it exists, otherwise 'IlyaGusev/saiga_llama3_8b' might be just the adapter.
# Better safe choice: 'Bartowski/Meta-Llama-3-8B-Instruct-GGUF' and a system prompt for Russian.
# Or 'IlyaGusev/saiga_llama3_8b' which is an adapter, we need the merged model.
# Let's go with 'IlyaGusev/saiga_llama3_8b_gguf' if it exists. 
# Actually, let's stick to a known working GGUF for now:
llm_model_id = "Bartowski/Meta-Llama-3-8B-Instruct-GGUF"
llm_dir = os.path.join(MODELS_DIR, "llm")
# Only download the specific quantization we want, e.g., Q4_K_M
download_model(llm_model_id, llm_dir, allow_patterns=["*Q4_K_M.gguf"])

# 4. TTS (CosyVoice)
# CosyVoice-300M-SFT
tts_model_id = "Alibaba-NLP/CosyVoice-300M-SFT"
tts_dir = os.path.join(MODELS_DIR, "tts", "CosyVoice-300M-SFT")
download_model(tts_model_id, tts_dir)

# Also need the base CosyVoice repo code or check if it's installable via pip.
# CosyVoice usually requires cloning the repo. We might need to clone it.
# For now, just downloading weights.

# 5. Avatar (MuseTalk)
# MuseTalk weights
avatar_model_id = "TMElyralab/MuseTalk"
avatar_dir = os.path.join(MODELS_DIR, "avatar", "MuseTalk")
download_model(avatar_model_id, avatar_dir)

# MuseTalk also needs other weights (dwpose, face-parse, etc.)
# These are usually downloaded by the MuseTalk scripts.
# We will download them here if we know the paths.
# For now, we'll stick to the main model.

print("Model download script completed.")
