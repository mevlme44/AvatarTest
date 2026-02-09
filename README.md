# Local AI Avatar Kiosk

Fully local, low-latency AI Avatar system for kiosk applications.
Supports Russian language, speech recognition, LLM processing, and text-to-speech with lip sync visualization.

## Architecture

- **VAD**: Silero VAD (Voice Activity Detection)
- **STT**: Faster-Whisper (Large-v3-turbo)
- **LLM**: Llama-3-8B-Instruct (GGUF via llama-cpp-python)
- **TTS**: Silero TTS (Fast streaming) / CosyVoice (High quality)
- **Avatar**: MuseTalk (Real-time lip sync) - *Placeholder included*

## Prerequisites

- NVIDIA GPU (16GB VRAM recommended)
- Python 3.10+
- CUDA drivers

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download models:
   ```bash
   python download_models.py
   ```
   *Note: Ensure you have `huggingface_hub` installed and configured if accessing gated models.*

## Usage

Run the kiosk application:

```bash
python run.py
```

## Configuration

- Edit `app/main.py` to change model paths or device settings.
- Place avatar video loops in `assets/`:
  - `idle_loop.mp4`
  - `talking_loop.mp4` (optional)

## Latency Optimization

- The system uses streaming for LLM and TTS.
- VAD pauses recording while the avatar is speaking to prevent echo.
- Use `llama-cpp-python` with GPU offloading (`n_gpu_layers=-1`) for best performance.
