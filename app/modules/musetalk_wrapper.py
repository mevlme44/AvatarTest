import sys
import os
import torch
import numpy as np
import cv2
import librosa
import queue
import threading
import time
import shutil
import pickle
import glob
import json
import subprocess
from tqdm import tqdm
from omegaconf import OmegaConf
import copy
from transformers import WhisperModel

# Add MuseTalk to path
sys.path.append(os.path.join(os.getcwd(), 'MuseTalk'))
sys.path.append(os.getcwd()) # Ensure root is also in path if needed

try:
    from musetalk.musetalk.utils.face_parsing import FaceParsing
    from musetalk.musetalk.utils.utils import datagen, load_all_model
    from musetalk.musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
    from musetalk.musetalk.utils.blending import get_image_prepare_material, get_image_blending
    from musetalk.musetalk.utils.audio_processor import AudioProcessor
except ImportError:
    # Try alternate import structure if "musetalk" is not a package inside MuseTalk folder directly but implicit
    try:
        from MuseTalk.musetalk.utils.face_parsing import FaceParsing
        from MuseTalk.musetalk.utils.utils import datagen, load_all_model
        from MuseTalk.musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
        from MuseTalk.musetalk.utils.blending import get_image_prepare_material, get_image_blending
        from MuseTalk.musetalk.utils.audio_processor import AudioProcessor
    except ImportError as e:
        print(f"Warning: MuseTalk modules not found. Ensure MuseTalk is in the project root. Error: {e}")

class MuseTalkAvatar:
    def __init__(self, 
                 avatar_id, 
                 video_path, 
                 bbox_shift=0, 
                 batch_size=4, 
                 preparation=False,
                 version="v15",
                 model_path="./models/avatar/MuseTalk",
                 device="cuda"):
        
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        self.preparation = preparation
        self.version = version
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Paths setup
        if self.version == "v15":
            self.base_path = f"./results/{self.version}/avatars/{avatar_id}"
        else:
            self.base_path = f"./results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": avatar_id,
            "video_path": video_path,
            "bbox_shift": bbox_shift,
            "version": version
        }

        # Initialize models
        print("Loading MuseTalk models...")
        unet_config = f"{model_path}/musetalkV15/musetalk.json"
        unet_model_path = f"{model_path}/musetalkV15/unet.pth"
        vae_type = "stabilityai/sd-vae-ft-mse" # Use huggingface ID instead of local path "sd-vae" if not present
        
        # Check if local sd-vae exists, otherwise use HF ID
        local_vae_path = f"{model_path}/sd-vae"
        abs_local_vae_path = os.path.abspath(local_vae_path)
        print(f"Checking for VAE at: {abs_local_vae_path}")
        
        if os.path.exists(local_vae_path) and os.path.exists(f"{local_vae_path}/config.json"):
            vae_type = abs_local_vae_path # Use absolute path to be safe
            print(f"Using local VAE: {vae_type}")
        else:
            print(f"Local VAE not found at {abs_local_vae_path}")
            # Try to fix path if it contains backslashes mixed with forward slashes incorrectly
            # Or force download if needed.
            # Here we just pass the ID string.
            print(f"Local VAE not found at {local_vae_path}, trying HuggingFace ID: {vae_type}")
            
            # Explicitly force download if not found to avoid weird path issues
            try:
                from diffusers import AutoencoderKL
                print(f"Downloading/Loading VAE: {vae_type}")
                # This call will cache it in ~/.cache/huggingface
                AutoencoderKL.from_pretrained(vae_type)
            except Exception as e:
                print(f"Warning: Failed to pre-load VAE from HF: {e}")
        if not os.path.exists(unet_model_path):
             print(f"Error: UNet model not found at {unet_model_path}")
        
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=unet_model_path,
            vae_type=vae_type,
            unet_config=unet_config,
            device=self.device
        )
        
        self.timesteps = torch.tensor([0], device=self.device)
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        
        # Audio Processor
        # Assuming whisper model is in models/whisper or handled by AudioProcessor
        whisper_dir = f"{model_path}/../stt/faster-whisper-large-v3-turbo" # Check this path!
        # Actually MuseTalk uses a specific whisper-tiny usually for features?
        # Let's check AudioProcessor default
        # It uses "openai/whisper-tiny" by default or path.
        # We should use a local path if possible or let it download.
        self.audio_processor = AudioProcessor(feature_extractor_path="openai/whisper-tiny")
        self.whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
        self.whisper = self.whisper.to(device=self.device, dtype=self.unet.model.dtype).eval()
        self.whisper.requires_grad_(False)
        
        # Face Parsing
        if self.version == "v15":
            self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90) # defaults
        else:
            self.fp = FaceParsing()

        self.idx = 0
        self.init() # Prepare data

    def _get_audio_feature_from_array(self, audio_data, sr, weight_dtype=None):
        """
        CPU pre-processing helper without temporary wav files.
        Keeps compatibility with AudioProcessor logic while removing disk I/O.
        """
        if audio_data is None:
            return None, 0

        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().cpu().numpy()

        audio_data = np.asarray(audio_data).astype(np.float32).reshape(-1)
        if audio_data.size == 0:
            return None, 0

        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

        sampling_rate = 16000
        segment_length = 30 * sampling_rate
        segments = [audio_data[i:i + segment_length] for i in range(0, len(audio_data), segment_length)]

        features = []
        for segment in segments:
            if len(segment) == 0:
                continue
            audio_feature = self.audio_processor.feature_extractor(
                segment,
                return_tensors="pt",
                sampling_rate=sampling_rate
            ).input_features
            if weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=weight_dtype)
            features.append(audio_feature)

        return features, len(audio_data)

    def init(self):
        # ... (Same logic as Avatar.init but using self attributes) ...
        # Simplified logic:
        if self.preparation:
             if os.path.exists(self.avatar_path):
                 shutil.rmtree(self.avatar_path)
             self._makedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
             self.prepare_material()
        else:
             if not os.path.exists(self.avatar_path):
                 print(f"Avatar {self.avatar_id} not prepared. Running preparation...")
                 self._makedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                 self.prepare_material()
             else:
                 # Load existing data
                 self.input_latent_list_cycle = torch.load(self.latents_out_path)
                 with open(self.coords_path, 'rb') as f:
                     self.coord_list_cycle = pickle.load(f)
                 input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
                 self.frame_list_cycle = read_imgs(input_img_list)
                 with open(self.mask_coords_path, 'rb') as f:
                     self.mask_coords_list_cycle = pickle.load(f)
                 input_mask_list = sorted(glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')))
                 self.mask_list_cycle = read_imgs(input_mask_list)

    def _makedirs(self, path_list):
        for path in path_list:
            os.makedirs(path, exist_ok=True)

    def prepare_material(self):
        # ... (Copy logic from realtime_inference.py prepare_material) ...
        print("Preparing data materials...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)

        if os.path.isfile(self.video_path):
            self._video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
             # Assume directory
             pass # Implement if needed

        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        print("Extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        
        input_latent_list = []
        idx = -1
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        for bbox, frame in zip(coord_list, frame_list):
            idx = idx + 1
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = y2 + 10 # extra_margin default
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]
            
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = self.vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)

        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        print("Generating masks...")
        for i, frame in enumerate(tqdm(self.frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)
            x1, y1, x2, y2 = self.coord_list_cycle[i]
            mode = "jaw" if self.version == "v15" else "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=self.fp, mode=mode)
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)

        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
        torch.save(self.input_latent_list_cycle, self.latents_out_path)
        print("Preparation done.")

    def _video2imgs(self, vid_path, save_path, ext='.png'):
        cap = cv2.VideoCapture(vid_path)
        count = 0
        while True:
            ret, frame = cap.read()
            if ret:
                # Ensure extension starts with dot
                if not ext.startswith('.'):
                    ext = f'.{ext}'
                cv2.imwrite(f"{save_path}/{count:08d}{ext}", frame)
                count += 1
            else:
                break
        cap.release()

    @torch.no_grad()
    def infer_audio_chunk(self, audio_data, sr=16000, fps=25):
        # Minimum audio length check (1 frame at 25fps = 1/25 sec = 0.04 sec)
        # 16000 * 0.04 = 640 samples. 
        # Let's say if less than 0.1s, we just return empty list or pad?
        # If we pad, we might generate static face.
        # If we return empty, video freezes?
        
        if len(audio_data) < 640: # Less than 1 frame
            print(f"Audio chunk too short ({len(audio_data)} samples), skipping inference.")
            return []

        # Extract features directly from in-memory audio buffer (no temp file).
        whisper_input_features, librosa_length = self._get_audio_feature_from_array(
            audio_data, sr, weight_dtype=self.unet.model.dtype
        )
        if not whisper_input_features or librosa_length <= 0:
            return []

        # Get whisper chunks
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.unet.model.dtype,
            self.whisper,
            librosa_length,
            fps=fps
        )
        
        # Generate frames
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        
        res_frames = []
        
        # We process frames sequentially to match audio length
        # Need to handle index cycling properly across chunks?
        # self.idx should be global across chunks? 
        # Yes, if we want continuous video loop.
        
        for i, (whisper_batch, latent_batch) in enumerate(gen):
            audio_feature_batch = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
            
            pred_latents = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
            recon = self.vae.decode_latents(pred_latents)
            
            for res_frame in recon:
                # Post-process frame (blending)
                bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
                ori_frame = self.frame_list_cycle[self.idx % len(self.frame_list_cycle)].copy()
                x1, y1, x2, y2 = bbox
                
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
                    mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
                    combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                    res_frames.append(combine_frame)
                except Exception as e:
                    print(f"Frame blending error: {e}")
                    res_frames.append(ori_frame) # Fallback
                    
                self.idx += 1
        
        return res_frames

if __name__ == "__main__":
    # Test
    avatar = MuseTalkAvatar(avatar_id="sun1", video_path="data/video/sun.mp4", preparation=False)
    # Test inference
    # audio = np.zeros(16000*2) # 2s silence
    # frames = avatar.infer_audio_chunk(audio)
    # print(f"Generated {len(frames)} frames")
