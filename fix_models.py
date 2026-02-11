import os
import requests
import sys

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return
        
    print(f"Downloading {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(save_path, 'wb') as f:
            downloaded = 0
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                # Simple progress
                if total_size > 0:
                    percent = int(50 * downloaded / total_size)
                    sys.stdout.write(f"\r[{'=' * percent}{' ' * (50 - percent)}] {downloaded}/{total_size} bytes")
                    sys.stdout.flush()
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nError downloading {url}: {e}")
        if os.path.exists(save_path):
            os.remove(save_path)

def main():
    # 1. DWPose
    dwpose_url = "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth"
    dwpose_path = "models/dwpose/dw-ll_ucoco_384.pth"
    download_file(dwpose_url, dwpose_path)

    # 2. Face Parsing (ResNet) - MuseTalk needs this too
    resnet_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    resnet_path = "models/face-parse-bisent/resnet18-5c106cde.pth"
    download_file(resnet_url, resnet_path)

    # 3. Face Parsing (Model)
    # This one is tricky as it's often on GDrive, but let's try a mirror or skip if manual needed.
    # Usually MuseTalk wrapper might download it or we need it manually.
    # For now let's hope DWPose is the main blocker.

def download_vae():
    import requests
    import os
    
    # Files needed for VAE
    files = ["config.json", "diffusion_pytorch_model.bin"]
    # Explicit URLs because sometimes resolving main/ works weirdly
    base_url = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main"
    save_dir = "models/avatar/MuseTalk/sd-vae"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Downloading VAE to {save_dir}...")
    
    for file in files:
        url = f"{base_url}/{file}"
        save_path = f"{save_dir}/{file}"
        
        # Check if file exists and has size > 0
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
             # Additional check for binary file
             if file.endswith(".bin"):
                 # Check if size is plausible (e.g. > 100MB)
                 # sd-vae is around 335MB usually
                 size_mb = os.path.getsize(save_path) / (1024 * 1024)
                 if size_mb < 100:
                     print(f"Warning: {file} size is {size_mb:.2f}MB, which seems too small. Re-downloading...")
                     os.remove(save_path)
                 else:
                     print(f"File {file} exists ({size_mb:.2f}MB), skipping.")
                     continue
             else:
                 print(f"File {file} exists, skipping.")
                 continue
            
        print(f"Downloading {file} from {url}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(save_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
            print(f"Downloaded {file}")
        except Exception as e:
            print(f"Error downloading {file}: {e}")
            if os.path.exists(save_path):
                os.remove(save_path)

def download_face_parsing():
    import requests
    import os
    
    # Files needed for Face Parsing
    base_url = "https://github.com/TMElyralab/MuseTalk/releases/download/v1.0/79999_iter.pth" 
    # Use reliable mirror if possible, but let's try direct link from MuseTalk releases or similar if available.
    # Actually MuseTalk provides GDrive link. Let's try to find a direct link or use a mirror.
    # Found a mirror on HuggingFace often used.
    
    url = "https://huggingface.co/SDInstant/models-moved/resolve/main/79999_iter.pth"
    save_path = "models/face-parse-bisent/79999_iter.pth"
    
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000000: # > 1MB
        print(f"Face parsing model exists, skipping.")
        return

    print(f"Downloading face parsing model to {save_path}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print("Downloaded face parsing model.")
    except Exception as e:
        print(f"Error downloading face parsing model: {e}")

if __name__ == "__main__":
    download_vae()
    download_face_parsing()
