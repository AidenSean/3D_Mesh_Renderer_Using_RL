from huggingface_hub import snapshot_download
import os

def download_model():
    model_id = "SimianLuo/LCM_Dreamshaper_v7"
    print(f"Downloading model: {model_id}...")
    print("This may take a while (approx 4-6 GB). Please wait.")
    
    try:
        path = snapshot_download(repo_id=model_id, resume_download=True)
        print(f"Model successfully downloaded to: {path}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Try running this script again to resume.")

if __name__ == "__main__":
    download_model()
