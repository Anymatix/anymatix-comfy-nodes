import os
import comfy # type: ignore
import folder_paths  # type: ignore
from .fetch import download_file

CHECKPOINTS_DIR =  os.path.join(folder_paths.models_dir, "checkpoints")
STORE = os.path.join(folder_paths.user_directory,"anymatix")

# Ensure checkpoints directory exists
if not os.path.exists(CHECKPOINTS_DIR):
    os.makedirs(CHECKPOINTS_DIR)

class AnymatixCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "ckpt_name": ("STRING", ),
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = ("The model used for denoising latents.", 
                       "The CLIP model used for encoding text prompts.", 
                       "The VAE model used for encoding and decoding images to and from latent space.")
    FUNCTION = "load_checkpoint"

    CATEGORY = "Anymatix"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, ckpt_name):
        ckpt_path = f"{CHECKPOINTS_DIR}/{ckpt_name}"
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[:3]

# Define the custom node class
class AnymatixCheckpointFetcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"}),                
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "download_model"
    CATEGORY = "Anymatix"

    def download_model(self, url):        
        model_name = "TEST.safetensors"
        # Destination path for the downloaded model
        model_path = os.path.join(CHECKPOINTS_DIR, model_name)

        # Download the model file from the provided URL
        print(f"Downloading {model_name} from {url} to {model_path}")   
        
        pbar = comfy.utils.ProgressBar(100)  
        model_name = download_file(url=url,store=STORE,dir=CHECKPOINTS_DIR,callback=lambda x,y: pbar.update_absolute(x,y))                           
        return(model_name,)
    
