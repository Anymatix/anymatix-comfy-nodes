import os
import re

import requests
import comfy
import comfy.sd
import comfy.utils
import folder_paths 
from .fetch import download_file

CHECKPOINTS_DIR =  os.path.join(folder_paths.models_dir, "checkpoints")
STORE = CHECKPOINTS_DIR #os.path.join(folder_paths.user_directory,"anymatix")

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
        pbar = comfy.utils.ProgressBar(1000)
        progress = 0        
        pbar.update_absolute(progress,1000)
        
        def callback(x,y): 
            import math           
            new_progress = round(1000*x/y)
            nonlocal progress
            if (new_progress != progress):
                progress = new_progress
                pbar.update_absolute(progress,1000)
        
        def expand_info_civitai(url):            
            # get the model id from the url using a regex that matches the first /.../ after https://civitai.com/api/download/models 
            pattern = r'https://civitai\.com/api/download/models/([^/]+)'
            match = re.search(pattern, url)
            if match:
                model_id = match.group(1)
            else:
                return None            
            model_info_url = f"https://civitai.com/api/v1/model-versions/{model_id}"
            with requests.Session() as session:
                return requests.get(model_info_url,allow_redirects=True).json()
            
        def expand_info(url):
            if url.startswith("https://civitai.com/api/download/models"):
                return expand_info_civitai(url)
            return None
        
        model_name = download_file(url=url,store=STORE,dir=CHECKPOINTS_DIR,callback=callback,expand_info=expand_info)                           
        return(model_name,)
    
