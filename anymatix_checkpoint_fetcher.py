import json
import os
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import re

import requests
import comfy
import comfy.sd
import comfy.utils
import folder_paths
import sys
from comfy_api.latest import io
from typing import Dict, Any, Tuple
from comfy_execution.utils import get_executing_context

try:
    # When loaded as a package inside ComfyUI, use relative import
    from .fetch import download_file, hash_string
except Exception:
    # When running this file directly for testing, fall back to absolute import
    try:
        from fetch import download_file, hash_string
    except Exception:
        # Provide a helpful error when import truly fails
        raise
from spandrel import ModelLoader, ImageModelDescriptor
from nodes import CLIPLoader, UNETLoader, VAELoader, CLIPVisionLoader, LoraLoaderModelOnly, DualCLIPLoader

# Try to load GGUF nodes module explicitly from the sibling ComfyUI-GGUF package
import importlib.util
import types

def verify_model_file_exists(file_path: str, model_type: str = "model") -> None:
    """
    Verify that a model file exists before attempting to load it.
    Raises FileNotFoundError with a helpful message if the file is missing.
    
    Args:
        file_path: Full path to the model file
        model_type: Type of model for error message (e.g., "lora", "checkpoint", "upscale")
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"{model_type.capitalize()} file not found: {file_path}\n"
            f"The model may have been deleted or moved. "
            f"Please ensure the model is downloaded and try again."
        )


gguf_nodes_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ComfyUI-GGUF", "nodes.py")
)

custom_nodes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if custom_nodes_path not in sys.path:
    sys.path.insert(0, custom_nodes_path)

# Load the sibling ComfyUI-GGUF package as a proper package so that its
# relative imports (e.g. `from .ops import ...`) work when we load the
# `nodes.py` file directly. The on-disk folder name contains a hyphen so we
# expose it under a valid Python package name `ComfyUI_GGUF` and register it
# in sys.modules with an appropriate __path__ before executing the module.
pkg_name = "ComfyUI_GGUF"
pkg_path = os.path.dirname(gguf_nodes_path)
if pkg_name not in sys.modules:
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [pkg_path]
    sys.modules[pkg_name] = pkg

# Import the nodes module as a submodule of our synthetic package so
# relative imports inside it resolve to files in the same directory.
nodes_mod_name = pkg_name + ".nodes"
spec = importlib.util.spec_from_file_location(nodes_mod_name, gguf_nodes_path)
gguf_nodes = importlib.util.module_from_spec(spec)
gguf_nodes.__package__ = pkg_name
sys.modules[nodes_mod_name] = gguf_nodes
spec.loader.exec_module(gguf_nodes)

# SeedVR2LoadDiTModel = None

# for mod in sys.modules.values():
#     if hasattr(mod, "SeedVR2LoadDiTModel"):
#         SeedVR2LoadDiTModel = mod.SeedVR2LoadDiTModel
#         break

# if SeedVR2LoadDiTModel is None:
#     raise RuntimeError("SeedVR2LoadDiTModel not loaded yet")

class AnymatixSeedVR2LoadDiTModel():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "seedvr2_ema_3b_fp8_e4m3fn.safetensors"}),
                "device": ("STRING", {"default": "cuda:0"}),
                "offload_device": ("STRING", {"default": "none"}),
                "cache_model": ("BOOLEAN", {"default": False}),
                "blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 36, "step": 1}),
                "swap_io_components": ("BOOLEAN", {"default": False}),
                "attention_mode": (["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"], {"default": "sdpa"}),
            }
        }
    CATEGORY = "Anymatix"
    FUNCTION = "execute"
    RETURN_TYPES = ("SEEDVR2_DIT",)

    def execute(self, model: str, device: str, offload_device: str = "none",
                      cache_model: bool = False, blocks_to_swap: int = 0, 
                      swap_io_components: bool = False, attention_mode: str = "sdpa"):
        
        """
        Create DiT model configuration for SeedVR2 main node
        
        Args:
            model: Model filename to load
            device: Target device for model execution
            offload_device: Device to offload model to when not in use
            cache_model: Whether to keep model loaded between runs
            blocks_to_swap: Number of transformer blocks to swap (requires offload_device != device)
            swap_io_components: Whether to offload I/O components (requires offload_device != device)
            attention_mode: Attention computation backend ('sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', or 'sageattn_3')
            torch_compile_args: Optional torch.compile configuration from settings node
            
        Returns:
            NodeOutput containing configuration dictionary for SeedVR2 main node
            
        Raises:
            ValueError: If cache_model is enabled but offload_device is not set
        """
        # Validate cache_model configuration
        if cache_model and offload_device == "none":
            raise ValueError(
                "Model caching (cache_model=True) requires offload_device to be set. "
                f"Current: offload_device='{offload_device}'. "
                "Please set offload_device to specify where the cached DiT model should be stored "
                "(e.g., 'cpu' or another device). Set cache_model=False if you don't want to cache the model."
            )
        
        config = {
            "model": model,
            "device": device,
            "offload_device": offload_device,
            "cache_model": cache_model,
            "blocks_to_swap": blocks_to_swap,
            "swap_io_components": swap_io_components,
            "attention_mode": attention_mode,
            "node_id": get_executing_context().node_id,
        }

        return io.NodeOutput(config)
    
class AnymatixSeedVR2LoadVAEModel():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("STRING", {"default": "ema_vae_fp16.safetensors"}),
                "device": ("STRING", {"default": "cuda:0"}),
                "offload_device": ("STRING", {"default": "none"}),
                "cache_model": ("BOOLEAN", {"default": False}),
                "encode_tiled": ("BOOLEAN", {"default": False}),
                "encode_tile_size": ("INT", {"default": 512, "min": 64, "step": 32}),
                "encode_tile_overlap": ("INT", {"default": 64, "min": 0, "step": 32}),
                "decode_tiled": ("BOOLEAN", {"default": False}),
                "decode_tile_size": ("INT", {"default": 512, "min": 64, "step": 32}),
                "decode_tile_overlap": ("INT", {"default": 64, "min": 0, "step": 32}),
                "tile_debug": (["false", "encode", "decode"], {"default": "false"}),
            }
        }
    CATEGORY = "Anymatix"
    FUNCTION = "execute"
    RETURN_TYPES = ("SEEDVR2_VAE",)

    def execute(self, model: str, device: str, offload_device: str = "none",
                     cache_model: bool = False, encode_tiled: bool = False,
                     encode_tile_size: int = 512, encode_tile_overlap: int = 64,
                     decode_tiled: bool = False, decode_tile_size: int = 512, 
                     decode_tile_overlap: int = 64, tile_debug: str = "false",
                     ):
        """
        Create VAE model configuration for SeedVR2 main node
        
        Args:
            model: Model filename to load
            device: Target device for model execution
            offload_device: Device to offload model to when not in use
            cache_model: Whether to keep model loaded between runs
            encode_tiled: Enable tiled encoding
            encode_tile_size: Tile size for encoding
            encode_tile_overlap: Tile overlap for encoding
            decode_tiled: Enable tiled decoding
            decode_tile_size: Tile size for decoding
            decode_tile_overlap: Tile overlap for decoding
            tile_debug: Tile visualization mode (false/encode/decode)
            torch_compile_args: Optional torch.compile configuration from settings node
            
        Returns:
            NodeOutput containing configuration dictionary for SeedVR2 main node
            
        Raises:
            ValueError: If cache_model is enabled but offload_device is invalid
        """
        # Validate cache_model configuration
        if cache_model and offload_device == "none":
            raise ValueError(
                "Model caching (cache_model=True) requires offload_device to be set. "
                f"Current: offload_device='{offload_device}'. "
                "Please set offload_device to specify where the cached VAE model should be stored "
                "(e.g., 'cpu' or another device). Set cache_model=False if you don't want to cache the model."
            )
        
        config = {
            "model": model,
            "device": device,
            "offload_device": offload_device,
            "cache_model": cache_model,
            "encode_tiled": encode_tiled,
            "encode_tile_size": encode_tile_size,
            "encode_tile_overlap": encode_tile_overlap,
            "decode_tiled": decode_tiled,
            "decode_tile_size": decode_tile_size,
            "decode_tile_overlap": decode_tile_overlap,
            "tile_debug": tile_debug,
            "node_id": get_executing_context().node_id,
        }
        return io.NodeOutput(config)

def get_anymatix_models_dir(type_name: str) -> str:
    """
    Get the primary directory for a model type, respecting extra_model_paths.yaml.
    """
    try:
        # folder_paths.get_folder_paths return a list of paths
        # The first path is the primary (default) one if is_default: true was used in YAML
        paths = folder_paths.get_folder_paths(type_name)
        if paths:
            return paths[0]
    except Exception:
        pass
    # Fallback to internal ComfyUI models directory
    return os.path.join(folder_paths.models_dir, type_name)

CHECKPOINTS_DIR = get_anymatix_models_dir("checkpoints")

# Ensure checkpoints directory exists
if not os.path.exists(CHECKPOINTS_DIR):
    os.makedirs(CHECKPOINTS_DIR)
class AnymatixCLIPVisionLoader(CLIPVisionLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": ("STRING", ),}}
    
    CATEGORY = "Anymatix"
    
    def load_clip(self, clip_name):
        return super().load_clip(os.path.basename(clip_name))

class AnymatixVAELoader(VAELoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": ("STRING", ),}}
    
    CATEGORY = "Anymatix"
    
    def load_vae(self, vae_name):
        return super().load_vae(os.path.basename(vae_name))

class AnymatixCLIPLoader(CLIPLoader):
    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        types["required"]["clip_name"] = ("STRING", )
        return types

    CATEGORY = "Anymatix"    

    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        return super().load_clip(os.path.basename(clip_name), type, device)

AnymatixCLIPLoader2=AnymatixCLIPLoader
class AnymatixDualCLIPLoader(DualCLIPLoader):
    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        types["required"]["clip_name1"] = ("STRING", )
        types["required"]["clip_name2"] = ("STRING", )
        return types

    CATEGORY = "Anymatix"

    def load_clip(self, clip_name1, clip_name2, type = "flux", device="default"):
        return super().load_clip(os.path.basename(clip_name1), os.path.basename(clip_name2), type, device)

class AnymatixTripleCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": ("STRING", ),
                "clip_name2": ("STRING", ),
                "clip_name3": ("STRING", ),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "Anymatix"

    def load_clip(self, clip_name1, clip_name2, clip_name3):
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", os.path.basename(clip_name1))
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", os.path.basename(clip_name2))
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", os.path.basename(clip_name3))
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3], embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (clip,)

class AnymatixQuadrupleCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": ("STRING", ),
                "clip_name2": ("STRING", ),
                "clip_name3": ("STRING", ),
                "clip_name4": ("STRING", ),
            }
        }
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "Anymatix"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4):
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", os.path.basename(clip_name1))
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", os.path.basename(clip_name2))
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", os.path.basename(clip_name3))
        clip_path4 = folder_paths.get_full_path_or_raise("text_encoders", os.path.basename(clip_name4))
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3, clip_path4], embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (clip,)

class AnymatixAudioEncoderLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio_encoder_name": ("STRING", ),
                             }}
    
    RETURN_TYPES = ("AUDIO_ENCODER",)
    FUNCTION = "load_model"
    CATEGORY = "Anymatix"
    
    def load_model(self, audio_encoder_name):
        audio_encoder_path = folder_paths.get_full_path_or_raise(
            "audio_encoders", 
            os.path.basename(audio_encoder_name)
        )
        sd = comfy.utils.load_torch_file(audio_encoder_path, safe_load=True)
        audio_encoder = comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(sd)
        if audio_encoder is None:
            raise RuntimeError("ERROR: audio encoder file is invalid")
        return (audio_encoder,)

class AnymatixUNETLoader(UNETLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": ("STRING", ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
                             }}

    CATEGORY = "Anymatix"

    def load_unet(self, unet_name, weight_dtype):
        return super().load_unet(os.path.basename(unet_name), weight_dtype)

class AnymatixUNETLoaderGGUF(gguf_nodes.UnetLoaderGGUF):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": ("STRING", )
                             }}

    CATEGORY = "Anymatix"

    def load_unet(self, unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None):
        return super().load_unet(os.path.basename(unet_name))

class AnymatixLoraLoaderModelOnly(LoraLoaderModelOnly):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "lora_name": ("STRING", ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }
                }

    CATEGORY = "Anymatix"

    def load_lora_model_only(self, model, lora_name, strength_model):
        return super().load_lora_model_only(model, os.path.basename(lora_name), strength_model)

class AnymatixUpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model_name": ("STRING", {})}}

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "Anymatix"
    DESCRIPTION = "Loads an upscale model for use with upscalers"

    def load_model(self, model_name):
        print("loading upscale model", model_name)
        model_path = model_name
        verify_model_file_exists(model_path, "upscale model")
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
        out = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(out, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")

        return (out,)


class AnymatixCheckpointLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": ("STRING",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    OUTPUT_TOOLTIPS = (
        "The model used for denoising latents.",
        "The CLIP model used for encoding text prompts.",
        "The VAE model used for encoding and decoding images to and from latent space.",
    )
    FUNCTION = "load_checkpoint"

    CATEGORY = "Anymatix"
    DESCRIPTION = "Loads a diffusion model checkpoint, diffusion models are used to denoise latents."

    def load_checkpoint(self, ckpt_name):
        print("loading checkpoint", ckpt_name)
        ckpt_path = ckpt_name
        verify_model_file_exists(ckpt_path, "checkpoint")
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return out[:3]


class AnymatixLoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "The diffusion model the LoRA will be applied to."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "The CLIP model the LoRA will be applied to."},
                ),
                "lora_name": ("STRING",),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                    },
                ),
                "strength_clip": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -100.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "How strongly to modify the CLIP model. This value can be negative.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        print("loading lora", lora_name)
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        # lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora_path = lora_name
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            verify_model_file_exists(lora_path, "lora")
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        return (model_lora, clip_lora)


# Define the custom node class


class AnymatixCheckpointFetcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
                    },
                ),
            },
            "optional": {
                # Optional auth query tail, e.g. "token=xxxx". Not persisted; only used at runtime.
                "auth": ("STRING", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "download_model"
    CATEGORY = "Anymatix"

    def download_model(self, url, auth=None):
        pbar = comfy.utils.ProgressBar(1000)
        progress = 0
        pbar.update_absolute(progress, 1000)

        def callback(x, y):
            import math

            new_progress = round(1000 * x / y)
            nonlocal progress
            if new_progress != progress:
                progress = new_progress
                pbar.update_absolute(progress, 1000)

        # Build base/effective URLs. Prefer explicit auth arg; else, preserve legacy token-in-URL behavior.
        try:
            p = urlparse(url)
            pairs = parse_qsl(p.query, keep_blank_values=True)
            non_auth = []
            legacy_auth_pairs = []
            for k, v in pairs:
                if k == "token":
                    legacy_auth_pairs.append((k, v))
                else:
                    non_auth.append((k, v))
            base_query = urlencode(non_auth)
            base_url = urlunparse(p._replace(query=base_query))

            # Decide which auth to use: explicit auth arg wins; otherwise use legacy token found in URL
            if auth is not None and len(str(auth)) > 0:
                to_add = parse_qsl(str(auth), keep_blank_values=True)
                effective_query = urlencode(non_auth + to_add)
                effective_url = urlunparse(p._replace(query=effective_query))
                auth_tail = str(auth)
            elif legacy_auth_pairs:
                effective_query = urlencode(non_auth + legacy_auth_pairs)
                effective_url = urlunparse(p._replace(query=effective_query))
                auth_tail = urlencode(legacy_auth_pairs)
            else:
                effective_url = base_url
                auth_tail = None
        except Exception:
            # Fallback to original behavior if parsing fails
            base_url = url
            effective_url = url
            auth_tail = None

        try:
            model_name = download_file(
                url=base_url,
                dir=CHECKPOINTS_DIR,
                callback=callback,
                expand_info=expand_info,
                effective_url=effective_url,
                redact_append=auth_tail,
            )
            return (model_name,)
        except Exception as e:
            # Enhance error message with context for better debugging
            error_msg = f"AnymatixCheckpointFetcher failed to download from {base_url}: {e}"
            print(f"[ANYMATIX ERROR] {error_msg}")
            raise Exception(error_msg) from e


dirmap = {
    "checkpoint": "checkpoints",
    "lora": "loras",
    "controlnet": "controlnet",
    "upscale": "upscale_models",
    "vae": "vae",
    "diffusion_model": "diffusion_models",
    "diffusion_models/GGUF": "diffusion_models",
    "text_encoders": "text_encoders",
    "clip_vision": "clip_vision",
    "audio_encoder": "audio_encoders",
    "sam2": "sam2",
    "SEEDVR2_model": "SEEDVR2",
    "SEEDVR2_vae_model": "SEEDVR2",
}


def expand_info_civitai(url):
    # get the model id from the url using a regex that matches the first /.../ after https://civitai.com/api/download/models
    pattern = r"https://civitai\.com/api/download/models/([^/]+)"
    match = re.search(pattern, url)
    if match:
        model_id = match.group(1)
    else:
        return None
    model_info_url = f"https://civitai.com/api/v1/model-versions/{model_id}"
    try:
        with requests.Session() as session:
            return requests.get(model_info_url, allow_redirects=True).json()
    except Exception:
        return None


def expand_info(url):
    if url.startswith("https://civitai.com/api/download/models"):
        return expand_info_civitai(url)
    return None


class AnymatixFetcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # "url": ("STRING", {"default": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"}),
                # Keep declared fields minimal; auth (if present) is accepted at runtime but not exposed in UI
                "url": ({"url": "STRING", "type": "STRING", "auth": "STRING"}, {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "download_model"
    CATEGORY = "Anymatix"

    def download_model(self, url):
        # Avoid printing tokens; only show safe info
        try:
            safe_preview = {k: ("<redacted>" if k == "auth" else v) for k, v in url.items()}
            print("download model", type(url), safe_preview)
        except Exception:
            pass
        if url["type"] in dirmap:
            dir = get_anymatix_models_dir(dirmap[url["type"]])
            pbar = comfy.utils.ProgressBar(1000)
            progress = 0
            pbar.update_absolute(progress, 1000)

            base_url = url.get("url")
            auth = url.get("auth")

            # PRE-DOWNLOAD DEDUPLICATION CHECK
            info = expand_info(base_url)
            if info and "files" in info:
                # Find the file that matches the download request (usually first or specific name)
                # Civitai provides SHA256 hashes in the files list
                target_hash = None
                for f in info["files"]:
                    if "hashes" in f and "SHA256" in f["hashes"]:
                        target_hash = f["hashes"]["SHA256"].lower()
                        break
                
                if target_hash:
                    # Check if we already have a sidecar with this hash
                    print(f"[ANYMATIX] Pre-checking hash for deduplication: {target_hash}")
                    for item in os.listdir(dir):
                        if item.endswith(".json"):
                            try:
                                with open(os.path.join(dir, item), 'r') as sidecar_f:
                                    sidecar_data = json.load(sidecar_f)
                                if sidecar_data.get("sha256") == target_hash:
                                    other_file_name = sidecar_data.get("file_name")
                                    if other_file_name:
                                        other_file_path = os.path.join(dir, other_file_name)
                                        if os.path.exists(other_file_path):
                                            print(f"[ANYMATIX] Model with hash {target_hash} already exists at {other_file_path}. Skipping download.")
                                            # Create sidecar for the current URL
                                            url_hash = hash_string(base_url) # Simple base_url hash for sidecar name
                                            new_sidecar_path = os.path.join(dir, f"{url_hash}.json")
                                            new_data = {
                                                "url": base_url,
                                                "file_name": other_file_name,
                                                "sha256": target_hash,
                                                "data": info
                                            }
                                            with open(new_sidecar_path, 'w') as f:
                                                json.dump(new_data, f, indent=4)
                                            return (other_file_path,)
                            except Exception:
                                pass

            def callback(x, y):
                import math

                new_progress = round(1000 * x / y)
                nonlocal progress
                if new_progress != progress:
                    progress = new_progress
                    pbar.update_absolute(progress, 1000)

            if auth is not None and len(auth) > 0:
                # Robustly append query parameters using urllib
                p = urlparse(base_url)
                existing = parse_qsl(p.query, keep_blank_values=True)
                to_add = parse_qsl(auth, keep_blank_values=True)
                new_query = urlencode(existing + to_add)
                effective = urlunparse(p._replace(query=new_query))
            else:
                effective = base_url

            try:
                model_name = download_file(
                    url=base_url,
                    dir=dir,
                    callback=callback,
                    expand_info=expand_info,
                    effective_url=effective,
                    redact_append=auth,
                )
                print("fetched model", model_name)
                return (model_name,)
            except Exception as e:
                # Enhance error message with context for better debugging
                model_type = url.get("type", "unknown")
                
                # Create user-friendly error messages based on common error patterns
                error_str = str(e)
                if "Expecting value: line 1 column 1 (char 0)" in error_str:
                    user_msg = f"Failed to download {model_type} model: The model information could not be retrieved from Civitai. This may be due to network issues, rate limiting, or the model being private/unavailable."
                elif "timeout" in error_str.lower():
                    user_msg = f"Failed to download {model_type} model: The download timed out. Please check your internet connection and try again."
                elif "connection" in error_str.lower():
                    user_msg = f"Failed to download {model_type} model: Could not connect to the download server. Please check your internet connection."
                elif "404" in error_str or "not found" in error_str.lower():
                    user_msg = f"Failed to download {model_type} model: The model was not found on the server. It may have been moved or deleted."
                elif "403" in error_str or "forbidden" in error_str.lower():
                    user_msg = f"Failed to download {model_type} model: Access denied. The model may be private or require authentication."
                elif "429" in error_str or "rate limit" in error_str.lower():
                    user_msg = f"Failed to download {model_type} model: Too many requests. Please wait a moment and try again."
                elif "Errno 28" in error_str or "No space left on device" in error_str:
                    storage_path = dir
                    user_msg = f"Failed to download {model_type} model: No space left on device.\n[ANYMATIX_STORAGE_FULL:{storage_path}]"
                else:
                    user_msg = f"Failed to download {model_type} model: {e}"
                
                print(f"[ANYMATIX ERROR] {user_msg}")
                print(f"[ANYMATIX DEBUG] Original error: {error_str}")
                print(f"[ANYMATIX DEBUG] URL: {base_url}")
                
                # Include URL in user message for better debugging
                user_msg_with_url = f"{user_msg}\nURL: {base_url}"
                raise Exception(user_msg_with_url) from e

    @classmethod
    def IS_CHANGED(cls, url):
        """
        Check if the model file exists. If not, return NaN to force re-download.
        This prevents ComfyUI from using cached output paths when the actual model
        file has been deleted.
        """
        if url["type"] not in dirmap:
            return float("NaN")
        
        base_url = url.get("url")
        auth = url.get("auth")
        if auth is not None and len(auth) > 0:
            p = urlparse(base_url)
            existing = parse_qsl(p.query, keep_blank_values=True)
            to_add = parse_qsl(auth, keep_blank_values=True)
            new_query = urlencode(existing + to_add)
            effective = urlunparse(p._replace(query=new_query))
        else:
            effective = base_url
        
        url_hash = hash_string(effective)
        dir_path = get_anymatix_models_dir(dirmap[url["type"]])
        json_path = os.path.join(dir_path, f"{url_hash}.json")
        
        # If JSON doesn't exist, file needs to be downloaded
        if not os.path.exists(json_path):
            return float("NaN")
        
        # Read JSON to get expected file name
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            file_name = data.get("file_name")
            if not file_name:
                return float("NaN")
            file_path = os.path.join(dir_path, file_name)
            
            # If model file doesn't exist, force re-download
            if not os.path.exists(file_path):
                print(f"[ANYMATIX IS_CHANGED] Model file missing, forcing re-download: {file_path}")
                return float("NaN")
            
            # File exists, return hash for stable caching
            return url_hash
        except Exception as e:
            print(f"[ANYMATIX IS_CHANGED] Error checking model file: {e}")
            return float("NaN")


# Attribution: Based on DownloadAndLoadSAM2Model from
# https://github.com/kijai/ComfyUI-segment-anything-2
# Original code licensed under Apache 2.0
# Modified for Anymatix fetcher pattern integration
class AnymatixSAM2Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam2_name": ("STRING", ),
                "segmentor": (["single_image", "video", "automaskgenerator"], ),
                "device": (["cuda", "cpu", "mps"], ),
                "precision": (["fp16", "bf16", "fp32"], ),
            }
        }

    RETURN_TYPES = ("SAM2MODEL",)
    RETURN_NAMES = ("sam2_model",)
    FUNCTION = "load_model"
    CATEGORY = "Anymatix"
    DESCRIPTION = "Loads a SAM2 segmentation model from a fetched file"

    def load_model(self, sam2_name, segmentor, device, precision):
        import torch
        import importlib.util
        import sys

        sam2_nodes_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "ComfyUI-segment-anything-2")
        )
        
        if not os.path.exists(sam2_nodes_path):
            raise Exception(
                "ComfyUI-segment-anything-2 not found. "
                "Please install it from https://github.com/kijai/ComfyUI-segment-anything-2"
            )

        # Import load_model as a module to support its relative imports
        load_model_path = os.path.join(sam2_nodes_path, "load_model.py")
        spec = importlib.util.spec_from_file_location("ComfyUI_segment_anything_2.load_model", load_model_path)
        if spec is None or spec.loader is None:
            raise Exception(f"Could not load load_model.py from {load_model_path}")
        
        load_model_module = importlib.util.module_from_spec(spec)
        sys.modules["ComfyUI_segment_anything_2.load_model"] = load_model_module
        sys.modules["ComfyUI_segment_anything_2"] = type(sys)("ComfyUI_segment_anything_2")
        sys.modules["ComfyUI_segment_anything_2"].__path__ = [sam2_nodes_path]
        
        try:
            spec.loader.exec_module(load_model_module)
            sam2_load_model = load_model_module.load_model
        except Exception as e:
            raise Exception(
                f"Failed to import load_model from ComfyUI-segment-anything-2: {e}"
            )


        if precision != 'fp32' and device == 'cpu':
            raise ValueError("fp16 and bf16 are not supported on cpu")

        if device == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        device_obj = {"cuda": torch.device("cuda"), "cpu": torch.device("cpu"), "mps": torch.device("mps")}[device]

        model_path = os.path.basename(sam2_name)

        if precision != 'fp32' and "2.1" in model_path:
            base_name, extension = model_path.rsplit('.', 1)
            model_path = f"{base_name}-fp16.{extension}"

        sam2_models_dir = get_anymatix_models_dir("sam2")
        full_model_path = os.path.join(sam2_models_dir, model_path)

        if not os.path.exists(full_model_path):
            raise FileNotFoundError(
                f"SAM2 model not found at {full_model_path}. "
                f"Make sure AnymatixFetcher has downloaded it first."
            )

        # Config files are in ComfyUI-segment-anything-2/sam2_configs/
        model_mapping = {
            "2.0": {
                "base": "sam2_hiera_b+.yaml",
                "large": "sam2_hiera_l.yaml",
                "small": "sam2_hiera_s.yaml",
                "tiny": "sam2_hiera_t.yaml"
            },
            "2.1": {
                "base": "sam2.1_hiera_b+.yaml",
                "large": "sam2.1_hiera_l.yaml",
                "small": "sam2.1_hiera_s.yaml",
                "tiny": "sam2.1_hiera_t.yaml"
            }
        }

        version = "2.1" if "2.1" in model_path else "2.0"
        model_cfg_path = next(
            (os.path.join(sam2_nodes_path, "sam2_configs", cfg)
             for key, cfg in model_mapping[version].items() if key in model_path),
            None
        )

        if not model_cfg_path:
            raise ValueError(f"Could not determine config for model: {model_path}")

        print(f"Loading SAM2 model from: {full_model_path}")
        print(f"Using config: {model_cfg_path}")

        model = sam2_load_model(full_model_path, model_cfg_path, segmentor, dtype, device_obj)

        sam2_model = {
            'model': model,
            'dtype': dtype,
            'device': device_obj,
            'segmentor': segmentor,
            'version': version
        }

        return (sam2_model,)


if __name__ == "__main__":
    # Quick test for AnymatixFetcher
    fetcher = AnymatixFetcher()
    test_url = {
        "url": "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
        "type": "checkpoint",
        "auth": ""
    }
    try:
        result = fetcher.download_model(test_url)
        print("Test fetch result:", result)
    except Exception as e:
        print("Test fetch failed:", e)

