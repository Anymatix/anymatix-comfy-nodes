import json
import os
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
import re

import requests
import comfy
import comfy.sd
import comfy.utils
import folder_paths
try:
    # When loaded as a package inside ComfyUI, use relative import
    from .fetch import download_file
except Exception:
    # When running this file directly for testing, fall back to absolute import
    try:
        from fetch import download_file
    except Exception:
        # Provide a helpful error when import truly fails
        raise
from spandrel import ModelLoader, ImageModelDescriptor
from nodes import CLIPLoader, UNETLoader, VAELoader, CLIPVisionLoader, LoraLoaderModelOnly, DualCLIPLoader
from comfy_extras.nodes_audio_encoder import AudioEncoderLoader

gguf_nodes_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ComfyUI-GGUF", "nodes.py")
)

import sys
custom_nodes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if custom_nodes_path not in sys.path:
    sys.path.insert(0, custom_nodes_path)

# Try to load GGUF nodes module explicitly from the sibling ComfyUI-GGUF package
import importlib.util
import types

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

CHECKPOINTS_DIR = os.path.join(folder_paths.models_dir, "checkpoints")

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
        self.vae_list()
        return super().load_vae(os.path.basename(vae_name))

class AnymatixCLIPLoader(CLIPLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": ("STRING", ),
                              "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image"], ),
                              },
                "optional": {
                              "device": (["default", "cpu"], {"advanced": True}),
    
                             }}

    CATEGORY = "Anymatix"    

    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        return super().load_clip(os.path.basename(clip_name), type, device)

AnymatixCLIPLoader2=AnymatixCLIPLoader
class AnymatixDualCLIPLoader(DualCLIPLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name1": ("STRING", ),
                              "clip_name2": ("STRING", ),
                              "type": (["sdxl", "sd3", "flux", "hunyuan_video", "hidream"], ),
                            },
                "optional": {
                    "device": (["default", "cpu"], {"advanced": True}),
                            }
                }

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

class AnymatixAudioEncoderLoader(AudioEncoderLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio_encoder_name": ("STRING", ),
                             }}
    CATEGORY = "Anymatix"
    
    def load_model(self, audio_encoder_name):
        return super().load_model(os.path.basename(audio_encoder_name))
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

        def expand_info_civitai(url):
            # get the model id from the url using a regex that matches the first /.../ after https://civitai.com/api/download/models
            pattern = r"https://civitai\.com/api/download/models/([^/]+)"
            match = re.search(pattern, url)
            if match:
                model_id = match.group(1)
            else:
                return None
            model_info_url = f"https://civitai.com/api/v1/model-versions/{model_id}"
            with requests.Session() as session:
                return requests.get(model_info_url, allow_redirects=True).json()

        def expand_info(url):
            if url.startswith("https://civitai.com/api/download/models"):
                return expand_info_civitai(url)
            return None

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
}


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
            dir = os.path.join(folder_paths.models_dir, dirmap[url["type"]])
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

            def expand_info_civitai(url):
                # get the model id from the url using a regex that matches the first /.../ after https://civitai.com/api/download/models
                pattern = r"https://civitai\.com/api/download/models/([^/]+)"
                match = re.search(pattern, url)
                if match:
                    model_id = match.group(1)
                else:
                    return None
                model_info_url = f"https://civitai.com/api/v1/model-versions/{model_id}"
                with requests.Session() as session:
                    return requests.get(model_info_url, allow_redirects=True).json()

            def expand_info(url):
                if url.startswith("https://civitai.com/api/download/models"):
                    return expand_info_civitai(url)
                return None

            base_url = url.get("url")
            auth = url.get("auth")
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
                else:
                    user_msg = f"Failed to download {model_type} model: {e}"
                
                print(f"[ANYMATIX ERROR] {user_msg}")
                print(f"[ANYMATIX DEBUG] Original error: {error_str}")
                print(f"[ANYMATIX DEBUG] URL: {base_url}")
                
                # Include URL in user message for better debugging
                user_msg_with_url = f"{user_msg}\nURL: {base_url}"
                raise Exception(user_msg_with_url) from e


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

        sam2_nodes_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "ComfyUI-segment-anything-2")
        )
        if os.path.exists(sam2_nodes_path) and sam2_nodes_path not in sys.path:
            sys.path.insert(0, sam2_nodes_path)

        try:
            from load_model import load_model as sam2_load_model
        except ImportError:
            raise Exception(
                "ComfyUI-segment-anything-2 not found. "
                "Please install it from https://github.com/kijai/ComfyUI-segment-anything-2"
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

        sam2_models_dir = os.path.join(folder_paths.models_dir, "sam2")
        full_model_path = os.path.join(sam2_models_dir, model_path)

        if not os.path.exists(full_model_path):
            raise FileNotFoundError(
                f"SAM2 model not found at {full_model_path}. "
                f"Make sure AnymatixFetcher has downloaded it first."
            )

        script_directory = os.path.dirname(sam2_nodes_path)
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
            (os.path.join(script_directory, "sam2_configs", cfg)
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

