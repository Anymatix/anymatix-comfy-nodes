import os
import torch
import numpy as np
from PIL import Image

import folder_paths

try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

CLIPSEG_MODEL_ID = "CIDAS/clipseg-rd64-refined"
CLIPSEG_BASE_URL = f"https://huggingface.co/{CLIPSEG_MODEL_ID}/resolve/main"

# Files to download from the HuggingFace repository (preserving original names)
CLIPSEG_CONFIG_FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]

# Weight files to try in order (stop after first successful download)
CLIPSEG_WEIGHT_FILES = [
    "model.safetensors",
    "pytorch_model.bin",
]


def get_clipseg_model_dir() -> str:
    model_dir = os.path.join(folder_paths.models_dir, "clip_seg", "CIDAS_clipseg-rd64-refined")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def _download_to_path(url: str, dest_path: str) -> None:
    """Download url to dest_path using requests. Raises on HTTP error."""
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library is required for downloading CLIPSeg model files")
    import comfy.utils
    pbar = comfy.utils.ProgressBar(1000)
    response = _requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pbar.update_absolute(round(1000 * downloaded / total), 1000)
    pbar.update_absolute(1000, 1000)


def ensure_clipseg_model() -> str:
    """Download all CLIPSeg model files to the Anymatix models directory if not present."""
    model_dir = get_clipseg_model_dir()

    for filename in CLIPSEG_CONFIG_FILES:
        dest = os.path.join(model_dir, filename)
        if not os.path.exists(dest):
            url = f"{CLIPSEG_BASE_URL}/{filename}"
            print(f"[AnymatixCLIPSeg] Downloading {filename} ...")
            try:
                _download_to_path(url, dest)
            except Exception as e:
                if hasattr(e, "response") and e.response is not None and e.response.status_code == 404:
                    print(f"[AnymatixCLIPSeg] {filename} not found in repo, skipping")
                else:
                    raise

    has_weights = any(
        os.path.exists(os.path.join(model_dir, wf)) for wf in CLIPSEG_WEIGHT_FILES
    )
    if not has_weights:
        downloaded_weights = False
        for weight_file in CLIPSEG_WEIGHT_FILES:
            url = f"{CLIPSEG_BASE_URL}/{weight_file}"
            dest = os.path.join(model_dir, weight_file)
            print(f"[AnymatixCLIPSeg] Downloading {weight_file} ...")
            try:
                _download_to_path(url, dest)
                downloaded_weights = True
                break
            except Exception as e:
                if hasattr(e, "response") and e.response is not None and e.response.status_code == 404:
                    continue
                raise
        if not downloaded_weights:
            raise RuntimeError(
                f"[AnymatixCLIPSeg] Could not download model weights for {CLIPSEG_MODEL_ID}. "
                "Tried: " + ", ".join(CLIPSEG_WEIGHT_FILES)
            )

    return model_dir


class AnymatixCLIPSeg:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": False}),
            },
            "optional": {
                "blur": ("FLOAT", {"min": 0, "max": 15, "step": 0.1, "default": 0}),
                "threshold": ("FLOAT", {"min": 0, "max": 1, "step": 0.05, "default": 0.4}),
                "dilation_factor": ("INT", {"min": 0, "max": 10, "step": 1, "default": 5}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "segment"
    CATEGORY = "Anymatix"

    def segment(self, image, text, blur=0, threshold=0.4, dilation_factor=5):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        import torch.nn.functional as F

        model_dir = ensure_clipseg_model()

        processor = CLIPSegProcessor.from_pretrained(model_dir, local_files_only=True)
        model = CLIPSegForImageSegmentation.from_pretrained(model_dir, local_files_only=True)
        model.eval()

        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        h, w = image.shape[1], image.shape[2]
        inputs = processor(text=[text], images=[pil_image], return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        preds = outputs.logits.squeeze()
        preds = torch.sigmoid(preds)

        preds = F.interpolate(
            preds.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        if blur > 0:
            kernel_size = int(blur) * 2 + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            padding = kernel_size // 2
            preds = F.avg_pool2d(
                preds.unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            ).squeeze()

        mask = (preds > threshold).float()

        if dilation_factor > 0:
            kernel_size = 2 * dilation_factor + 1
            mask = F.max_pool2d(
                mask.unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=dilation_factor,
            ).squeeze()

        return (mask,)


NODE_CLASS_MAPPINGS = {"AnymatixCLIPSeg": AnymatixCLIPSeg}
NODE_DISPLAY_NAME_MAPPINGS = {"AnymatixCLIPSeg": "Anymatix CLIPSeg"}
