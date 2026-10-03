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
# Pinned, not `main`: bugs/create-mask-fetches-clipseg-from-moving-main.
# `AnymatixCLIPSeg` takes no url input (its INPUT_TYPES has none), so a
# pinned constant is the fix, same standing rule as every other shipped url
# (todos/shipped-model-urls-name-branch-not-commit, 2026-09-16): "pin them
# all, and we will update the app at least once per month".
CLIPSEG_REVISION = "999e0328d9e10b484360c477313983f9afdd7050"
CLIPSEG_BASE_URL = f"https://huggingface.co/{CLIPSEG_MODEL_ID}/resolve/{CLIPSEG_REVISION}"

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


def blur_heatmap(preds, blur):
    """Box-blur a 2-D CLIPSeg heat map, WITHOUT inventing pixels outside the frame.

    `F.avg_pool2d` pads with zeros, and PyTorch's `count_include_pad` defaults
    to **True**, so those invented zeros land in the denominator. Every pixel
    within `kernel_size // 2` of an edge is then averaged against a strip of
    nothing and pulled towards 0 — and the `> threshold` on the next line in
    `segment()` cuts exactly there. Measured on a uniform heat map of 0.9 at
    the shipped `blur = 6` (kernel 13): a corner came out **0.261** and a
    mid-edge pixel **0.485**, against the shipped threshold of 0.4. So any
    object touching the frame lost its border from the mask.

    `count_include_pad=False` divides by the number of REAL pixels the window
    covered, which is what "smooth the heat map" means at a boundary: the same
    0.9 comes out 0.900 everywhere, edges and corners included. Nothing away
    from the border moves by a bit.

    See `TRACKERS/BUGS/mask-blur-eats-mask-frame-border` in the app repository,
    and `tests/test_clipseg_blur.py` here, which fails under the old keyword.
    """
    import torch.nn.functional as F

    if blur <= 0:
        return preds

    kernel_size = int(blur) * 2 + 1
    padding = kernel_size // 2
    return F.avg_pool2d(
        preds.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        count_include_pad=False,
    ).squeeze()


def pad_convolutions_by_replication(model):
    """Make CLIPSeg's padded convolutions continue the picture past the frame, not pad it with zeros.

    The pinned `clipseg-rd64-refined` decoder ends in one 3x3
    `nn.Conv2d(padding=1)` over its 22x22 token grid
    (`decoder.transposed_convolution[0]`), and PyTorch pads with ZEROS unless
    told otherwise. Every token on the border of the grid is averaged against
    a ring of invented nothing, and the two 4x4/stride-4 transposed
    convolutions after it turn each token into its own 16x16 block of the
    352x352 heat map -- so the heat map sags over the outer block on every
    side, whatever the picture shows there. Measured on the pinned weights
    (CPU, torch 2.11, transformers 5.6), a 1024x1024 picture filled with one
    orange, prompt "orange": 0.95 inside, and 0.08, 0.12, 0.28, 0.48, 0.62
    over the five heat-map rows nearest the top edge. The 0.4 threshold cuts
    the first three, so the mask of an object crossing the frame stopped short
    of the crossed edge, up to 15 px on a 1024-px picture, and lost the corners
    -- at Mask Blur 0 as at 6. The resize, blur, threshold and dilation after
    the model all keep a heat map that reaches the frame on the frame (see the
    tests); the margin was made here.

    `padding_mode="replicate"` pads the grid with its own border tokens: the
    picture continues past the frame as it is at the frame. The same five rows
    read 0.93 each. Only the outer ring of tokens has a padded neighbour, so
    the heat map more than 16 px inside the border is BIT-identical with and
    without this (measured with `torch.equal` on the pumpkin the shipped card
    bakes, and on the test pictures). Replicate, not reflect: reflect pads a
    border token with the token inside it, and then an object ending 24 px
    before the bottom of a 1024-px picture reached the frame; with replicate it
    stops short, as it should. Mirroring the picture itself past the frame was
    measured too and rejected: it reaches the frame as well, but it changes
    the token grid, so it moves the interior (on a half-teal, half-orange
    picture: the heat map up to 0.72 off inside, the mask's IoU against the
    shipped one 0.80).

    Returns how many convolutions it changed: 1 on the pinned weights. The
    patch embedding is a convolution too, unpadded, and stays as it is.

    See `TRACKERS/BUGS/create-mask-stops-11-15-px-short` in the app
    repository, and `tests/test_clipseg_frame_edge.py` here.
    """
    import torch.nn as nn

    padded = [m for m in model.modules() if isinstance(m, nn.Conv2d) and any(m.padding)]
    for conv in padded:
        conv.padding_mode = "replicate"
    return len(padded)


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
        pad_convolutions_by_replication(model)

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

        preds = blur_heatmap(preds, blur)

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
