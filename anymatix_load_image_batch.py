"""
AnymatixLoadImageBatch: a batch of uploaded frames, as one IMAGE batch.

Why it exists (app/specifications/multi-machine-runs-v1.md §3.3, §7; output-materialization-v1.0.md §12b, §13):
a run can be split across machines, and a value crosses from one machine to another through the app. The app
uploads every frame as a content-addressed input file (`<sha256>.png`, the upload endpoint refuses bytes whose
digest is not the name), and this node turns the list of names back into the `IMAGE` batch the consuming node
reads, in order. The first user is the 3D Code card: the app renders its colour and depth frames, and the host's
`AnymatixImageToVideo` encodes them, its `images` input fed by this node.

Each frame is decoded exactly as ComfyUI's own `LoadImage` decodes a picture (EXIF orientation applied, converted
to RGB, float32 in [0, 1], shape [H, W, 3]), so a batch of N frames is the tensor N `LoadImage`s would give, stacked.
A grey depth frame becomes three equal channels, as `LoadImage` makes it.

What it refuses, each naming the frame:
  - an empty list;
  - a name that is not a plain file name (a path could read outside the input folder);
  - a name whose bytes are not that digest: an address must mean the bytes it names;
  - frames of different sizes: a batch is one tensor.

`load_image_batch` imports neither `comfy` nor `folder_paths`, so it is tested without a ComfyUI; the node only
resolves the names in the input folder and calls it.
"""

import hashlib
import os
import re

import numpy as np
from PIL import Image, ImageOps

_CONTENT_NAME = re.compile(r"^([0-9a-f]{64})\.[A-Za-z0-9]+$")


class FrameBatchError(ValueError):
    pass


def frame_names(files):
    """The names in `files`, one per line, blank lines ignored. Refuses anything that is not a plain file name."""
    names = [line.strip() for line in str(files).splitlines() if line.strip()]
    if not names:
        raise FrameBatchError("AnymatixLoadImageBatch: no frames were given")
    for i, name in enumerate(names, start=1):
        if os.path.basename(name) != name or name in (".", "..") or "\\" in name:
            raise FrameBatchError(f"AnymatixLoadImageBatch: frame {i} is not a file name: {name!r}")
    return names


def _decode(path):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return np.array(image).astype(np.float32) / 255.0


def load_image_batch(paths, names=None, verify_digest=True):
    """[N, H, W, 3] float32 in [0, 1], frame i from paths[i]. `names` (default: the basenames) are used in errors
    and, when a name is a content address `<sha256>.<ext>`, checked against the bytes."""
    names = names if names is not None else [os.path.basename(p) for p in paths]
    if len(paths) == 0:
        raise FrameBatchError("AnymatixLoadImageBatch: no frames were given")
    frames = []
    for i, (path, name) in enumerate(zip(paths, names), start=1):
        if not os.path.isfile(path):
            raise FrameBatchError(f"AnymatixLoadImageBatch: frame {i} ({name}) was not uploaded to this machine")
        match = _CONTENT_NAME.match(name)
        if verify_digest and match:
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            if digest != match.group(1):
                raise FrameBatchError(
                    f"AnymatixLoadImageBatch: frame {i} ({name}) holds other bytes (sha256 {digest})"
                )
        frame = _decode(path)
        if frames and frame.shape != frames[0].shape:
            raise FrameBatchError(
                f"AnymatixLoadImageBatch: frame {i} ({name}) is {frame.shape[1]}x{frame.shape[0]}, "
                f"frame 1 is {frames[0].shape[1]}x{frames[0].shape[0]}"
            )
        frames.append(frame)
    return np.stack(frames, axis=0)


class AnymatixLoadImageBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "files": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load"
    CATEGORY = "Anymatix"
    DESCRIPTION = (
        "A batch of uploaded frames, one input file name per line, in order, as one IMAGE batch. "
        "Each frame is decoded as LoadImage decodes it."
    )

    def load(self, files):
        import torch
        import folder_paths

        names = frame_names(files)
        paths = [folder_paths.get_annotated_filepath(name) for name in names]
        batch = load_image_batch(paths, names)
        print(f"AnymatixLoadImageBatch: {batch.shape[0]} frames of {batch.shape[2]}x{batch.shape[1]}")
        return (torch.from_numpy(batch),)
