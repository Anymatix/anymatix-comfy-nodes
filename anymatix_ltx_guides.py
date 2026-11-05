"""
ANY NUMBER OF LTX KEYFRAMES, EACH AT ITS OWN FRAME, IN ONE NODE.

`AnymatixAddGuides` is ComfyUI's `LTXVAddGuide` applied once per keyframe, in
order, to the same conditioning and latent. The keyframes arrive as ONE input:
a JSON list of pairs, `[{"image": "<file in input/>", "frame": 18}, ...]`.

Why a list of pairs and not N inputs: KJNodes' `LTXVAddGuideMulti` stops at a
fixed count (its dynamic combo) and a card had to bind every place of it. A
Final Cut timeline has as many cuts as it has (Franco's sh010: 19 on one shot).
The pair is the record the app will one day type; until then the app builds
the list at compile time (`KeyframePairs`) from the card's picture collection
and its list of frames, so neither the node nor the list changes when it does.

Why the images are file names: the app uploads every picture of a collection
to ComfyUI's input folder under `<hash>.<ext>` before the run, the same files
a `LoadImage` would read; this node reads them with `LoadImage` itself, so a
keyframe is loaded exactly as a picture input is.

Each pair keeps its exact frame. `LTXVAddGuide.get_latent_index` snaps a
frame only for a multi-frame guide; a still lands where the timeline put it.
"""

import json


def parse_keyframes(text):
    """The pairs, checked: a non-empty list of {"image": str, "frame": int}."""
    try:
        pairs = json.loads(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"keyframes is not JSON: {error}") from error
    if not isinstance(pairs, list) or len(pairs) == 0:
        raise ValueError("keyframes must be a non-empty list of {image, frame} pairs")
    out = []
    for position, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict):
            raise ValueError(f"keyframe {position} is not an {{image, frame}} pair")
        image = pair.get("image")
        frame = pair.get("frame")
        if not isinstance(image, str) or image == "":
            raise ValueError(f"keyframe {position} names no image")
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise ValueError(f"keyframe {position}'s frame is not a whole number: {frame!r}")
        out.append((image, frame))
    return out


def _result(output):
    """`LTXVAddGuide.execute` returns an `io.NodeOutput` (V3) or a tuple (V1)."""
    values = getattr(output, "result", output)
    return values[0], values[1], values[2]


class AnymatixAddGuides:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "keyframes": ("STRING", {"default": "[]", "multiline": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "add"
    CATEGORY = "Anymatix"

    def add(self, positive, negative, vae, latent, keyframes, strength):
        from comfy_extras.nodes_lt import LTXVAddGuide
        from nodes import LoadImage

        loader = LoadImage()
        for image_name, frame in parse_keyframes(keyframes):
            image = loader.load_image(image_name)[0]
            positive, negative, latent = _result(
                LTXVAddGuide.execute(
                    positive=positive,
                    negative=negative,
                    vae=vae,
                    latent=latent,
                    image=image,
                    frame_idx=frame,
                    strength=strength,
                )
            )
        return (positive, negative, latent)

    @classmethod
    def VALIDATE_INPUTS(cls, keyframes):
        import folder_paths

        try:
            pairs = parse_keyframes(keyframes)
        except ValueError as error:
            return str(error)
        for image_name, _frame in pairs:
            if not folder_paths.exists_annotated_filepath(image_name):
                return f"keyframe picture not on the machine: {image_name}"
        return True
