"""
ANY NUMBER OF MINIMAX H3 KEYFRAMES, EACH AT ITS OWN FRAME, IN ONE NODE.

`AnymatixH3AddGuides` is ComfyUI's `MiniMaxH3AddGuide` applied once per
keyframe, in order, to the same conditioning, as `AnymatixAddGuides` is
`LTXVAddGuide` for LTX. The keyframes arrive as the same ONE input: a JSON list
of pairs, `[{"image": "<file in input/>", "frame": 18}, ...]`, built by the
app at compile time (`KeyframePairs`) from a card's picture collection and its
list of frames. Parsing is `anymatix_ltx_guides.parse_keyframes`, shared, so
the two nodes accept and refuse exactly the same lists.

How it differs from the LTX node, both from `MiniMaxH3AddGuide` itself:
- an H3 guide changes only the positive conditioning (it records the keyframe
  under `minimax_keyframes`); the latent is read for its size, never written,
  so there is no negative and no latent output;
- there is no strength.

`skip` leaves out the first pairs. The measured whole-shot run (sh010, 18 keys,
2026-11-27) sent the first key as `MiniMaxH3ImageToVideo`'s `first_frame` and
only the further keys through `MiniMaxH3AddGuide`; a card that does the same
binds `skip` to 1, so the first picture is not anchored twice.

Why the images are file names: the app uploads every picture of a collection
to ComfyUI's input folder under `<hash>.<ext>` before the run; this node reads
them with `LoadImage`, so a keyframe is loaded exactly as a picture input is.
"""

try:
    from .anymatix_ltx_guides import parse_keyframes
except ImportError:
    # Loaded standalone (tests via spec_from_file_location), with no package
    # context for a relative import to resolve against.
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from anymatix_ltx_guides import parse_keyframes


def _positive(output):
    """`MiniMaxH3AddGuide.execute` returns an `io.NodeOutput` (V3) or a tuple (V1)."""
    values = getattr(output, "result", output)
    return values[0]


class AnymatixH3AddGuides:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "keyframes": ("STRING", {"default": "[]", "multiline": True}),
            },
            "optional": {
                "skip": ("INT", {"default": 0, "min": 0, "max": 9999}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "add"
    CATEGORY = "Anymatix"

    def add(self, positive, latent, vae, keyframes, skip=0):
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
        from nodes import LoadImage

        loader = LoadImage()
        for image_name, frame in parse_keyframes(keyframes)[skip:]:
            image = loader.load_image(image_name)[0]
            positive = _positive(
                MiniMaxH3AddGuide.execute(
                    positive=positive,
                    latent=latent,
                    frame_idx=frame,
                    vae=vae,
                    image=image,
                )
            )
        return (positive,)

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
