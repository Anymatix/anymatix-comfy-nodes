import math

import comfy.utils


class AnymatixLTXResizeToClosestValidSize:
    """
    Resize an input image to an LTX-friendly output size while preserving aspect ratio.

    This helper is intended for the two-pass LTX 2.3 I2V workflows that render a
    half-resolution base pass and then apply a spatial x2 upscale. To keep the
    half-resolution latent canvas aligned with LTX's 32px grid, the final output
    size is snapped to 64px multiples.
    """

    SNAP_MULTIPLE = 64
    SCALE_METHOD = "lanczos"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_height": ("INT", {"default": 720, "min": cls.SNAP_MULTIPLE, "max": 16384, "step": 1}),
                "max_longer_edge": ("INT", {"default": 1920, "min": cls.SNAP_MULTIPLE, "max": 16384, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Anymatix"

    @classmethod
    def _snap_to_multiple(cls, value: float) -> int:
        snapped = int(math.floor((value / cls.SNAP_MULTIPLE) + 0.5)) * cls.SNAP_MULTIPLE
        return max(cls.SNAP_MULTIPLE, snapped)

    @classmethod
    def _snap_down_to_multiple(cls, value: float) -> int:
        snapped = int(math.floor(value / cls.SNAP_MULTIPLE)) * cls.SNAP_MULTIPLE
        return max(cls.SNAP_MULTIPLE, snapped)

    @classmethod
    def _compute_target_size(cls, width: int, height: int, target_height: int, max_longer_edge: int) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {width}x{height}")

        aspect_ratio = width / height
        scaled_height = float(target_height)
        scaled_width = scaled_height * aspect_ratio

        longer_edge = max(scaled_width, scaled_height)
        if longer_edge > max_longer_edge:
            cap_scale = max_longer_edge / longer_edge
            scaled_width *= cap_scale
            scaled_height *= cap_scale

        if scaled_width >= scaled_height:
            snapped_width = min(
                cls._snap_to_multiple(scaled_width),
                cls._snap_down_to_multiple(max_longer_edge),
            )
            snapped_height = cls._snap_to_multiple(snapped_width / aspect_ratio)
            if snapped_height > max_longer_edge:
                snapped_height = cls._snap_down_to_multiple(max_longer_edge)
                snapped_width = cls._snap_to_multiple(snapped_height * aspect_ratio)
        else:
            snapped_height = min(
                cls._snap_to_multiple(scaled_height),
                cls._snap_down_to_multiple(max_longer_edge),
            )
            snapped_width = cls._snap_to_multiple(snapped_height * aspect_ratio)
            if snapped_width > max_longer_edge:
                snapped_width = cls._snap_down_to_multiple(max_longer_edge)
                snapped_height = cls._snap_to_multiple(snapped_width / aspect_ratio)

        if max(snapped_width, snapped_height) > max_longer_edge:
            if snapped_width >= snapped_height:
                snapped_width = cls._snap_down_to_multiple(max_longer_edge)
                snapped_height = cls._snap_down_to_multiple(snapped_width / aspect_ratio)
            else:
                snapped_height = cls._snap_down_to_multiple(max_longer_edge)
                snapped_width = cls._snap_down_to_multiple(snapped_height * aspect_ratio)

        return snapped_width, snapped_height

    def process(self, image, target_height, max_longer_edge):
        if len(image.shape) != 4:
            raise ValueError(f"Expected IMAGE tensor with shape [batch, height, width, channels], got {image.shape}")

        _, height, width, _ = image.shape
        output_width, output_height = self._compute_target_size(width, height, target_height, max_longer_edge)

        resized = comfy.utils.common_upscale(
            image.movedim(-1, 1),
            output_width,
            output_height,
            self.SCALE_METHOD,
            "disabled",
        ).movedim(1, -1)

        return (resized,)
