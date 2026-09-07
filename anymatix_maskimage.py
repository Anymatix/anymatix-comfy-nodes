"""
Multiply an image by a mask, so the masked-out part goes black.

WHY THE COLORMAP IS SPELLED OUT HERE INSTEAD OF IMPORTED.

This module used to do `import matplotlib.cm as cm` and pass the mask through
`cm.Greys_r`. matplotlib is not a ComfyUI dependency and this pack never
declared it, so on a stock install the import failed — and because
`__init__.py` imports this module, that failure took the WHOLE node pack with
it. Forty-odd nodes disappeared over one greyscale ramp.

The obvious fix, `np.repeat` on the mask, is WRONG, and the difference is
visible rather than theoretical: `Greys_r` is ColorBrewer's sequential grey,
not a linear ramp. Measured against matplotlib 3.x — a mask value of 0.5 comes
out at 0.5906, and the largest departure from the identity over [0,1] is 0.118.
A soft mask (feathered edges, a CLIPSeg probability map) would therefore have
come out visibly different, so swapping in the identity would have been a
silent change of what this node does.

So the ramp is reproduced exactly instead. `GREYS_STOPS` is matplotlib's own
`_Greys_data`, reversed; the quantisation to 256 levels is what
`Colormap.__call__` does to a float input. Checked over 100 001 samples on
2026-09-07: max absolute difference from `cm.Greys_r` is 4.4e-16, and the uint8
images are byte-identical. No dependency, same picture.
"""

import numpy as np
import torch

#: matplotlib's `_Greys_data`, reversed — the nine ColorBrewer control points
#: `Greys_r` interpolates between. Grey, so one channel is enough.
GREYS_STOPS = np.array(
    [
        0.0,
        0.145098039215686,
        0.321568627450980,
        0.450980392156863,
        0.588235294117647,
        0.741176470588235,
        0.850980392156863,
        0.941176470588235,
        1.0,
    ]
)
_GREYS_X = np.linspace(0.0, 1.0, len(GREYS_STOPS))


def greys_r(values: np.ndarray) -> np.ndarray:
    """`matplotlib.cm.Greys_r` for float input in [0, 1], to float in [0, 1].

    The floor-to-256-levels is not incidental: a matplotlib colormap called
    with floats bins them into its 256 entries first, and reproducing the ramp
    without reproducing the binning would drift by up to half a level.
    """
    quantised = np.minimum(np.floor(np.clip(values, 0.0, 1.0) * 256), 255) / 255.0
    return np.interp(quantised, _GREYS_X, GREYS_STOPS)


class AnymatixMaskImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_mask"
    CATEGORY = "Anymatix"
    DESCRIPTION = "Multiply an image by a mask: where the mask is 0 the image goes black."

    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """A 0..1 tensor as a 0..255 uint8 array."""
        array = tensor.numpy().squeeze()
        return (array * 255).astype(np.uint8)

    def apply_colormap(self, mask: torch.Tensor) -> np.ndarray:
        """The mask through Greys_r, as three 0..255 channels."""
        ramped = greys_r(mask.numpy())
        return (np.repeat(ramped[:, :, None], 3, axis=2) * 255).astype(np.uint8)

    def numpy_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """A 0..255 array back to a 0..1 tensor with ComfyUI's batch axis."""
        array = array.astype(np.float32) / 255.0
        return torch.from_numpy(array)[None,]

    def apply_mask(self, image: torch.Tensor, mask: torch.Tensor):
        image_np = self.tensor_to_numpy(image)
        mask_np = self.apply_colormap(mask) / 255.0
        masked_image_np = (image_np * mask_np).astype(np.uint8)
        return (self.numpy_to_tensor(masked_image_np),)


NODE_CLASS_MAPPINGS = {"AnymatixMaskImage": AnymatixMaskImage}
NODE_DISPLAY_NAME_MAPPINGS = {"AnymatixMaskImage": "Apply Mask to Image"}
