import numpy as np
import torch
from PIL import Image
import matplotlib.cm as cm

class AnymatixMaskImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_mask"
    CATEGORY = "Anymatix"

    def tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:#CLIPSeg function
        #Convert a tensor to a numpy array and scale its values to 0-255
        array = tensor.numpy().squeeze()
        return (array * 255).astype(np.uint8)
    
    def apply_colormap(self, mask: torch.Tensor, colormap) -> np.ndarray:#CLIPSeg function
        """Apply a colormap to a tensor and convert it to a numpy array."""
        colored_mask = colormap(mask.numpy())[:, :, :3]
        return (colored_mask * 255).astype(np.uint8)
    
    def numpy_to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """Convert a numpy array to a tensor and scale its values from 0-255 to 0-1."""
        array = array.astype(np.float32) / 255.0
        return torch.from_numpy(array)[None,]   
    
    def apply_mask(self, image: torch.Tensor, mask: torch.Tensor):
        image_np = self.tensor_to_numpy(image)
        mask_np = self.apply_colormap(mask,cm.Greys_r)
        mask_np = mask_np/255.0
        
        masked_image_np = (image_np * mask_np).astype(np.uint8)
        
        masked_image = self.numpy_to_tensor(masked_image_np)
        return (masked_image,)

NODE_CLASS_MAPPINGS = {"AnymatixMaskImage": AnymatixMaskImage}
NODE_DISPLAY_NAME_MAPPINGS = {"AnymatixMaskImage": "Apply Mask to Image"}
