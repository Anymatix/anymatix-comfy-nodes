import os
import numpy as np
import json
from PIL import Image
import folder_paths
from comfy.cli_args import args


class AnymatixSaveAnimatedWEBP:
    """
    Custom SaveAnimatedWEBP node that uses the exact filename provided.
    Based on ComfyUI's SaveAnimatedWEBP but without any counter suffixes.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    methods = {"default": 4, "fastest": 0, "slowest": 6}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "fps": ("FLOAT", {"default": 6.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
                "lossless": ("BOOLEAN", {"default": True}),
                "quality": ("INT", {"default": 80, "min": 0, "max": 100}),
                "method": (list(cls.methods.keys()),),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Anymatix"

    def save_images(self, images, fps, filename_prefix, lossless, quality, method, 
                   prompt=None, extra_pnginfo=None):
        """
        Save images as animated WEBP with simple filename.
        """
        method = self.methods.get(method, 4)
        
        # Create output directory
        output_path = os.path.join(self.output_dir, "anymatix", "results")
        os.makedirs(output_path, exist_ok=True)
        
        print(f"Anymatix: saving animated WEBP to {output_path}")
        
        results = []
        pil_images = []
        
        # Convert images to PIL format
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            pil_images.append(img)

        # Handle metadata
        metadata = pil_images[0].getexif()
        if not args.disable_metadata:
            if prompt is not None:
                metadata[0x0110] = "prompt:{}".format(json.dumps(prompt))
            if extra_pnginfo is not None:
                inital_exif = 0x010f
                for x in extra_pnginfo:
                    metadata[inital_exif] = "{}:{}".format(x, json.dumps(extra_pnginfo[x]))
                    inital_exif -= 1

        # Use filename_prefix directly as the filename
        if not filename_prefix.endswith('.webp'):
            filename = f"{filename_prefix}.webp"
        else:
            filename = filename_prefix
        
        # Save all images as a single animated WEBP
        file_path = os.path.join(output_path, filename)
        pil_images[0].save(
            file_path,
            save_all=True,
            duration=int(1000.0/fps),
            append_images=pil_images[1:],
            exif=metadata,
            lossless=lossless,
            quality=quality,
            method=method
        )
        
        print(f"Animated WEBP saved to: {file_path}")
        
        results.append({
            "filename": filename,
            "subfolder": "anymatix/results",
            "type": self.type
        })

        return {"ui": {"images": results, "animated": (True,)}}


# Node export for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AnymatixSaveAnimatedWEBP": AnymatixSaveAnimatedWEBP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnymatixSaveAnimatedWEBP": "Anymatix Save Animated WEBP",
}
