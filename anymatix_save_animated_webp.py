import os
import numpy as np
import json
from PIL import Image
import folder_paths
from comfy.cli_args import args


class AnymatixSaveAnimatedWEBP:
    """
    Custom SaveAnimatedWEBP node that provides better filename control.
    Based on ComfyUI's SaveAnimatedWEBP but allows configurable filename patterns
    without the hardcoded counter suffix.
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
                "filename_counter_format": ("STRING", {"default": "_{counter:05}_", "multiline": False}),
                "use_original_filename": ("BOOLEAN", {"default": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Anymatix"

    def save_images(self, images, fps, filename_prefix, lossless, quality, method, 
                   filename_counter_format="_{counter:05}_", use_original_filename=False,
                   num_frames=0, prompt=None, extra_pnginfo=None):
        """
        Save images as animated WEBP with configurable filename format.
        
        Args:
            images: Image tensor batch
            fps: Frame rate for animation
            filename_prefix: Base filename prefix
            lossless: Whether to use lossless compression
            quality: WEBP quality (0-100)
            method: Compression method
            num_frames: Number of frames per file (0 = all frames in one file)
            prompt: Prompt metadata
            extra_pnginfo: Extra PNG info metadata
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

        if num_frames == 0:
            num_frames = len(pil_images)

        # Determine filename strategy
        if use_original_filename:
            # Use filename_prefix as the exact filename (with .webp extension)
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
        else:
            # Use counter-based naming (similar to original but configurable)
            counter = 1
            
            # Find existing files to avoid conflicts
            existing_files = [f for f in os.listdir(output_path) if f.startswith(filename_prefix) and f.endswith('.webp')]
            if existing_files:
                # Extract counter from existing files and start from max + 1
                counters = []
                for f in existing_files:
                    try:
                        # Try to extract number from filename
                        base = f[len(filename_prefix):].replace('.webp', '')
                        if base.startswith('_') and base.endswith('_'):
                            num_str = base[1:-1]
                            counters.append(int(num_str))
                    except:
                        pass
                if counters:
                    counter = max(counters) + 1
            
            # Split images into chunks based on num_frames
            c = len(pil_images)
            for i in range(0, c, num_frames):
                try:
                    # Format the counter according to the format string
                    counter_str = filename_counter_format.format(counter=counter)
                except:
                    # Fallback to default format if formatting fails
                    counter_str = f"_{counter:05}_"
                
                filename = f"{filename_prefix}{counter_str}.webp"
                file_path = os.path.join(output_path, filename)
                
                # Get the chunk of images for this file
                chunk_images = pil_images[i:i + num_frames]
                
                chunk_images[0].save(
                    file_path,
                    save_all=True,
                    duration=int(1000.0/fps),
                    append_images=chunk_images[1:],
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
                counter += 1

        animated = num_frames != 1
        return {"ui": {"images": results, "animated": (animated,)}}


# Node export for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AnymatixSaveAnimatedWEBP": AnymatixSaveAnimatedWEBP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnymatixSaveAnimatedWEBP": "Anymatix Save Animated WEBP",
}
