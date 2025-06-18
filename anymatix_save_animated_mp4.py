import os
import numpy as np
import folder_paths

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV (cv2) not available. MP4 saving will not work.")


class AnymatixSaveAnimatedMP4:
    """
    Custom SaveAnimatedMP4 node that uses the exact filename provided.
    Saves image sequences as MP4 video files using OpenCV.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    # MP4 codec options
    codecs = {
        "libx264": cv2.VideoWriter_fourcc(*'H264'),  # Better browser compatibility
        "h264": cv2.VideoWriter_fourcc(*'H264'),
        "xvid": cv2.VideoWriter_fourcc(*'XVID'),
        "mjpeg": cv2.VideoWriter_fourcc(*'MJPG')
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": "anymatix/results", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 120.0, "step": 0.01}),
                "codec": (list(cls.codecs.keys()), {"default": "libx264"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Anymatix"

    def save_images(self, images, output_path, filename_prefix, fps, codec):
        """
        Save images as MP4 video with simple filename.
        """
        if not CV2_AVAILABLE:
            print("Error: OpenCV (cv2) is not available. Cannot save MP4 videos.")
            return {"ui": {"images": [], "animated": (True,)}}
            
        fourcc = self.codecs.get(codec, self.codecs["libx264"])
        
        # Create output directory using the same pattern as image save
        output_path = os.path.join(folder_paths.get_output_directory(), output_path)
        os.makedirs(output_path, exist_ok=True)
        
        print(f"Anymatix: saving animated MP4 to {output_path}")
        
        results = []
        
        # Convert images to numpy arrays in BGR format for OpenCV
        video_frames = []
        for image in images:
            # Convert from torch tensor to numpy array (0-1 range to 0-255)
            img_np = (255.0 * image.cpu().numpy()).astype(np.uint8)
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            video_frames.append(img_bgr)
        
        if not video_frames:
            print("No frames to save")
            return {"ui": {"images": [], "animated": (True,)}}
        
        # Get frame dimensions
        height, width = video_frames[0].shape[:2]
        
        # Use filename_prefix directly as the filename
        if not filename_prefix.endswith('.mp4'):
            filename = f"{filename_prefix}.mp4"
        else:
            filename = filename_prefix
        
        # Create video writer
        file_path = os.path.join(output_path, filename)
        
        # Create VideoWriter object
        video_writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"Error: Could not open video writer for {file_path}")
            return {"ui": {"images": [], "animated": (True,)}}
        
        # Write frames to video
        for frame in video_frames:
            video_writer.write(frame)
        
        # Release the video writer
        video_writer.release()
        
        print(f"Animated MP4 saved to: {file_path}")
        
        results.append({
            "filename": filename,
            "subfolder": output_path.replace(folder_paths.get_output_directory(), "").strip(os.sep),
            "type": self.type
        })
        
        return {"ui": {"images": results, "animated": (True,)}}


# Node export for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AnymatixSaveAnimatedMP4": AnymatixSaveAnimatedMP4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnymatixSaveAnimatedMP4": "Anymatix Save Animated MP4",
}
