import torch


class AnymatixImageToVideo:
    """
    A custom node that combines image batches with FPS parameters for video generation.
    This node takes an IMAGE input (batch of images) and a fps parameter,
    then outputs both values in a format that can be consumed by video save nodes.
    
    Since observer nodes injected by the 'read' function can't take parameters directly,
    this node serves as a bridge between image generation and video saving functionality.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {
                    "default": 6.0,
                    "min": 0.01,
                    "max": 1000.0,
                    "step": 0.01
                }),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "AUDIO")
    RETURN_NAMES = ("images", "fps", "audio")
    FUNCTION = "process"
    CATEGORY = "Anymatix"

    def process(self, images, fps, audio=None):
        """
        Process the input images, fps parameter, and optional audio.
        
        Args:
            images: Batch of images (IMAGE tensor)
            fps: Frame rate for video generation (float)
            audio: Optional audio data to mux with the video (AUDIO tensor)
            
        Returns:
            tuple: (images, fps, audio) - passes through the inputs for downstream consumption
        """
        # Validate inputs
        if images is None:
            raise ValueError("Images input cannot be None")
        
        if fps <= 0:
            raise ValueError("FPS must be greater than 0")
            
        # Validate that we have a batch of images
        if len(images.shape) != 4:  # [batch, height, width, channels]
            raise ValueError(f"Expected 4D image tensor, got shape {images.shape}")
            
        batch_size = images.shape[0]
        if batch_size < 1:
            raise ValueError("Image batch must contain at least 1 image")
            
        # Log processing info
        audio_info = " with audio" if audio is not None else ""
        print(f"AnymatixImageToVideo: Processing {batch_size} images at {fps} FPS{audio_info}")
        
        # Simply pass through the inputs - the actual video creation will be handled
        # by the SaveAnimatedMP4 node that gets injected by the read function
        return (images, fps, audio)
