from fractions import Fraction
from comfy_api.latest import InputImpl, Types


class AnymatixImageToVideo:
    """
    A custom node that combines image batches with FPS parameters for video generation.
    This node takes an IMAGE input (batch of images) and a fps parameter,
    then outputs a VIDEO object compatible with ComfyUI's video abstraction.
    
    The VIDEO output can be consumed by AnymatixSaveAnimatedMP4 or GetVideoComponents.
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

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
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
            tuple: (VIDEO,) - a VideoFromComponents object wrapping the inputs
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
        
        # Create a VIDEO object using ComfyUI's VideoFromComponents abstraction
        # This is zero-cost - just wraps references without copying data
        video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps))
        )
        return (video,)
