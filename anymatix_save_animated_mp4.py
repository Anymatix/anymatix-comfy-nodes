import os
import numpy as np
import folder_paths
import subprocess
import tempfile
import shutil

# Try to import imageio-ffmpeg for automatic FFmpeg management
try:
    import imageio_ffmpeg
    IMAGEIO_FFMPEG_AVAILABLE = True
    print("Using imageio-ffmpeg for automatic FFmpeg management")
except ImportError:
    IMAGEIO_FFMPEG_AVAILABLE = False
    print("imageio-ffmpeg not available, falling back to system FFmpeg")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV (cv2) not available. Will use PIL for image processing.")

def get_ffmpeg_exe():
    """Get FFmpeg executable path, with automatic download via imageio-ffmpeg if available"""
    if IMAGEIO_FFMPEG_AVAILABLE:
        try:
            # This will automatically download FFmpeg if not present
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"Using imageio-ffmpeg: {ffmpeg_path}")
            return ffmpeg_path
        except Exception as e:
            print(f"imageio-ffmpeg failed: {e}, falling back to system FFmpeg")
    
    # Fallback to system FFmpeg
    return 'ffmpeg'

def check_ffmpeg():
    """Check if FFmpeg is available (either via imageio-ffmpeg or system)"""
    try:
        ffmpeg_exe = get_ffmpeg_exe()
        subprocess.run([ffmpeg_exe, '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True, ffmpeg_exe
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, None

FFMPEG_AVAILABLE, FFMPEG_EXE = check_ffmpeg()


class AnymatixSaveAnimatedMP4:
    """
    Custom SaveAnimatedMP4 node that uses FFmpeg for maximum browser compatibility.
    Saves image sequences as MP4 video files using FFmpeg with H.264 baseline profile.
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    # Codec options for different quality/compatibility tradeoffs
    codec_presets = {
        "web_compatible": {
            "codec": "libx264", 
            "profile": "baseline", 
            "level": "3.0",
            "pix_fmt": "yuv420p",
            "crf": "23"
        },
        "high_quality": {
            "codec": "libx264", 
            "profile": "high", 
            "level": "4.0",
            "pix_fmt": "yuv420p",
            "crf": "18"
        },
        "fast_encode": {
            "codec": "libx264", 
            "profile": "baseline", 
            "level": "3.0",
            "pix_fmt": "yuv420p",
            "crf": "28",
            "preset": "ultrafast"
        }
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": "anymatix/results", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 120.0, "step": 0.01}),
                "quality": (list(cls.codec_presets.keys()), {"default": "web_compatible"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Anymatix"

    def save_images(self, images, output_path, filename_prefix, fps, quality):
        """
        Save images as MP4 video using FFmpeg for maximum browser compatibility.
        """
        if not FFMPEG_AVAILABLE:
            print("=" * 70)
            print("ERROR: FFmpeg is not available!")
            print("=" * 70)
            print("Anymatix requires FFmpeg for browser-compatible MP4 video encoding.")
            print("")
            print("RECOMMENDED: Install imageio-ffmpeg for automatic FFmpeg management:")
            print("  pip install imageio-ffmpeg")
            print("")
            print("This will automatically download FFmpeg - no manual installation needed!")
            print("")
            print("ALTERNATIVE: Install FFmpeg manually:")
            print("  macOS:    brew install ffmpeg")
            print("  Ubuntu:   sudo apt install ffmpeg") 
            print("  Windows:  Download from https://ffmpeg.org/download.html")
            print("")
            print("After installation, restart ComfyUI and try again.")
            print("=" * 70)
            return {"ui": {"images": [], "animated": (True,)}}
            
        preset = self.codec_presets.get(quality, self.codec_presets["web_compatible"])
        
        # Create output directory
        output_path = os.path.join(folder_paths.get_output_directory(), output_path)
        os.makedirs(output_path, exist_ok=True)
        
        ffmpeg_source = "imageio-ffmpeg" if IMAGEIO_FFMPEG_AVAILABLE else "system"
        print(f"Anymatix: Encoding MP4 with FFmpeg ({ffmpeg_source}, quality: {quality})")
        print(f"Output path: {output_path}")
        
        results = []
        
        # Check if we have images to process
        if images is None or len(images) == 0:
            print("No frames to save")
            return {"ui": {"images": [], "animated": (True,)}}
        
        # Prepare filename
        if not filename_prefix.endswith('.mp4'):
            filename = f"{filename_prefix}.mp4"
        else:
            filename = filename_prefix
        
        file_path = os.path.join(output_path, filename)
        
        print(f"Creating video: {filename} ({len(images)} frames at {fps} fps)")
        
        # Create temporary directory for frame images
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Save frames as individual images
                frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
                
                print(f"Preparing {len(images)} frames...")
                for i, image in enumerate(images):
                    # Convert from torch tensor to numpy array
                    img_np = (255.0 * image.cpu().numpy()).astype(np.uint8)
                    
                    if CV2_AVAILABLE:
                        # Use OpenCV for image saving (faster)
                        frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                        cv2.imwrite(frame_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                    else:
                        # Fallback to PIL
                        try:
                            from PIL import Image as PILImage
                            frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                            pil_image = PILImage.fromarray(img_np)
                            pil_image.save(frame_path)
                        except ImportError:
                            print("Error: Neither OpenCV nor PIL is available for image processing")
                            print("Please install: pip install opencv-python pillow")
                            return {"ui": {"images": [], "animated": (True,)}}
                
                # Build FFmpeg command for maximum browser compatibility
                ffmpeg_cmd = [
                    FFMPEG_EXE,  # Use the detected FFmpeg executable
                    '-y',  # Overwrite output file
                    '-framerate', str(fps),
                    '-i', frame_pattern,
                    '-c:v', preset['codec'],
                    '-profile:v', preset['profile'],
                    '-level', preset['level'],
                    '-pix_fmt', preset['pix_fmt'],
                    '-crf', preset['crf'],
                    '-movflags', '+faststart',  # Enable fast start for web streaming
                    '-tune', 'stillimage'  # Optimize for still image sequences
                ]
                
                # Add preset if specified
                if 'preset' in preset:
                    ffmpeg_cmd.extend(['-preset', preset['preset']])
                
                # Add output file
                ffmpeg_cmd.append(file_path)
                
                print("Running FFmpeg with browser-optimized settings...")
                print(f"Using: {FFMPEG_EXE}")
                
                # Run FFmpeg
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    print("FFmpeg encoding failed!")
                    print(f"Error output: {result.stderr}")
                    return {"ui": {"images": [], "animated": (True,)}}
                
                # Verify the output file exists and has content
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    file_size = os.path.getsize(file_path)
                    duration = len(images) / fps
                    print(f"✓ MP4 created successfully!")
                    print(f"  File: {filename}")
                    print(f"  Size: {file_size:,} bytes")
                    print(f"  Duration: {duration:.2f} seconds")
                    print(f"  Quality: {quality}")
                    print(f"  FFmpeg: {ffmpeg_source}")
                    
                    results.append({
                        "filename": filename,
                        "subfolder": output_path.replace(folder_paths.get_output_directory(), "").strip(os.sep),
                        "type": self.type
                    })
                else:
                    print(f"Error: Output file {file_path} was not created or is empty")
                    return {"ui": {"images": [], "animated": (True,)}}
                
            except subprocess.TimeoutExpired:
                print("Error: FFmpeg encoding timed out (>5 minutes)")
                print("This might happen with very long videos or slow systems.")
                return {"ui": {"images": [], "animated": (True,)}}
            except Exception as e:
                print(f"Error during FFmpeg encoding: {str(e)}")
                return {"ui": {"images": [], "animated": (True,)}}
        
        return {"ui": {"images": results, "animated": (True,)}}


# Node export for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AnymatixSaveAnimatedMP4": AnymatixSaveAnimatedMP4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnymatixSaveAnimatedMP4": "Anymatix Save Animated MP4",
}
