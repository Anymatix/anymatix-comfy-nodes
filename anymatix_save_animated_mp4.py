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
    Saves VIDEO objects as MP4 video files using FFmpeg with H.264 baseline profile.
    Accepts ComfyUI VIDEO input (from AnymatixImageToVideo, LoadVideo, CreateVideo, etc.)
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
        },
        "high quality": {
            "codec": "libx264",
            "profile": "high",
            "level": "4.0", 
            "pix_fmt": "yuv420p",
            "crf": "15",  # Very high quality for client-side processing
            "preset": "slow",  # Better compression
            "keyint": "24",  # GOP size of 24 frames (1 second at 24fps)
            "min_keyint": "12",  # Minimum keyframe interval
            "sc_threshold": "0",  # Disable scene change detection for consistent GOP
            "g": "24"  # Explicit GOP size
        }
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "output_path": ("STRING", {"default": "anymatix/results", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "quality": (list(cls.codec_presets.keys()), {"default": "web_compatible"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "Anymatix"

    def save_video(self, video, output_path, filename_prefix, quality):
        """
        Save VIDEO object as MP4 using FFmpeg pipe input for efficient on-the-fly encoding.
        No temporary files - frames are piped directly to FFmpeg stdin as raw RGB data.
        """
        # Extract components from VIDEO object
        components = video.get_components()
        images = components.images
        fps = float(components.frame_rate)
        audio = components.audio
        
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
        
        # Create output directory (output_path is the hash subdirectory)
        full_output_path = os.path.join(folder_paths.get_output_directory(), output_path)
        os.makedirs(full_output_path, exist_ok=True)
        
        ffmpeg_source = "imageio-ffmpeg" if IMAGEIO_FFMPEG_AVAILABLE else "system"
        print(f"Anymatix: Encoding MP4 with FFmpeg pipe ({ffmpeg_source}, quality: {quality})")
        print(f"Output path: {full_output_path}")
        
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
        
        file_path = os.path.join(full_output_path, filename)
        
        # Get frame dimensions from first image
        height, width = images[0].shape[:2]
        total_frames = len(images)
        
        print(f"Creating video: {filename} ({total_frames} frames at {fps} fps, {width}x{height})")
        
        try:
            # Handle audio - need temp file for audio only (can't pipe two streams)
            audio_file_path = None
            temp_dir = None
            
            if audio is not None:
                print("Audio provided - will mux with video")
                temp_dir = tempfile.mkdtemp()
                audio_file_path = os.path.join(temp_dir, "audio.wav")
                try:
                    import av
                    waveform = audio['waveform']
                    sample_rate = audio['sample_rate']
                    
                    if len(waveform.shape) == 3:
                        waveform = waveform[0]
                    
                    num_channels = waveform.shape[0]
                    layout = 'mono' if num_channels == 1 else 'stereo'
                    
                    output_container = av.open(audio_file_path, mode='w', format='wav')
                    out_stream = output_container.add_stream('pcm_s16le', rate=sample_rate, layout=layout)
                    
                    frame = av.AudioFrame.from_ndarray(
                        waveform.movedim(0, 1).reshape(1, -1).float().cpu().numpy(),
                        format='flt',
                        layout=layout
                    )
                    frame.sample_rate = sample_rate
                    frame.pts = 0
                    
                    for packet in out_stream.encode(frame):
                        output_container.mux(packet)
                    for packet in out_stream.encode(None):
                        output_container.mux(packet)
                    
                    output_container.close()
                    print(f"Prepared audio: {sample_rate}Hz, {num_channels} channel(s)")
                except Exception as e:
                    print(f"Warning: Could not process audio: {e}")
                    audio_file_path = None
            
            # Build FFmpeg command with pipe input (rawvideo)
            ffmpeg_cmd = [
                FFMPEG_EXE,
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',
                '-pix_fmt', 'rgb24',
                '-r', str(fps),
                '-i', '-',  # Read from stdin
            ]
            
            # Add audio input if available
            if audio_file_path is not None:
                ffmpeg_cmd.extend(['-i', audio_file_path])
            
            # Video encoding options
            ffmpeg_cmd.extend([
                '-c:v', preset['codec'],
                '-profile:v', preset['profile'],
                '-level', preset['level'],
                '-pix_fmt', preset['pix_fmt'],
                '-crf', preset['crf'],
                '-movflags', '+faststart',
            ])
            
            # Audio encoding options
            if audio_file_path is not None:
                ffmpeg_cmd.extend([
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest'
                ])
            
            # GOP settings for high quality
            if quality == "high quality":
                ffmpeg_cmd.extend([
                    '-g', preset['g'],
                    '-keyint_min', preset['min_keyint'],
                    '-sc_threshold', preset['sc_threshold'],
                ])
            
            if 'preset' in preset:
                ffmpeg_cmd.extend(['-preset', preset['preset']])
            
            ffmpeg_cmd.append(file_path)
            
            print(f"Starting FFmpeg pipe encoding...")
            
            # Start FFmpeg process with pipe
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Stream frames directly to FFmpeg - no temp files!
            import comfy.utils
            pbar = comfy.utils.ProgressBar(total_frames)
            
            for i, image in enumerate(images):
                # Convert to uint8 RGB and get raw bytes
                img_np = (255.0 * image.cpu().numpy()).astype(np.uint8)
                # Write raw RGB bytes directly to FFmpeg stdin
                process.stdin.write(img_np.tobytes())
                pbar.update(1)
            
            # Close stdin and wait for FFmpeg to finish
            process.stdin.close()
            stdout, stderr = process.communicate(timeout=300)
            
            # Cleanup temp audio file
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            if process.returncode != 0:
                print("FFmpeg encoding failed!")
                print(f"Error: {stderr.decode()}")
                return {"ui": {"images": [], "animated": (True,)}}
            
            # Verify output
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                file_size = os.path.getsize(file_path)
                duration = total_frames / fps
                audio_info = " with audio" if audio_file_path is not None else ""
                print(f"MP4 created successfully{audio_info}!")
                print(f"  File: {filename}")
                print(f"  Size: {file_size:,} bytes")
                print(f"  Duration: {duration:.2f} seconds")
                print(f"  Quality: {quality}")
                
                results.append({
                    "filename": filename,
                    "subfolder": output_path,
                    "type": self.type
                })
            else:
                print(f"Error: Output file {file_path} was not created or is empty")
                return {"ui": {"images": [], "animated": (True,)}}
                
        except subprocess.TimeoutExpired:
            process.kill()
            print("Error: FFmpeg encoding timed out (>5 minutes)")
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
