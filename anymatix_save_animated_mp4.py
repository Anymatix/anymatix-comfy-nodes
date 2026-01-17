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
        Save VIDEO object as MP4 using FFmpeg for maximum browser compatibility.
        Extracts images, fps, and audio from the VIDEO's components.
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
                
                # Add progress bar for frame preparation
                import comfy.utils
                total_frames = len(images)
                pbar = comfy.utils.ProgressBar(total_frames)
                print(f"Preparing {total_frames} frames...")
                
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
                    
                    # Update progress bar after each frame
                    pbar.update(1)
                
                # Handle audio input if provided
                audio_file_path = None
                if audio is not None:
                    print("Audio provided - will mux with video")
                    # Save audio to temporary file using PyAV (consistent with ComfyUI)
                    audio_file_path = os.path.join(temp_dir, "audio.wav")
                    # ComfyUI audio format: dict with 'waveform' and 'sample_rate'
                    # waveform shape: [channels, samples] or [batch, channels, samples]
                    try:
                        import av
                        waveform = audio['waveform']
                        sample_rate = audio['sample_rate']
                        
                        # Handle batch dimension if present
                        if len(waveform.shape) == 3:
                            waveform = waveform[0]  # Take first item from batch
                        
                        # Determine audio layout
                        num_channels = waveform.shape[0]
                        layout = 'mono' if num_channels == 1 else 'stereo'
                        
                        # Save audio as WAV file using PyAV
                        output_container = av.open(audio_file_path, mode='w', format='wav')
                        out_stream = output_container.add_stream('pcm_s16le', rate=sample_rate, layout=layout)
                        
                        # Convert waveform to numpy array and create audio frame
                        # PyAV expects shape [1, samples * channels] with channels interleaved
                        # ComfyUI format: [channels, samples] -> move channels to end -> flatten to [1, samples * channels]
                        frame = av.AudioFrame.from_ndarray(
                            waveform.movedim(0, 1).reshape(1, -1).float().cpu().numpy(),
                            format='flt',
                            layout=layout
                        )
                        frame.sample_rate = sample_rate
                        frame.pts = 0
                        
                        # Encode and write
                        for packet in out_stream.encode(frame):
                            output_container.mux(packet)
                        
                        # Flush encoder
                        for packet in out_stream.encode(None):
                            output_container.mux(packet)
                        
                        output_container.close()
                        print(f"Saved audio: {sample_rate}Hz, {num_channels} channel(s)")
                    except Exception as e:
                        print(f"Warning: Could not process audio: {e}")
                        audio_file_path = None
                
                # Build FFmpeg command for maximum browser compatibility
                ffmpeg_cmd = [
                    FFMPEG_EXE,  # Use the detected FFmpeg executable
                    '-y',  # Overwrite output file
                    '-framerate', str(fps),
                    '-i', frame_pattern,
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
                    '-movflags', '+faststart',  # Enable fast start for web streaming
                    '-tune', 'stillimage'  # Optimize for still image sequences
                ])
                
                # Audio encoding options (if audio is present)
                if audio_file_path is not None:
                    ffmpeg_cmd.extend([
                        '-c:a', 'aac',  # AAC audio codec for broad compatibility
                        '-b:a', '192k',  # Audio bitrate
                        '-shortest'  # End encoding when shortest input ends (video or audio)
                    ])
                
                # Add GOP structure settings for high quality preset
                if quality == "high quality":
                    ffmpeg_cmd.extend([
                        '-g', preset['g'],  # GOP size
                        '-keyint_min', preset['min_keyint'],  # Minimum keyframe interval
                        '-sc_threshold', preset['sc_threshold'],  # Scene change threshold
                        '-force_key_frames', f'expr:gte(t,n_forced*{1.0/fps})'  # Force keyframes at regular intervals
                    ])
                
                # Add preset if specified
                if 'preset' in preset:
                    ffmpeg_cmd.extend(['-preset', preset['preset']])
                
                # Generate metadata file for high quality preset
                metadata_path = None
                if quality == "high quality":
                    metadata_path = file_path.replace('.mp4', '_metadata.json')
                    gop_size = int(preset['g'])
                    total_frames = len(images)
                    keyframe_positions = list(range(0, total_frames, gop_size))
                    
                    import json
                    metadata = {
                        "gop_size": gop_size,
                        "fps": fps,
                        "total_frames": total_frames,
                        "keyframe_positions": keyframe_positions,
                        "duration": total_frames / fps,
                        "encoding": {
                            "codec": preset['codec'],
                            "profile": preset['profile'],
                            "crf": preset['crf']
                        }
                    }
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    print(f"Generated metadata file: {os.path.basename(metadata_path)}")
                
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
                    audio_info = " with audio" if audio_file_path is not None else ""
                    print(f"✓ MP4 created successfully{audio_info}!")
                    print(f"  File: {filename}")
                    print(f"  Size: {file_size:,} bytes")
                    print(f"  Duration: {duration:.2f} seconds")
                    print(f"  Quality: {quality}")
                    print(f"  FFmpeg: {ffmpeg_source}")
                    if audio_file_path is not None:
                        print(f"  Audio: muxed")
                    
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
