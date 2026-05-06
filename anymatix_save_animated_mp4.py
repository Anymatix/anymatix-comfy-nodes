import os
import numpy as np
import folder_paths
import subprocess
import tempfile
import shutil
import sys

# Dev/instrumentation: one tag for grepping ComfyUI logs
_DBG = "[ANYMATIX_SAVE_MP4]"


def _save_mp4_dbg(tag: str, detail: str) -> None:
    print(f"{_DBG} {tag} | {detail}")

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


def list_available_encoders(ffmpeg_exe):
    try:
        result = subprocess.run(
            [ffmpeg_exe, '-hide_banner', '-encoders'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout + result.stderr
    except Exception:
        return ""


def get_video_encoder_candidates(ffmpeg_exe, preset):
    encoders_output = list_available_encoders(ffmpeg_exe)
    candidates = []

    if sys.platform == 'darwin' and 'h264_videotoolbox' in encoders_output:
        candidates.append({
            'name': 'h264_videotoolbox',
            'args': [
                '-c:v', 'h264_videotoolbox',
                '-b:v', '12M',
                '-maxrate', '18M',
                '-bufsize', '24M'
            ]
        })

    candidates.append({
        'name': preset['codec'],
        'args': [
            '-c:v', preset['codec'],
            '-profile:v', preset['profile'],
            '-level', preset['level'],
            '-crf', preset['crf']
        ]
    })

    return candidates


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
                "quality": (list(cls.codec_presets.keys()), {"default": "high_quality"}),
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
        _save_mp4_dbg(
            "ENTER",
            f"output_path={output_path!r} filename_prefix={filename_prefix!r} quality={quality!r} "
            f"ffmpeg_available={FFMPEG_AVAILABLE} comfy_output_root={folder_paths.get_output_directory()!r}",
        )
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
            _save_mp4_dbg("RETURN", "path=EARLY_NO_FFMPEG ui_images=0")
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
            _save_mp4_dbg("RETURN", "path=EARLY_NO_FRAMES ui_images=0")
            return {"ui": {"images": [], "animated": (True,)}}
        
        # Prepare filename
        if not filename_prefix.endswith('.mp4'):
            filename = f"{filename_prefix}.mp4"
        else:
            filename = filename_prefix
        
        file_path = os.path.join(full_output_path, filename)
        temp_file_path = os.path.join(
            full_output_path,
            f".{filename}.tmp-{os.getpid()}.mp4"
        )
        _save_mp4_dbg(
            "PATHS",
            f"file_path={file_path!r} temp_file_path={temp_file_path!r}",
        )
        
        # Get frame dimensions from first image
        height, width = images[0].shape[:2]
        total_frames = len(images)
        
        print(f"Creating video: {filename} ({total_frames} frames at {fps} fps, {width}x{height})")
        try:
            _t0 = images[0].detach().cpu().float().numpy()
            _save_mp4_dbg(
                "FRAME0_STATS",
                f"shape={getattr(_t0, 'shape', '?')} min={float(np.nanmin(_t0)):.6g} "
                f"max={float(np.nanmax(_t0)):.6g} all_finite={bool(np.isfinite(_t0).all())}",
            )
        except Exception as _e:
            _save_mp4_dbg("FRAME0_STATS", f"failed_to_compute err={_e!r}")
        
        try:
            # Handle audio - need temp file for audio only (can't pipe two streams)
            audio_file_path = None
            temp_dir = None
            
            if audio is not None:
                print("Audio provided - will mux with video")
                _save_mp4_dbg("AUDIO", "branch=MUX temp_wav")
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
                    _save_mp4_dbg("AUDIO_OK", f"path={audio_file_path!r}")
                except Exception as e:
                    print(f"Warning: Could not process audio: {e}")
                    _save_mp4_dbg("AUDIO_FAIL", f"err={e!r} continuing_video_only")
                    audio_file_path = None
            else:
                _save_mp4_dbg("AUDIO", "branch=NONE")
            
            process = None
            stdout = b""
            stderr = b""
            last_error = None
            _cands = get_video_encoder_candidates(FFMPEG_EXE, preset)
            _save_mp4_dbg("ENCODER_LIST", f"candidates={[c['name'] for c in _cands]!r}")

            for encoder_candidate in _cands:
                ffmpeg_cmd = [
                    FFMPEG_EXE,
                    '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{width}x{height}',
                    '-pix_fmt', 'rgb24',
                    '-r', str(fps),
                    '-i', '-',
                ]

                if audio_file_path is not None:
                    ffmpeg_cmd.extend(['-i', audio_file_path])

                ffmpeg_cmd.extend([
                    '-sws_flags', 'lanczos+accurate_rnd+full_chroma_int',
                    '-pix_fmt', preset['pix_fmt'],
                    '-fps_mode', 'cfr',
                    '-r', str(fps),
                    '-movflags', '+faststart',
                ])

                ffmpeg_cmd.extend(encoder_candidate['args'])

                if audio_file_path is not None:
                    ffmpeg_cmd.extend([
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-shortest'
                    ])

                gop = str(max(12, int(round(fps * 2))))
                keyint_min = str(max(12, int(round(fps))))
                ffmpeg_cmd.extend([
                    '-g', preset.get('g', gop),
                    '-keyint_min', preset.get('min_keyint', keyint_min),
                    '-sc_threshold', preset.get('sc_threshold', '0'),
                ])

                if encoder_candidate['name'] == preset['codec'] and 'preset' in preset:
                    ffmpeg_cmd.extend(['-preset', preset['preset']])

                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except OSError:
                    pass

                ffmpeg_cmd.append(temp_file_path)

                print(f"Starting FFmpeg pipe encoding with {encoder_candidate['name']}...")
                _save_mp4_dbg("ENCODER_TRY", f"name={encoder_candidate['name']!r}")

                process = subprocess.Popen(
                    ffmpeg_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                import comfy.utils
                pbar = comfy.utils.ProgressBar(total_frames)

                try:
                    for _fi, image in enumerate(images):
                        _arr = image.detach().cpu().float().numpy()
                        if not np.isfinite(_arr).all():
                            _bad_count = int(np.sum(~np.isfinite(_arr)))
                            _save_mp4_dbg(
                                "FRAME_NONFINITE",
                                f"encoder={encoder_candidate['name']!r} frame_index={_fi} "
                                f"bad_count={_bad_count}",
                            )
                            _arr = np.nan_to_num(_arr, nan=0.0, posinf=1.0, neginf=0.0)
                        _arr = np.clip(_arr, 0.0, 1.0)
                        img_np = np.rint(255.0 * _arr).astype(np.uint8)
                        process.stdin.write(img_np.tobytes())
                        pbar.update(1)

                    process.stdin.close()
                    process.stdin = None
                    stdout, stderr = process.communicate(timeout=300)
                except Exception as encoder_error:
                    last_error = encoder_error
                    _stderr_txt = ""
                    try:
                        if process is not None and process.stdin is not None:
                            try:
                                process.stdin.close()
                            except Exception:
                                pass
                            process.stdin = None
                        if process is not None:
                            try:
                                stdout, stderr = process.communicate(timeout=5)
                            except Exception:
                                stderr = stderr or b""
                        _stderr_txt = stderr.decode(errors="replace") if stderr else ""
                    except Exception:
                        _stderr_txt = "<stderr unavailable>"
                    try:
                        process.kill()
                    except Exception:
                        pass
                    try:
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                    except OSError:
                        pass
                    print(f"Encoder {encoder_candidate['name']} failed: {encoder_error}")
                    if _stderr_txt:
                        print(f"Encoder stderr: {_stderr_txt}")
                    _tail = _stderr_txt[-800:] if len(_stderr_txt) > 800 else _stderr_txt
                    _save_mp4_dbg(
                        "ENCODER_PIPE_EXCEPTION",
                        f"name={encoder_candidate['name']!r} err={encoder_error!r} stderr_tail={_tail!r}",
                    )
                    continue

                if process.returncode == 0:
                    print(f"FFmpeg encoding succeeded with {encoder_candidate['name']}")
                    _sz = os.path.getsize(temp_file_path) if os.path.exists(temp_file_path) else -1
                    _save_mp4_dbg(
                        "ENCODER_OK",
                        f"name={encoder_candidate['name']!r} temp_bytes={_sz}",
                    )
                    break

                _stderr_txt = ""
                try:
                    _stderr_txt = stderr.decode(errors="replace") if stderr else ""
                except Exception:
                    _stderr_txt = "<decode_err>"
                last_error = RuntimeError(_stderr_txt)
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except OSError:
                    pass
                print(f"Encoder {encoder_candidate['name']} failed, trying fallback...")
                _tail = _stderr_txt[-800:] if len(_stderr_txt) > 800 else _stderr_txt
                _save_mp4_dbg(
                    "ENCODER_FFMPEG_FAIL",
                    f"name={encoder_candidate['name']!r} returncode={process.returncode} stderr_tail={_tail!r}",
                )
            else:
                if temp_dir is not None:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                print("FFmpeg encoding failed!")
                try:
                    print(f"Error: {(stderr or b'').decode(errors='replace')}")
                except Exception:
                    print("Error: <stderr unavailable>")
                if last_error is not None:
                    print(f"Last error: {last_error}")
                _save_mp4_dbg("RETURN", "path=ALL_ENCODERS_FAILED ui_images=0")
                return {"ui": {"images": [], "animated": (True,)}}
            
            # Cleanup temp audio file
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Publish atomically: the final URL must not exist until FFmpeg has
            # closed and synced a complete MP4.
            _tmp_exists = os.path.exists(temp_file_path)
            _tmp_sz = os.path.getsize(temp_file_path) if _tmp_exists else 0
            _save_mp4_dbg("ATOMIC_PRE", f"temp_exists={_tmp_exists} temp_size={_tmp_sz}")
            if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
                with open(temp_file_path, 'r+b') as f:
                    os.fsync(f.fileno())

                os.replace(temp_file_path, file_path)
                _save_mp4_dbg("ATOMIC_REPLACE", f"final_path={file_path!r}")

                try:
                    dir_fd = os.open(full_output_path, os.O_DIRECTORY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except Exception:
                    pass
                
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
                _save_mp4_dbg("RETURN", "path=ATOMIC_FAIL_EMPTY_OR_MISSING_TEMP ui_images=0")
                return {"ui": {"images": [], "animated": (True,)}}
                
        except subprocess.TimeoutExpired:
            process.kill()
            print("Error: FFmpeg encoding timed out (>5 minutes)")
            _save_mp4_dbg("RETURN", "path=FFMPEG_TIMEOUT ui_images=0")
            return {"ui": {"images": [], "animated": (True,)}}
        except Exception as e:
            print(f"Error during FFmpeg encoding: {str(e)}")
            _save_mp4_dbg("RETURN", f"path=TOP_EXCEPTION err={e!r} ui_images=0")
            return {"ui": {"images": [], "animated": (True,)}}
        
        _save_mp4_dbg("RETURN", f"path=SUCCESS_FINAL results_count={len(results)}")
        return {"ui": {"images": results, "animated": (True,)}}


# Node export for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AnymatixSaveAnimatedMP4": AnymatixSaveAnimatedMP4,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnymatixSaveAnimatedMP4": "Anymatix Save Animated MP4",
}
