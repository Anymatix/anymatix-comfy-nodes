import os
import io
import av
import folder_paths

from .anymatix_output_formats import AUDIO_FORMATS, audio_format


class AnymatixSaveAudio:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "output_path": (
                    "STRING",
                    {"default": "anymatix/results", "multiline": False},
                ),
                "filename_prefix": ("STRING", {"default": "audio"}),
            },
            # Optional for the same reason as the image node's: every address
            # already computed asks for a save with no `format`, and MP3 is
            # what it has always meant. `quality` is a bit rate, which is a
            # thing only the lossy codecs have — WAV and FLAC have none to
            # state and are not made to invent one.
            "optional": {
                "format": (list(AUDIO_FORMATS.keys()), {"default": "mp3"}),
                "quality": (["320k", "256k", "192k", "128k", "V0"], {"default": "320k"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "Anymatix"

    def save_audio(
        self,
        audio,
        output_path="anymatix/results",
        filename_prefix="audio",
        quality="320k",
        format="mp3",
    ):
        # Handle None audio (e.g., from videos without audio tracks)
        if audio is None or (isinstance(audio, dict) and audio.get("waveform") is None):
            print(f"anymatix: skipping audio save - no audio data available")
            return {"ui": {"audio": []}}
        
        full_output_path = os.path.join(self.output_dir, output_path)
        os.makedirs(full_output_path, exist_ok=True)

        print(f"anymatix: saving audio to {full_output_path}")

        results = []

        # WHAT THE RUNG MEANS IN BYTES lives in `anymatix_output_formats`, so
        # that the extension the renderer derives from the type and the one
        # written here come from one table. MP3 stays the default: a card that
        # never chose a format keeps the file name it has always had.
        spec = audio_format(format)
        extension = spec["extension"]

        for batch_number, waveform in enumerate(audio["waveform"].cpu()):
            file = f"{filename_prefix}.{extension}"
            output_file = os.path.abspath(os.path.join(full_output_path, file))

            sample_rate = audio["sample_rate"]

            try:
                output_buffer = io.BytesIO()
                output_container = av.open(
                    output_buffer, mode="w", format=spec["container"]
                )

                layout = "mono" if waveform.shape[0] == 1 else "stereo"
                stream_rate = spec.get("sample_rate", sample_rate)
                out_stream = output_container.add_stream(
                    spec["codec"], rate=stream_rate, layout=layout
                )
                if "sample_fmt" in spec:
                    out_stream.format = spec["sample_fmt"]

                # Only the lossy codecs have a bit rate to argue about.
                if spec["codec"] == "libmp3lame":
                    if quality == "V0":
                        out_stream.codec_context.qscale = 1
                    else:
                        out_stream.bit_rate = int(quality.rstrip("k")) * 1000
                elif spec["codec"] == "aac":
                    out_stream.bit_rate = 256000

                frame = av.AudioFrame.from_ndarray(
                    waveform.movedim(0, 1).reshape(1, -1).float().numpy(),
                    format="flt",
                    layout=layout,
                )
                frame.sample_rate = sample_rate
                frame.pts = 0

                # A codec whose rate differs from the source needs the frames
                # resampled; PyAV does that when the stream is asked to encode
                # a frame whose rate does not match, via its own resampler.
                for packet in out_stream.encode(frame):
                    output_container.mux(packet)
                for packet in out_stream.encode(None):
                    output_container.mux(packet)

                output_container.close()

                # Write the output to file and fsync so it is visible to the
                # serving layer immediately after execution_success.
                output_buffer.seek(0)
                with open(output_file, "wb") as f:
                    f.write(output_buffer.getbuffer())
                    f.flush()
                    os.fsync(f.fileno())

                if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                    print(f"Error: Audio output {output_file} was not created or is empty")
                    return {"ui": {"audio": []}}

                file_size = os.path.getsize(output_file)
                print(f"Audio file saved to: {output_file} ({file_size:,} bytes)")

                results.append({
                    "filename": file,
                    "subfolder": "",
                    "type": self.type
                })

            except Exception as e:
                print(f"Error encoding audio to {output_file}: {e}")
                return {"ui": {"audio": []}}

        return {"ui": {"audio": results}}
