import os
import json
import io
import av
import folder_paths


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
                "quality": (["320k", "256k", "192k", "128k", "V0"], {"default": "320k"}),
            }
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
    ):
        # Handle None audio (e.g., from videos without audio tracks)
        if audio is None or (isinstance(audio, dict) and audio.get("waveform") is None):
            print(f"anymatix: skipping audio save - no audio data available")
            return {"ui": {"audio": []}}
        
        output_path = os.path.join(self.output_dir, output_path)
        os.makedirs(output_path, exist_ok=True)

        print(f"anymatix: saving audio to {output_path}")

        results = []

        for batch_number, waveform in enumerate(audio["waveform"].cpu()):
            # Use filename_prefix directly without counter
            file = f"{filename_prefix}.mp3"
            output_file = os.path.abspath(os.path.join(output_path, file))

            sample_rate = audio["sample_rate"]

            # Create output with MP3 format
            output_buffer = io.BytesIO()
            output_container = av.open(output_buffer, mode='w', format='mp3')

            layout = 'mono' if waveform.shape[0] == 1 else 'stereo'
            out_stream = output_container.add_stream("libmp3lame", rate=sample_rate, layout=layout)
            
            # Set quality/bitrate
            if quality == "V0":
                out_stream.codec_context.qscale = 1
            elif quality == "128k":
                out_stream.bit_rate = 128000
            elif quality == "192k":
                out_stream.bit_rate = 192000
            elif quality == "256k":
                out_stream.bit_rate = 256000
            elif quality == "320k":
                out_stream.bit_rate = 320000

            frame = av.AudioFrame.from_ndarray(
                waveform.movedim(0, 1).reshape(1, -1).float().numpy(),
                format='flt',
                layout=layout
            )
            frame.sample_rate = sample_rate
            frame.pts = 0
            output_container.mux(out_stream.encode(frame))

            # Flush encoder
            output_container.mux(out_stream.encode(None))

            # Close container
            output_container.close()

            # Write the output to file
            output_buffer.seek(0)
            with open(output_file, 'wb') as f:
                f.write(output_buffer.getbuffer())

            print(f"Audio file saved to: {output_file}")

            results.append({
                "filename": file,
                "subfolder": "",
                "type": self.type
            })

        return {"ui": {"audio": results}}
