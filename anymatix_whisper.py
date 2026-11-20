"""
AnymatixWhisperTranscribe: speech in, text out, with the time of every word.

The model is a transformers Whisper checkpoint directory that AnymatixFetcher
has already put on disk (`url/whisper_pack`, see WHISPER_PACKS in
anymatix_checkpoint_fetcher.py); the node never downloads anything. transformers
is a ComfyUI requirement, so the node adds no dependency of its own.

Spec: app/specifications/listening-cards-v1.md (anymatix repo).
"""

from __future__ import annotations

import os

import torch

import comfy.audio
import comfy.model_management

from .anymatix_checkpoint_fetcher import model_pack_dir_by_name
from .anymatix_whisper_text import LANGUAGE_OPTIONS, AUTO_LANGUAGE, format_timings, language_code

WHISPER_SAMPLE_RATE = 16000

# One model at a time, keyed by what decides its weights and where they live.
_loaded = {"key": None, "pipe": None}


def _pipeline(model_dir: str):
    device = comfy.model_management.get_torch_device()
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    key = (model_dir, str(device), str(dtype))
    if _loaded["key"] == key:
        return _loaded["pipe"]
    _loaded["key"], _loaded["pipe"] = None, None
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True, local_files_only=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )
    _loaded["key"], _loaded["pipe"] = key, pipe
    return pipe


def _mono_16k(audio: dict):
    """ComfyUI AUDIO ({waveform [B, C, T], sample_rate}) -> 1-D float32 numpy at 16 kHz."""
    waveform = audio["waveform"]
    if waveform.dim() == 3:
        waveform = waveform[0]
    waveform = waveform.float().cpu()
    if waveform.dim() == 2:
        waveform = waveform.mean(dim=0, keepdim=True)
    else:
        waveform = waveform.unsqueeze(0)
    sample_rate = int(audio["sample_rate"])
    if sample_rate != WHISPER_SAMPLE_RATE:
        waveform = comfy.audio.resample(waveform, sample_rate, WHISPER_SAMPLE_RATE)
    return waveform.squeeze(0).contiguous().numpy()


class AnymatixWhisperTranscribe:
    DESCRIPTION = "Transcribe speech with Whisper: the text, and the time of every word or sentence."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_pack": ("STRING", {"forceInput": True, "tooltip": "Wire AnymatixFetcher (url/whisper_pack) here."}),
                "language": (LANGUAGE_OPTIONS, {"default": AUTO_LANGUAGE}),
                "word_timings": ("BOOLEAN", {"default": True, "tooltip": "Time every word; off times each sentence."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "timings")
    FUNCTION = "transcribe"
    CATEGORY = "Anymatix/audio"

    def transcribe(self, audio, model_pack, language, word_timings):
        model_dir = model_pack_dir_by_name(model_pack)
        if not os.path.isfile(os.path.join(model_dir, "model.safetensors")):
            raise FileNotFoundError(f"Whisper model missing at {model_dir}: the fetcher did not finish.")
        samples = _mono_16k(audio)
        if samples.size == 0:
            raise ValueError("The audio is empty: there is nothing to transcribe.")
        generate_kwargs = {"task": "transcribe"}
        code = language_code(language)
        if code is not None:
            generate_kwargs["language"] = code
        result = _pipeline(model_dir)(
            {"raw": samples, "sampling_rate": WHISPER_SAMPLE_RATE},
            chunk_length_s=30,
            batch_size=1,
            return_timestamps="word" if word_timings else True,
            generate_kwargs=generate_kwargs,
        )
        return (str(result["text"]).strip(), format_timings(result.get("chunks") or []))
