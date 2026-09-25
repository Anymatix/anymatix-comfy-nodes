"""
The two audio-encoder dtype fixes, applied to stand-in modules.

ComfyUI is not importable here, so each test builds a module shaped like the
part of ComfyUI the fix touches — a model whose `encode_audio` sends float32
audio, a wav2vec2 whose positional conv ignores the encoder's dtype — and
checks what the patched classes produce with real torch tensors.
"""

import importlib.util
import os
import threading
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "anymatix_comfy_compat", os.path.join(_HERE, "..", "anymatix_comfy_compat.py")
)
compat = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compat)


# --- audio encoder ----------------------------------------------------------

class _HalfModel(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.lin = nn.Linear(4, 4, dtype=dtype)
        self.seen = []

    def forward(self, x):
        self.seen.append(x.dtype)
        return self.lin(x), ()


def _audio_encoders_module():
    class AudioEncoderModel:
        def __init__(self, dtype):
            self.load_device = torch.device("cpu")
            self.model = _HalfModel(dtype)

        def encode_audio(self, audio, sample_rate):
            out, layers = self.model(audio.to(self.load_device))
            return {"encoded_audio": out, "audio_samples": audio.shape[-1]}

    return types.SimpleNamespace(AudioEncoderModel=AudioEncoderModel)


def test_unpatched_encoder_fails_on_half_precision():
    mod = _audio_encoders_module()
    enc = mod.AudioEncoderModel(torch.bfloat16)
    with pytest.raises(RuntimeError):
        enc.encode_audio(torch.zeros(1, 4), 16000)


def test_encode_audio_feeds_the_model_its_own_dtype():
    mod = _audio_encoders_module()
    assert compat.patch_audio_encoder(mod) is True
    enc = mod.AudioEncoderModel(torch.bfloat16)
    out = enc.encode_audio(torch.zeros(1, 4), 16000)
    assert enc.model.seen == [torch.bfloat16]
    assert out["encoded_audio"].dtype == torch.bfloat16
    assert out["audio_samples"] == 4


def test_float32_encoder_is_unchanged():
    mod = _audio_encoders_module()
    compat.patch_audio_encoder(mod)
    enc = mod.AudioEncoderModel(torch.float32)
    enc.encode_audio(torch.zeros(1, 4), 16000)
    assert enc.model.seen == [torch.float32]


def test_cast_is_scoped_to_encode_audio():
    mod = _audio_encoders_module()
    compat.patch_audio_encoder(mod)
    enc = mod.AudioEncoderModel(torch.bfloat16)
    enc.encode_audio(torch.zeros(1, 4), 16000)
    assert len(enc.model._forward_pre_hooks) == 0
    enc.model(torch.zeros(1, 4, dtype=torch.bfloat16))
    assert enc.model.seen[-1] == torch.bfloat16


def test_audio_encoder_patch_is_idempotent():
    mod = _audio_encoders_module()
    assert compat.patch_audio_encoder(mod) is True
    first = mod.AudioEncoderModel.encode_audio
    assert compat.patch_audio_encoder(mod) is False
    assert mod.AudioEncoderModel.encode_audio is first


def test_missing_encoder_class_is_left_alone():
    assert compat.patch_audio_encoder(types.SimpleNamespace()) is False


# --- wav2vec2 ---------------------------------------------------------------

class _Ops:
    """Stands in for comfy.ops.manual_cast: a distinguishable Conv1d."""

    class Conv1d(nn.Conv1d):
        pass


def _wav2vec2_module():
    class PositionalConvEmbedding(nn.Module):
        def __init__(self, embed_dim=768, kernel_size=128, groups=16):
            super().__init__()
            self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size,
                                  padding=kernel_size // 2, groups=groups)
            self.conv = torch.nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
            self.activation = nn.GELU()

    class TransformerEncoder(nn.Module):
        def __init__(self, embed_dim=768, num_layers=1, dtype=None, device=None, operations=None):
            super().__init__()
            self.pos_conv_embed = PositionalConvEmbedding(embed_dim=embed_dim)
            self.layer_norm = nn.LayerNorm(embed_dim, device=device, dtype=dtype)

    return types.SimpleNamespace(
        PositionalConvEmbedding=PositionalConvEmbedding,
        TransformerEncoder=TransformerEncoder,
    )


def _conv_dtype(encoder):
    return encoder.pos_conv_embed.conv.parametrizations.weight.original1.dtype


def test_unpatched_positional_conv_ignores_the_encoder_dtype():
    mod = _wav2vec2_module()
    enc = mod.TransformerEncoder(embed_dim=32, dtype=torch.float16, operations=_Ops)
    assert _conv_dtype(enc) == torch.float32
    assert type(enc.pos_conv_embed.conv).__mro__[1] is nn.Conv1d


def test_positional_conv_follows_the_encoder():
    mod = _wav2vec2_module()
    assert compat.patch_wav2vec2(mod) is True
    enc = mod.TransformerEncoder(embed_dim=32, dtype=torch.float16, device="cpu", operations=_Ops)
    conv = enc.pos_conv_embed.conv
    assert _conv_dtype(enc) == torch.float16
    assert conv.bias.dtype == torch.float16
    assert isinstance(conv, _Ops.Conv1d)
    assert conv.kernel_size == (128,) and conv.padding == (64,) and conv.groups == 16
    assert isinstance(enc.pos_conv_embed.activation, nn.GELU)


def test_state_dict_keys_are_unchanged():
    before = set(_wav2vec2_module().TransformerEncoder(embed_dim=32).state_dict())
    mod = _wav2vec2_module()
    compat.patch_wav2vec2(mod)
    after = set(mod.TransformerEncoder(embed_dim=32, dtype=torch.float16, operations=_Ops).state_dict())
    assert before == after


def test_positional_conv_built_alone_keeps_upstream_defaults():
    mod = _wav2vec2_module()
    compat.patch_wav2vec2(mod)
    pce = mod.PositionalConvEmbedding(embed_dim=32)
    assert type(pce.conv).__mro__[1] is nn.Conv1d
    assert pce.conv.bias.dtype == torch.float32


def test_encoder_factory_does_not_leak_across_threads():
    mod = _wav2vec2_module()
    compat.patch_wav2vec2(mod)
    results = {}

    def build(name, dtype):
        results[name] = _conv_dtype(mod.TransformerEncoder(embed_dim=32, dtype=dtype, operations=_Ops))

    threads = [threading.Thread(target=build, args=(f"t{i}", d))
               for i, d in enumerate([torch.float16, torch.bfloat16, torch.float32] * 3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [results[f"t{i}"] for i in range(9)] == [torch.float16, torch.bfloat16, torch.float32] * 3
    assert compat._POS_CONV_FACTORY.get() is None


def test_wav2vec2_patch_is_idempotent():
    mod = _wav2vec2_module()
    assert compat.patch_wav2vec2(mod) is True
    init = mod.PositionalConvEmbedding.__init__
    assert compat.patch_wav2vec2(mod) is False
    assert mod.PositionalConvEmbedding.__init__ is init


def test_upstream_fix_is_detected_and_left_alone():
    mod = _wav2vec2_module()

    class PositionalConvEmbedding(nn.Module):
        def __init__(self, embed_dim=768, kernel_size=128, groups=16, dtype=None, device=None, operations=None):
            super().__init__()

    mod.PositionalConvEmbedding = PositionalConvEmbedding
    encoder_init = mod.TransformerEncoder.__init__
    assert compat.patch_wav2vec2(mod) is False
    assert mod.TransformerEncoder.__init__ is encoder_init


def test_apply_never_raises_without_comfy():
    messages = []
    compat.apply(log=messages.append, warn=messages.append)
    assert all("could not patch" in m for m in messages)
