"""
Two dtype fixes to ComfyUI's audio encoders, applied to the loaded classes.

Both used to be text substitutions that the Anymatix setup script wrote into
ComfyUI's own files after every checkout. They live here instead because this
pack is GPL-3.0 and published in source form, like ComfyUI: the right home for
code that adapts ComfyUI, and it keeps the installer free of ComfyUI's source.
ComfyUI's files now stay exactly as checked out.

1. `AudioEncoderModel.encode_audio` moves the resampled audio to the load
   device but keeps it float32. A half-precision encoder (fp16/bf16 on CUDA and
   MPS) then fails on a dtype mismatch in its first layer. The fix casts the
   input to the dtype of the model's first parameter, read at call time, just
   before the model runs.

2. `wav2vec2.PositionalConvEmbedding` builds a plain `nn.Conv1d` and ignores
   the `dtype`, `device` and `operations` its `TransformerEncoder` was given,
   so that one layer stayed float32 on the CPU, outside ComfyUI's casting,
   while the rest of the model followed it. The fix builds the conv with them
   (`operations.Conv1d`, i.e. ComfyUI's manual-cast op, when given).

Each fix is applied only where it is still needed, once per process, and a
failure is logged, never raised: an encoder that cannot be patched is exactly
as it was before, and must not stop the pack from loading.
"""

import contextvars
import functools
import inspect
import logging

_LOG = logging.getLogger(__name__)
_MARK = "_anymatix_dtype_compat"

# The dtype/device/operations of the TransformerEncoder being built, handed to
# the PositionalConvEmbedding it builds. A context variable, not a global:
# two models loading at once on two threads must not see each other's.
_POS_CONV_FACTORY = contextvars.ContextVar("anymatix_pos_conv_factory", default=None)


def _cast_input_to_model_dtype(module, args):
    """Forward pre-hook: cast the model's first input to its parameter dtype."""
    if not args:
        return None
    first = args[0]
    to = getattr(first, "to", None)
    if to is None:
        return None
    try:
        dtype = next(module.parameters()).dtype
    except StopIteration:
        return None
    return (to(dtype=dtype),) + tuple(args[1:])


def patch_audio_encoder(audio_encoders_module) -> bool:
    """Make `encode_audio` feed the model its own dtype. True when applied now."""
    cls = getattr(audio_encoders_module, "AudioEncoderModel", None)
    original = getattr(cls, "encode_audio", None) if cls is not None else None
    if original is None or getattr(original, _MARK, False):
        return False

    @functools.wraps(original)
    def encode_audio(self, *args, **kwargs):
        model = getattr(self, "model", None)
        register = getattr(model, "register_forward_pre_hook", None)
        if register is None:
            return original(self, *args, **kwargs)
        # Scoped to this one call: the model's other callers are untouched.
        handle = register(_cast_input_to_model_dtype)
        try:
            return original(self, *args, **kwargs)
        finally:
            handle.remove()

    setattr(encode_audio, _MARK, True)
    cls.encode_audio = encode_audio
    return True


def patch_wav2vec2(wav2vec2_module, nn=None, weight_norm=None) -> bool:
    """Build wav2vec2's positional conv with the encoder's dtype, device and ops."""
    pce = getattr(wav2vec2_module, "PositionalConvEmbedding", None)
    encoder = getattr(wav2vec2_module, "TransformerEncoder", None)
    if pce is None or encoder is None:
        return False
    if getattr(pce.__init__, _MARK, False):
        return False
    try:
        pce_params = inspect.signature(pce.__init__).parameters
    except (TypeError, ValueError):
        return False
    if "operations" in pce_params:
        return False  # upstream already takes them; nothing to fix
    if nn is None:
        import torch.nn as nn
    if weight_norm is None:
        import torch
        weight_norm = torch.nn.utils.parametrizations.weight_norm

    original_encoder_init = encoder.__init__
    try:
        encoder_sig = inspect.signature(original_encoder_init)
    except (TypeError, ValueError):
        return False

    def pce_init(self, embed_dim=768, kernel_size=128, groups=16, dtype=None, device=None, operations=None):
        if dtype is None and device is None and operations is None:
            dtype, device, operations = _POS_CONV_FACTORY.get() or (None, None, None)
        nn.Module.__init__(self)
        conv_class = nn.Conv1d if operations is None else operations.Conv1d
        self.conv = conv_class(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
            device=device,
            dtype=dtype,
        )
        self.conv = weight_norm(self.conv, name="weight", dim=2)
        self.activation = nn.GELU()

    @functools.wraps(original_encoder_init)
    def encoder_init(self, *args, **kwargs):
        try:
            bound = encoder_sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            factory = tuple(bound.arguments.get(k) for k in ("dtype", "device", "operations"))
        except TypeError:
            factory = None
        token = _POS_CONV_FACTORY.set(factory)
        try:
            original_encoder_init(self, *args, **kwargs)
        finally:
            _POS_CONV_FACTORY.reset(token)

    setattr(pce_init, _MARK, True)
    setattr(encoder_init, _MARK, True)
    pce.__init__ = pce_init
    encoder.__init__ = encoder_init
    return True


def apply(log=_LOG.info, warn=_LOG.warning) -> None:
    """Patch the loaded ComfyUI audio encoders. Never raises."""
    try:
        import comfy.audio_encoders.wav2vec2 as wav2vec2
        if patch_wav2vec2(wav2vec2):
            log("anymatix: applied wav2vec2 positional convolution dtype compatibility patch")
    except Exception as e:
        warn(f"anymatix: could not patch wav2vec2 dtype handling: {e}")
    try:
        import comfy.audio_encoders.audio_encoders as audio_encoders
        if patch_audio_encoder(audio_encoders):
            log("anymatix: applied audio encoder dtype compatibility patch")
    except Exception as e:
        warn(f"anymatix: could not patch audio encoder dtype handling: {e}")
