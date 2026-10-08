"""
FP8 weights on Apple Silicon: decode them without a cast kernel MPS lacks.

The claim: **a card whose checkpoint stores fp8 weights runs on a Mac.**

It did not. On an Apple M5 Max (torch 2.14, comfy-kitchen 0.2.35) Krea 2 and
Krea 2 Image to Image failed on their first sampling step with
`RuntimeError: Undefined type Float8_e4m3fn`
(`bugs/krea-2-cards-fail-apple-silicon-undefined-type-float8`). ComfyUI loads
an fp8-scaled checkpoint as quantized tensors, moves them to the GPU as fp8
(this works: MPS stores the bytes), and dequantizes each layer on the fly with
comfy-kitchen's eager `dequantize_per_tensor_fp8`, whose first line is
`x.to(dtype=output_type)`. MPS has no conversion kernel for either fp8 dtype,
so that cast is what raises. ComfyUI's own `supports_cast` already knows MPS
cannot cast fp8 -- which is why a plain fp8 checkpoint is upcast at load and
runs -- but the mixed-precision path never asks it.

The fix keeps the weights in fp8 on the GPU (half the memory of bf16, which is
what lets a 13 GB checkpoint fit beside its text encoder) and decodes them
with a 256-entry table instead of a cast: an fp8 value IS its byte, so
`table[byte]` is exact for every code, NaN included. The table is built on the
CPU, where the cast exists, once per (fp8 dtype, output dtype, device).

It is registered as a comfy-kitchen backend, through the registry's own public
`register`/`set_priority`, constrained to tensors on `mps`: every other
device keeps whatever backend it had, and nothing in comfy-kitchen or ComfyUI
is rewritten.
"""

import logging

import torch

BACKEND_NAME = "anymatix_mps"
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_TABLES = {}


def fp8_decode_table(fp8_dtype: torch.dtype, out_dtype: torch.dtype, device) -> torch.Tensor:
    """Every fp8 code, in byte order, as `out_dtype` on `device`."""
    key = (fp8_dtype, out_dtype, str(device))
    table = _TABLES.get(key)
    if table is None:
        codes = torch.arange(256, dtype=torch.uint8).view(fp8_dtype)
        table = codes.to(torch.float32).to(out_dtype).to(device)
        _TABLES[key] = table
    return table


def fp8_to(x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """`x.to(out_dtype)` for an fp8 tensor, on a device with no fp8 cast."""
    table = fp8_decode_table(x.dtype, out_dtype, x.device)
    return table[x.view(torch.uint8).to(torch.int32)]


def dequantize_per_tensor_fp8(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """comfy-kitchen's eager `dequantize_per_tensor_fp8`, minus the fp8 cast."""
    return fp8_to(x, output_type) * scale.to(dtype=output_type)


def _constraints():
    from comfy_kitchen.constraints import FunctionConstraints, ParamConstraint

    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    return {
        "dequantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=frozenset(FP8_DTYPES)),
                "scale": ParamConstraint(dtypes=floats),
                "output_type": ParamConstraint(dtypes=floats),
            },
            default_devices=frozenset({"mps"}),
        ),
    }


def register(registry=None, mps_available=None) -> bool:
    """Put the fp8 decoder first in comfy-kitchen's backend order, on a Mac.

    Returns whether it registered. Never raises: a comfy-kitchen without this
    registry API, or a machine without MPS, leaves everything as it was.
    """
    if mps_available is None:
        mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if not mps_available:
        return False
    try:
        if registry is None:
            from comfy_kitchen.registry import registry
        from types import SimpleNamespace

        registry.register(
            name=BACKEND_NAME,
            module=SimpleNamespace(dequantize_per_tensor_fp8=dequantize_per_tensor_fp8),
            capabilities=_constraints(),
        )
        order = [b for b in getattr(registry, "_priority", ["cuda", "triton", "eager"]) if b != BACKEND_NAME]
        registry.set_priority([BACKEND_NAME, *order])
        logging.info("[anymatix] fp8 weights on MPS: decoded by table (backend %s first)", BACKEND_NAME)
        return True
    except Exception as e:  # noqa: BLE001 -- a node pack must not take ComfyUI down
        logging.warning("[anymatix] fp8-on-MPS backend not registered: %s", e)
        return False
