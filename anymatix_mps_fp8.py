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

The same cast sits at the end of the two ENCODE paths, `quantize_per_tensor_fp8`
and `stochastic_rounding_fp8`, which ComfyUI calls when it patches an fp8
weight (a Style LoRA on Krea 2 Image to Image failed there, after the decode
was fixed). Those are re-implemented with the eager arithmetic on the device
and the final cast done on the CPU.

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


def fp8_from(x: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    """`x.to(fp8_dtype)` on a device with no fp8 cast: cast on the CPU, move back.

    Only the encode direction pays the round trip, and ComfyUI encodes once per
    weight when it patches one (a LoRA), never per sampling step.
    """
    return x.to("cpu").to(fp8_dtype).to(x.device)


def quantize_per_tensor_fp8(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.float8_e4m3fn
) -> torch.Tensor:
    """comfy-kitchen's eager `quantize_per_tensor_fp8`, minus the fp8 cast."""
    lp_max = torch.finfo(output_type).max
    temp = x * (1.0 / scale).to(x.dtype)
    temp = torch.clamp(temp, -lp_max, lp_max, out=temp)
    return fp8_from(temp, output_type)


def stochastic_rounding_fp8(
    x: torch.Tensor, rng: torch.Tensor, output_type: torch.dtype = torch.float8_e4m3fn
) -> torch.Tensor:
    """comfy-kitchen's eager `stochastic_rounding_fp8`, minus the fp8 cast.

    The arithmetic is the eager backend's, line for line, on the tensor's own
    device; the result is already an exact fp8 value, so the CPU cast at the end
    only re-encodes it. A Style LoRA on Krea 2 Image to Image patches the fp8
    weights through this path, and its last line was the failing cast.
    """
    from comfy_kitchen.backends.eager.quantization import calc_mantissa

    if output_type == torch.float8_e4m3fn:
        exponent_bits, mantissa_bits, exponent_bias = 4, 3, 7
    elif output_type == torch.float8_e5m2:
        exponent_bits, mantissa_bits, exponent_bias = 5, 2, 15
    else:
        raise ValueError(f"Unsupported output_type: {output_type}")

    x = x.half()
    sign = torch.sign(x)
    abs_x = x.abs()
    sign = torch.where(abs_x == 0, 0, sign)
    exponent = torch.clamp(torch.floor(torch.log2(abs_x)) + exponent_bias, 0, 2**exponent_bits - 1)
    normal_mask = ~(exponent == 0)
    abs_x[:] = calc_mantissa(abs_x, exponent, normal_mask, mantissa_bits, exponent_bias, rng)
    sign *= torch.where(
        normal_mask,
        (2.0 ** (exponent - exponent_bias)) * (1.0 + abs_x),
        (2.0 ** (-exponent_bias + 1)) * abs_x,
    )
    info = torch.finfo(output_type)
    torch.clamp(sign, min=info.min, max=info.max, out=sign)
    return fp8_from(sign, output_type)


def dequantize_per_tensor_fp8(
    x: torch.Tensor, scale: torch.Tensor, output_type: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """comfy-kitchen's eager `dequantize_per_tensor_fp8`, minus the fp8 cast."""
    return fp8_to(x, output_type) * scale.to(dtype=output_type)


def _constraints():
    from comfy_kitchen.constraints import FunctionConstraints, ParamConstraint

    floats = frozenset({torch.float32, torch.float16, torch.bfloat16})
    fp8 = frozenset(FP8_DTYPES)
    mps = frozenset({"mps"})
    return {
        "quantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "scale": ParamConstraint(dtypes=frozenset({torch.float32})),
                "output_type": ParamConstraint(dtypes=fp8),
            },
            default_devices=mps,
        ),
        "stochastic_rounding_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=floats),
                "rng": ParamConstraint(dtypes=frozenset({torch.uint8})),
                "output_type": ParamConstraint(dtypes=fp8),
            },
            default_devices=mps,
        ),
        "dequantize_per_tensor_fp8": FunctionConstraints(
            params={
                "x": ParamConstraint(dtypes=fp8),
                "scale": ParamConstraint(dtypes=floats),
                "output_type": ParamConstraint(dtypes=floats),
            },
            default_devices=mps,
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
            module=SimpleNamespace(
                dequantize_per_tensor_fp8=dequantize_per_tensor_fp8,
                quantize_per_tensor_fp8=quantize_per_tensor_fp8,
                stochastic_rounding_fp8=stochastic_rounding_fp8,
            ),
            capabilities=_constraints(),
        )
        order = [b for b in getattr(registry, "_priority", ["cuda", "triton", "eager"]) if b != BACKEND_NAME]
        registry.set_priority([BACKEND_NAME, *order])
        logging.info("[anymatix] fp8 weights on MPS: decoded by table (backend %s first)", BACKEND_NAME)
        return True
    except Exception as e:  # noqa: BLE001 -- a node pack must not take ComfyUI down
        logging.warning("[anymatix] fp8-on-MPS backend not registered: %s", e)
        return False
