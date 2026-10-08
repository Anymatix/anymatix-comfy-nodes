"""
FP8 weights decoded without a cast kernel (`anymatix_mps_fp8.py`).

The claim: **an fp8-scaled checkpoint runs on Apple Silicon.** It failed on its
first sampling step with `Undefined type Float8_e4m3fn` because MPS has no fp8
cast (`bugs/krea-2-cards-fail-apple-silicon-undefined-type-float8`).

These tests run anywhere: the table decode is checked against the CPU cast for
all 256 codes of both fp8 formats, and the registration is checked against a
stand-in registry. The MPS tests run only on a Mac and reproduce the original
failure first, so they cannot pass without discriminating anything.
"""

import importlib.util
import os

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "anymatix_mps_fp8", os.path.join(_HERE, "..", "anymatix_mps_fp8.py")
)
M = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(M)

MPS = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


@pytest.mark.parametrize("fp8", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("out", [torch.float32, torch.float16, torch.bfloat16])
def test_every_code_decodes_as_the_cpu_cast_does(fp8, out):
    codes = torch.arange(256, dtype=torch.uint8).view(fp8)
    expected = codes.to(torch.float32).to(out)
    got = M.fp8_to(codes, out)
    assert got.dtype == out
    assert torch.equal(torch.isnan(got), torch.isnan(expected))
    finite = ~torch.isnan(expected)
    assert torch.equal(got[finite], expected[finite])


def test_dequantize_matches_the_eager_formula_on_cpu():
    x = torch.randn(64, 32).to(torch.float8_e4m3fn)
    scale = torch.tensor(0.37, dtype=torch.float32)
    expected = x.to(torch.bfloat16) * scale.to(torch.bfloat16)
    got = M.dequantize_per_tensor_fp8(x, scale, torch.bfloat16)
    assert got.shape == x.shape
    assert torch.equal(got, expected)


@pytest.mark.parametrize("fp8", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_encode_paths_match_the_eager_backend_on_cpu(fp8):
    eager = pytest.importorskip("comfy_kitchen.backends.eager.quantization")
    torch.manual_seed(0)
    x = torch.randn(256, 64, dtype=torch.bfloat16) * 3
    rng = torch.randint(0, 256, x.shape, dtype=torch.uint8)
    got = M.stochastic_rounding_fp8(x.clone(), rng, fp8)
    want = eager.stochastic_rounding_fp8(x.clone(), rng, fp8)
    assert got.dtype == fp8 and torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    scale = torch.tensor(0.05, dtype=torch.float32)
    got = M.quantize_per_tensor_fp8(x.float(), scale, fp8)
    want = eager.quantize_per_tensor_fp8(x.float(), scale, fp8)
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


class _Registry:
    def __init__(self):
        self._priority = ["cuda", "triton", "eager"]
        self.registered = {}

    def register(self, name, module, capabilities):
        self.registered[name] = (module, capabilities)

    def set_priority(self, order):
        self._priority = list(order)


def test_registers_first_on_a_mac_and_only_on_mps():
    reg = _Registry()
    pytest.importorskip("comfy_kitchen")
    assert M.register(registry=reg, mps_available=True) is True
    assert reg._priority == ["anymatix_mps", "cuda", "triton", "eager"]
    module, caps = reg.registered["anymatix_mps"]
    assert module.dequantize_per_tensor_fp8 is M.dequantize_per_tensor_fp8
    assert set(caps) == {"dequantize_per_tensor_fp8", "quantize_per_tensor_fp8", "stochastic_rounding_fp8"}
    for name in caps:
        assert caps[name].default_devices == frozenset({"mps"})
        assert getattr(module, name) is getattr(M, name)
    # registering twice does not stack the name
    M.register(registry=reg, mps_available=True)
    assert reg._priority.count("anymatix_mps") == 1


def test_does_nothing_without_mps():
    reg = _Registry()
    assert M.register(registry=reg, mps_available=False) is False
    assert reg.registered == {} and reg._priority == ["cuda", "triton", "eager"]


@pytest.mark.skipif(not MPS, reason="needs Apple Silicon")
def test_the_mps_cast_is_what_failed():
    x = torch.randn(8).to(torch.float8_e4m3fn).to("mps")
    with pytest.raises(RuntimeError, match="Float8_e4m3fn"):
        x.to(torch.bfloat16)


@pytest.mark.skipif(not MPS, reason="needs Apple Silicon")
def test_comfy_kitchen_dequantizes_on_mps_once_registered():
    ck = pytest.importorskip("comfy_kitchen")
    assert M.register() is True
    x_cpu = torch.randn(128, 64).to(torch.float8_e4m3fn)
    scale = torch.tensor(1.5, dtype=torch.float32)
    got = ck.dequantize_per_tensor_fp8(x_cpu.to("mps"), scale.to("mps"), torch.bfloat16)
    assert got.device.type == "mps"
    assert torch.equal(got.cpu(), x_cpu.to(torch.bfloat16) * scale.to(torch.bfloat16))


@pytest.mark.skipif(not MPS, reason="needs Apple Silicon")
def test_a_lora_patch_requantizes_on_mps_once_registered():
    """Krea 2 Image to Image's Style LoRA failed here, in the encode direction."""
    ck = pytest.importorskip("comfy_kitchen")
    eager = pytest.importorskip("comfy_kitchen.backends.eager.quantization")
    assert M.register() is True
    x = torch.randn(128, 64, dtype=torch.bfloat16)
    rng = torch.randint(0, 256, x.shape, dtype=torch.uint8)
    got = ck.stochastic_rounding_fp8(x.to("mps"), rng.to("mps"), torch.float8_e4m3fn)
    assert got.device.type == "mps" and got.dtype == torch.float8_e4m3fn
    want = eager.stochastic_rounding_fp8(x, rng, torch.float8_e4m3fn)
    assert torch.equal(got.cpu().view(torch.uint8), want.view(torch.uint8))
