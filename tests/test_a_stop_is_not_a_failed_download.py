#!/usr/bin/env python3
"""
A STOP DURING A FETCH TRAVELS UP AS A STOP, NOT AS "Failed to download ...".

bugs/a-paused-download-re-fetches-up-256-mib (app repo), second half.
`AnymatixFetcher.download_model` wrapped every `Exception` from `download_file`
into "Failed to download <type> model: ...". `PrefetchCancelled` is an
`Exception`, so a Pause reached `anymatix_prefetch` as a failure and the
prefetch ended `failed`, not `cancelled`; on a pod with an NVMe cache the inner
cache-first fallback even started the download again against the volume.
ComfyUI's `InterruptProcessingException` is a `BaseException` in the pinned
ComfyUI (#13523) and an `Exception` before it: both must pass untouched.

The module imports ComfyUI, so ComfyUI's modules are stubbed for the import
and `download_file` is replaced by one that raises what a real download
raises. A real connection error still becomes the user-facing failure.

    python3 -m pytest tests/test_a_stop_is_not_a_failed_download.py -q
"""
import importlib.util
import os
import sys
import tempfile
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import pytest
import requests

import fetch


class _Stub(types.ModuleType):
    """Any name asked of it is a fresh, inert class."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (), {"__init__": lambda self, *a, **k: None})
        setattr(self, name, cls)
        return cls


class _ProgressBar:
    def __init__(self, total):
        self.hook = None

    def update_absolute(self, value, total=None, preview=None):
        pass


def _load(monkeypatch, interrupt_cls):
    models = tempfile.mkdtemp()
    names = [
        "comfy", "comfy.sd", "comfy.utils", "comfy.model_management", "folder_paths",
        "comfy_api", "comfy_api.latest", "comfy_execution", "comfy_execution.utils",
        "comfy_extras", "comfy_extras.nodes_hunyuan", "comfy_extras.nodes_lt_audio",
        "comfy_extras.nodes_model_patch", "spandrel", "nodes",
    ]
    for name in names:
        monkeypatch.setitem(sys.modules, name, _Stub(name))
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, sys.modules[name])
    sys.modules["comfy.utils"].ProgressBar = _ProgressBar
    mm = sys.modules["comfy.model_management"]
    mm.InterruptProcessingException = interrupt_cls
    mm.throw_exception_if_processing_interrupted = lambda: None
    fp = sys.modules["folder_paths"]
    fp.models_dir = models
    fp.get_folder_paths = lambda t: [os.path.join(models, t)]
    sys.modules["comfy_execution.utils"].get_executing_context = lambda: None
    monkeypatch.delenv("ANYMATIX_NVME_MODEL_CACHE", raising=False)

    spec = importlib.util.spec_from_file_location(
        "anymatix_checkpoint_fetcher_under_test", os.path.join(_ROOT, "anymatix_checkpoint_fetcher.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InterruptProcessingException(BaseException):
    """ComfyUI's, as pinned (a BaseException since ComfyUI #13523)."""


class _OldInterrupt(Exception):
    """ComfyUI's before #13523, when it was an Exception."""


_OldInterrupt.__name__ = "InterruptProcessingException"

URL = {"type": "checkpoint", "url": "https://example.invalid/weights.safetensors"}


def _raising(exc, calls):
    def download_file(**kwargs):
        calls.append(kwargs["dir"])
        raise exc
    return download_file


@pytest.mark.parametrize("make", [
    lambda: fetch.PrefetchCancelled("prefetch p1 stopped"),
    lambda: InterruptProcessingException(),
    lambda: _OldInterrupt(),
], ids=["prefetch-cancelled", "comfy-interrupt", "comfy-interrupt-pre-13523"])
def test_a_stop_leaves_download_model_unwrapped(monkeypatch, make):
    module = _load(monkeypatch, InterruptProcessingException)
    stop = make()
    calls = []
    monkeypatch.setattr(module, "download_file", _raising(stop, calls))
    with pytest.raises(BaseException) as raised:
        module.AnymatixFetcher().download_model(dict(URL))
    assert raised.value is stop, f"the Stop arrived as {type(raised.value).__name__}: {raised.value}"
    assert len(calls) == 1


def test_a_stop_on_the_cache_disk_is_not_retried_against_the_volume(monkeypatch):
    module = _load(monkeypatch, InterruptProcessingException)
    cache = tempfile.mkdtemp()
    monkeypatch.setenv("ANYMATIX_NVME_MODEL_CACHE", cache)
    if module._nvme_cache_dir_for(module.get_anymatix_models_dir("checkpoints")) is None:
        pytest.skip("the temp disk has under 10 GiB free, so no cache dir is offered")
    stop = fetch.PrefetchCancelled("prefetch p1 stopped")
    calls = []
    monkeypatch.setattr(module, "download_file", _raising(stop, calls))
    with pytest.raises(fetch.PrefetchCancelled):
        module.AnymatixFetcher().download_model(dict(URL))
    assert calls == [os.path.join(cache, "checkpoints")], \
        f"a Stop started the download again elsewhere: {calls}"


def test_a_real_network_error_is_still_a_failed_download(monkeypatch):
    module = _load(monkeypatch, InterruptProcessingException)
    calls = []
    try:
        requests.get("http://127.0.0.1:9/weights.safetensors", timeout=2)
    except requests.exceptions.ConnectionError as e:
        refused = e
    else:
        pytest.skip("something answers on port 9")
    monkeypatch.setattr(module, "download_file", _raising(refused, calls))
    with pytest.raises(Exception) as raised:
        module.AnymatixFetcher().download_model(dict(URL))
    assert raised.value is not refused
    assert str(raised.value).startswith("Failed to download checkpoint model:"), str(raised.value)


def test_the_deprecated_checkpoint_fetcher_lets_a_stop_through_too(monkeypatch):
    module = _load(monkeypatch, InterruptProcessingException)
    stop = fetch.PrefetchCancelled("prefetch p1 stopped")
    monkeypatch.setattr(module, "download_file", _raising(stop, []))
    with pytest.raises(fetch.PrefetchCancelled):
        module.AnymatixCheckpointFetcher().download_model(URL["url"])
