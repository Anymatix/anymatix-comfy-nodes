"""A CLIPSeg download in progress must not wear the name of a finished model.

bugs/create-mask-s-clipseg-fetch-writes-model: the fetch wrote `model.safetensors`
in place, so a run stopped mid-download left a truncated file that every later
run trusted ('incomplete metadata'). Loaded against stubs for folder_paths/comfy.
"""
import importlib.util
import json
import os
import struct
import sys
import types

import pytest

pytest.importorskip("torch")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(models_dir):
    fp = types.ModuleType("folder_paths")
    fp.models_dir = str(models_dir)
    sys.modules["folder_paths"] = fp
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")

    class PB:
        def __init__(self, *a): pass
        def update_absolute(self, *a): pass
    utils.ProgressBar = PB
    comfy.utils = utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils
    spec = importlib.util.spec_from_file_location("anymatix_clipseg", os.path.join(_HERE, "..", "anymatix_clipseg.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def safetensors_bytes(n=64):
    header = json.dumps({"w": {"dtype": "U8", "shape": [n], "data_offsets": [0, n]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"x" * n


class FakeResponse:
    def __init__(self, data, declared=None, die_after=None):
        self.data, self.die_after = data, die_after
        self.headers = {"content-length": str(len(data) if declared is None else declared)}

    def raise_for_status(self): pass

    def iter_content(self, chunk_size):
        for i in range(0, len(self.data), 16):
            if self.die_after is not None and i >= self.die_after:
                raise KeyboardInterrupt("stopped mid-download")
            yield self.data[i:i + 16]


def test_killed_download_leaves_no_file_at_the_final_name(tmp_path):
    m = _load(tmp_path)
    dest = str(tmp_path / "model.safetensors")
    m._requests = types.SimpleNamespace(get=lambda *a, **k: FakeResponse(safetensors_bytes(), die_after=48))
    with pytest.raises(KeyboardInterrupt):
        m._download_to_path("http://x/model.safetensors", dest)
    assert not os.path.exists(dest)
    assert not os.path.exists(dest + ".part")


def test_short_transfer_is_refused(tmp_path):
    m = _load(tmp_path)
    dest = str(tmp_path / "model.safetensors")
    data = safetensors_bytes()
    m._requests = types.SimpleNamespace(get=lambda *a, **k: FakeResponse(data[:50], declared=len(data)))
    with pytest.raises(IOError):
        m._download_to_path("http://x", dest)
    assert not os.path.exists(dest)


def test_complete_transfer_is_renamed_into_place(tmp_path):
    m = _load(tmp_path)
    dest = str(tmp_path / "model.safetensors")
    data = safetensors_bytes()
    m._requests = types.SimpleNamespace(get=lambda *a, **k: FakeResponse(data))
    m._download_to_path("http://x", dest)
    assert open(dest, "rb").read() == data and not os.path.exists(dest + ".part")


def test_truncated_safetensors_is_detected_and_refetched(tmp_path):
    m = _load(tmp_path)
    d = m.get_clipseg_model_dir()
    data = safetensors_bytes()
    for name in m.CLIPSEG_CONFIG_FILES:
        with open(os.path.join(d, name), "w") as f:
            f.write("{}" if name.endswith(".json") else "x")
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(data[:40])
    assert not m._is_complete_file(os.path.join(d, "model.safetensors"))
    m._requests = types.SimpleNamespace(get=lambda *a, **k: FakeResponse(data))
    m.ensure_clipseg_model()
    assert open(os.path.join(d, "model.safetensors"), "rb").read() == data


def test_whole_files_are_kept(tmp_path):
    m = _load(tmp_path)
    p = tmp_path / "model.safetensors"
    p.write_bytes(safetensors_bytes())
    assert m._is_complete_file(str(p))
    (tmp_path / "c.json").write_text('{"a":1}')
    assert m._is_complete_file(str(tmp_path / "c.json"))
    (tmp_path / "bad.json").write_text('{"a":')
    assert not m._is_complete_file(str(tmp_path / "bad.json"))
