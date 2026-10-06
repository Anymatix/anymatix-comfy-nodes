"""A scan of sibling sidecars must not refresh their access time.

bootstrap.py's volume auto-clean treats a sidecar's atime as "a card last
asked for this model". A scan that touches every sibling destroys that.
"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch

OLD = 1_600_000_000


class _Spy:
    """Filesystems differ on whether a read moves atime (APFS never does), so
    the proof is also that the read puts the old atime back."""
    def __enter__(self):
        self.calls = []
        self.real = os.utime
        def spy(path, *a, **k):
            self.calls.append((path, k.get("ns")))
            return self.real(path, *a, **k)
        os.utime = spy
        return self
    def __exit__(self, *e):
        os.utime = self.real


def _make(d, name, data):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        json.dump(data, f)
    os.utime(p, (OLD, OLD))
    return p


def test_read_sidecar_keeps_atime():
    with tempfile.TemporaryDirectory() as d:
        p = _make(d, "a.json", {"file_name": "w.safetensors"})
        before = os.stat(p).st_atime_ns
        with _Spy() as spy:
            assert fetch.read_sidecar(p) == {"file_name": "w.safetensors"}
        assert (p, (before, os.stat(p).st_mtime_ns)) in spy.calls
        assert os.stat(p).st_atime == OLD


def test_adoption_scan_keeps_sibling_atimes():
    with tempfile.TemporaryDirectory() as d:
        sib = _make(d, "b.json", {"file_name": "other.safetensors", "sha256": "ab" * 32})
        with _Spy() as spy:
            list(fetch.adoption_candidates([d], "w.safetensors", "cd" * 32, "self.json", None))
        assert any(c[0] == sib for c in spy.calls)
        assert os.stat(sib).st_atime == OLD


if __name__ == "__main__":
    test_read_sidecar_keeps_atime()
    test_adoption_scan_keeps_sibling_atimes()
    print("2 passed")
