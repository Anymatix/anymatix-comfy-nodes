"""End-to-end test for the results cache expiry, against a real ComfyUI.

Boots ComfyUI with this repo as its only custom node, in a throwaway base
directory, and drives the cache over HTTP the way the app does. Nothing is
mocked: the sweep runs inside the server, on files on disk.

    python3 test_cache_gc_e2e.py

Set ANYMATIX_COMFY_ROOT to the installation to test against (default
~/anymatix, which must contain ComfyUI/ and a python with ComfyUI's deps).
Boot is CPU-only and loads no model, so no GPU is needed.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ANYMATIX_COMFY_ROOT", Path.home() / "anymatix"))

# Short enough that the test finishes, long enough that boot cannot expire the
# fixtures before they are asserted on.
TTL_RESULTS = 5
TTL_INPUTS = 4
MIN_AGE = 1
GC_INTERVAL = 5

BOOT_TIMEOUT = 240
failures: list[str] = []


def check(condition: bool, description: str):
    print(f"{'PASS' if condition else 'FAIL'}  {description}")
    if not condition:
        failures.append(description)


def find_python() -> str | None:
    """First interpreter that can actually import what ComfyUI needs at boot.

    An installation can carry more than one environment, and the one the app
    launches is not necessarily complete for this ComfyUI checkout — so probe
    rather than assume.
    """
    explicit = os.environ.get("ANYMATIX_TEST_PYTHON")
    candidates = (
        [Path(explicit)]
        if explicit
        else [ROOT / "venv/bin/python3", ROOT / "python/bin/python3", Path(sys.executable)]
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        probe = subprocess.run(
            [str(candidate), "-c", "import torch, sqlalchemy, aiohttp"],
            capture_output=True,
        )
        if probe.returncode == 0:
            return str(candidate)
    return None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get(url: str, timeout: float = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read()


def post(url: str, payload: dict, timeout: float = 30):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as r:
        body = r.read()
        try:
            return r.status, json.loads(body)
        except json.JSONDecodeError:
            return r.status, body


def make_result(results_dir: Path, name: str, age_seconds: float) -> Path:
    """A result as the save node writes it: a directory of files under its hash."""
    d = results_dir / name
    d.mkdir(parents=True, exist_ok=True)
    manifest = d / f"{name}.json"
    manifest.write_text(json.dumps({"count": 1}))
    (d / f"{name}_1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    when = time.time() - age_seconds
    for path in (manifest, d / f"{name}_1.png", d):
        os.utime(path, (when, when))
    return d


def make_input(input_dir: Path, name: str, age_seconds: float) -> Path:
    f = input_dir / f"{name}.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n")
    when = time.time() - age_seconds
    os.utime(f, (when, when))
    return f


def wait_until_gone(path: Path, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not path.exists():
            return True
        time.sleep(0.5)
    return not path.exists()


def main() -> int:
    comfy_dir = ROOT / "ComfyUI"
    main_py = comfy_dir / "main.py"
    if not main_py.exists():
        print(f"SKIP: no ComfyUI at {comfy_dir} (set ANYMATIX_COMFY_ROOT)")
        return 0

    python = find_python()
    if python is None:
        print(f"SKIP: no interpreter under {ROOT} can import ComfyUI's dependencies")
        return 0
    port = free_port()
    base = Path(tempfile.mkdtemp(prefix="anymatix-cache-e2e-"))
    custom_nodes = base / "custom_nodes"
    custom_nodes.mkdir()
    # This repo under test, plus its siblings from the installation — the nodes
    # module imports ComfyUI-GGUF at load time and fails to register anything
    # without it.
    os.symlink(REPO, custom_nodes / REPO.name)
    for sibling in (comfy_dir / "custom_nodes").iterdir():
        if sibling.name in {REPO.name, "__pycache__"} or not sibling.is_dir():
            continue
        os.symlink(sibling, custom_nodes / sibling.name)
    results_dir = base / "output/anymatix/results"
    input_dir = base / "input"
    results_dir.mkdir(parents=True)
    input_dir.mkdir(parents=True)

    env = dict(os.environ)
    env.update(
        ANYMATIX_CACHE_TTL_RESULTS=str(TTL_RESULTS),
        ANYMATIX_CACHE_TTL_INPUTS=str(TTL_INPUTS),
        ANYMATIX_CACHE_MIN_AGE=str(MIN_AGE),
        ANYMATIX_CACHE_GC_INTERVAL=str(GC_INTERVAL),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )

    log_path = base / "comfyui.log"
    log = open(log_path, "w")
    # cwd is the base directory because serve_file resolves paths against the
    # working directory — the same arrangement bootstrap.py sets up in production.
    server = subprocess.Popen(
        [
            python,
            str(main_py),
            "--cpu",
            "--port",
            str(port),
            "--base-directory",
            str(base),
            "--disable-auto-launch",
        ],
        cwd=str(base),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    api = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + BOOT_TIMEOUT
        booted = False
        while time.time() < deadline:
            if server.poll() is not None:
                print(f"ComfyUI exited during boot (rc={server.returncode}); log:")
                print(log_path.read_text()[-3000:])
                return 1
            try:
                get(f"{api}/anymatix/cache_size", timeout=2)
                booted = True
                break
            except urllib.error.HTTPError:
                # The server is answering but has no anymatix routes: our module
                # failed to import. Waiting will not change that.
                print("ComfyUI is up but the anymatix node did not load; log:")
                print(log_path.read_text()[-3000:])
                return 1
            except (urllib.error.URLError, ConnectionError, socket.timeout, OSError):
                time.sleep(1)
        if not booted:
            print(f"ComfyUI did not answer within {BOOT_TIMEOUT}s; log:")
            print(log_path.read_text()[-3000:])
            return 1
        print(f"ComfyUI up on {api}, base={base}\n")

        H = lambda n: f"{n:064x}"
        expired, fresh, served, autonomous = H(1), H(2), H(3), H(4)

        # --- expiry, driven explicitly -------------------------------------
        expired_dir = make_result(results_dir, expired, age_seconds=3600)
        fresh_dir = make_result(results_dir, fresh, age_seconds=0)
        expired_input = make_input(input_dir, H(0x11), age_seconds=3600)
        fresh_input = make_input(input_dir, H(0x12), age_seconds=0)

        status, swept = post(f"{api}/anymatix/cache_gc", {})
        check(status == 200, "cache_gc answers 200")
        check(not expired_dir.exists(), "a result past its TTL is deleted")
        check(fresh_dir.exists(), "a result inside its TTL is kept")
        check(not expired_input.exists(), "an input asset past its TTL is deleted")
        check(fresh_input.exists(), "an input asset inside its TTL is kept")
        check(
            expired in swept.get("results", []) and fresh not in swept.get("results", []),
            "the sweep reports exactly what it deleted",
        )

        # --- serving a result renews it ------------------------------------
        served_dir = make_result(results_dir, served, age_seconds=3600)
        status, _ = get(f"{api}/anymatix/output/anymatix/results/{served}/{served}.json")
        check(status == 200, "an expired-but-present result is still served")
        post(f"{api}/anymatix/cache_gc", {})
        check(served_dir.exists(), "serving a result renews its TTL (download in progress survives)")

        # --- active release -------------------------------------------------
        status, released = post(
            f"{api}/anymatix/release_results", {"hashes": [fresh, "not-a-hash", H(0xDEAD)]}
        )
        check(status == 200, "release_results answers 200")
        check(not fresh_dir.exists(), "a released result is deleted at once, TTL or not")
        check(released.get("deleted") == [fresh], "release reports only what it deleted")
        check(served_dir.exists(), "release touches nothing it was not given")

        status, size = get(f"{api}/anymatix/cache_size")
        check(json.loads(size) == 1, "cache_size counts what is left")

        # --- the sweep runs on its own, with no client asking ---------------
        autonomous_dir = make_result(results_dir, autonomous, age_seconds=3600)
        check(
            wait_until_gone(autonomous_dir, timeout=GC_INTERVAL * 3 + 5),
            "an expired result disappears with no client involved",
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        if failures:
            print(f"\nComfyUI log kept at {log_path}")
        else:
            shutil.rmtree(base, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
