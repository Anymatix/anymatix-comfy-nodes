"""
The parallel downloader: where the bytes go, and what survives a failure.

Run it with the ComfyUI venv, which has aiohttp/aiofiles:

    /path/to/anymatix/python/bin/python3 tests/test_parallel_download.py

Two things are checked, and both were broken:

1. MEMORY. Segments used to accumulate in RAM (`segment_data += chunk`) and
   `asyncio.gather` held all of them before anything was written, so the peak
   was the whole file plus the copies that `+=` makes — measured at +1.6 GB for
   a 128 MB download. A remote ComfyUI died mid-fetch of a 2.14 GB weight.
2. WHAT SURVIVES. Because nothing was written until the end, an interrupted
   parallel download left NO file — and `download_file` picks the parallel path
   precisely when the file is missing, so every retry started from zero and the
   resume path could never engage.
"""
import contextlib
import hashlib
import http.server
import io
import re
import os
import resource
import socketserver
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch as F  # noqa: E402


def serve(blob, refuse_late_ranges=0):
    """A Range-capable server. Slices are sent from a memoryview so the SERVER
    never shows up in the memory measurement."""
    size = len(blob)
    # `gets` records every GET the server actually SERVED, as (start, end).
    # It is what separates "resumed" from "started again": a resume is a GET
    # whose start is the prefix, and a restart is a GET whose start is 0.
    state = {"refused": 0, "gets": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _range(self):
            header = self.headers.get("Range")
            if not header:
                return None
            start, _, end = header.replace("bytes=", "").partition("-")
            return int(start), (int(end) if end else size - 1)

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            rng = self._range()
            if rng and rng[0] > 0 and state["refused"] < refuse_late_ranges:
                state["refused"] += 1
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = rng if rng else (0, size - 1)
            state["gets"].append((start, end))
            view = memoryview(blob)[start:end + 1]
            self.send_response(206 if rng else 200)
            if rng:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(len(view)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            for offset in range(0, len(view), 1 << 20):
                self.wfile.write(view[offset:offset + (1 << 20)])

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], state


def rss_mb():
    # macOS reports ru_maxrss in bytes, Linux in kilobytes.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def test_streams_to_disk():
    size = 128 * 1024 * 1024
    blob = os.urandom(1 << 20) * (size >> 20)
    server, port, _ = serve(blob)
    out = os.path.join(tempfile.mkdtemp(), "weights.bin")
    try:
        before = rss_mb()
        assert F.fetch_parallel(f"http://127.0.0.1:{port}/weights.bin", out) is True
        growth = rss_mb() - before
        assert os.path.getsize(out) == size, os.path.getsize(out)
        assert hashlib.sha256(open(out, "rb").read()).hexdigest() == hashlib.sha256(blob).hexdigest()
        assert not os.path.exists(out + ".part"), "part file left behind"
        # The old code peaked at ~12x the file size here. Anything near the
        # file size means the bytes are being buffered again.
        assert growth < size / (1024 * 1024) / 2, f"peak RSS grew {growth:.0f} MB for a {size >> 20} MB file"
        print(f"  streams to disk: peak RSS +{growth:.1f} MB for {size >> 20} MB, sha256 matches")
    finally:
        server.shutdown()


def test_failed_segments_leave_a_resumable_prefix():
    size = 64 * 1024 * 1024
    blob = os.urandom(1 << 20) * (size >> 20)
    server, port, state = serve(blob, refuse_late_ranges=4)
    directory = tempfile.mkdtemp()
    try:
        path = F.download_file(f"http://127.0.0.1:{port}/weights.bin", directory)
        assert state["refused"] == 4
        assert os.path.getsize(path) == size
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == hashlib.sha256(blob).hexdigest()
        assert not any(name.endswith(".part") for name in os.listdir(directory))
        print("  lost segments: prefix kept, single stream resumed, sha256 matches")
    finally:
        server.shutdown()


def test_the_kept_prefix_is_what_the_next_attempt_resumes_FROM():
    """The salvage is worth nothing unless the single stream STARTS at it.

    `test_failed_segments_leave_a_resumable_prefix` above passes whether or not
    the resume happens: a download that restarts from byte zero also ends with
    the right bytes and the right sha256. That is exactly how
    `bugs/an-interrupted-parallel-download-restarted-from-zero` survived — the
    only visible difference is 27 GB of traffic, and no assertion looked at it.

    So this one asserts the OFFSET, at the server, where it cannot be faked:

      * the salvage kept N bytes (N > 0);
      * the fallback said so — "Resuming from the N bytes the parallel attempt
        left" — with the same N;
      * the server was asked for `bytes=N-` and NEVER for the whole file again.

    The last is the falsifier that has no opinion about log lines: one GET for
    (0, size-1) after the parallel attempt IS the bug, whatever anything prints.
    """
    size = 64 * 1024 * 1024
    blob = os.urandom(1 << 20) * (size >> 20)
    server, port, state = serve(blob, refuse_late_ranges=4)
    directory = tempfile.mkdtemp()
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            path = F.download_file(f"http://127.0.0.1:{port}/weights.bin", directory)
        out = log.getvalue()

        kept = re.search(r"Keeping the (\d+) bytes that landed", out)
        assert kept, "the parallel attempt never salvaged a prefix:\n" + out[-2000:]
        prefix = int(kept.group(1))
        assert prefix > 0, prefix

        resumed = re.search(
            r"Resuming from the (\d+) bytes the parallel attempt left", out
        )
        assert resumed, (
            f"the salvage kept {prefix} bytes and the fallback resumed from none of "
            f"them — it started again from zero\n" + out[-2000:]
        )
        assert int(resumed.group(1)) == prefix, (int(resumed.group(1)), prefix)

        assert (prefix, size - 1) in state["gets"], (
            f"no GET for bytes={prefix}-{size - 1}; the single stream did not pick "
            f"up where the segments stopped. served: {state['gets']}"
        )
        # ONE range-less GET is expected and it is the FIRST: `fetch_headers`
        # opens the URL to read Content-Length and Content-Disposition, then
        # drops the connection. A SECOND one is the failure this test exists
        # for — the single stream asking for the whole file again.
        whole_file = [i for i, g in enumerate(state["gets"]) if g == (0, size - 1)]
        assert whole_file == [0], (
            f"the whole file was requested again after the salvage (GETs at "
            f"{whole_file}) — this is the download restarting from zero"
        )

        assert os.path.getsize(path) == size, os.path.getsize(path)
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == \
            hashlib.sha256(blob).hexdigest()
        assert not any(name.endswith(".part") for name in os.listdir(directory))
        served = sum(end - start + 1 for start, end in state["gets"])
        print(
            f"  resumed at byte {prefix} ({prefix * 100 // size}% already on disk); "
            f"{served / size:.2f} file-lengths served in total"
        )
    finally:
        server.shutdown()


if __name__ == "__main__":
    print("parallel downloader")
    test_streams_to_disk()
    test_failed_segments_leave_a_resumable_prefix()
    test_the_kept_prefix_is_what_the_next_attempt_resumes_FROM()
    print("OK")
