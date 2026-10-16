#!/usr/bin/env python3
"""
A FILE WHOSE SIZE IS NOT A MULTIPLE OF THE CONNECTION COUNT DOWNLOADS IN ONE RUN.

bugs/a-parallel-download-s-last-segment-stalls (app repo). On RunPod,
2026-10-16, the 21 701 367 310-byte LTX 2.3 FP4 checkpoint from Civitai failed
with `segment 8 stopped at byte 21701367304 of 21701367310`, five times, and
the Gemma encoder from Hugging Face did the same 2 bytes from its end. Eight
connections, `size // 8` per segment, so the remainder (6 bytes, 2 bytes)
became a NINTH segment; the pool has eight connections, the ninth waited
behind eight multi-GB transfers, and aiohttp counts that wait against the
connect timeout — the segment spent its whole retry budget in the queue and
never received a byte. The `Range` itself is fine: Civitai answers
`bytes=21701367304-21701367309` with exactly those 6 bytes.

The server below honours inclusive ranges and sends each one slowly (never
silently), so a segment queued behind the others waits longer than the
connect timeout, exactly as on the pod.

    python3 -m pytest tests/test_parallel_download_tail_segment.py -q
"""
import http.server
import json
import os
import socketserver
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import fetch

pytest.importorskip("aiohttp")
pytest.importorskip("aiofiles")


def serve_slowly(blob, seconds_per_range=2.5, slices=25):
    """A Range server that honours `bytes=a-b` INCLUSIVELY and spreads every
    multi-byte range over `seconds_per_range`, a slice at a time: slow, never
    silent. One-byte ranges (the address probe) are answered at once."""
    size = len(blob)
    gets = []
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            header = self.headers.get("Range")
            start, end = 0, size - 1
            if header:
                a, _, b = header.replace("bytes=", "").partition("-")
                start, end = int(a), (min(int(b), size - 1) if b else size - 1)
            with lock:
                gets.append((start, end))
            view = memoryview(blob)[start:end + 1]
            self.send_response(206 if header else 200)
            if header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(len(view)))
            self.end_headers()
            try:
                if end == start:
                    self.wfile.write(view)
                    return
                step = max(1, len(view) // slices)
                for i in range(0, len(view), step):
                    self.wfile.write(view[i:i + step])
                    self.wfile.flush()
                    time.sleep(seconds_per_range / slices)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], gets


@pytest.fixture
def quick(monkeypatch):
    # A connect timeout well under the time one range takes to serve, as on
    # the pod (30 s against minutes per 2.7 GB segment).
    monkeypatch.setattr(fetch, "DOWNLOAD_CONNECT_SECONDS", 0.5)
    monkeypatch.setattr(fetch, "DOWNLOAD_STALL_SECONDS", 2)
    monkeypatch.setattr(fetch, "DOWNLOAD_ATTEMPTS", 3)
    monkeypatch.setattr(fetch, "DOWNLOAD_RETRY_PAUSES", (0,))


@pytest.mark.parametrize("size,connections", [
    (21_701_367_310, 8),   # the Civitai LTX 2.3 FP4 checkpoint: remainder 6
    (9_447_702_218, 8),    # the Hugging Face Gemma encoder: remainder 2
    (8 * (1 << 20) + 7, 2),
    (5 * (1 << 20), 3),
])
def test_the_layout_has_one_segment_per_connection_and_covers_every_byte(size, connections):
    segments = fetch.segment_layout(size, fetch.segment_step(size, connections))
    assert len(segments) <= connections, f"{len(segments)} segments for {connections} connections"
    assert [s[0] for s in segments] == list(range(len(segments)))
    assert segments[0][1] == 0
    assert segments[-1][2] == size - 1, "the last inclusive end is not the last byte"
    for (_, _, end), (_, start, _) in zip(segments, segments[1:]):
        assert start == end + 1, "segments are not contiguous"
    lengths = [e - s + 1 for _, s, e in segments]
    assert sum(lengths) == size
    assert max(lengths) - min(lengths) < connections, f"the remainder is a stub segment: {lengths}"


def test_a_size_not_divisible_by_the_connections_downloads_in_one_run(quick):
    blob = os.urandom(8 * (1 << 20) + 7)   # 8 MiB + 7: odd, so `// 2` leaves 1 byte over
    server, port, gets = serve_slowly(blob)
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        assert fetch.fetch_parallel(f"http://127.0.0.1:{port}/weights.bin", part, max_connections=2) is True
        with open(part, "rb") as f:
            assert f.read() == blob, "the file is not the served file"
        segments = sorted(g for g in gets if g != (0, 0))  # not the range probe
        assert len(segments) == 2, f"one GET per connection, no tail segment: {segments}"
        assert segments[-1][1] == len(blob) - 1, f"the last range does not end on the last byte: {segments}"
    finally:
        server.shutdown()


def test_a_resumed_layout_with_more_segments_than_connections_still_finishes(quick):
    """A journal fixes the layout of a resumed download, so it may hold more
    unfinished segments than there are connections. Those wait for a
    connection without spending their attempts."""
    blob = os.urandom(6 * (1 << 20) + 5)
    server, port, gets = serve_slowly(blob)
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    with open(part, "wb") as f:
        f.truncate(len(blob))
    with open(fetch.segment_journal_for(part), "w") as j:
        json.dump({"size": len(blob), "segment_size": 2 * (1 << 20), "done": {}}, j)
    try:
        assert fetch.fetch_parallel(f"http://127.0.0.1:{port}/weights.bin", part, max_connections=2) is True
        with open(part, "rb") as f:
            assert f.read() == blob, "the resumed file is not the served file"
        segments = sorted(g for g in gets if g != (0, 0))
        assert len(segments) == 4, f"each of 4 journaled segments fetched once, none retried: {segments}"
    finally:
        server.shutdown()
