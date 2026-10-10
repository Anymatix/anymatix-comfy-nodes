#!/usr/bin/env python3
"""
A COLD FILE ANSWERS ITS FIRST RANGE REQUESTS WITH 200; ASK AGAIN, DO NOT GIVE UP.

bugs/a-resumed-civitai-download-re-fetches-whole-file (app repo). Measured
2026-10-10 against Civitai's live API, no key: `/api/download/models/<id>`
redirects to a signed Backblaze B2 URL, and on a file nobody fetched lately B2
answers the first one or two range requests with `200` (the whole file) and
every later one with `206` — 40 cold files of 40. The downloader took the first
`200` as final, so on sleipnir every segment of a resume failed and a 1.45 GB
VAE started again from byte 0.

The server below behaves like that: `/api/<name>` redirects to
`/signed/<name>?sig=<n>`, the first `cold` range requests get `200` and the
whole body, later ones `206`; a signature can be made to expire (403).

    python3 -m pytest tests/test_range_warmup.py -q
"""
import http.server
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


def serve_cold(blob, cold=2, expire_after=None, stalls=0, stall_after=256 * 1024, stall_from=2):
    """`cold`: range requests answered 200 before the file is 'warm'.
    `expire_after`: signed GETs served before the first signature expires.
    `stalls`: GETs, from the `stall_from`-th signed GET on (the first is the
    header read), that send `stall_after` bytes and then drop the connection."""
    size = len(blob)
    state = {"api": 0, "gets": [], "ranges_answered_200": 0, "signed_gets": 0, "valid": {1}, "stalled": 0}
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _redirect(self):
            with lock:
                state["api"] += 1
                sig = max(state["valid"])
            self.send_response(307)
            self.send_header("Location", f"/signed/weights.bin?sig={sig}")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _signed_ok(self):
            sig = int(self.path.split("sig=")[1])
            if sig not in state["valid"]:
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return False
            return True

        def do_HEAD(self):
            if self.path.startswith("/api/"):
                return self._redirect()
            if not self._signed_ok():
                return
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")  # B2 says so even when it ignores them
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/api/"):
                return self._redirect()
            if not self._signed_ok():
                return
            header = self.headers.get("Range")
            start, end = 0, size - 1
            if header:
                a, _, b = header.replace("bytes=", "").partition("-")
                start, end = int(a), (int(b) if b else size - 1)
            with lock:
                state["signed_gets"] += 1
                if expire_after is not None and state["signed_gets"] == expire_after:
                    state["valid"] = {max(state["valid"]) + 1}
                ignore = bool(header) and state["ranges_answered_200"] < cold
                if ignore:
                    state["ranges_answered_200"] += 1
                stall = not ignore and state["signed_gets"] >= stall_from and state["stalled"] < stalls
                if stall:
                    state["stalled"] += 1
                state["gets"].append((start, end, 200 if ignore or not header else 206))
            if ignore:
                start, end = 0, size - 1
            view = memoryview(blob)[start:end + 1]
            self.send_response(206 if header and not ignore else 200)
            if header and not ignore:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(view)))
            self.end_headers()
            try:
                if stall:
                    self.wfile.write(view[:stall_after])
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.shutdown(2)
                    return
                self.wfile.write(view)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], state


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(fetch, "RANGE_WARMUP_PAUSES", (0.05, 0.05, 0.05))
    monkeypatch.setattr(fetch, "DOWNLOAD_STALL_SECONDS", 2)
    monkeypatch.setattr(fetch, "DOWNLOAD_ATTEMPTS", 3)
    monkeypatch.setattr(fetch, "DOWNLOAD_RETRY_PAUSES", (0,))


def test_a_cold_file_is_downloaded_in_parallel_through_one_resolved_address(quick):
    blob = os.urandom(1 << 20) * 8
    server, port, state = serve_cold(blob, cold=2)
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        assert fetch.fetch_parallel(f"http://127.0.0.1:{port}/api/weights.bin", part, max_connections=4) is True
        with open(part, "rb") as f:
            assert f.read() == blob
        segments = [g for g in state["gets"] if g[:2] != (0, 0)]
        assert segments and all(code == 206 for _, _, code in segments), \
            f"a segment met the cold answer, so the probe did not warm the file: {state['gets']}"
        assert state["api"] == 1, f"the redirect was followed {state['api']} times, not once"
    finally:
        server.shutdown()


def test_a_segment_that_meets_the_cold_answer_asks_again(quick):
    import asyncio
    import aiohttp

    blob = os.urandom(1 << 20) * 4
    server, port, state = serve_cold(blob, cold=3)
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    with open(part, "wb") as f:
        f.truncate(len(blob))
    start, end = 1 << 20, (2 << 20) - 1

    async def go():
        async with aiohttp.ClientSession() as session:
            url = fetch.DownloadUrl(f"http://127.0.0.1:{port}/api/weights.bin",
                                    f"http://127.0.0.1:{port}/signed/weights.bin?sig=1")
            return await fetch.fetch_async_segment_resuming(session, url, start, end, 1, part_path=part)

    try:
        assert asyncio.run(go()) == 1
        with open(part, "rb") as f:
            f.seek(start)
            assert f.read(end - start + 1) == blob[start:end + 1]
        assert os.path.getsize(part) == len(blob), "a 200 body was written into the part file"
        assert [code for _, _, code in state["gets"]] == [200, 200, 200, 206]
    finally:
        server.shutdown()


def test_an_expired_signature_is_resolved_again(quick):
    blob = os.urandom(1 << 20) * 8
    # Probe + 1 segment GET on the first signature, then it expires.
    server, port, state = serve_cold(blob, cold=0, expire_after=2)
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        assert fetch.fetch_parallel(f"http://127.0.0.1:{port}/api/weights.bin", part, max_connections=4) is True
        with open(part, "rb") as f:
            assert f.read() == blob
        assert state["api"] == 2, f"expected one resolution and one re-resolution, got {state['api']}"
    finally:
        server.shutdown()


def test_a_server_that_never_honours_ranges_gets_one_stream(quick):
    blob = os.urandom(1 << 20) * 8
    server, port, _ = serve_cold(blob, cold=10 ** 6)
    try:
        assert fetch.check_range_support(f"http://127.0.0.1:{port}/api/weights.bin") == (False, len(blob))
        path = fetch.download_file(f"http://127.0.0.1:{port}/api/weights.bin", tempfile.mkdtemp())
        with open(path, "rb") as f:
            assert f.read() == blob
    finally:
        server.shutdown()


def test_the_single_stream_resume_waits_out_the_cold_answer(quick, monkeypatch):
    blob = os.urandom(1 << 20) * 3
    # The first GET drops after 256 KiB; its resume meets two cold answers.
    server, port, state = serve_cold(blob, cold=2, stalls=1)
    monkeypatch.setattr(fetch, "fetch_parallel", lambda *a, **k: False)
    try:
        path = fetch.download_file(f"http://127.0.0.1:{port}/api/weights.bin", tempfile.mkdtemp())
        with open(path, "rb") as f:
            assert f.read() == blob
        resumed = [g for g in state["gets"] if g[0] > 0]
        assert resumed and resumed[-1][2] == 206, f"the resume did not carry on from its byte: {state['gets']}"
    finally:
        server.shutdown()


def test_the_single_stream_starts_over_when_the_server_never_resumes(quick, monkeypatch):
    blob = os.urandom(1 << 20) * 3
    server, port, state = serve_cold(blob, cold=10 ** 6, stalls=1)
    monkeypatch.setattr(fetch, "fetch_parallel", lambda *a, **k: False)
    try:
        path = fetch.download_file(f"http://127.0.0.1:{port}/api/weights.bin", tempfile.mkdtemp())
        with open(path, "rb") as f:
            assert f.read() == blob, "the restart from byte 0 did not produce the file"
    finally:
        server.shutdown()
