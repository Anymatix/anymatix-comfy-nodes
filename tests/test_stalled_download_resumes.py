#!/usr/bin/env python3
"""
A DOWNLOAD WHOSE LINK GOES SILENT IS DETECTED, RESUMED, AND ONLY THEN REPORTED.

bugs/a-local-run-stalled-model-download-given (app repo). On sleipnir,
2026-10-10, a Civitai VAE stopped at 591M of 1.45G and nothing happened for 28
minutes: the parallel downloader had no read timeout, so a connection that went
silent without a reset was waited on for ever, and ComfyUI's interrupt could
not reach the thread doing it.

The silence is reproduced, not described: the server below sends part of a
range and then holds the connection open without a byte.

    python3 -m pytest tests/test_stalled_download_resumes.py -q
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


def serve_stalling(blob, stalls, stall_after=256 * 1024, hold=30.0):
    """A Range server whose first `stalls` GETs send `stall_after` bytes and
    then go silent for `hold` seconds with the connection open."""
    size = len(blob)
    state = {"stalled": 0, "gets": []}
    lock = threading.Lock()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            header = self.headers.get("Range")
            start, end = 0, size - 1
            if header:
                a, _, b = header.replace("bytes=", "").partition("-")
                start, end = int(a), (int(b) if b else size - 1)
            with lock:
                state["gets"].append((start, end))
                stall = state["stalled"] < stalls
                if stall:
                    state["stalled"] += 1
            view = memoryview(blob)[start:end + 1]
            self.send_response(206 if header else 200)
            if header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(len(view)))
            self.end_headers()
            try:
                if stall:
                    self.wfile.write(view[:stall_after])
                    self.wfile.flush()
                    time.sleep(hold)
                    return
                self.wfile.write(view)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], state


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(fetch, "DOWNLOAD_STALL_SECONDS", 1)
    monkeypatch.setattr(fetch, "DOWNLOAD_ATTEMPTS", 3)
    monkeypatch.setattr(fetch, "DOWNLOAD_RETRY_PAUSES", (0,))


def test_a_stalled_segment_resumes_from_the_byte_it_reached(quick):
    blob = os.urandom(1 << 20) * 8
    server, port, state = serve_stalling(blob, stalls=2)
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        started = time.monotonic()
        assert fetch.fetch_parallel(f"http://127.0.0.1:{port}/weights.bin", part, max_connections=2) is True
        assert time.monotonic() - started < 20, "a stall was waited out instead of detected"
        with open(part, "rb") as f:
            assert f.read() == blob, "the resumed file is not the served file"
        segment_starts = {0, len(blob) // 2}
        resumed = [s for s, e in state["gets"] if s not in segment_starts]
        assert resumed, f"no GET resumed mid-segment: {state['gets']}"
        assert all(s % (256 * 1024) == 0 for s in resumed)
    finally:
        server.shutdown()


def test_a_link_that_never_comes_back_fails_bounded_and_keeps_the_bytes(quick):
    blob = os.urandom(1 << 20) * 8
    server, port, state = serve_stalling(blob, stalls=10 ** 6)
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        started = time.monotonic()
        with pytest.raises(Exception) as raised:
            fetch.fetch_parallel(f"http://127.0.0.1:{port}/weights.bin", part, max_connections=2)
        assert time.monotonic() - started < 30
        stall = fetch.stall_in(raised.value)
        assert stall is not None, f"not reported as a stall: {raised.value}"
        assert stall.reason == "no data for 1s, 3 attempts"
        assert os.path.getsize(part) == len(blob), "the part file was not kept for a resume"
        journal = fetch.read_segment_journal(part, len(blob))
        assert journal is not None, "the journal was not kept for a resume"
        segments = [g for g in state["gets"] if g != (0, 0)]  # not the range probe
        assert len(segments) == 2 * 3, f"each of 2 segments tried 3 times: {state['gets']}"
    finally:
        server.shutdown()


def test_download_file_names_the_file_when_it_gives_up(quick):
    blob = os.urandom(1 << 20) * 8
    server, port, _ = serve_stalling(blob, stalls=10 ** 6)
    directory = tempfile.mkdtemp()
    try:
        with pytest.raises(fetch.DownloadStalled) as raised:
            fetch.download_file(f"http://127.0.0.1:{port}/weights.bin", directory)
        message = str(raised.value)
        assert message.startswith("Download of weights"), message
        assert "stopped: no data for 1s, 3 attempts" in message
        assert any(name.endswith(".part") for name in os.listdir(directory)), "the bytes that arrived were deleted"
    finally:
        server.shutdown()


def test_the_single_stream_resumes_after_a_stall(quick, monkeypatch):
    blob = os.urandom(1 << 20) * 3
    server, port, state = serve_stalling(blob, stalls=2)
    # Too small for the parallel path: the single stream does it all.
    monkeypatch.setattr(fetch, "fetch_parallel", lambda *a, **k: False)
    directory = tempfile.mkdtemp()
    try:
        path = fetch.download_file(f"http://127.0.0.1:{port}/weights.bin", directory)
        with open(path, "rb") as f:
            assert f.read() == blob
        starts = [s for s, e in state["gets"]]
        assert any(s > 0 for s in starts), f"the stream started again from zero: {starts}"
    finally:
        server.shutdown()


def test_stop_reaches_a_stalled_parallel_download(monkeypatch):
    class InterruptProcessingException(Exception):
        pass

    stop = threading.Event()

    def check_interrupted():
        if stop.is_set():
            stop.clear()
            raise InterruptProcessingException()

    monkeypatch.setattr(fetch, "check_interrupted", check_interrupted)
    blob = os.urandom(1 << 20) * 8
    server, port, _ = serve_stalling(blob, stalls=10 ** 6, hold=60)
    part = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        threading.Timer(1.0, stop.set).start()
        started = time.monotonic()
        with pytest.raises(InterruptProcessingException):
            fetch.fetch_parallel(f"http://127.0.0.1:{port}/weights.bin", part, max_connections=2)
        assert time.monotonic() - started < 5, "Stop waited for the stalled transfer"
        assert os.path.getsize(part) == len(blob), "Stop threw the part file away"
    finally:
        server.shutdown()
