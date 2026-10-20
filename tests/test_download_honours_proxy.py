"""
The parallel downloader goes through the person's proxy.

`bugs/the-model-fetcher-ignores-http-s-proxy` (app repo). aiohttp's `trust_env`
defaults to False and `requests` (the header probe, the single-stream fallback)
defaults to True, so behind a proxy-only network the probe reached the proxy and
the gigabytes of the segments tried to go straight out.

The origin here has a name that resolves nowhere (`origin.proxy-test.invalid`):
the only way its bytes can arrive is through the proxy, which is a forward proxy
of our own that LOGS every request it carries. `NO_PROXY` is checked the other
way round: an origin named in it is reached directly and the proxy logs nothing.

Run it with the ComfyUI venv, which has aiohttp/aiofiles:

    /path/to/anymatix/python/bin/python3 tests/test_download_honours_proxy.py
"""
import hashlib
import http.client
import http.server
import os
import socketserver
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fetch as F  # noqa: E402

if not (F.AIOHTTP_AVAILABLE and F.AIOFILES_AVAILABLE):
    _why = "needs aiohttp and aiofiles (the ComfyUI venv has them): the async downloader is not what would run"
    try:
        import pytest
        pytest.skip(_why, allow_module_level=True)
    except ImportError:
        raise SystemExit(_why)

FAKE_ORIGIN = "origin.proxy-test.invalid"


def serve_origin(blob):
    size = len(blob)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            header = self.headers.get("Range")
            if header:
                start, _, end = header.replace("bytes=", "").partition("-")
                start, end = int(start), (int(end) if end else size - 1)
            else:
                start, end = 0, size - 1
            view = memoryview(blob)[start:end + 1]
            self.send_response(206 if header else 200)
            if header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(len(view)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(view)

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def serve_proxy(origin_port):
    """A forward HTTP proxy: sends everything to the one origin, logs what it carried."""
    carried = []

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_GET(self):
            carried.append((self.path, self.headers.get("Range")))
            upstream = http.client.HTTPConnection("127.0.0.1", origin_port)
            headers = {k: v for k, v in self.headers.items() if k.lower() in ("range",)}
            upstream.request("GET", "/" + self.path.split("/", 3)[3], headers=headers)
            response = upstream.getresponse()
            body = response.read()
            self.send_response(response.status)
            for name in ("Content-Range", "Accept-Ranges"):
                if response.getheader(name):
                    self.send_header(name, response.getheader(name))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            upstream.close()

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], carried


PROXY_VARIABLES = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy", "ALL_PROXY", "all_proxy")


class Environment:
    """Sets the proxy variables for the test and puts the person's own back."""

    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {name: os.environ.pop(name, None) for name in PROXY_VARIABLES}
        os.environ.update(self.values)

    def __exit__(self, *exc):
        for name in PROXY_VARIABLES:
            os.environ.pop(name, None)
        os.environ.update({k: v for k, v in self.saved.items() if v is not None})


def test_the_segments_go_through_the_proxy():
    size = 32 * 1024 * 1024
    blob = os.urandom(1 << 20) * (size >> 20)
    origin, origin_port = serve_origin(blob)
    proxy, proxy_port, carried = serve_proxy(origin_port)
    out = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        with Environment(HTTP_PROXY=f"http://127.0.0.1:{proxy_port}", http_proxy=f"http://127.0.0.1:{proxy_port}"):
            assert F.fetch_parallel(f"http://{FAKE_ORIGIN}:{origin_port}/weights.bin", out) is True
        assert os.path.getsize(out) == size
        assert hashlib.sha256(open(out, "rb").read()).hexdigest() == hashlib.sha256(blob).hexdigest()
        ranged = [r for _, r in carried if r and r != "bytes=0-0"]
        assert len(ranged) > 1, f"the segments did not go through the proxy: carried {carried}"
        print(f"  proxy carried {len(carried)} requests, {len(ranged)} of them segment ranges, sha256 matches")
    finally:
        origin.shutdown()
        proxy.shutdown()


def test_no_proxy_is_honoured():
    size = 32 * 1024 * 1024
    blob = os.urandom(1 << 20) * (size >> 20)
    origin, origin_port = serve_origin(blob)
    proxy, proxy_port, carried = serve_proxy(origin_port)
    out = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        env = dict(
            HTTP_PROXY=f"http://127.0.0.1:{proxy_port}",
            http_proxy=f"http://127.0.0.1:{proxy_port}",
            NO_PROXY="127.0.0.1",
            no_proxy="127.0.0.1",
        )
        with Environment(**env):
            assert F.fetch_parallel(f"http://127.0.0.1:{origin_port}/weights.bin", out) is True
        assert os.path.getsize(out) == size
        assert carried == [], f"an address in NO_PROXY went through the proxy: {carried}"
        print("  an origin in NO_PROXY is reached directly, the proxy carried nothing")
    finally:
        origin.shutdown()
        proxy.shutdown()


def test_without_a_proxy_nothing_changes():
    size = 32 * 1024 * 1024
    blob = os.urandom(1 << 20) * (size >> 20)
    origin, origin_port = serve_origin(blob)
    out = os.path.join(tempfile.mkdtemp(), "weights.bin.part")
    try:
        with Environment():
            assert F.fetch_parallel(f"http://127.0.0.1:{origin_port}/weights.bin", out) is True
        assert os.path.getsize(out) == size
    finally:
        origin.shutdown()


if __name__ == "__main__":
    print("the downloader and the person's proxy")
    test_the_segments_go_through_the_proxy()
    test_no_proxy_is_honoured()
    test_without_a_proxy_nothing_changes()
    print("OK")
