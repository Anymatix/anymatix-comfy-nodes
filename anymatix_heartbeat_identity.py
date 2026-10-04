"""
WHOSE HEARTBEAT ANSWERED?

The app reads "the side channel answers and ComfyUI's own port does not" as
"this machine is alive and busy loading", and waits up to fifteen minutes for
it (`app/src/machines/ComfyUI/ComfyUI-local.ts`, the quick-reconnect loop).
On 2026-10-03 the answer on that port came from ANOTHER Anymatix's ComfyUI --
a second app on the same Mac had started its own ComfyUI there -- and the app
waited on a dead remote as if it were busy
(`bugs/a-second-app-s-local-machine-takes` in the app repository).

So every heartbeat answer now says which ComfyUI it speaks for: the port that
ComfyUI was launched on (`--port`). The app compares it with the port it
launched, or tunnels to, and trusts nothing else.

And it works the other way too. A heartbeat may name the port it is meant for;
one meant for a different ComfyUI is answered but never touches this one's
watchdog, so a stranger's probe cannot arm it, nor a stranger's one-second
"shut down" end it. A heartbeat that names nothing is obeyed as before.

Its own module, importing nothing from ComfyUI, so the rule is tested without
one (`tests/test_heartbeat_identity.py`).
"""

import http.server
import json


def heartbeat_reply(comfy_port, **fields):
    """The body of every heartbeat answer: what it reports, plus whose it is."""
    reply = {"status": "ok"}
    reply.update(fields)
    reply["comfy_port"] = comfy_port
    return reply


def meant_for_another(request_body, comfy_port):
    """True when the heartbeat names a ComfyUI port and it is not this one."""
    if not isinstance(request_body, dict):
        return False
    claimed = request_body.get("comfy_port")
    if claimed is None:
        return False
    try:
        return int(claimed) != int(comfy_port)
    except (TypeError, ValueError):
        return True


def side_channel_handler(touch, comfy_port):
    """The request handler of the side channel, which answers from its own thread.

    `touch(timeout)` refreshes the watchdog; it is called only for a heartbeat
    meant for this ComfyUI.
    """

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw or b"{}")
            except Exception:
                body = {}
            if meant_for_another(body, comfy_port):
                reply = heartbeat_reply(comfy_port, armed=False)
            else:
                try:
                    timeout = int(body.get("timeout", 60)) if isinstance(body, dict) else 60
                except (TypeError, ValueError):
                    timeout = 60
                touch(timeout)
                reply = heartbeat_reply(comfy_port, seconds_until_death=timeout)
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            self.do_POST()

        def log_message(self, *_args):
            pass  # one line per heartbeat would bury the log it shares

    return Handler
