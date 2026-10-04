"""
Every heartbeat answer says whose it is, and a heartbeat meant for another
ComfyUI is answered but never obeyed.

The app trusts a side channel's answer only when it names the port the app
launched, or tunnels to (`bugs/a-second-app-s-local-machine-takes` in the app
repository: another app's ComfyUI answered on that port and a dead remote was
waited on as busy for a quarter of an hour).

The module under test imports nothing from ComfyUI, so it is loaded directly,
the same way `test_atomic_write.py` loads its module.
"""

import http.server
import importlib.util
import json
import os
import threading
import urllib.request

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "anymatix_heartbeat_identity", os.path.join(_HERE, "..", "anymatix_heartbeat_identity.py")
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

heartbeat_reply = _MODULE.heartbeat_reply
meant_for_another = _MODULE.meant_for_another
side_channel_handler = _MODULE.side_channel_handler


# ------------------------------------------------------------ the reply ----


def test_a_reply_names_the_port_this_comfyui_was_launched_on():
    assert heartbeat_reply(10101, seconds_until_death=42) == {
        "status": "ok",
        "seconds_until_death": 42,
        "comfy_port": 10101,
    }


# ------------------------------------------------ who a heartbeat is for ----


def test_a_heartbeat_that_names_nothing_is_obeyed_as_before():
    assert meant_for_another({"timeout": 60}, 10101) is False
    assert meant_for_another({}, 10101) is False
    assert meant_for_another(None, 10101) is False


def test_a_heartbeat_for_this_port_is_obeyed():
    assert meant_for_another({"timeout": 60, "comfy_port": 10101}, 10101) is False
    assert meant_for_another({"comfy_port": "10101"}, 10101) is False


def test_a_heartbeat_for_another_port_is_not():
    assert meant_for_another({"timeout": 1, "comfy_port": 10102}, 10101) is True
    assert meant_for_another({"comfy_port": "not a port"}, 10101) is True


# ------------------------------------------------- the side channel, live ----


@pytest.fixture
def side_channel():
    touched = []
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), side_channel_handler(touched.append, 10101)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], touched
    finally:
        server.shutdown()
        server.server_close()


def _post(port, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/anymatix/heartbeat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def test_the_side_channel_answers_with_whose_it_is_and_refreshes_its_watchdog(side_channel):
    port, touched = side_channel
    status, reply = _post(port, {"timeout": 42})
    assert status == 200
    assert reply == {"status": "ok", "seconds_until_death": 42, "comfy_port": 10101}
    assert touched == [42]


def test_a_heartbeat_meant_for_another_comfyui_never_touches_the_watchdog(side_channel):
    port, touched = side_channel
    status, reply = _post(port, {"timeout": 1, "comfy_port": 10102})
    assert status == 200
    assert reply == {"status": "ok", "armed": False, "comfy_port": 10101}
    assert touched == []


def test_a_heartbeat_naming_this_port_is_obeyed(side_channel):
    port, touched = side_channel
    _post(port, {"timeout": 30, "comfy_port": 10101})
    assert touched == [30]


def test_a_get_still_counts_as_a_heartbeat(side_channel):
    port, touched = side_channel
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
        reply = json.loads(response.read())
    assert reply["comfy_port"] == 10101
    assert touched == [60]


# ------------------------------------------- both routes use the rule ----


def test_both_heartbeat_routes_in_the_pack_carry_the_identity():
    with open(os.path.join(_HERE, "..", "__init__.py"), encoding="utf-8") as f:
        source = f.read()
    assert "side_channel_handler(_hb_touch, comfy_args.port)" in source
    route = source[source.index("async def serve_heartbeat"):]
    route = route[: route.index("\n@routes.")] if "\n@routes." in route else route
    assert "meant_for_another(data, comfy_args.port)" in route
    assert '{"status": "ok"' not in route
    assert route.count("heartbeat_reply(comfy_args.port") == 3
