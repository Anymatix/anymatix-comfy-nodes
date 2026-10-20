"""The lifecycle journal lives where the bootstrap says, never inside the ComfyUI checkout.

bugs/the-node-pack-writes-lifecycle-journal-inside: the old path was derived from
this package's location, which is the ComfyUI git repository, so every boot
force-checked ComfyUI out. The module imports nothing from ComfyUI and is loaded
directly, like test_heartbeat_identity.py.
"""

import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "anymatix_state_dir", os.path.join(_HERE, "..", "anymatix_state_dir.py")
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_the_journal_is_inside_the_directory_the_bootstrap_named(tmp_path):
    target = tmp_path / "state"
    path = _MODULE.lifecycle_journal_path({"ANYMATIX_STATE_DIR": str(target)})
    assert path == str(target / "lifecycle.jsonl")
    assert target.is_dir()


def test_without_a_named_directory_there_is_no_file_journal():
    assert _MODULE.lifecycle_journal_path({}) == ""
    assert _MODULE.lifecycle_journal_path({"ANYMATIX_STATE_DIR": "   "}) == ""


def test_an_uncreatable_directory_gives_no_journal_instead_of_raising(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert _MODULE.lifecycle_journal_path({"ANYMATIX_STATE_DIR": str(blocker / "sub")}) == ""


def test_the_package_location_is_never_the_base():
    # The cause of the bug, as a tripwire: a __file__-relative base must not return.
    with open(os.path.join(_HERE, "..", "__init__.py"), encoding="utf-8") as fh:
        src = fh.read()
    start = src.index("def _lifecycle_journal_path")
    body = src[start:src.index("def _journal(")]
    code = body.split('"""')[-1]
    assert "__file__" not in code
    with open(os.path.join(_HERE, "..", "anymatix_state_dir.py"), encoding="utf-8") as fh:
        module_code = fh.read().split('"""')[2]
    assert "__file__" not in module_code
