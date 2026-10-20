"""Where the node pack keeps files of its own that are not models and not outputs.

ONE PATH, DECIDED BY THE PARTY THAT KNOWS IT.

The lifecycle journal used to be put at `<three levels above this file>/.anymatix`.
Three levels above `custom_nodes/anymatix-comfy-nodes/` is the ComfyUI git
checkout, so on every machine that is not a RunPod pod the journal landed inside
a repository the bootstrap resets: ComfyUI never read as unchanged and every
boot force-checked it out (bugs/the-node-pack-writes-lifecycle-journal-inside).

The bootstrap launches ComfyUI and knows the install root, so IT decides the
directory (`anymatix_state_dir()` in bootstrap.py) and exports it as
`ANYMATIX_STATE_DIR`, the way it exports the heartbeat port. The node pack only
reads it. There is deliberately no fallback derived from this file's location:
a path guessed from inside a checkout is how the bug happened, and a ComfyUI
started by hand loses nothing but the file copy of a line that is also printed.

Imports nothing from ComfyUI, so the tests load it directly.
"""
import os

STATE_DIR_ENV = "ANYMATIX_STATE_DIR"
LIFECYCLE_JOURNAL = "lifecycle.jsonl"


def state_dir(environ=None) -> str:
    """The directory the bootstrap named, or "" when it named none."""
    env = os.environ if environ is None else environ
    return (env.get(STATE_DIR_ENV) or "").strip()


def lifecycle_journal_path(environ=None) -> str:
    """The journal file inside the state directory (created), or "" when there is no usable one."""
    d = state_dir(environ)
    if not d:
        return ""
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        return ""
    return os.path.join(d, LIFECYCLE_JOURNAL)
