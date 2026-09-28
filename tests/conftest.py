"""
Every save the suite makes also signs a C2PA manifest (`anymatix_c2pa.py`).
Two things that must never happen while it does, whichever test file runs:

  * a test writing a signing credential into the REAL per-user data
    directory -- the suite gets a throwaway one, removed at the end;
  * a test depending on the network -- the timestamp authority is off unless
    a test turns it on for itself.

Environment variables, not monkeypatching: the save modules are loaded more
than once under different names (standalone and as a package), and an
override in the environment reaches every copy.
"""

import os
import shutil
import tempfile

_CREDENTIAL_DIR = tempfile.mkdtemp(prefix="anymatix-c2pa-test-")
os.environ["ANYMATIX_C2PA_DIR"] = _CREDENTIAL_DIR
os.environ["ANYMATIX_C2PA_TSA"] = ""


def pytest_unconfigure(config):
    shutil.rmtree(_CREDENTIAL_DIR, ignore_errors=True)
