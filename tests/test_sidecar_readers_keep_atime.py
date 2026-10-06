"""Every sidecar reader that is not a USE goes through fetch.read_sidecar.

On a RunPod network volume (MooseFS) a plain read persists atime=now whenever
the sidecar's atime is not newer than its ctime -- and every atime restore
(bootstrap's auto-clean pass, read_sidecar itself) moves ctime. So ONE plain
reader that lists every sidecar marks every weight as used today, and the
volume auto-clean never evicts. Measured 2026-10-06 on pod pgzbm4s8dd4tn6:
`/anymatix/resources` (serve_resources) was that reader.

The only plain json.load left in the ComfyUI-facing modules is the fetcher's
IS_CHANGED, which reads the sidecar of the url the card asked for: a use.
"""
import ast, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["__init__.py", "anymatix_checkpoint_fetcher.py"]
USES = {"AnymatixFetcher.IS_CHANGED"}


def _json_load_sites(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []

    def walk(node, stack):
        for c in ast.iter_child_nodes(node):
            st = stack + [c.name] if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else stack
            if (isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "load" and getattr(c.func.value, "id", None) == "json"):
                out.append((c.lineno, ".".join(stack)))
            walk(c, st)

    walk(tree, [])
    return out


def test_only_uses_read_sidecars_plainly():
    bad = []
    for name in FILES:
        for line, where in _json_load_sites(os.path.join(ROOT, name)):
            if where not in USES:
                bad.append(f"{name}:{line} in {where or '<module>'}")
    assert not bad, "plain json.load outside a use -- read through fetch.read_sidecar: " + ", ".join(bad)


def test_resources_listing_reads_through_read_sidecar():
    src = open(os.path.join(ROOT, "__init__.py"), encoding="utf-8").read()
    body = src[src.index("async def serve_resources"):]
    body = body[:body.index("return web.json_response(result)")]
    assert "read_sidecar(path)" in body


if __name__ == "__main__":
    test_only_uses_read_sidecars_plainly()
    test_resources_listing_reads_through_read_sidecar()
    print("2 passed")
