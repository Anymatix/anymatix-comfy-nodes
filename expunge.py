import re
import shutil
import os
import time
from pathlib import Path

pattern = re.compile(r"^.*anymatix/results/[a-f0-9]{64}$")
hash_pattern = re.compile(r"^[a-f0-9]{64}$")
input_asset_pattern = re.compile(
    r"^[a-f0-9]{64}\.[a-zA-Z0-9]+$"
)  # hash.extension format


def delete_results_entry(path: Path):
    if not path.exists():
        return

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=False)
        return

    path.unlink()


async def find_expunge_computation_results(
    computation_results: list[str], results_dir: str
):
    """
    Find computation result directories to expunge
    """
    keep_set = set(computation_results)
    return filter(
        lambda x: hash_pattern.match(x) and x not in keep_set, os.listdir(results_dir)
    )


async def find_expunge_input_assets(input_assets: list[str], input_dir: str):
    """
    Find input asset files to expunge
    """
    keep_set = set(input_assets)
    return filter(
        lambda x: input_asset_pattern.match(x) and x.split(".")[0] not in keep_set,
        os.listdir(input_dir),
    )


async def expunge_differentiated(
    input_assets: list[str],
    computation_results: list[str],
    results_dir: str,
    input_dir: str,
):
    """
    Enhanced expunge that handles both input assets and computation results
    """
    # Find items to expunge in both directories
    computation_hashes = await find_expunge_computation_results(
        computation_results, results_dir
    )
    input_files = await find_expunge_input_assets(input_assets, input_dir)

    for h in computation_hashes:
        results_path = Path(results_dir) / h
        posixpath = results_path.as_posix()
        if pattern.match(posixpath) and results_path.exists():
            delete_results_entry(results_path)

    for filename in input_files:
        input_path = Path(input_dir) / filename
        if input_path.exists():
            os.remove(input_path)


async def clear_anymatix_cache(results_dir: str, input_dir: str):
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    for child in list(results_path.iterdir()):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=False)
        else:
            child.unlink()

    input_path = Path(input_dir)
    if not input_path.exists():
        return

    for child in list(input_path.iterdir()):
        if input_asset_pattern.match(child.name) or child.name.endswith(".tmp"):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=False)
            else:
                child.unlink()


def last_touch(path: Path) -> float:
    """Most recent mtime of an entry, looking one level inside a directory.

    A result is a directory of files written one by one, and reads renew the
    directory's own mtime (see touch_result), so the newest of the two is the
    moment the result was last written *or* served.
    """
    try:
        newest = path.stat().st_mtime
    except OSError:
        return 0.0

    if path.is_dir() and not path.is_symlink():
        try:
            for child in path.iterdir():
                try:
                    newest = max(newest, child.stat().st_mtime)
                except OSError:
                    continue
        except OSError:
            pass

    return newest


def touch_result(results_dir: str, name: str):
    """Renew a result's TTL — serving it counts as using it, so the sweep is LRU.

    Called on every GET of a file under anymatix/results, which is what keeps a
    result that is still being downloaded (file by file, with resume) from being
    swept out from under the client.
    """
    if not hash_pattern.match(name):
        return
    now = time.time()
    try:
        os.utime(Path(results_dir) / name, (now, now))
    except OSError:
        pass


def sweep_expired(
    results_dir: str,
    input_dir: str,
    results_ttl: int,
    inputs_ttl: int,
    min_age: int,
    protected: set[str],
):
    """Delete cache entries untouched for longer than their TTL.

    The remote cache is a cache: every result is recomputable from its inputs
    (the hash *is* the computation) and every input asset is re-uploadable by
    the app, which checks hashedModelAssetExists before sending. So expiry is
    the primary reclaim path and it runs here, with no client involved.

    Never touched: entries younger than min_age (a run may still be writing
    them) and hashes referenced by the queue (protected).
    """
    now = time.time()
    deleted_results: list[str] = []
    deleted_inputs: list[str] = []

    def expired(path: Path, ttl: int) -> bool:
        age = now - last_touch(path)
        return age > ttl and age > min_age

    if results_ttl > 0:
        try:
            names = os.listdir(results_dir)
        except OSError:
            names = []
        for name in names:
            if not hash_pattern.match(name) or name in protected:
                continue
            path = Path(results_dir) / name
            if not expired(path, results_ttl):
                continue
            if not pattern.match(path.as_posix()):
                continue
            try:
                delete_results_entry(path)
                deleted_results.append(name)
            except OSError as e:
                print(f"anymatix: cache gc failed on result {name}: {e}")

    if inputs_ttl > 0:
        try:
            names = os.listdir(input_dir)
        except OSError:
            names = []
        for name in names:
            if not input_asset_pattern.match(name):
                continue
            if name.split(".")[0] in protected:
                continue
            path = Path(input_dir) / name
            if not expired(path, inputs_ttl):
                continue
            try:
                os.remove(path)
                deleted_inputs.append(name)
            except OSError as e:
                print(f"anymatix: cache gc failed on input {name}: {e}")

    return {"results": deleted_results, "inputs": deleted_inputs}


def delete_result_hashes(results_dir: str, hashes: list[str], protected: set[str]):
    """Delete whole result directories by hash — the app's active release path.

    Whole directories only: a result is a manifest plus its payload files, and
    the app releases it once, after the complete local copy exists.
    """
    deleted: list[str] = []
    skipped: list[str] = []

    for h in hashes:
        if not isinstance(h, str) or not hash_pattern.match(h):
            continue
        if h in protected:
            skipped.append(h)
            continue
        path = Path(results_dir) / h
        if not pattern.match(path.as_posix()) or not path.exists():
            continue
        try:
            delete_results_entry(path)
            deleted.append(h)
        except OSError as e:
            print(f"anymatix: failed to delete result {h}: {e}")

    return {"deleted": deleted, "skipped": skipped}


async def count_outputs(dir: str):
    if not os.path.isdir(dir):
        return 0
    return len([name for name in os.listdir(dir) if hash_pattern.match(name)])
