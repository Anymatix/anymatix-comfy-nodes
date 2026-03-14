import re
import shutil
import os
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


async def count_outputs(dir: str):
    if not os.path.isdir(dir):
        return 0
    return len([name for name in os.listdir(dir) if hash_pattern.match(name)])
