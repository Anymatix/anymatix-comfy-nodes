def delete_file_and_cleanup_dir(file_path: Path, results_dir: str):
    """
    Delete a single file and, if its parent output directory is empty, delete the directory.
    TODO: If output folders can have more than one level of hierarchy, make this logic more robust.
    """
    try:
        file_path.unlink()
        with open(Path(results_dir) / "expunge_log.txt", "a") as f:
            f.write(f"Deleted file: {file_path}\n")
    except Exception as e:
        with open(Path(results_dir) / "error.txt", "a") as f:
            f.write(f"Failed to remove file: {file_path} - {e}\n")
    parent_dir = file_path.parent
    # TODO: If output folders can have more than one level of hierarchy, make this logic more robust.
    remaining_files = [f for f in parent_dir.iterdir()]
    if not remaining_files:
        try:
            parent_dir.rmdir()
            with open(Path(results_dir) / "expunge_log.txt", "a") as f:
                f.write(f"Deleted empty output directory: {parent_dir}\n")
        except Exception as e:
            with open(Path(results_dir) / "error.txt", "a") as f:
                f.write(f"Failed to remove output directory: {parent_dir} - {e}\n")
import re
import shutil
import os
from pathlib import Path

pattern = re.compile(r"^.*anymatix/results/[a-f0-9]{64}$")
hash_pattern = re.compile(r"^[a-f0-9]{64}$")
input_asset_pattern = re.compile(
    r"^[a-f0-9]{64}\.[a-zA-Z0-9]+$"
)  # hash.extension format


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

    # Log what we're keeping for debugging
    with open(Path(results_dir) / "expunge_log.txt", "a") as f:
        f.write(f"Keeping {len(input_assets)} input assets\n")
        f.write(f"Keeping {len(computation_results)} computation results\n")
        f.write(f"Input assets: {input_assets}\n")
        f.write(f"Computation results: {computation_results}\n")

    # Remove computation result directories
    for h in computation_hashes:
        results_path = Path(results_dir) / h
        posixpath = results_path.as_posix()
        with open(Path(results_dir) / "try_remove.txt", "a") as f:
            f.write(f"Computation result: {results_path}\n")
        if pattern.match(posixpath):  # Extra sanity check
            if results_path.exists() and results_path.is_dir():
                # Remove files inside the output directory
                for file in results_path.iterdir():
                    try:
                        file.unlink()
                    except Exception as e:
                        with open(Path(results_dir) / "error.txt", "a") as f:
                            f.write(f"Failed to remove file: {file} in {results_path} - {e}\n")
                # Check if directory is empty after file deletion
                if not any(results_path.iterdir()):
                    try:
                        results_path.rmdir()
                        with open(Path(results_dir) / "expunge_log.txt", "a") as f:
                            f.write(f"Deleted empty output directory: {results_path}\n")
                    except Exception as e:
                        with open(Path(results_dir) / "error.txt", "a") as f:
                            f.write(f"Failed to remove output directory: {results_path} - {e}\n")
        else:
            with open(Path(results_dir) / "error.txt", "a") as f:
                f.write(f"Failed to remove computation result: {results_path}\n")

    # Remove input asset files
    for filename in input_files:
        input_path = Path(input_dir) / filename
        with open(Path(results_dir) / "try_remove.txt", "a") as f:
            f.write(f"Input asset: {input_path}\n")
        try:
            os.remove(input_path)
        except Exception as e:
            with open(Path(results_dir) / "error.txt", "a") as f:
                f.write(f"Failed to remove input asset: {input_path} - {e}\n")


async def count_outputs(dir: str):
    return len(os.listdir(dir))
