import re
import shutil
from pathlib import Path
import os

pattern = re.compile(r"^.*anymatix/results/[a-f0-9]{64}$")
hash_pattern = re.compile(r"^[a-f0-9]{64}$")


async def find_expunge(keep: list[str], dir: str):
    keep_set = set(keep)
    return filter(
        lambda x: hash_pattern.match(x) and x not in keep_set, os.listdir(dir)
    )


async def expunge(keep: list[str], dir: str):
    hashes = await find_expunge(keep, dir)
    for h in hashes:
        dir = Path(dir)
        path = dir / h
        posixpath = path.as_posix()
        with open(dir / "try_remove.txt", "a") as f:
            f.write(str(path) + "\n")
        if pattern.match(posixpath): # Extra sanity check
            shutil.rmtree(path)
        else:
            with open(dir / "error.txt", "a") as f:
                f.write(str(path) + "\n")


async def count_outputs(dir: str):
    return len(os.listdir(dir))
