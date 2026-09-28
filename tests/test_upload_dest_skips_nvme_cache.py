#!/usr/bin/env python3
"""
An uploaded user model must land on the durable volume, not the NVMe cache.

bugs/an-uploaded-user-model-may-land-nvme: `/anymatix/uploadAsset` picked
`folder_paths.get_folder_paths(folder_paths_key)[0]` as the write destination.
`extra_model_paths.yaml` puts the NVMe cache first in that list whenever one is
configured (see `generate_extra_model_paths_config` in bootstrap.py), so a
model a user imported themselves — no URL to re-fetch it from — landed in the
cache. A pod STOP wipes the cache; only the volume survives.

`pick_durable_dir` (fetch.py) is the fix: it picks the first candidate that is
NOT under the NVMe cache, exactly like `get_anymatix_models_dir` already does
for downloads. These tests pin `is_under_nvme_cache` / `pick_durable_dir`
directly (no ComfyUI boot needed — they have no ComfyUI dependency) and check
that `__init__.py`'s uploader calls through them instead of indexing [0].

    python3 tests/test_upload_dest_skips_nvme_cache.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fetch import is_under_nvme_cache, pick_durable_dir

failures = []


def check(label, condition):
    print(f"  {'ok' if condition else 'FAIL'}   {label}")
    if not condition:
        failures.append(label)


def main() -> int:
    cache = "/root/.anymatix/model-cache"
    volume = "/workspace/anymatix/models"

    print("is_under_nvme_cache")
    check("the cache root itself counts", is_under_nvme_cache(cache, cache_root=cache))
    check(
        "a subdirectory of the cache counts",
        is_under_nvme_cache(cache + "/loras", cache_root=cache),
    )
    check(
        "the volume does not count",
        not is_under_nvme_cache(volume + "/loras", cache_root=cache),
    )
    check(
        "a path that merely shares the cache's prefix as a string does not count",
        not is_under_nvme_cache("/root/.anymatix/model-cache-other/loras", cache_root=cache),
    )
    check(
        "no cache configured (empty root) never matches",
        not is_under_nvme_cache(cache + "/loras", cache_root=""),
    )

    print("\npick_durable_dir")
    # This is the exact shape folder_paths.get_folder_paths returns on a pod:
    # the cache first (ComfyUI moves every is_default path to the front, and
    # the cache block is written after the volume block on purpose), then the
    # volume.
    dirs = [cache + "/loras", volume + "/loras"]
    check(
        "skips the cache and picks the volume when the cache is listed first",
        pick_durable_dir(dirs, cache_root=cache) == volume + "/loras",
    )
    check(
        "the naive bug ([0]) would have picked the cache — confirms the fixture models the regression",
        dirs[0] == cache + "/loras",
    )
    check(
        "no cache configured: falls back to the first entry, same as before",
        pick_durable_dir(dirs, cache_root="") == dirs[0],
    )
    check(
        "a single-entry list with no cache active returns that entry",
        pick_durable_dir([volume + "/loras"], cache_root=cache) == volume + "/loras",
    )
    check("an empty list returns None", pick_durable_dir([], cache_root=cache) is None)
    check(
        "every candidate under the cache: falls back to the first rather than raising",
        pick_durable_dir([cache + "/a", cache + "/b"], cache_root=cache) == cache + "/a",
    )

    # ------------------------------------------------------------------
    # THE CALL SITE ITSELF: `/anymatix/uploadAsset` must resolve its
    # destination through `pick_durable_dir`, not `dest_dirs[0]`. A test that
    # only exercises the helper cannot catch a future edit that reintroduces
    # the naive index at the call site, so the source is checked directly —
    # same technique test_part_file_downloads.py uses for "one place builds a
    # part path".
    print("\nthe upload route itself")
    init_source = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__init__.py"),
        encoding="utf-8",
    ).read()
    check(
        "uploadAsset no longer indexes the folder_paths list at [0]",
        "dest_dirs[0]" not in init_source,
    )
    check(
        "uploadAsset resolves its destination through pick_durable_dir",
        "dest_dir = pick_durable_dir(dest_dirs)" in init_source,
    )

    if failures:
        print(f"\n{len(failures)} FAILED")
        return 1
    print("\nall good")
    return 0


if __name__ == "__main__":
    sys.exit(main())
