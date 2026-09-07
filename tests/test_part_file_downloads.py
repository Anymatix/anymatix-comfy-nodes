#!/usr/bin/env python3
"""
A download in progress must not wear the name of a finished one.

A beta tester found `ltx-2-19b-dev-fp8.safetensors` on disk at 141 MB of a
declared 27 GB — 0.5% — sitting there looking like a model, because the
downloader wrote straight to the final filename and used the bytes already
there as its resume offset. The downloader itself would have resumed next time;
anything else in the system, asked to load that file in the meantime, fails on
something that has nothing to do with the real cause.

These tests pin the rule: the `.part` earns the name, it is not given it.

    python3 test_part_file_downloads.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import part_path_for, finalize_download

failures = []


def check(label, condition):
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        failures.append(label)


def main():
    with tempfile.TemporaryDirectory() as d:
        final = os.path.join(d, "model.safetensors")
        part = part_path_for(final)

        check("the part file sits beside the final name", part == final + ".part")

        # 1. complete → renamed
        with open(part, "wb") as f:
            f.write(b"x" * 100)
        finalize_download(part, final, 100, "model.safetensors")
        check("a complete download takes the final name", os.path.exists(final))
        check("and its .part is gone", not os.path.exists(part))
        check("with every byte", os.path.getsize(final) == 100)

        # 2. short → refused, and the bytes are KEPT to resume from
        os.remove(final)
        with open(part, "wb") as f:
            f.write(b"x" * 40)
        raised = ""
        try:
            finalize_download(part, final, 100, "model.safetensors")
        except Exception as e:
            raised = str(e)
        check("a short download is refused", "Incomplete download" in raised)
        check("it never takes the final name", not os.path.exists(final))
        check("and it KEEPS its bytes to resume from", os.path.getsize(part) == 40)
        check("the message says how short", "40 of 100" in raised)

        # 3. no declared size (a sidecar json) → still atomic, no size check
        os.remove(part)
        final2 = os.path.join(d, "meta.json")
        part2 = part_path_for(final2)
        with open(part2, "wb") as f:
            f.write(b"{}")
        finalize_download(part2, final2, None, "meta.json")
        check("a file with no declared size is still renamed", os.path.exists(final2))

        # 4. nothing downloaded → said plainly
        raised = ""
        try:
            finalize_download(os.path.join(d, "absent.part"), os.path.join(d, "absent"), 10, "absent")
        except Exception as e:
            raised = str(e)
        check("no data at all is an error that says so", "produced no data" in raised)

    # ------------------------------------------------------------------
    # ONE PLACE MAKES A PART PATH, AND THE PARALLEL PATH IS NOT IT.
    #
    # Every check above exercises `part_path_for` and `finalize_download`, and
    # all ten passed for months while the PARALLEL downloader wrote to
    # `<name>.safetensors.part.part`: `download_file` built the part path with
    # `part_path_for` and handed it to `fetch_parallel`, whose downloader
    # appended `.part` a second time. A completed transfer then failed to
    # finalise ("Download produced no data") and no later run could resume,
    # because it looked for the single-suffix name that never existed.
    # Measured 2026-09-05 on a remote: a complete 323 MB SAM2 weight wearing
    # two suffixes. bugs/the-parallel-download-writes-part-part-can
    #
    # A behavioural test cannot reach that code without a server and a real
    # transfer, so the module is read instead — the property is "nobody else
    # derives a part path", which is a statement about the whole file.
    print("\none place makes a part path")
    source = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch.py"),
        encoding="utf-8",
    ).read()
    check(
        "the parallel downloader writes where it was told, and appends nothing",
        'f"{self.file_path}.part"' not in source,
    )
    check(
        "only part_path_for builds a part path",
        source.count('+ ".part"') == 1,
    )

    if failures:
        print(f"\n{len(failures)} FAILED")
        return 1
    print("\nall good")
    return 0


if __name__ == "__main__":
    sys.exit(main())
