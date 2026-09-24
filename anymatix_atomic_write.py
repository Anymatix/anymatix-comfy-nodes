"""
Write a result the app will read by URL without ever letting that URL point
at a half-written file.

A viewer or `ComfyRun.ts` polls a filename the moment the prompt's history
entry names it — sometimes while the writer is still mid-write. Every saver
in this pack (`anymatix_image_save.py`, `anymatix_save_json.py`,
`anymatix_save_audio.py`, `anymatix_save_animated_mp4.py`) therefore writes to
a temporary name IN THE SAME DIRECTORY as the final file and only
`os.replace()`s it into place once the bytes are flushed and fsynced —
`os.replace` is atomic on both POSIX and Windows, so a reader never observes
a partial file at the final name.

The temp name starts with `.` so it is excluded from every glob/listdir scan
this pack does over a result directory (`Anymatix_Image_Save`'s counter scan
matches `{prefix}{delimiter}(\\d+)` at the start of the basename, which a
leading dot never does) and from anything ComfyUI's own `/view` endpoint would
serve (it is asked for by exact final filename, never discovered by listing).

This module imports neither `comfy` nor `folder_paths`, so it can be tested
outside a running ComfyUI — see `tests/test_atomic_write.py`.
"""

import os
import uuid


def temp_path_for(final_path):
    """
    The hidden, per-attempt name a writer stages `final_path` under.

    The original extension is kept at the END of the name — not just
    appended after `.tmp-...` — because PIL and OpenCV both pick the encoder
    from the path's suffix; a temp name that buried `.png` in the middle
    would make `Image.save()`/`cv2.imwrite()` fail to write it at all.
    """
    directory = os.path.dirname(final_path)
    basename = os.path.basename(final_path)
    stem, ext = os.path.splitext(basename)
    unique = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
    return os.path.join(directory, f".{stem}.tmp-{unique}{ext}")


def fsync_path(path):
    """
    fsync the file's data at `path` without disturbing its mode/flags.

    Opened READ-WRITE: on Windows `os.fsync` is `_commit`, which refuses a
    read-only descriptor with EBADF — every save on Windows failed with
    "[Errno 9] Bad file descriptor" (anymatix bugs/windows-fsync-read-only-
    descriptor-every-save-fails). POSIX accepts either; the temp file is ours
    and writable, so O_RDWR costs nothing there.
    """
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(directory):
    """
    fsync the directory entry so the rename itself survives a crash.

    Best-effort: some platforms (Windows) do not support `O_DIRECTORY`, and a
    missing directory fsync only widens the crash window, it never produces a
    torn file — `os.replace` itself stays atomic either way.
    """
    try:
        dir_fd = os.open(directory or ".", os.O_DIRECTORY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def publish(temp_path, final_path):
    """fsync `temp_path`'s data, then atomically rename it onto `final_path`."""
    fsync_path(temp_path)
    os.replace(temp_path, final_path)
    fsync_dir(os.path.dirname(final_path))


def cleanup_temp(temp_path):
    """Remove a temp file a failed write left behind. Never raises."""
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    except OSError:
        pass


class atomic_output(object):
    """
    Context manager: open a temp file next to `final_path`, hand it to the
    caller, and on a clean exit flush + fsync + `os.replace()` it into place.
    On an exception, the temp file is removed and the exception propagates —
    the final path is never created and nothing half-written is left behind.

        with atomic_output(final_path, "wb") as (f, temp_path):
            f.write(data)
        # final_path now exists, complete, or the exception propagated and
        # neither final_path nor temp_path exists.
    """

    def __init__(self, final_path, mode="wb"):
        self.final_path = final_path
        self.mode = mode
        self.temp_path = temp_path_for(final_path)
        self._file = None

    def __enter__(self):
        self._file = open(self.temp_path, self.mode)
        return self._file, self.temp_path

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
                self._file = None
                publish(self.temp_path, self.final_path)
            else:
                try:
                    self._file.close()
                except Exception:
                    pass
                cleanup_temp(self.temp_path)
        except Exception:
            cleanup_temp(self.temp_path)
            raise
        return False
