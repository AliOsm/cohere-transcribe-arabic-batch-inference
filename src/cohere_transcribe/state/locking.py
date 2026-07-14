"""Scalable advisory locks for independent transcript stems."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import stat
import tempfile
import threading
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ..models import AudioJob
from .io import output_parent_and_stem

LOCK_NAME = "outputs.lock"
_REGISTRY_GUARD = threading.RLock()
_LOCK_FILES: dict[Path, _RegistryLockFile] = {}
_ACTIVE_OUTPUT_LOCKS: weakref.WeakSet[OutputSetLock]


@dataclass(frozen=True, slots=True)
class OutputLockTarget:
    """One byte-range lease within the per-user output registry."""

    path: Path
    offset: int
    identity: str

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (os.fspath(self.path).casefold(), self.offset, self.identity.casefold())


def lock_target_for_outputs(output_paths: Mapping[str, Path]) -> OutputLockTarget:
    parent, stem = output_parent_and_stem(output_paths)
    identity = os.fspath((parent / stem).resolve(strict=False))
    digest = hashlib.sha256(os.fsencode(os.path.normcase(identity))).digest()
    # Keep the byte offset within the platform lock API's signed range. Offset
    # zero is reserved so a whole-file API cannot masquerade as a stem lock.
    max_start = (1 << 30) - 1 if os.name == "nt" else (1 << 63) - 2
    offset = (int.from_bytes(digest[:8], "big") % max_start) + 1
    return OutputLockTarget(_registry_path(), offset, identity)


def _registry_path() -> Path:
    if hasattr(os, "getuid"):
        root = Path("/tmp")
        scope = str(os.getuid())
    else:  # pragma: no cover - the release-tested platform is Linux
        root = Path(tempfile.gettempdir())
        home = os.path.normcase(os.path.expanduser("~"))
        scope = hashlib.sha256(os.fsencode(home)).hexdigest()[:16]
    return root / f"cohere-transcribe-{scope}" / LOCK_NAME


def _validate_lock_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, exist_ok=True)
        opened = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot prepare output lock directory {path}: {exc}"
        ) from exc
    if not stat.S_ISDIR(opened.st_mode):
        raise RuntimeError(f"Output lock directory is not a real directory: {path}")
    if hasattr(os, "getuid"):
        if opened.st_uid != os.getuid():
            raise RuntimeError(
                f"Output lock directory is not owned by the current user: {path}"
            )
        if stat.S_IMODE(opened.st_mode) & 0o077:
            raise RuntimeError(
                f"Output lock directory permissions must be private (0700): {path}"
            )


def _same_inode(path: Path, opened: os.stat_result) -> bool:
    current = os.lstat(path)
    return (
        stat.S_ISREG(current.st_mode)
        and current.st_dev == opened.st_dev
        and current.st_ino == opened.st_ino
    )


def _open_lock_file(path: Path) -> BinaryIO:
    _validate_lock_directory(path.parent)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if nofollow:
        flags |= nofollow
    elif path.exists() or path.is_symlink():
        try:
            before = os.lstat(path)
        except OSError as exc:
            raise RuntimeError(f"Cannot inspect output lock {path}: {exc}") from exc
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"Output lock is not a regular file: {path}")
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EISDIR, errno.ENXIO}:
            raise RuntimeError(f"Output lock is not a regular file: {path}") from exc
        raise RuntimeError(f"Cannot open output lock {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_inode(path, opened):
            raise RuntimeError(
                f"Output lock changed while it was being opened or is not regular: {path}"
            )
        if hasattr(os, "getuid") and opened.st_uid != os.getuid():
            raise RuntimeError(f"Output lock is not owned by the current user: {path}")
        if os.name != "nt" and stat.S_IMODE(opened.st_mode) & 0o077:
            raise RuntimeError(
                f"Output lock permissions must be private (0600): {path}"
            )
        return os.fdopen(descriptor, "r+b", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


class _RegistryLockFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = _open_lock_file(path)
        self.offsets: set[int] = set()

    def _set_lock(self, offset: int, *, acquire: bool) -> None:
        if os.name == "nt":
            import msvcrt

            if acquire and os.fstat(self.handle.fileno()).st_size <= offset:
                os.ftruncate(self.handle.fileno(), offset + 1)
            self.handle.seek(offset)
            mode = msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK
            msvcrt.locking(self.handle.fileno(), mode, 1)
        else:
            import fcntl

            mode = fcntl.LOCK_EX | fcntl.LOCK_NB if acquire else fcntl.LOCK_UN
            fcntl.lockf(self.handle.fileno(), mode, 1, offset, os.SEEK_SET)

    def acquire(self, target: OutputLockTarget) -> None:
        if target.offset in self.offsets:
            raise RuntimeError(
                f"Another transcription job owns output set {target.identity}"
            )
        locked = False
        try:
            self._set_lock(target.offset, acquire=True)
            locked = True
            if not _same_inode(self.path, os.fstat(self.handle.fileno())):
                raise RuntimeError(
                    f"Output lock changed while acquiring {target.identity}"
                )
        except BaseException as exc:
            if locked:
                with contextlib.suppress(OSError):
                    self._set_lock(target.offset, acquire=False)
            if isinstance(exc, OSError):
                raise RuntimeError(
                    f"Another transcription process owns output set {target.identity} "
                    f"(lock {self.path}, byte {target.offset})"
                ) from exc
            raise
        self.offsets.add(target.offset)

    def release(self, offset: int) -> None:
        if offset not in self.offsets:
            return
        try:
            self._set_lock(offset, acquire=False)
        finally:
            self.offsets.remove(offset)

    def close(self) -> None:
        self.handle.close()


class OutputSetLock:
    """A per-stem lease backed by one per-user registry descriptor."""

    def __init__(self, target: OutputLockTarget, owner: _RegistryLockFile) -> None:
        self.target = target
        self.path = target.path
        self._owner = owner
        self._locked = True

    @classmethod
    def acquire(cls, target: OutputLockTarget) -> OutputSetLock:
        with _REGISTRY_GUARD:
            owner = _LOCK_FILES.get(target.path)
            created = owner is None
            if owner is None:
                owner = _RegistryLockFile(target.path)
                _LOCK_FILES[target.path] = owner
            try:
                owner.acquire(target)
            except BaseException:
                if created and not owner.offsets:
                    owner.close()
                    _LOCK_FILES.pop(target.path, None)
                raise
            lock = cls(target, owner)
            _ACTIVE_OUTPUT_LOCKS.add(lock)
            return lock

    def release(self) -> None:
        with _REGISTRY_GUARD:
            if not self._locked:
                return
            try:
                self._owner.release(self.target.offset)
            finally:
                self._locked = False
                _ACTIVE_OUTPUT_LOCKS.discard(self)
                if not self._owner.offsets:
                    self._owner.close()
                    _LOCK_FILES.pop(self.target.path, None)

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.release()


_ACTIVE_OUTPUT_LOCKS = weakref.WeakSet()


def release_all_output_locks() -> None:
    for lock in list(_ACTIVE_OUTPUT_LOCKS):
        lock.release()


def release_output_locks(jobs: Sequence[AudioJob]) -> None:
    locks = {
        id(job.output_lock): job.output_lock
        for job in jobs
        if job.output_lock is not None
    }
    for lock in locks.values():
        release = getattr(lock, "release", None)
        if callable(release):
            release()
    for job in jobs:
        job.output_lock = None
