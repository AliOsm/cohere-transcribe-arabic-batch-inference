"""Input discovery and transcription job construction."""

from __future__ import annotations

import errno
import json
import math
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .cancellation import (
    cancellable_executor,
    raise_if_cancelled,
    registered_process,
    terminate_process,
)
from .models import (
    AUDIO_EXTENSIONS,
    AudioJob,
    SourceSnapshot,
    TranscriptionConfig,
    info,
)
from .state import (
    OutputLockTarget,
    OutputSetLock,
    asr_contract_key,
    checkpoint_path_for_outputs,
    lock_target_for_outputs,
    release_output_locks,
    render_contract_key,
    restore_asr_checkpoint,
    state_path_for_outputs,
    verify_published_outputs,
)


@dataclass(frozen=True, slots=True)
class _JobPlan:
    path: Path
    relative_path: Path
    output_paths: dict[str, Path]
    state_path: Path
    checkpoint_path: Path
    lock_target: OutputLockTarget
    snapshot: SourceSnapshot


def _resolve_path(path: Path, *, strict: bool) -> Path:
    """Resolve a path while detecting loops even when its tail may be missing."""
    if strict:
        return path.resolve(strict=True)
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        return path.resolve(strict=False)


def _resolution_error(exc: BaseException) -> str:
    if isinstance(exc, RuntimeError) or (
        isinstance(exc, OSError) and exc.errno == errno.ELOOP
    ):
        return "Symlink loop"
    return str(exc)


def _resolve_planner_path(
    raw: str | os.PathLike[str], *, label: str, strict: bool
) -> Path:
    """Expand and canonicalize a user path with a concise planner error."""
    try:
        raw_text = os.fspath(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid {label} path: {exc}") from exc
    try:
        expanded = Path(raw_text).expanduser()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid {label} path {raw_text!r}: {exc}") from exc
    try:
        return _resolve_path(expanded, strict=strict)
    except FileNotFoundError as exc:
        if label == "input":
            raise SystemExit(f"Input does not exist: {expanded}") from exc
        raise SystemExit(f"Invalid {label} path {raw_text!r}: {exc}") from exc
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid {label} path {raw_text!r}: {_resolution_error(exc)}"
        ) from exc


def _resolve_output_parent(parent: Path, output_root: Path | None) -> Path:
    """Resolve one output parent without allowing an output-root symlink escape."""
    try:
        resolved = _resolve_path(parent, strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"Cannot resolve output directory {parent}: {_resolution_error(exc)}"
        ) from exc
    if output_root is not None and not resolved.is_relative_to(output_root):
        raise SystemExit(
            f"Output directory escapes output root: {parent} -> {resolved}"
        )
    try:
        resolved.mkdir(parents=True, exist_ok=True)
        resolved = _resolve_path(resolved, strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"Cannot create output directory {parent}: {_resolution_error(exc)}"
        ) from exc
    if output_root is not None and not resolved.is_relative_to(output_root):
        raise SystemExit(
            f"Output directory escapes output root: {parent} -> {resolved}"
        )
    return resolved


def probe_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=start_time,duration:format=start_time,duration",
        "-of",
        "json",
        os.fspath(path),
    ]
    try:
        raise_if_cancelled()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            with registered_process(process):
                try:
                    stdout, stderr = process.communicate(timeout=30)
                except BaseException:
                    terminate_process(process)
                    raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
        raise_if_cancelled()
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode,
                command,
                output=stdout,
                stderr=stderr,
            )
        payload = json.loads(stdout)

        def seconds(value: object) -> float | None:
            if value in (None, "N/A"):
                return None
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            return result if math.isfinite(result) and result >= 0 else None

        streams = payload.get("streams", [])
        if isinstance(streams, list):
            for stream in streams:
                if not isinstance(stream, dict):
                    continue
                # FFprobe stream.duration is a duration, independent of start_time.
                if (duration := seconds(stream.get("duration"))) is not None:
                    return duration

        format_metadata = payload.get("format", {})
        if isinstance(format_metadata, dict):
            duration = seconds(format_metadata.get("duration"))
            start = seconds(format_metadata.get("start_time"))
            if duration is not None:
                # Some containers report the absolute end timestamp as their
                # duration when the stream begins after zero. Stream duration is
                # preferred above; normalize this container-only fallback.
                if start is not None and start > 0 and duration > start:
                    return duration - start
                return duration
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return None


def expand_inputs(inputs: Sequence[str], recursive: bool) -> list[tuple[Path, Path]]:
    expanded: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for raw in inputs:
        source = _resolve_planner_path(raw, label="input", strict=True)

        try:
            if source.is_file():
                candidates = [(source, Path(source.name))]
            elif source.is_dir():
                iterator = source.rglob("*") if recursive else source.iterdir()
                paths = sorted(
                    (
                        path
                        for path in iterator
                        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
                    ),
                    key=lambda path: os.fspath(path).casefold(),
                )
                candidates = [
                    (_resolve_path(path, strict=True), path.relative_to(source))
                    for path in paths
                ]
            else:
                raise SystemExit(f"Input is not a regular file or directory: {source}")
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"Cannot inspect input {source}: {exc}") from exc

        for path, relative_path in candidates:
            try:
                canonical = _resolve_path(path, strict=True)
            except (OSError, RuntimeError, ValueError) as exc:
                raise SystemExit(
                    f"Cannot access input {path}: {_resolution_error(exc)}"
                ) from exc
            if canonical in seen:
                continue
            seen.add(canonical)
            expanded.append((canonical, relative_path))

    if not expanded:
        raise SystemExit("No audio files found in the supplied inputs.")
    return expanded


def segmentation_parameters(args: TranscriptionConfig) -> dict[str, int | float]:
    """Return the behavior-affecting segmentation settings for provenance."""
    parameters: dict[str, int | float] = {
        "max_duration_seconds": args.max_dur,
    }
    if args.vad == "silero":
        parameters.update(
            {
                "min_duration_seconds": args.min_dur,
                "threshold": args.vad_threshold,
                "min_silence_ms": args.min_silence_ms,
                "speech_pad_ms": args.speech_pad_ms,
            }
        )
    elif args.vad == "auditok":
        parameters.update(
            {
                "min_duration_seconds": args.min_dur,
                "max_silence_seconds": args.max_silence,
                "energy_threshold": args.energy_threshold,
            }
        )
    return parameters


def _probe_duration_hints(jobs: Sequence[AudioJob]) -> None:
    """Populate inexpensive duration hints without changing job order."""
    probe_jobs = [job for job in jobs if not job.asr_checkpoint_loaded]
    if not probe_jobs:
        return
    probe_workers = min(len(probe_jobs), 8, max(1, (os.cpu_count() or 2) // 2))
    if probe_workers == 1:
        probe_jobs[0].duration_hint = probe_duration(probe_jobs[0].path)
        return
    with cancellable_executor(
        max_workers=probe_workers, thread_name_prefix="duration-probe"
    ) as executor:
        durations = executor.map(probe_duration, (job.path for job in probe_jobs))
        for job, duration in zip(probe_jobs, durations, strict=True):
            job.duration_hint = duration


def build_jobs(
    args: TranscriptionConfig,
    *,
    contract_args: TranscriptionConfig | None = None,
    publication_enabled: bool = True,
    capture_results: bool = False,
    retain_skipped: bool = False,
) -> list[AudioJob]:
    entries = expand_inputs(args.audio, args.recursive)
    if not publication_enabled:
        memory_jobs: list[AudioJob] = []
        for index, (path, relative_path) in enumerate(entries):
            try:
                snapshot = SourceSnapshot.capture(path)
            except OSError as exc:
                raise SystemExit(f"Cannot snapshot input {path}: {exc}") from exc
            memory_jobs.append(
                AudioJob(
                    index=index,
                    path=path,
                    relative_path=relative_path,
                    snapshot=snapshot,
                    duration_hint=None,
                    language=args.language,
                    vad_mode=args.vad,
                    alignment_mode=args.alignment,
                    capture_result=capture_results,
                    vad_engine_requested=(
                        args.vad_engine if args.vad == "silero" else None
                    ),
                    vad_merge=args.vad == "silero" and args.vad_merge,
                    segmentation_parameters=segmentation_parameters(args),
                )
            )
        _probe_duration_hints(memory_jobs)
        return memory_jobs

    if args.formats is None:
        raise RuntimeError("Output formats must be normalized before building jobs")
    output_root = (
        _resolve_planner_path(args.output_dir, label="output directory", strict=False)
        if args.output_dir
        else None
    )
    if output_root is not None:
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            output_root = output_root.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(
                f"Cannot create output directory {output_root}: {exc}"
            ) from exc

    plans: list[_JobPlan] = []
    jobs: list[AudioJob] = []
    locks: list[OutputSetLock] = []
    claimed_outputs: dict[Path, Path] = {}
    claimed_reserved: dict[Path, Path] = {}
    input_paths = {path for path, _ in entries}
    profile_candidate = (
        _resolve_planner_path(args.profile_json, label="profile", strict=False)
        if args.profile_json is not None
        else None
    )
    if profile_candidate in input_paths:
        raise SystemExit(
            f"Profile path collides with an input audio file: {profile_candidate}"
        )
    contract_configuration = contract_args or args
    asr_key = asr_contract_key(contract_configuration)
    render_key = render_contract_key(contract_configuration)

    try:
        for path, relative_path in entries:
            if output_root is None:
                parent = path.parent
            else:
                parent = output_root / relative_path.parent
            parent = _resolve_output_parent(parent, output_root)
            output_paths = {
                fmt: parent / f"{relative_path.stem}.{fmt}" for fmt in args.formats
            }
            state_path = state_path_for_outputs(output_paths)
            checkpoint_path = checkpoint_path_for_outputs(output_paths)
            lock_target = lock_target_for_outputs(output_paths)

            for reserved_path, label in (
                (state_path, "state marker"),
                (checkpoint_path, "ASR checkpoint"),
                (lock_target.path, "output lock"),
            ):
                if reserved_path.is_symlink() or (
                    reserved_path.exists() and not reserved_path.is_file()
                ):
                    raise SystemExit(
                        f"Reserved {label} is not a regular file: {reserved_path}"
                    )
                key = _resolve_path(reserved_path, strict=False)
                if key == profile_candidate:
                    raise SystemExit(
                        f"Profile path collides with a reserved {label}: {reserved_path}"
                    )
                if key in input_paths:
                    raise SystemExit(
                        f"Reserved {label} collides with an input: {reserved_path}"
                    )

            for reserved_path in (state_path, checkpoint_path):
                key = _resolve_path(reserved_path, strict=False)
                previous_reserved = claimed_reserved.get(key)
                if previous_reserved is not None and previous_reserved != path:
                    raise SystemExit(
                        f"Reserved state collision between {previous_reserved} and "
                        f"{path}: {reserved_path}"
                    )
                claimed_reserved[key] = path

            for output in output_paths.values():
                if output.is_symlink() or (output.exists() and not output.is_file()):
                    raise SystemExit(f"Output path is not a regular file: {output}")
                key = _resolve_path(output, strict=False)
                previous = claimed_outputs.get(key)
                if previous is not None and previous != path:
                    raise SystemExit(
                        "Output collision detected before model loading:\n"
                        f"  {previous}\n  {path}\n  -> {output}\n"
                        "Use separate output directories or preserve distinct relative paths."
                    )
                if key in input_paths:
                    raise SystemExit(
                        f"Output path collides with an input audio file: {output}"
                    )
                if key == profile_candidate:
                    raise SystemExit(
                        f"Profile path collides with a transcript output: {output}"
                    )
                claimed_outputs[key] = path
                if not os.access(output.parent, os.W_OK):
                    raise SystemExit(
                        f"Output directory is not writable: {output.parent}"
                    )

            try:
                snapshot = SourceSnapshot.capture(path)
            except OSError as exc:
                raise SystemExit(f"Cannot snapshot input {path}: {exc}") from exc
            plans.append(
                _JobPlan(
                    path=path,
                    relative_path=relative_path,
                    output_paths=output_paths,
                    state_path=state_path,
                    checkpoint_path=checkpoint_path,
                    lock_target=lock_target,
                    snapshot=snapshot,
                )
            )

        reserved_output_collisions = set(claimed_reserved).intersection(claimed_outputs)
        if reserved_output_collisions:
            collision = min(reserved_output_collisions, key=os.fspath)
            raise SystemExit(
                f"Transcript output collides with reserved state path: {collision}"
            )

        targets_by_range: dict[tuple[Path, int], OutputLockTarget] = {}
        for plan in plans:
            range_key = (plan.lock_target.path, plan.lock_target.offset)
            previous_target = targets_by_range.get(range_key)
            if (
                previous_target is not None
                and previous_target.identity != plan.lock_target.identity
            ):
                raise SystemExit(
                    "Output lock hash collision; process these stems in separate runs: "
                    f"{previous_target.identity}, {plan.lock_target.identity}"
                )
            targets_by_range[range_key] = plan.lock_target

        locks_by_identity: dict[str, OutputSetLock] = {}
        for target in sorted(targets_by_range.values(), key=lambda item: item.sort_key):
            try:
                lock = OutputSetLock.acquire(target)
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
            locks.append(lock)
            locks_by_identity[target.identity] = lock

        for plan in plans:
            path = plan.path
            output_paths = plan.output_paths
            lock = locks_by_identity[plan.lock_target.identity]
            job = AudioJob(
                index=len(jobs),
                path=path,
                relative_path=plan.relative_path,
                snapshot=plan.snapshot,
                duration_hint=None,
                language=args.language,
                vad_mode=args.vad,
                alignment_mode=args.alignment,
                output_paths=output_paths,
                state_path=plan.state_path,
                checkpoint_path=plan.checkpoint_path,
                asr_contract_key=asr_key,
                render_contract_key=render_key,
                output_lock=lock,
                capture_result=capture_results,
                vad_engine_requested=(
                    args.vad_engine if args.vad == "silero" else None
                ),
                vad_merge=args.vad == "silero" and args.vad_merge,
                segmentation_parameters=segmentation_parameters(args),
            )

            existing_outputs = [
                output for output in output_paths.values() if output.exists()
            ]
            if existing_outputs and args.existing == "error":
                paths = "\n".join(f"  {output}" for output in existing_outputs)
                raise SystemExit(
                    f"Output already exists:\n{paths}\n"
                    "Use --existing overwrite to replace it or --existing skip to keep complete sets."
                )
            if args.existing == "skip" and len(existing_outputs) == len(output_paths):
                verified, reason = verify_published_outputs(job)
                if verified:
                    info(f"skipping {path}: verified output generation is complete")
                    job.skipped = True
                    job.result_completed = True
                    job.written.extend(output_paths.values())
                    lock.release()
                    job.output_lock = None
                    if retain_skipped:
                        jobs.append(job)
                    continue
                info(f"rebuilding {path}: existing output set is unverified ({reason})")
            elif existing_outputs and args.existing == "skip":
                info(f"rebuilding {path}: requested output set is incomplete")

            if plan.checkpoint_path.exists():
                restored, reason = restore_asr_checkpoint(job)
                if restored:
                    job.duration_hint = job.duration
                    info(f"resuming {path}: verified ASR checkpoint loaded")
                else:
                    info(f"ignoring ASR checkpoint for {path}: {reason}")

            jobs.append(job)

        _probe_duration_hints(jobs)
        return jobs
    except BaseException:
        release_output_locks(jobs)
        for lock in locks:
            lock.release()
        raise
