"""Measure small-chunk latency and allocations during a retained wake candidate."""

from __future__ import annotations

import time
import argparse
import platform
import statistics
import tracemalloc
from pathlib import Path

import numpy as np

from lumivox_wakelab import InputChunk, StreamConfig, VadPolicyConfig, WakePolicyConfig, WakeStreamProcessor
from lumivox_wakelab.wakeword import OpenWakeWord


class _Logger:
    def bind(self, **values: object) -> _Logger:
        return self

    def debug(self, event: str, **values: object) -> None:
        pass

    def info(self, event: str, **values: object) -> None:
        pass

    def warning(self, event: str, **values: object) -> None:
        pass

    def error(self, event: str, **values: object) -> None:
        pass

    def critical(self, event: str, **values: object) -> None:
        pass

    def exception(self, event: str, **values: object) -> None:
        pass


class _Backend:
    sample_rate = 16_000

    def __init__(self, frame_samples: int, score: float) -> None:
        self.frame_samples = frame_samples
        self._score = score

    def infer(self, frame: np.ndarray[tuple[int], np.dtype[np.int16]]) -> float:
        return self._score

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


def _run(sample_count: int) -> tuple[list[int], int]:
    consecutive = sample_count // 1_280 + 2
    required = consecutive * 1_280 + 512
    config = StreamConfig(
        VadPolicyConfig(0.6, 0.4, 1, 1, 0, 0),
        WakePolicyConfig(0.7, consecutive, 0, 0),
        required,
    )
    processor = WakeStreamProcessor(config, _Backend(512, 1.0), _Backend(1_280, 1.0), logger=_Logger())
    source = np.arange(sample_count, dtype=np.dtype("<i2"))
    durations: list[int] = []
    outputs = []
    tracemalloc.start()
    try:
        for sample in source:
            chunk = InputChunk(np.asarray([sample], dtype=np.dtype("<i2")), 0, 0, 0, False)
            started = time.perf_counter_ns()
            outputs.extend(processor.process(chunk))
            durations.append(time.perf_counter_ns() - started)
        outputs.extend(processor.finish())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        processor.close()
    np.testing.assert_array_equal(np.concatenate([output.samples for output in outputs]), source)
    return durations, peak


def _run_openwakeword(feature_dir: Path, classifier: Path, frame_count: int) -> tuple[list[int], int]:
    backend = OpenWakeWord.load(feature_dir, classifier, logger=_Logger())
    frame = np.zeros(1_280, dtype=np.dtype("<i2"))
    durations: list[int] = []
    tracemalloc.start()
    try:
        for _ in range(frame_count):
            started = time.perf_counter_ns()
            backend.infer(frame)
            durations.append(time.perf_counter_ns() - started)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        backend.close()
    return durations, peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--classifier", type=Path)
    parser.add_argument("--openwakeword-frames", type=int, default=0)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")

    if (args.feature_dir is None) != (args.classifier is None):
        parser.error("--feature-dir and --classifier must be supplied together")
    if args.openwakeword_frames < 0:
        parser.error("--openwakeword-frames must be nonnegative")
    if args.openwakeword_frames:
        if args.feature_dir is None or args.classifier is None:
            parser.error("openWakeWord benchmarking requires --feature-dir and --classifier")
        durations, peak = _run_openwakeword(args.feature_dir, args.classifier, args.openwakeword_frames)
        workload = f"openwakeword_frames={args.openwakeword_frames}"
    else:
        durations, peak = _run(args.samples)
        workload = f"samples={args.samples}"
    ordered = sorted(durations)
    p99 = ordered[min(len(ordered) - 1, (len(ordered) * 99 + 99) // 100 - 1)]
    print(f"platform={platform.platform()}")
    print(workload)
    print(f"median_us={statistics.median(durations) / 1_000:.3f}")
    print(f"p99_us={p99 / 1_000:.3f}")
    print(f"peak_bytes={peak}")


if __name__ == "__main__":
    main()
