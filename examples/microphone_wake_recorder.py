"""Record a bounded activated region from a Devicelab microphone."""

from __future__ import annotations

import os
import wave
import argparse
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import Event
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from lumivox_devicelab import (
    AudioDevice,
    AudioFormat,
    CapturedChunk,
    PipelineState,
    CaptureHandler,
    DeviceSnapshot,
    MicrophoneCapturePipeline,
    MicrophoneDeviceDiscovery,
)
from lumivox_core.logger import Logger, LogMode, LoggingConfig, get_logger, shutdown_logging, configure_logging

from lumivox_wakelab import (
    InputChunk,
    OutputChunk,
    StreamProcessor,
    StreamCloseError,
)
from tools.wake_pipeline_support.profile import (
    MicrophoneProfile,
)
from tools.wake_pipeline_support.profile import (
    load_profile as load_profile,
)
from tools.wake_pipeline_support.profile import (
    build_processor as build_processor_bundle,
)

SAMPLE_RATE = 16_000
DEFAULT_OUTPUT = Path("wake-recording.wav")
DEFAULT_PROFILE = Path(__file__).with_name("profile.json")


def select_microphone(snapshot: DeviceSnapshot, requested_id: str | None) -> AudioDevice:
    """Select an explicit microphone or Devicelab's reported default."""

    if requested_id is None:
        if snapshot.default is None:
            raise ValueError("no default microphone is configured; use --list-microphones and --microphone-id")
        return snapshot.default
    for device in snapshot.devices:
        if device.id == requested_id:
            return device
    raise ValueError(f"microphone ID not found: {requested_id}")


class ActivatedAudioHandler(CaptureHandler):
    """Run the stream processor and retain exactly the requested activated PCM."""

    def __init__(
        self,
        processor: StreamProcessor,
        target_samples: int,
        *,
        status: Callable[[str], None] = print,
    ) -> None:
        if target_samples <= 0:
            raise ValueError("target_samples must be positive")
        self._processor = processor
        self._target_samples = target_samples
        self._status = status
        self._parts: list[NDArray[np.int16]] = []
        self._recorded_samples = 0
        self._activation_announced = False
        self.complete = Event()

    @property
    def recorded_samples(self) -> int:
        return self._recorded_samples

    def on_chunk(self, chunk: CapturedChunk) -> None:
        if self.complete.is_set():
            return
        outputs = self._processor.process(
            InputChunk(
                samples=chunk.samples,
                running_time_ns=chunk.running_time_ns,
                captured_at_ns=chunk.captured_at_ns,
                generation=chunk.generation,
                discontinuity=chunk.discontinuity,
            )
        )
        self.accept(outputs)

    def accept(self, outputs: Sequence[OutputChunk]) -> None:
        """Retain activated output from normal processing or finalization."""

        for output in outputs:
            if not output.is_activated or self.complete.is_set():
                continue
            if not self._activation_announced:
                self._activation_announced = True
                self._status("Wake word detected. Recording activated audio...")
            remaining = self._target_samples - self._recorded_samples
            part = output.samples[:remaining].copy()
            self._parts.append(part)
            self._recorded_samples += part.size
            if self._recorded_samples == self._target_samples:
                self.complete.set()
                self._status("Requested recording duration reached. Stopping microphone...")

    def samples(self) -> NDArray[np.int16]:
        if not self._parts:
            return np.empty(0, dtype=np.dtype("<i2"))
        return np.concatenate(self._parts)


def build_processor(profile: MicrophoneProfile, logger: Logger) -> StreamProcessor:
    """Acquire, verify, load, and warm real backends before capture begins."""

    return build_processor_bundle(profile, logger).processor


def write_wave(path: Path, samples: NDArray[np.int16]) -> None:
    """Publish one mono PCM16 WAV without silently replacing an existing file."""

    if samples.size == 0:
        raise ValueError("cannot write an empty recording")
    if path.suffix.lower() != ".wav":
        raise ValueError("output path must end with .wav")
    if not path.parent.is_dir():
        raise ValueError(f"output directory does not exist: {path.parent}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w+b", dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
            temporary_path = Path(output.name)
            with wave.open(output, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(samples.astype(np.dtype("<i2"), copy=False).tobytes())
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _duration(value: str) -> Decimal:
    try:
        duration = Decimal(value)
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("duration must be a number") from error
    if not duration.is_finite() or duration <= 0:
        raise argparse.ArgumentTypeError("duration must be finite and positive")
    if duration * SAMPLE_RATE < 1:
        raise argparse.ArgumentTypeError(f"duration must contain at least one sample at {SAMPLE_RATE} Hz")
    return duration


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-microphones", action="store_true", help="list microphone names and IDs, then exit")
    parser.add_argument("--microphone-id", help="Devicelab microphone ID; defaults to the reported default device")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help=f"output WAV path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--duration", type=_duration, default=Decimal("10"), help="activated audio seconds (default: 10)"
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE, help="model and policy JSON profile")
    return parser


def _list_microphones(snapshot: DeviceSnapshot) -> None:
    if not snapshot.devices:
        print("No microphones found.")
        return
    print("Available microphones:")
    for device in snapshot.devices:
        marker = " (default)" if device.is_default else ""
        print(f"  {device.name}{marker}\n    ID: {device.id}")


def run(args: argparse.Namespace, logger: Logger) -> int:
    discovery = MicrophoneDeviceDiscovery(logger=logger)
    snapshot = discovery.snapshot()
    if args.list_microphones:
        _list_microphones(snapshot)
        return 0

    device = select_microphone(snapshot, args.microphone_id)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if output.suffix.lower() != ".wav":
        raise ValueError("output path must end with .wav")
    if not output.parent.is_dir():
        raise ValueError(f"output directory does not exist: {output.parent}")

    profile = load_profile(args.profile)
    target_samples = int(args.duration * SAMPLE_RATE)
    print(f"Loading profile {profile.name!r} and warming models...")
    processor = build_processor(profile, logger)
    handler = ActivatedAudioHandler(processor, target_samples)
    try:
        pipeline = MicrophoneCapturePipeline(
            logger=logger,
            handler=handler,
            audio_format=AudioFormat(sample_rate=SAMPLE_RATE, channels=1),
            device_id=device.id,
        )
    except Exception:
        processor.close()
        raise

    interrupted = False
    try:
        print(f"Using microphone: {device.name} ({device.id})")
        print("Listening for the wake word. Press Ctrl+C to stop.")
        pipeline.start()
        while not handler.complete.wait(0.1):
            if pipeline.state is PipelineState.STOPPED:
                pipeline.wait()
                raise RuntimeError("microphone capture stopped before recording completed")
        pipeline.stop()
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping on user request...")
        if pipeline.state is not PipelineState.STOPPED:
            pipeline.stop()
    finally:
        if pipeline.state is not PipelineState.STOPPED:
            pipeline.stop(immediate=True)
        try:
            handler.accept(processor.close())
        except StreamCloseError as error:
            handler.accept(error.outputs)
            raise

    samples = handler.samples()
    if samples.size == 0:
        print("No activated audio was recorded; no file was created.")
        return 130 if interrupted else 1
    write_wave(output, samples)
    duration = samples.size / SAMPLE_RATE
    qualifier = "partial " if samples.size < target_samples else ""
    print(f"Saved {qualifier}recording ({duration:.3f} s) to {output}")
    return 130 if interrupted else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    configure_logging(LoggingConfig(application="wakelab-microphone-example", mode=LogMode.DEV, level="WARNING"))
    logger = get_logger(component="microphone_example")
    try:
        return run(args, logger)
    except Exception as error:
        print(f"Error: {error}")
        return 1
    finally:
        shutdown_logging()


if __name__ == "__main__":
    raise SystemExit(main())
