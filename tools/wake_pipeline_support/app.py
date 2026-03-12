"""NiceGUI presentation layer for the local wake-pipeline workbench."""

from __future__ import annotations

import asyncio
from typing import Any, cast
from pathlib import Path
from argparse import Namespace

from nicegui import ui, app, run
from lumivox_devicelab import SpeakerDeviceDiscovery, MicrophoneDeviceDiscovery
from lumivox_core.logger import Logger, LogMode, LoggingConfig, get_logger, shutdown_logging, configure_logging

from lumivox_wakelab import VadPolicyConfig, WakePolicyConfig

from .charts import chart_options
from .profile import ProfileDraft, load_profile, save_profile, resolve_profile
from .help_text import HelpLanguage, parameter_help, language_from_locale
from .controller import SAMPLE_RATE, DEFAULT_RECORDING_PATH, DisplayEvent, SessionState, SessionController


def event_rows(events: tuple[DisplayEvent, ...]) -> list[dict[str, int | str]]:
    return [
        {"row": index, "sample": item.sample, "kind": item.kind, "detail": item.detail}
        for index, item in reversed(tuple(enumerate(events)))
    ]


def run_app(args: Namespace) -> None:
    configure_logging(LoggingConfig(application="wakelab-pipeline-lab", mode=LogMode.DEV, level="INFO"))
    logger = get_logger(tool="wake_pipeline_lab")
    controller = SessionController(logger, recording_path=DEFAULT_RECORDING_PATH)
    app.config.socket_io_js_transports = ["polling"]
    app.on_shutdown(controller.shutdown)

    @ui.page("/")
    def index() -> None:
        build_ui(controller, logger, cast(Path, args.profile))

    try:
        ui.run(host="127.0.0.1", port=cast(int, args.port), title="Wake Pipeline Lab", reload=False)
    except KeyboardInterrupt:
        pass
    finally:
        controller.shutdown()
        shutdown_logging()


def build_ui(controller: SessionController, logger: Logger, initial_profile: Path) -> None:
    ui.dark_mode().enable()
    client_id = ui.context.client.id
    draft = load_profile(initial_profile).draft
    current_profile_path = initial_profile.expanduser().resolve()
    viewport = {"samples": 30 * SAMPLE_RATE, "follow": True}
    selected_device: dict[str, str | None] = {"id": None}
    selected_speaker: dict[str, str | None] = {"id": None}
    dirty = {"value": False}
    help_language_selected = {"value": False}

    ui.add_css(
        """
        body { background: #0d1217; color: #e8edf1; }
        .lab-card { background: #151d24; border: 1px solid #26343f; border-radius: 12px; }
        .metric { min-width: 9rem; }
        .metric-active {
            background: #14251d;
            border-color: #22c55e;
            color: #86efac;
            box-shadow: 0 0 0 1px rgb(34 197 94 / 25%);
        }
        .parameter-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(100%, 17rem), 1fr));
            width: 100%;
            gap: 0.75rem;
        }
        .parameter-help-link {
            color: #b2ebf2 !important;
            font-size: 1.125rem;
            line-height: 1.5rem;
            pointer-events: auto;
            text-decoration: underline dotted;
            text-underline-offset: 0.2rem;
        }
        .parameter-grid .q-field__label { pointer-events: auto; }
        """
    )

    with ui.dialog() as help_dialog, ui.card().classes("w-full max-w-xl lab-card"):
        help_title = ui.label().classes("text-lg font-medium text-cyan-200")
        help_body = ui.label().classes("text-sm leading-relaxed whitespace-normal")
        with ui.row().classes("w-full justify-end"):
            help_close = ui.button("Close")

    with ui.column().classes("w-full max-w-[1500px] mx-auto p-3 gap-3"):
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            ui.label("WAKE PIPELINE LAB").classes("text-xl font-mono tracking-widest text-cyan-300")
            state_badge = ui.badge("ready", color="blue-grey")
            dirty_badge = ui.badge("saved", color="green")
            device_select = ui.select({}, label="Microphone").classes("min-w-[18rem] grow")
            speaker_select = ui.select({}, label="Speaker").classes("min-w-[18rem] grow")
            refresh_button = ui.button("Refresh")
            start_button = ui.button("Start", color="positive")
            stop_button = ui.button("Stop", color="negative")
            help_language = ui.select(
                {"en": "English", "ru": "Русский"}, label="Help language / Язык справки", value="en"
            ).classes("min-w-[12rem]")

        def selected_help_language() -> HelpLanguage:
            return "ru" if help_language.value == "ru" else "en"

        def open_parameter_help(key: str) -> None:
            entry = parameter_help(key, selected_help_language())
            help_title.text = entry.title
            help_body.text = entry.body
            help_close.text = "Закрыть" if selected_help_language() == "ru" else "Close"
            help_dialog.open()

        def parameter_heading(title: str, key: str) -> None:
            heading = ui.label(title).classes("parameter-help-link cursor-pointer").props('role="button" tabindex="0"')
            heading.on("click", lambda: open_parameter_help(key))
            heading.on("keydown.enter", lambda: open_parameter_help(key))

        def add_parameter_help(component: Any, title: str, key: str) -> Any:
            component.props("label-slot stack-label")
            with component.add_slot("label"):
                parameter_heading(title, key)
            return component

        with ui.expansion("Evaluation profile", icon="tune", value=True).classes("w-full lab-card"):
            profile_path = ui.input("Server-local profile path", value=str(current_profile_path)).classes("w-full")
            with ui.row().classes("w-full gap-3 flex-wrap"):
                name = ui.input("Profile name", value=draft.name).classes("grow")
                model_root = ui.input("Model root", value=draft.model_root).classes("grow")
                allow_download = ui.switch("Allow artifact download", value=draft.allow_download)
            classifier_path = ui.input("Classifier path", value=draft.classifier_path).classes("w-full")
            classifier_sha = ui.input("Classifier SHA-256", value=draft.classifier_sha256).classes("w-full font-mono")
            with ui.element("div").classes("parameter-grid"):
                speech_threshold = add_parameter_help(
                    ui.number(
                        "VAD speech threshold", value=draft.vad_policy.speech_threshold, min=0, max=1, step=0.01
                    ).classes("w-full min-w-0"),
                    "VAD speech threshold",
                    "vad_speech_threshold",
                )
                silence_threshold = add_parameter_help(
                    ui.number(
                        "VAD silence threshold", value=draft.vad_policy.silence_threshold, min=0, max=1, step=0.01
                    ).classes("w-full min-w-0"),
                    "VAD silence threshold",
                    "vad_silence_threshold",
                )
                wake_threshold = add_parameter_help(
                    ui.number(
                        "Wake threshold", value=draft.wake_policy.score_threshold, min=0, max=1, step=0.01
                    ).classes("w-full min-w-0"),
                    "Wake threshold",
                    "wake_threshold",
                )
                wake_streak = add_parameter_help(
                    ui.number("Wake streak", value=draft.wake_policy.consecutive_score_count, min=1, step=1).classes(
                        "w-full min-w-0"
                    ),
                    "Wake streak",
                    "wake_streak",
                )
            with ui.element("div").classes("parameter-grid"):
                min_speech = add_parameter_help(
                    ui.number(
                        "Minimum speech samples", value=draft.vad_policy.minimum_speech_samples, min=1, step=1
                    ).classes("w-full min-w-0"),
                    "Minimum speech samples",
                    "minimum_speech",
                )
                min_silence = add_parameter_help(
                    ui.number(
                        "Minimum silence samples", value=draft.vad_policy.minimum_silence_samples, min=1, step=1
                    ).classes("w-full min-w-0"),
                    "Minimum silence samples",
                    "minimum_silence",
                )
                left_padding = add_parameter_help(
                    ui.number(
                        "Left padding samples", value=draft.vad_policy.left_padding_samples, min=0, step=1
                    ).classes("w-full min-w-0"),
                    "Left padding samples",
                    "left_padding",
                )
                right_padding = add_parameter_help(
                    ui.number(
                        "Right padding samples", value=draft.vad_policy.right_padding_samples, min=0, step=1
                    ).classes("w-full min-w-0"),
                    "Right padding samples",
                    "right_padding",
                )
                silence_bridge = add_parameter_help(
                    ui.number(
                        "Silence bridge samples", value=draft.wake_policy.silence_bridge_samples, min=0, step=1
                    ).classes("w-full min-w-0"),
                    "Silence bridge samples",
                    "silence_bridge",
                )
                pre_roll = add_parameter_help(
                    ui.number("Pre-roll samples", value=draft.wake_policy.pre_roll_samples, min=0, step=1).classes(
                        "w-full min-w-0"
                    ),
                    "Pre-roll samples",
                    "pre_roll",
                )
                max_retained = add_parameter_help(
                    ui.number(
                        "Maximum retained samples", value=draft.max_retained_audio_samples, min=1, step=1
                    ).classes("w-full min-w-0"),
                    "Maximum retained samples",
                    "maximum_retained",
                )
            timing_summary = ui.label().classes("text-xs text-blue-grey-3")
            validation = ui.label().classes("text-sm text-red-300")
            with ui.row().classes("gap-2"):
                load_button = ui.button("Load")
                save_button = ui.button("Save")
                started_summary = ui.label("No session started yet").classes("text-sm text-blue-grey-3")

        with ui.row().classes("w-full gap-2 flex-wrap"):
            metrics = {
                key: ui.label(f"{label}: -").classes("lab-card metric p-2 font-mono text-xs")
                for key, label in (
                    ("time", "Audio position"),
                    ("generation", "Generation"),
                    ("queue", "Queue / drops"),
                    ("retained", "Retained"),
                    ("capture_latency", "Capture latency (500 ms avg)"),
                    ("processing_latency", "DSP processing (500 ms avg)"),
                    ("providers", "ONNX providers"),
                    ("vad", "VAD"),
                    ("wake", "Wake"),
                    ("activation", "Activation"),
                    ("recording", "Activation recording"),
                )
            }

        with ui.row().classes("w-full items-center gap-2"):
            rearm_button = ui.button("Re-arm", color="warning")
            play_button = ui.button("Play capture", icon="play_arrow")
            stop_playback_button = ui.button("Stop playback", icon="stop", color="negative")
            follow = ui.switch("Follow live", value=True)
            for seconds in (10, 30, 60):
                ui.button(f"{seconds}s", on_click=lambda _, seconds=seconds: set_viewport(seconds)).props("flat dense")
            reset_zoom = ui.button("Reset zoom").props("flat dense")
        chart = ui.echart({}).classes("w-full h-[520px] lab-card")
        event_table = ui.table(
            columns=[
                {"name": "sample", "label": "Sample", "field": "sample"},
                {"name": "kind", "label": "Event", "field": "kind"},
                {"name": "detail", "label": "Detail", "field": "detail"},
            ],
            rows=[],
            row_key="row",
            pagination={"rowsPerPage": 10},
        ).classes("w-full lab-card")

    fields = (
        name,
        model_root,
        allow_download,
        classifier_path,
        classifier_sha,
        speech_threshold,
        silence_threshold,
        wake_threshold,
        wake_streak,
        min_speech,
        min_silence,
        left_padding,
        right_padding,
        silence_bridge,
        pre_roll,
        max_retained,
    )

    def integer(component: Any, label: str) -> int:
        value = component.value
        if isinstance(value, bool) or value is None or int(value) != value:
            raise ValueError(f"{label} must be an exact integer sample count")
        return int(value)

    def read_draft() -> ProfileDraft:
        return ProfileDraft(
            str(name.value),
            str(model_root.value),
            bool(allow_download.value),
            str(classifier_path.value),
            str(classifier_sha.value),
            VadPolicyConfig(
                float(speech_threshold.value),
                float(silence_threshold.value),
                integer(min_speech, "minimum speech"),
                integer(min_silence, "minimum silence"),
                integer(left_padding, "left padding"),
                integer(right_padding, "right padding"),
            ),
            WakePolicyConfig(
                float(wake_threshold.value),
                integer(wake_streak, "wake streak"),
                integer(silence_bridge, "silence bridge"),
                integer(pre_roll, "pre-roll"),
            ),
            integer(max_retained, "maximum retained audio"),
        )

    def mark_dirty() -> None:
        dirty["value"] = True
        dirty_badge.text = "draft changed"
        dirty_badge.props("color=orange")
        validate()

    def validate() -> ProfileDraft | None:
        try:
            value = read_draft()
            resolve_profile(value, Path(str(profile_path.value)))
        except (TypeError, ValueError) as error:
            validation.text = str(error)
            return None
        validation.text = ""
        samples = [
            value.vad_policy.minimum_speech_samples,
            value.vad_policy.minimum_silence_samples,
            value.vad_policy.left_padding_samples,
            value.vad_policy.right_padding_samples,
            value.wake_policy.silence_bridge_samples,
            value.wake_policy.pre_roll_samples,
            value.max_retained_audio_samples,
        ]
        timing_summary.text = "Timing: " + ", ".join(f"{item / SAMPLE_RATE * 1000:.1f} ms" for item in samples)
        return value

    def apply_draft(value: ProfileDraft, path: Path) -> None:
        nonlocal current_profile_path
        current_profile_path = path
        profile_path.value = str(path)
        values = (
            value.name,
            value.model_root,
            value.allow_download,
            value.classifier_path,
            value.classifier_sha256,
            value.vad_policy.speech_threshold,
            value.vad_policy.silence_threshold,
            value.wake_policy.score_threshold,
            value.wake_policy.consecutive_score_count,
            value.vad_policy.minimum_speech_samples,
            value.vad_policy.minimum_silence_samples,
            value.vad_policy.left_padding_samples,
            value.vad_policy.right_padding_samples,
            value.wake_policy.silence_bridge_samples,
            value.wake_policy.pre_roll_samples,
            value.max_retained_audio_samples,
        )
        for component, field_value in zip(fields, values, strict=True):
            cast(Any, component).value = field_value
            component.update()
        dirty["value"] = False
        dirty_badge.text = "saved"
        dirty_badge.props("color=green")
        validate()

    async def refresh_devices() -> None:
        try:
            snapshots = await run.io_bound(
                lambda: (
                    MicrophoneDeviceDiscovery(logger=logger).snapshot(),
                    SpeakerDeviceDiscovery(logger=logger).snapshot(),
                )
            )
            if snapshots is None:
                return
            snapshot, speakers = snapshots
            options = {device.id: device.name for device in snapshot.devices}
            preferred = selected_device["id"] if selected_device["id"] in options else None
            if preferred is None and snapshot.default is not None:
                preferred = snapshot.default.id
            if preferred is None and snapshot.devices:
                preferred = snapshot.devices[0].id
            selected_device["id"] = preferred
            device_select.options = options
            device_select.value = preferred
            device_select.update()
            speaker_options = {device.id: device.name for device in speakers.devices}
            preferred_speaker = selected_speaker["id"] if selected_speaker["id"] in speaker_options else None
            if preferred_speaker is None and speakers.default is not None:
                preferred_speaker = speakers.default.id
            if preferred_speaker is None and speakers.devices:
                preferred_speaker = speakers.devices[0].id
            selected_speaker["id"] = preferred_speaker
            speaker_select.options = speaker_options
            speaker_select.value = preferred_speaker
            speaker_select.update()
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error("pipeline_lab_discovery_failed", error=str(error))
            ui.notify(str(error), type="negative")

    async def select_browser_help_language() -> None:
        try:
            browser_locale = await ui.run_javascript("navigator.language || 'en'", timeout=2.0)
        except (asyncio.CancelledError, TimeoutError):
            return
        if not help_language_selected["value"]:
            help_language.value = language_from_locale(browser_locale)
            help_language.update()

    async def start_session() -> None:
        value = validate()
        device_id = device_select.value
        if value is None or not isinstance(device_id, str):
            return
        profile = resolve_profile(value, Path(str(profile_path.value)))
        try:
            await run.io_bound(controller.start, client_id, device_id, profile)
            started_summary.text = f"Started: {value.name} ({profile.path})"
        except asyncio.CancelledError:
            controller.request_stop()
            raise
        except Exception as error:
            ui.notify(str(error), type="negative")

    async def stop_session() -> None:
        try:
            await run.io_bound(controller.stop)
        except asyncio.CancelledError:
            controller.request_stop()
            raise

    async def rearm_session() -> None:
        try:
            await run.io_bound(controller.rearm)
        except asyncio.CancelledError:
            controller.request_stop()
            raise
        except Exception as error:
            ui.notify(str(error), type="negative")

    async def play_recording() -> None:
        device_id = speaker_select.value
        if not isinstance(device_id, str):
            return
        try:
            completed = await run.io_bound(controller.play, device_id)
            ui.notify(
                "Playback finished" if completed else "Playback stopped", type="positive" if completed else "info"
            )
        except asyncio.CancelledError:
            controller.stop_playback()
            raise
        except Exception as error:
            ui.notify(str(error), type="negative")

    async def stop_playback() -> None:
        try:
            await run.io_bound(controller.stop_playback)
        except asyncio.CancelledError:
            controller.stop_playback()
            raise

    async def load_action() -> None:
        try:
            loaded = await run.io_bound(load_profile, Path(str(profile_path.value)))
            if loaded is not None:
                apply_draft(loaded.draft, loaded.path)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            validation.text = str(error)

    async def save_action() -> None:
        value = validate()
        if value is None:
            return
        path = Path(str(profile_path.value)).expanduser().resolve()
        replace_existing = path.exists()
        if replace_existing:
            with ui.dialog() as dialog, ui.card():
                ui.label(f"Replace existing profile {path}?")
                with ui.row():
                    ui.button("Replace", on_click=lambda: dialog.submit(True), color="negative")
                    ui.button("Cancel", on_click=lambda: dialog.submit(False))
            if not await dialog:
                return
        try:
            saved = await run.io_bound(save_profile, path, value, replace=replace_existing)
            if saved is not None:
                apply_draft(saved.draft, saved.path)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            validation.text = str(error)

    def set_viewport(seconds: int) -> None:
        viewport["samples"] = seconds * SAMPLE_RATE
        viewport["follow"] = True
        follow.value = True

    def refresh_display() -> None:
        snapshot = controller.snapshot()
        state_badge.text = snapshot.state.value
        state_badge.props(f"color={'red' if snapshot.state is SessionState.ERROR else 'cyan'}")
        transition = snapshot.state in {SessionState.LOADING, SessionState.STOPPING}
        start_button.set_enabled(not snapshot.active and device_select.value is not None and validate() is not None)
        stop_button.set_enabled(snapshot.active)
        rearm_button.set_enabled(snapshot.state is SessionState.LISTENING)
        play_button.set_enabled(
            snapshot.recording_available
            and not snapshot.recording_active
            and not snapshot.playback_active
            and speaker_select.value is not None
        )
        stop_playback_button.set_enabled(snapshot.playback_active)
        speaker_select.set_enabled(not snapshot.playback_active)
        load_button.set_enabled(not transition)
        save_button.set_enabled(not transition)
        refresh_button.set_enabled(not transition)
        metrics[
            "time"
        ].text = f"Audio position: {snapshot.current_sample} ({snapshot.current_sample / SAMPLE_RATE:.3f}s)"
        metrics["generation"].text = f"Generation: {snapshot.generation if snapshot.generation is not None else '-'}"
        metrics["queue"].text = f"Queue / drops: {snapshot.queue_depth} / {snapshot.dropped_chunks}"
        metrics["retained"].text = f"Retained: {snapshot.retained_samples}"
        metrics["capture_latency"].text = (
            f"Capture latency (500 ms avg): {snapshot.capture_latency_ms:.2f} ms"
            if snapshot.capture_latency_ms is not None
            else "Capture latency (500 ms avg): -"
        )
        metrics["processing_latency"].text = (
            f"DSP processing (500 ms avg): {snapshot.processing_latency_ms:.2f} ms"
            if snapshot.processing_latency_ms is not None
            else "DSP processing (500 ms avg): -"
        )
        metrics["providers"].text = f"ONNX providers: {', '.join(snapshot.providers) or '-'}"
        metrics["vad"].text = (
            f"VAD: {snapshot.vad_probability:.4f} / {'speech' if snapshot.vad_is_speech else 'quiet'}"
            if snapshot.vad_probability is not None
            else "VAD: waiting"
        )
        metrics["wake"].text = (
            f"Wake: {snapshot.wake_score:.4f} / streak {snapshot.wake_hit_count}"
            if snapshot.wake_score is not None
            else f"Wake: {snapshot.wake_state}"
        )
        metrics["activation"].text = f"Activation: {'latched' if snapshot.activated else 'armed'}"
        metrics["activation"].classes(add="metric-active" if snapshot.activated else None)
        metrics["activation"].classes(remove=None if snapshot.activated else "metric-active")
        recording_state = (
            "recording" if snapshot.recording_active else "ready" if snapshot.recording_available else "none"
        )
        metrics["recording"].text = (
            f"Activation recording: {recording_state}, {snapshot.recording_samples / SAMPLE_RATE:.3f}s, "
            f"{snapshot.recording_path}"
        )
        value = validate()
        if value is not None and bool(viewport["follow"]):
            chart.options.clear()
            chart.options.update(chart_options(snapshot, value, int(viewport["samples"])))
            chart.update()
        event_table.rows = event_rows(snapshot.events)
        event_table.update()
        if snapshot.error:
            validation.text = snapshot.error

    async def disconnect() -> None:
        try:
            snapshot = controller.snapshot()
            if snapshot.owner_id == client_id and snapshot.active:
                await run.io_bound(controller.stop)
        except asyncio.CancelledError:
            controller.request_stop()

    async def navigate_to_event(event: Any) -> None:
        args = event.args
        row = args.get("data", args) if isinstance(args, dict) else {}
        sample = row.get("sample") if isinstance(row, dict) else None
        if isinstance(sample, int):
            viewport["follow"] = False
            follow.value = False
            half = int(viewport["samples"]) // 2
            await chart.run_chart_method(
                "dispatchAction",
                {"type": "dataZoom", "startValue": max(0, sample - half), "endValue": sample + half},
            )

    for field in fields:
        field.on("update:model-value", lambda _: mark_dirty())
    help_close.on_click(help_dialog.close)
    help_language.on_value_change(lambda _: help_language_selected.update(value=True))
    profile_path.on("update:model-value", lambda _: mark_dirty())
    device_select.on_value_change(lambda event: selected_device.update(id=event.value))
    speaker_select.on_value_change(lambda event: selected_speaker.update(id=event.value))
    refresh_button.on_click(refresh_devices)
    start_button.on_click(start_session)
    stop_button.on_click(stop_session)
    rearm_button.on_click(rearm_session)
    play_button.on_click(play_recording)
    stop_playback_button.on_click(stop_playback)
    load_button.on_click(load_action)
    save_button.on_click(save_action)
    follow.on_value_change(lambda event: viewport.update(follow=bool(event.value)))
    reset_zoom.on_click(lambda: set_viewport(60))
    event_table.on("rowClick", navigate_to_event)
    ui.context.client.on_disconnect(disconnect)
    ui.timer(0.1, refresh_display)
    ui.timer(0.1, select_browser_help_language, once=True)
    ui.timer(0.1, refresh_devices, once=True)
    validate()
