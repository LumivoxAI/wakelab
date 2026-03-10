from __future__ import annotations

import subprocess
from pathlib import Path
from dataclasses import replace

from lumivox_wakelab import (
    VadPolicyConfig,
    WakePolicyConfig,
    VadFrameDiagnostic,
    WakeFrameDiagnostic,
    ActivationDiagnostic,
)
from tools.wake_pipeline_support.app import event_rows
from tools.wake_pipeline_support.charts import chart_options
from tools.wake_pipeline_support.profile import ProfileDraft
from tools.wake_pipeline_support.help_text import parameter_help, language_from_locale
from tools.wake_pipeline_support.controller import DisplayEvent, SessionState, SessionSnapshot


def snapshot() -> SessionSnapshot:
    return SessionSnapshot(
        state=SessionState.LISTENING,
        owner_id="client",
        profile_name="test",
        generation=0,
        current_sample=64_000,
        retained_samples=0,
        queue_depth=0,
        queue_samples=0,
        dropped_chunks=0,
        dropped_samples=0,
        diagnostic_drops=0,
        capture_latency_ms=2.0,
        processing_latency_ms=1.0,
        vad_probability=0.5,
        vad_is_speech=True,
        wake_score=None,
        wake_hit_count=0,
        wake_state="armed",
        activated=False,
        recording_path="/tmp/wake-pipeline-capture.wav",
        recording_samples=0,
        recording_active=False,
        recording_available=False,
        playback_active=False,
        providers=("CPUExecutionProvider",),
        diagnostics=tuple(VadFrameDiagnostic(0, index, index + 1, (index % 7) / 7) for index in range(10_000)),
        events=(),
        error=None,
    )


def profile() -> ProfileDraft:
    return ProfileDraft(
        "test",
        "models",
        False,
        "wake.onnx",
        "0" * 64,
        VadPolicyConfig(0.6, 0.4, 4, 4, 0, 0),
        WakePolicyConfig(0.7, 1, 4, 2),
        16,
    )


def test_chart_projection_is_bounded_and_wake_series_stays_absent() -> None:
    options = chart_options(snapshot(), profile(), 30 * 16_000)
    series = options["series"]

    assert isinstance(series, list)
    vad = next(item for item in series if isinstance(item, dict) and item.get("name") == "VAD probability")
    wake = next(item for item in series if isinstance(item, dict) and item.get("name") == "Wake score")
    assert len(vad["data"]) <= 8
    assert wake["data"] == []
    assert options["dataZoom"]
    grids = options["grid"]
    axes = options["yAxis"]
    assert isinstance(grids, list)
    assert isinstance(axes, list)
    assert len(grids) == 2
    assert [axis["name"] for axis in axes if isinstance(axis, dict)] == ["VAD", "Wake"]
    x_axes = options["xAxis"]
    assert isinstance(x_axes, list)
    assert x_axes[1]["nameLocation"] == "middle"
    assert x_axes[1]["nameGap"] == 28


def test_event_rows_show_newest_event_first_with_unique_keys() -> None:
    rows = event_rows(
        (
            DisplayEvent(2_560, "candidate_start", "wake evaluation started"),
            DisplayEvent(2_560, "wake_hit", "score 0.8000, streak 1"),
        )
    )

    assert [row["kind"] for row in rows] == ["wake_hit", "candidate_start"]
    assert [row["row"] for row in rows] == [1, 0]


def test_activated_wake_chart_keeps_shared_timeline_without_fake_zero_scores() -> None:
    diagnostics = (
        WakeFrameDiagnostic(0, 31_000, 32_000, 0.95, True, 1),
        ActivationDiagnostic(0, 16_000, 31_000, 32_000, 0.95, 31_000, 32_000, 0.95),
    )
    active = replace(
        snapshot(),
        current_sample=160_000,
        wake_state="bypassed",
        activated=True,
        diagnostics=diagnostics,
    )

    options = chart_options(active, profile(), 10 * 16_000)
    axes = options["xAxis"]
    series = options["series"]
    assert isinstance(axes, list)
    assert axes[0]["min"] == axes[1]["min"] == 0
    assert axes[0]["max"] == axes[1]["max"] == active.current_sample
    assert isinstance(series, list)
    wake = next(item for item in series if isinstance(item, dict) and item.get("name") == "Wake score")
    assert wake["data"] == [[32_000, 0.95]]
    assert wake["markArea"]["data"][-1][1]["xAxis"] == active.current_sample


def test_cli_help_does_not_load_gui_or_hardware() -> None:
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        ["uv", "run", "python", "tools/wake_pipeline_lab.py", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Wake-pipeline diagnostic" not in completed.stderr
    assert "--profile" in completed.stdout


def test_help_language_follows_supported_browser_locale() -> None:
    assert language_from_locale("ru-RU") == "ru"
    assert language_from_locale("ru_RU") == "ru"
    assert language_from_locale("en-US") == "en"
    assert language_from_locale("de-DE") == "en"
    assert language_from_locale(None) == "en"


def test_parameter_help_is_available_in_both_languages() -> None:
    english = parameter_help("vad_speech_threshold", "en")
    russian = parameter_help("vad_speech_threshold", "ru")

    assert english.title == "VAD speech threshold"
    assert russian.title == "Порог речи VAD"
    assert english.body != russian.body
