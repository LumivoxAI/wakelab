"""Pure bounded chart projection for the diagnostic GUI."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

from lumivox_wakelab import (
    RearmDiagnostic,
    VadFrameDiagnostic,
    VadSpeechDiagnostic,
    WakeFrameDiagnostic,
    ActivationDiagnostic,
    ContinuityBoundaryDiagnostic,
    WakeCandidateEndedDiagnostic,
    WakeCandidateStartedDiagnostic,
)

from .profile import ProfileDraft
from .controller import DEFAULT_HISTORY_SAMPLES, DEFAULT_DISPLAY_BUCKET_SAMPLES, SessionSnapshot


def downsample_extrema[T](
    values: Sequence[T],
    *,
    sample: Callable[[T], int],
    value: Callable[[T], float],
    bucket_samples: int,
) -> tuple[T, ...]:
    """Keep each sample bucket's extrema in their original order."""

    if bucket_samples <= 0:
        raise ValueError("bucket_samples must be positive")
    buckets: dict[int, list[tuple[int, T]]] = defaultdict(list)
    for index, item in enumerate(values):
        buckets[sample(item) // bucket_samples].append((index, item))
    result: list[T] = []
    for bucket in sorted(buckets):
        items = buckets[bucket]
        low = min(items, key=lambda pair: value(pair[1]))
        high = max(items, key=lambda pair: value(pair[1]))
        result.extend(item for _, item in sorted({low[0]: low, high[0]: high}.values()))
    return tuple(result)


def chart_options(snapshot: SessionSnapshot, profile: ProfileDraft, viewport_samples: int) -> dict[str, object]:
    """Build one bounded ECharts option with linked VAD and wake grids."""

    vad = tuple(item for item in snapshot.diagnostics if isinstance(item, VadFrameDiagnostic))
    wake = tuple(item for item in snapshot.diagnostics if isinstance(item, WakeFrameDiagnostic))
    vad_points = downsample_extrema(
        vad,
        sample=lambda item: item.end_sample,
        value=lambda item: item.probability,
        bucket_samples=DEFAULT_DISPLAY_BUCKET_SAMPLES,
    )
    wake_points = downsample_extrema(
        wake,
        sample=lambda item: item.end_sample,
        value=lambda item: item.score,
        bucket_samples=DEFAULT_DISPLAY_BUCKET_SAMPLES,
    )
    speech = _spans(snapshot, VadSpeechDiagnostic)
    candidates = _candidate_spans(snapshot)
    activated = _activation_spans(snapshot)
    gaps = [[item.sample, 0] for item in snapshot.diagnostics if isinstance(item, ContinuityBoundaryDiagnostic)]
    zoom_start = max(0, snapshot.current_sample - viewport_samples)
    axis_start = max(0, snapshot.current_sample - DEFAULT_HISTORY_SAMPLES)
    axis_end = max(viewport_samples, snapshot.current_sample)
    common_zoom: dict[str, object] = {
        "type": "inside",
        "xAxisIndex": [0, 1],
        "startValue": zoom_start,
        "endValue": max(viewport_samples, snapshot.current_sample),
        "filterMode": "none",
    }
    return {
        "animation": False,
        "textStyle": {"color": "#d6e2ea"},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
        "legend": {"top": 0, "textStyle": {"color": "#d6e2ea"}},
        "grid": [
            {"left": 58, "right": 24, "top": 42, "height": "34%"},
            {"left": 58, "right": 24, "top": "53%", "height": "34%"},
        ],
        "xAxis": [
            _axis(0, False, axis_start, axis_end),
            _axis(1, True, axis_start, axis_end),
        ],
        "yAxis": [
            _probability_axis(0, "VAD"),
            _probability_axis(1, "Wake"),
        ],
        "dataZoom": [common_zoom, {**common_zoom, "type": "slider", "bottom": 4}],
        "series": [
            _line(
                "VAD probability",
                0,
                [[item.end_sample, item.probability] for item in vad_points],
                "#35b89a",
                marks=(profile.vad_policy.speech_threshold, profile.vad_policy.silence_threshold),
                areas=speech,
            ),
            _line(
                "Wake score",
                1,
                [[item.end_sample, item.score] for item in wake_points],
                "#bf72e9",
                marks=(profile.wake_policy.score_threshold,),
                areas=[*candidates, *activated],
                connect_nulls=False,
            ),
            {
                "name": "Boundaries",
                "type": "scatter",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "symbol": "diamond",
                "symbolSize": 10,
                "itemStyle": {"color": "#ef5350"},
                "data": gaps,
            },
        ],
    }


def _axis(index: int, labels: bool, minimum: int, maximum: int) -> dict[str, object]:
    axis: dict[str, object] = {
        "type": "value",
        "gridIndex": index,
        "min": minimum,
        "max": maximum,
        "axisLabel": {"show": labels, "color": "#aebfca"},
        "axisLine": {"lineStyle": {"color": "#607786"}},
        "splitLine": {"lineStyle": {"color": "#263943"}},
        "axisPointer": {"label": {"formatter": "{@[0]} samples", "color": "#e8edf1"}},
    }
    if labels:
        axis["name"] = "processor samples"
        axis["nameLocation"] = "middle"
        axis["nameGap"] = 28
        axis["nameTextStyle"] = {"color": "#c7d5de"}
    return axis


def _probability_axis(index: int, name: str) -> dict[str, object]:
    return {
        "gridIndex": index,
        "name": name,
        "nameTextStyle": {"color": "#c7d5de"},
        "min": 0,
        "max": 1,
        "axisLabel": {"color": "#aebfca"},
        "axisLine": {"lineStyle": {"color": "#607786"}},
        "splitLine": {"lineStyle": {"color": "#263943"}},
    }


def _line(
    name: str,
    axis: int,
    data: list[list[int | float]],
    color: str,
    *,
    marks: tuple[float, ...] = (),
    areas: list[list[dict[str, object]]] | None = None,
    connect_nulls: bool = True,
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "type": "line",
        "xAxisIndex": axis,
        "yAxisIndex": axis,
        "showSymbol": False,
        "connectNulls": connect_nulls,
        "lineStyle": {"width": 1.5, "color": color},
        "data": data,
    }
    if marks:
        result["markLine"] = {
            "silent": True,
            "symbol": "none",
            "data": [{"yAxis": value} for value in marks],
        }
    if areas:
        result["markArea"] = {"silent": True, "data": areas}
    return result


def _spans(snapshot: SessionSnapshot, event_type: type[VadSpeechDiagnostic]) -> list[list[dict[str, object]]]:
    spans: list[list[dict[str, object]]] = []
    start: int | None = None
    for item in snapshot.diagnostics:
        if isinstance(item, event_type):
            if item.is_speech:
                start = item.sample
            elif start is not None:
                spans.append([{"xAxis": start, "itemStyle": {"color": "rgba(53,184,154,.12)"}}, {"xAxis": item.sample}])
                start = None
    if start is not None:
        spans.append(
            [{"xAxis": start, "itemStyle": {"color": "rgba(53,184,154,.12)"}}, {"xAxis": snapshot.current_sample}]
        )
    return spans


def _candidate_spans(snapshot: SessionSnapshot) -> list[list[dict[str, object]]]:
    spans: list[list[dict[str, object]]] = []
    start: int | None = None
    for item in snapshot.diagnostics:
        if isinstance(item, WakeCandidateStartedDiagnostic):
            start = item.start_sample
        elif isinstance(item, WakeCandidateEndedDiagnostic) and start is not None:
            spans.append(
                [{"xAxis": start, "itemStyle": {"color": "rgba(191,114,233,.11)"}}, {"xAxis": item.end_sample}]
            )
            start = None
    if start is not None:
        spans.append(
            [{"xAxis": start, "itemStyle": {"color": "rgba(191,114,233,.11)"}}, {"xAxis": snapshot.current_sample}]
        )
    return spans


def _activation_spans(snapshot: SessionSnapshot) -> list[list[dict[str, object]]]:
    spans: list[list[dict[str, object]]] = []
    start: int | None = None
    for item in snapshot.diagnostics:
        if isinstance(item, ActivationDiagnostic):
            start = item.activation_start_sample
        elif isinstance(item, RearmDiagnostic) and start is not None:
            spans.append(
                [{"xAxis": start, "itemStyle": {"color": "rgba(238,190,72,.10)"}}, {"xAxis": item.effective_sample}]
            )
            start = None
    if start is not None:
        spans.append(
            [{"xAxis": start, "itemStyle": {"color": "rgba(238,190,72,.10)"}}, {"xAxis": snapshot.current_sample}]
        )
    return spans
