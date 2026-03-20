# Development

## Repository Setup

Wakelab uses `uv` for environments, dependency resolution, and builds, `just` for
workflows, Ruff for formatting/linting, mypy in strict mode, and pytest.

```console
git clone https://github.com/LumivoxAI/wakelab.git
cd wakelab
just postclone
just precommit
```

Useful recipes:

| Command | Purpose |
| --- | --- |
| `just sync` | Synchronize normal and development dependencies. |
| `just fmt` | Format the repository. |
| `just lint` | Run Ruff lint checks. |
| `just typecheck` | Run strict mypy checks. |
| `just test` | Run the default test suite. |
| `just lock_check` | Verify that `uv.lock` matches project metadata. |
| `just build` | Build source and wheel artifacts without local source overrides. |
| `just precommit` | Run formatting, lint, typing, tests, and lock checks. |

The default pytest configuration excludes tests marked `model`. Real openWakeWord
tests additionally need a compatible classifier configured by the test environment.

## Architecture

The production pipeline is:

```text
InputChunk validation and continuity handling
  -> bounded sample timeline
  -> native VAD framing and probability inference
  -> VAD hysteresis, debounce, and padding policy
  -> VAD-gated native wake-word framing and inference
  -> activation lifecycle
  -> OutputChunk segmentation and metadata
```

Inference adapters produce probabilities. Temporal policies decide speech,
candidates, and activation. The processor alone owns pending PCM and decides when
metadata is final enough to emit it.

Logical positions are processor-lifetime sample counts. Internal and diagnostic
ranges are half-open `[start, end)`. This prevents timestamp drift and ring-buffer
wraparound from affecting order or conservation.

## Required Invariants

- Preserve each accepted sample exactly once and in order.
- Never cross discontinuity or generation boundaries with model frames.
- Keep VAD and activation metadata independent.
- Keep activation latched until explicit re-arm.
- Skip wake inference while activated.
- Keep network access, downloads, session creation, and warm-up outside the hot path.
- Keep model-native sizes and tensor names out of the public input contract.
- Keep stateful processing synchronous, serial, and bounded.
- Never import production code from `src/tmp`.

`src/tmp` is read-only research material. Do not edit it, package it, import it, or
preserve compatibility with its prototype APIs.

## Testing Changes

Tests should prefer deterministic fake backends and recognizable synthetic sample
sequences. Cover different input chunk partitions for the same audio. Verify emitted
ranges are ordered, non-overlapping, gap-free, and concatenate to the original input.

Backend changes also require reset determinism, transactional inference failure,
provider validation, schema validation, and close idempotency. Artifact changes need
cache, corruption, digest mismatch, interrupted download, and atomic publication
coverage without requiring network access in the default suite.

Run `just precommit` after implementation changes. For public exports or package
layout changes, also run `just build` and inspect artifacts for caches, models,
prototype files, and unsupported content.

## Public API Rules

Supported imports are re-exported from `lumivox_wakelab`, `lumivox_wakelab.vad`, or
`lumivox_wakelab.wakeword`. Modules and names beginning with an underscore are
internal and may change without compatibility support. Repository tools and profile
helpers under `tools/` and `examples/` are not package APIs.

The package receives `lumivox_core.logger.Logger` instances from applications and
binds Wakelab context. Library code must not configure global logging and must not log
normal per-frame events at ordinary levels.
