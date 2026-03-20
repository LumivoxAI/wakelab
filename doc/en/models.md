# Models And Artifacts

## Separation Of Responsibilities

Wakelab ships inference adapters but no model artifacts. Shared upstream artifacts
have code-owned manifests. The wake-word classifier remains a caller-owned trusted
input. Model acquisition, verification, ONNX session creation, and warm-up are
explicit setup operations outside streaming.

## Silero VAD

The VAD adapter pins the standard Silero VAD v6.2.1 ONNX model:

- cache path: `<model-root>/silero-vad/v6.2.1/silero_vad.onnx`;
- native frame: 512 samples (32 ms at 16 kHz);
- execution provider: ONNX Runtime CPU;
- model SHA-256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.

```python
from lumivox_wakelab.vad import SileroVad, ensure_silero_vad_model

model_path = ensure_silero_vad_model("model-data", logger=logger)
vad = SileroVad.load(model_path, logger=logger)
```

`load()` validates the graph and provider, warms inference, and resets recurrent
state. `infer()` only returns a speech probability; thresholds and temporal policy
belong to the stream configuration.

## openWakeWord

The wake backend pins the openWakeWord v0.5.1 mel-spectrogram and embedding feature
ABI:

- cache directory: `<model-root>/openwakeword-features/v0.5.1/`;
- shared files: `melspectrogram.onnx` and `embedding_model.onnx`;
- native frame: 1,280 samples (80 ms at 16 kHz);
- execution provider: ONNX Runtime CPU.

```python
from lumivox_wakelab.wakeword import OpenWakeWord, ensure_openwakeword_feature_models

feature_dir = ensure_openwakeword_feature_models("model-data", logger=logger)
wake = OpenWakeWord.load(feature_dir, "models/my-wake-word.onnx", logger=logger)
```

The caller's classifier must be a binary ONNX model with one float32 input shaped
`[1, N, 96]`, where `1 <= N <= 120`, and one float32 output shaped `[1, 1]`.
Wakelab validates this tensor contract but does not establish classifier provenance,
integrity, quality, threshold, temporal calibration, or license.

For caller-side SHA-256 verification, read and verify the classifier once, then pass
the same bytes to `load()`:

```python
import hashlib

classifier_path = "models/my-wake-word.onnx"
classifier_bytes = open(classifier_path, "rb").read()
actual = hashlib.sha256(classifier_bytes).hexdigest()
if actual != expected_sha256:
    raise RuntimeError("wake-word classifier digest mismatch")

wake = OpenWakeWord.load(
    feature_dir,
    classifier_path,
    logger=logger,
    classifier_bytes=classifier_bytes,
)
```

`load()` snapshots all models, validates schemas and providers, computes deterministic
silence history, warms the complete pipeline, and resets it. `reset()` restores that
history without ONNX inference.

## Artifact Cache Integrity

The `ensure_*` functions validate actual file bytes against a hard-coded expected
size and SHA-256 on every use. A sibling `.sha256` file is cache metadata, not a
trust source. Downloads are staged, verified, flushed, and atomically published.
An invalid or partial download does not replace a valid artifact.

Acquisition uses a cooperating process lock on Linux. It makes one bounded attempt;
applications own retry scheduling.

After online setup, request offline reuse explicitly:

```python
vad_path = ensure_silero_vad_model(
    model_root,
    logger=logger,
    allow_download=False,
)
feature_dir = ensure_openwakeword_feature_models(
    model_root,
    logger=logger,
    allow_download=False,
)
```

Offline mode prevents downloads. Cache metadata repair and lock-file operations may
still write below the model root.

## Licensing

Silero's pinned model is distributed by its authors under MIT. The narrow
openWakeWord inference implementation is under Apache-2.0. Upstream bundled
openWakeWord classifiers use CC BY-NC-SA 4.0 and are intentionally not downloaded or
redistributed by Wakelab. Review [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)
and the license of every classifier supplied by the application.
