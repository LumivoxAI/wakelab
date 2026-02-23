"""Internal wake-word backend contract used by the streaming orchestrator."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class WakeWordBackend(Protocol):
    """One-stream native-frame backend with transactional ``infer`` calls.

    An implementation must leave its stream state unchanged when ``infer`` raises.
    ``reset`` restores deterministic stream state and ``close`` is terminal and
    idempotent.
    """

    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_samples(self) -> int: ...

    def infer(self, frame: NDArray[np.int16]) -> float: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...
