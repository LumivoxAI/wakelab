"""Internal VAD backend contract used by the streaming orchestrator."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class VadBackend(Protocol):
    @property
    def sample_rate(self) -> int: ...

    @property
    def frame_samples(self) -> int: ...

    def infer(self, frame: NDArray[np.int16]) -> float: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...
