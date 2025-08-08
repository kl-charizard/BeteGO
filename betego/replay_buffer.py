from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class Sample:
    planes: np.ndarray  # [C, 8, 8]
    policy: np.ndarray  # [4672]
    value: float        # z in [-1, 0, 1]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: Deque[Sample] = deque(maxlen=capacity)

    def add(self, planes: np.ndarray, policy: np.ndarray, value: float) -> None:
        self.data.append(Sample(planes, policy, float(value)))

    def extend(self, samples: List[Sample]) -> None:
        for s in samples:
            self.data.append(s)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        x = np.stack([b.planes for b in batch], axis=0)
        pi = np.stack([b.policy for b in batch], axis=0)
        z = np.array([b.value for b in batch], dtype=np.float32)
        return x, pi, z

    def __len__(self) -> int:
        return len(self.data) 