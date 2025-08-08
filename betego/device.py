from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def get_device(prefer: str = "auto") -> torch.device:
    prefer = (prefer or "auto").lower()
    if prefer == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        print("[BeteGo] Requested MPS but not available; falling back to CPU")
        return torch.device("cpu")
    if prefer == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[BeteGo] Requested CUDA but not available; falling back to CPU")
        return torch.device("cpu")
    if prefer == "cpu":
        return torch.device("cpu")

    # auto
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed) 