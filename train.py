from __future__ import annotations

import argparse
import os

import torch
import yaml

from betego.device import get_device
from betego.model import ModelConfig
from betego.mcts import MCTSConfig
from betego.self_play import SelfPlayConfig
from betego.trainer import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="BeteGo Training")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--device", type=str, default="auto", help="auto/mps/cuda/cpu")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--resume", type=str, default=None)
    return ap.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    device = get_device(args.device if args.device != "auto" else cfg.get("device", "auto"))
    print(f"[BeteGo] Using device: {device}")

    seed = cfg.get("seed", 42)

    model_cfg = ModelConfig(**cfg.get("network", {}))

    sp_dict = cfg.get("self_play", {})
    mcts_keys = ["num_simulations", "cpuct", "dirichlet_alpha", "dirichlet_epsilon"]
    mcts_dict = {k: sp_dict[k] for k in mcts_keys if k in sp_dict}

    mcts_cfg = MCTSConfig(**mcts_dict)
    sp_cfg = SelfPlayConfig(**sp_dict)
    train_cfg = TrainingConfig(**cfg.get("training", {}))

    trainer = Trainer(
        device=device,
        model_cfg=model_cfg,
        sp_cfg=sp_cfg,
        mcts_cfg=mcts_cfg,
        train_cfg=train_cfg,
        seed=seed,
    )

    if args.resume:
        print(f"[BeteGo] Resuming from {args.resume}")
        trainer.load(args.resume)

    trainer.fit(args.iterations)


if __name__ == "__main__":
    main() 