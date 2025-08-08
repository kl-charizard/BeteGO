from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .device import get_device, seed_everything
from .model import PolicyValueNet, ModelConfig
from .mcts import MCTS, MCTSConfig
from .self_play import SelfPlayConfig, play_game
from .replay_buffer import ReplayBuffer
from .utils import POLICY_SIZE


@dataclass
class TrainingConfig:
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 2
    replay_buffer_size: int = 50000
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    checkpoint_dir: str = "checkpoints"
    save_every_iterations: int = 1


class Trainer:
    def __init__(
        self,
        device: torch.device,
        model_cfg: ModelConfig,
        sp_cfg: SelfPlayConfig,
        mcts_cfg: MCTSConfig,
        train_cfg: TrainingConfig,
        seed: Optional[int] = None,
    ) -> None:
        seed_everything(seed)
        self.device = device
        self.model = PolicyValueNet(model_cfg).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        self.train_cfg = train_cfg
        self.sp_cfg = sp_cfg
        self.mcts_cfg = mcts_cfg
        self.buffer = ReplayBuffer(train_cfg.replay_buffer_size)
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
        self.latest_ckpt = os.path.join(train_cfg.checkpoint_dir, "latest.pt")

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.latest_ckpt
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"]) 
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"]) 

    def iteration(self, it: int) -> None:
        # Self-play to generate games
        mcts = MCTS(self.model, self.device, self.mcts_cfg)
        num_games = self.sp_cfg.games_per_iteration
        new_samples = 0
        for g in range(num_games):
            traj, z_white = play_game(
                mcts,
                resign_threshold=self.sp_cfg.resign_threshold,
                temperature_initial=self.sp_cfg.temperature_initial,
                temperature_moves=self.sp_cfg.temperature_moves,
            )
            # Convert trajectory to samples with final outcome perspective
            for planes, pi, player in traj:
                z = z_white * player  # if player was black, invert
                self.buffer.add(planes, pi.astype(np.float32), float(z))
                new_samples += 1

        # Training epochs
        if len(self.buffer) == 0:
            return
        self.model.train()
        pbar = tqdm(range(self.train_cfg.epochs), desc=f"Train it{it}")
        for _ in pbar:
            x, pi, z = self.buffer.sample_batch(self.train_cfg.batch_size)
            x_t = torch.from_numpy(x).to(self.device)
            pi_t = torch.from_numpy(pi).to(self.device)
            z_t = torch.from_numpy(z).to(self.device)

            p_logits, v = self.model(x_t)
            policy_loss = nn.CrossEntropyLoss()(p_logits, torch.argmax(pi_t, dim=1))
            value_loss = nn.MSELoss()(v, z_t)
            loss = self.train_cfg.policy_loss_weight * policy_loss + self.train_cfg.value_loss_weight * value_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            pbar.set_postfix({
                "loss": float(loss.item()),
                "policy": float(policy_loss.item()),
                "value": float(value_loss.item()),
                "buf": len(self.buffer),
                "new": new_samples,
            })

    def fit(self, iterations: int) -> None:
        for it in range(1, iterations + 1):
            self.iteration(it)
            if it % self.train_cfg.save_every_iterations == 0:
                self.save() 