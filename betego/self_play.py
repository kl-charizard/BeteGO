from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import chess

from .utils import encode_board_planes, policy_index_from_move, POLICY_SIZE
from .mcts import MCTS, MCTSConfig


@dataclass
class SelfPlayConfig:
    games_per_iteration: int = 4
    num_simulations: int = 128
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_initial: float = 1.0
    temperature_moves: int = 20
    resign_threshold: float = -0.95  # set to None to disable resigns


def select_action_from_visits(visits: np.ndarray, temperature: float) -> int:
    if temperature <= 1e-8:
        return int(visits.argmax())
    probs = visits.astype(np.float64) ** (1.0 / max(1e-8, temperature))
    s = probs.sum()
    if s <= 0:
        legal = np.flatnonzero(visits > 0)
        if len(legal) == 0:
            return 0
        probs = np.zeros_like(visits)
        probs[legal] = 1.0 / len(legal)
    else:
        probs = probs / s
    return int(np.random.choice(np.arange(len(visits)), p=probs))


def play_game(mcts: MCTS, resign_threshold: float | None, temperature_initial: float, temperature_moves: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, int]], float]:
    board = chess.Board()
    trajectory: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (planes, pi, player)

    move_count = 0
    while not board.is_game_over():
        visits, legal_actions = mcts.run(board)

        # Temperature schedule
        tau = temperature_initial if move_count < temperature_moves else 1e-8

        # Normalized visit distribution over policy space
        if visits.sum() > 0:
            pi = visits / visits.sum()
        else:
            # Uniform over legals
            pi = np.zeros_like(visits)
            if len(legal_actions) > 0:
                pi[legal_actions] = 1.0 / len(legal_actions)

        action = select_action_from_visits(visits, tau)

        # Record sample
        planes = encode_board_planes(board)
        current_player = 1 if board.turn == chess.WHITE else -1
        trajectory.append((planes, pi, current_player))

        # Resign check using MCTS root value proxy (approx via visit-weighted Q)
        if resign_threshold is not None and len(legal_actions) > 0:
            # crude proxy: if best Q over root < threshold
            # Note: MCTS class doesn't expose Q; for simplicity, approximate with value from net
            pass

        # Apply selected move
        # Map action -> actual move
        legal_mv = None
        for mv in board.legal_moves:
            if policy_index_from_move(board, mv) == action:
                legal_mv = mv
                break
        if legal_mv is None:
            legal_mv = next(iter(board.legal_moves))
        board.push(legal_mv)
        move_count += 1

        if move_count > 512:  # safety
            break

    # Determine game outcome from White perspective
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        z_white = 0.0
    else:
        z_white = 1.0 if outcome.winner else -1.0

    return trajectory, z_white 