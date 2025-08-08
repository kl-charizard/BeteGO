from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import chess

from .utils import POLICY_SIZE, policy_index_from_move, legal_policy_mask


@dataclass
class MCTSConfig:
    num_simulations: int = 128
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.30
    dirichlet_epsilon: float = 0.25


class MCTSNode:
    def __init__(self, board: chess.Board) -> None:
        self.board = board
        self.is_expanded = False
        self.P: Dict[int, float] = {}
        self.N: Dict[int, int] = {}
        self.W: Dict[int, float] = {}
        self.Q: Dict[int, float] = {}
        self.children: Dict[int, "MCTSNode"] = {}
        self.terminal_value: Optional[float] = None  # value from current player's perspective

    def is_terminal(self) -> bool:
        return self.board.is_game_over()

    def expand(self, policy_logits: np.ndarray) -> None:
        assert not self.is_expanded
        legal_mask = legal_policy_mask(self.board)
        # softmax over legal moves
        logits = policy_logits.copy()
        logits[legal_mask < 0.5] = -1e9
        logits = logits - logits.max()
        exp = np.exp(logits)
        exp *= legal_mask
        s = exp.sum()
        if s <= 1e-8:
            P = legal_mask / max(1, legal_mask.sum())
        else:
            P = exp / s
        legal_indices = np.where(legal_mask > 0.5)[0]
        for a in legal_indices:
            self.P[a] = float(P[a])
            self.N[a] = 0
            self.W[a] = 0.0
            self.Q[a] = 0.0
        self.is_expanded = True

    def add_dirichlet_noise(self, alpha: float, eps: float) -> None:
        if not self.is_expanded or len(self.P) == 0:
            return
        actions = list(self.P.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for a, n in zip(actions, noise):
            self.P[a] = (1 - eps) * self.P[a] + eps * float(n)


class MCTS:
    def __init__(self, net, device: torch.device, cfg: MCTSConfig):
        self.net = net
        self.device = device
        self.cfg = cfg

    def run(self, root_board: chess.Board) -> Tuple[np.ndarray, List[int]]:
        root = MCTSNode(root_board.copy())

        # If terminal immediately, return empty visits
        if root.is_terminal():
            visits = np.zeros(POLICY_SIZE, dtype=np.float32)
            return visits, []

        # Fast path: if only one legal move, skip simulations
        if sum(1 for _ in root.board.legal_moves) == 1:
            visits = np.zeros(POLICY_SIZE, dtype=np.float32)
            only_move = next(iter(root.board.legal_moves))
            a = policy_index_from_move(root.board, only_move)
            visits[a] = 1.0
            return visits, [a]

        # Initial expansion
        p, v = self._infer(root.board)
        root.expand(p)
        root.add_dirichlet_noise(self.cfg.dirichlet_alpha, self.cfg.dirichlet_epsilon)

        for _ in range(self.cfg.num_simulations):
            node = root
            path: List[Tuple[MCTSNode, int]] = []

            # Selection
            while node.is_expanded and not node.is_terminal():
                a = self._select_action(node)
                path.append((node, a))
                if a in node.children:
                    node = node.children[a]
                else:
                    # Apply move to get next node
                    move = self._action_to_move(node.board, a)
                    next_board = node.board.copy()
                    next_board.push(move)
                    child = MCTSNode(next_board)
                    node.children[a] = child
                    node = child

            # Expansion or terminal
            if node.is_terminal():
                value = self._terminal_value(node.board)
                node.terminal_value = value
            else:
                p, v = self._infer(node.board)
                node.expand(p)
                value = float(v)

            # Backup with alternating perspective
            for parent, action in reversed(path):
                parent.N[action] += 1
                parent.W[action] += value
                parent.Q[action] = parent.W[action] / parent.N[action]
                value = -value

        # Collect visit counts at root
        visits = np.zeros(POLICY_SIZE, dtype=np.float32)
        for a, n in root.N.items():
            visits[a] = float(n)
        return visits, list(root.N.keys())

    def _infer(self, board: chess.Board) -> Tuple[np.ndarray, float]:
        from .utils import encode_board_planes

        planes = encode_board_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        p_logits, v = self.net.predict(x)
        p = p_logits.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return p, float(v.item())

    def _select_action(self, node: MCTSNode) -> int:
        # PUCT: argmax_a Q + U, where U = cpuct * P * sqrt(sumN) / (1 + n)
        total_N = sum(node.N.values()) + 1
        best_score = -1e9
        best_action = None
        sqrt_total = math.sqrt(total_N)
        for a in node.P.keys():
            n = node.N[a]
            q = node.Q[a]
            p = node.P[a]
            u = self.cfg.cpuct * p * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = a
        assert best_action is not None
        return best_action

    def _action_to_move(self, board: chess.Board, action: int) -> chess.Move:
        # Recover a legal move corresponding to the given action index
        for mv in board.legal_moves:
            if policy_index_from_move(board, mv) == action:
                return mv
        # Fallback: choose any legal move (should not happen)
        return next(iter(board.legal_moves))

    def _terminal_value(self, board: chess.Board) -> float:
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0.0
        # Value from side-to-move at terminal node: if terminal, side-to-move has just been checkmated or stalemated
        # However, board.is_game_over() means side to move has no legal moves; outcome.winner is the previous mover.
        return 1.0 if outcome.winner == board.turn else -1.0 