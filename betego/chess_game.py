from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import chess


@dataclass
class GameResult:
    outcome: Optional[float]  # 1.0 white win, 0.0 draw, -1.0 black win, None if ongoing
    reason: str


class ChessGame:
    def __init__(self) -> None:
        self.board = chess.Board()

    def legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    def push(self, move: chess.Move) -> None:
        self.board.push(move)

    def pop(self) -> chess.Move:
        return self.board.pop()

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def result(self) -> GameResult:
        if not self.board.is_game_over():
            return GameResult(None, "ongoing")
        outcome = self.board.outcome()
        if outcome is None:
            return GameResult(0.0, "draw (unknown)")
        if outcome.winner is None:
            return GameResult(0.0, f"draw ({outcome.termination.name.lower()})")
        return GameResult(1.0 if outcome.winner else -1.0, outcome.termination.name.lower())

    def turn(self) -> bool:
        return self.board.turn

    def copy(self) -> "ChessGame":
        g = ChessGame()
        g.board = self.board.copy(stack=True)
        return g

    def fen(self) -> str:
        return self.board.fen() 