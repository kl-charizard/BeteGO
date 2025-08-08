from __future__ import annotations

from typing import Dict, List, Tuple

import chess
import numpy as np

# 64 squares * 73 moves per square = 4672 action space (AlphaZero-style)
NUM_SQUARES = 64
MOVES_PER_SQUARE = 73
POLICY_SIZE = NUM_SQUARES * MOVES_PER_SQUARE

# Directions for queen-like moves ordered as: N, E, S, W, NE, NW, SE, SW
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]

# Knight move deltas ordered clockwise starting from NNE
KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]

# Underpromotion order: [to R, to B, to N]
UNDERPROMO_PIECES = [chess.ROOK, chess.BISHOP, chess.KNIGHT]

# Forward direction mapping (from current-player/white perspective):
# 0 = straight (same file), 1 = capture-left (file -1), 2 = capture-right (file +1)
FORWARD_MAP = {0: 0, -1: 1, 1: 2}


def square_to_file_rank(square: int) -> Tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


def encode_board_planes(board: chess.Board) -> np.ndarray:
    """Encode board into 18 planes (12 piece planes + 4 castling + 1 stm + 1 move count/256).

    Canonical orientation: from the current player's perspective (white-to-move).
    If it's black to move, we mirror the board so the encoder always sees white.
    """
    b = board
    mirror = not b.turn

    if mirror:
        b = b.mirror()

    planes: List[np.ndarray] = []

    # 12 piece planes: white P,N,B,R,Q,K, black p,n,b,r,q,k (in canonical board, stm is white)
    piece_order = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]
    for color in [chess.WHITE, chess.BLACK]:
        for pt in piece_order:
            bb = b.pieces(pt, color)
            plane = np.zeros((8, 8), dtype=np.float32)
            for sq in bb:
                file, rank = square_to_file_rank(sq)
                plane[rank, file] = 1.0
            planes.append(plane)

    # Castling rights (Wk, Wq, Bk, Bq) on canonical board
    w_ks = 1.0 if b.has_kingside_castling_rights(chess.WHITE) else 0.0
    w_qs = 1.0 if b.has_queenside_castling_rights(chess.WHITE) else 0.0
    b_ks = 1.0 if b.has_kingside_castling_rights(chess.BLACK) else 0.0
    b_qs = 1.0 if b.has_queenside_castling_rights(chess.BLACK) else 0.0
    for v in [w_ks, w_qs, b_ks, b_qs]:
        planes.append(np.full((8, 8), v, dtype=np.float32))

    # Side to move (always 1 since canonicalized), keep for symmetry-breaking
    planes.append(np.ones((8, 8), dtype=np.float32))

    # Move count plane (scaled)
    move_count_scaled = min(b.fullmove_number, 256) / 256.0
    planes.append(np.full((8, 8), move_count_scaled, dtype=np.float32))

    stacked = np.stack(planes, axis=0)  # [C, 8, 8]
    return stacked


def policy_index_from_move(board: chess.Board, move: chess.Move) -> int:
    """Map a legal move to AlphaZero's 4672-index under canonical orientation.

    Steps:
    - Mirror when black to move so current player is white
    - Compute from-square base = s * 73
    - Sliding moves (56), knight moves (8), underpromotions (9)
    """
    b = board
    if not b.turn:
        # Mirror board and move to white perspective
        from_sq = chess.square_mirror(move.from_square)
        to_sq = chess.square_mirror(move.to_square)
        promo = move.promotion
    else:
        from_sq = move.from_square
        to_sq = move.to_square
        promo = move.promotion

    from_file, from_rank = square_to_file_rank(from_sq)
    to_file, to_rank = square_to_file_rank(to_sq)
    df = to_file - from_file
    dr = to_rank - from_rank

    base = from_sq * MOVES_PER_SQUARE

    # Knight
    if (df, dr) in KNIGHT_DELTAS:
        knight_idx = KNIGHT_DELTAS.index((df, dr))
        return base + 56 + knight_idx

    # Underpromotions
    if promo in UNDERPROMO_PIECES:
        promo_type_idx = UNDERPROMO_PIECES.index(promo)
        forward_dir = FORWARD_MAP.get(np.sign(df), None)
        if forward_dir is None:
            # Should not happen for valid promotions, but keep safe fallback
            forward_dir = 0
        return base + 64 + promo_type_idx * 3 + forward_dir

    # Sliding (includes queen promotions treated as straight move forward)
    if df == 0 and dr == 0:
        raise ValueError("Invalid move: zero displacement")

    # Normalize to a direction among DIRS and a step count
    step_df = np.sign(df)
    step_dr = np.sign(dr)

    # Ensure the direction is one of the 8 primary directions
    if (step_df, step_dr) not in DIRS:
        # Non-matching direction implies illegal (shouldn't occur)
        # For safety, map to first direction with magnitude 1
        step_df, step_dr = (np.sign(df), np.sign(dr))

    try:
        dir_idx = DIRS.index((step_df, step_dr))
    except ValueError:
        # Fallback: map to N
        dir_idx = 0

    steps = max(abs(df), abs(dr))
    if steps < 1 or steps > 7:
        steps = max(1, min(7, steps))
    return base + dir_idx * 7 + (steps - 1)


def legal_policy_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
    for mv in board.legal_moves:
        idx = policy_index_from_move(board, mv)
        mask[idx] = 1.0
    return mask


def softmax_masked(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = logits.copy()
    x[mask < 0.5] = -1e9
    if temperature != 1.0:
        x = x / max(1e-6, temperature)
    x = x - np.max(x)
    exp = np.exp(x)
    exp *= mask
    s = exp.sum()
    if s <= 0:
        # Uniform over legal moves
        legal = np.flatnonzero(mask > 0.5)
        if len(legal) == 0:
            return np.zeros_like(logits)
        out = np.zeros_like(logits)
        out[legal] = 1.0 / len(legal)
        return out
    return exp / s 