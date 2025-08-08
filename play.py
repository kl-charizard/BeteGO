from __future__ import annotations

import argparse

import chess
import torch

from betego.device import get_device
from betego.model import PolicyValueNet, ModelConfig
from betego.mcts import MCTS, MCTSConfig
from betego.utils import policy_index_from_move


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Play against BeteGo")
    ap.add_argument("--model", type=str, default="checkpoints/latest.pt")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--sims", type=int, default=256)
    return ap.parse_args()


def load_model(path: str, device: torch.device) -> PolicyValueNet:
    cfg = ModelConfig()
    net = PolicyValueNet(cfg).to(device)
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"[BeteGo] Using device: {device}")

    net = load_model(args.model, device)
    mcts = MCTS(net, device, MCTSConfig(num_simulations=args.sims))

    board = chess.Board()
    print(board)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # Engine plays White
            visits, _ = mcts.run(board)
            action = int(visits.argmax())
            chosen = None
            for mv in board.legal_moves:
                if policy_index_from_move(board, mv) == action:
                    chosen = mv
                    break
            if chosen is None:
                chosen = next(iter(board.legal_moves))
            print(f"BeteGo plays: {chosen.uci()}")
            board.push(chosen)
        else:
            # Human plays Black
            move_str = input("Your move (uci): ")
            try:
                mv = chess.Move.from_uci(move_str)
                if mv not in board.legal_moves:
                    print("Illegal move. Try again.")
                    continue
                board.push(mv)
            except Exception:
                print("Invalid input. Use UCI like e2e4, e7e8q.")
                continue
        print(board)

    print("Game over:", board.result())


if __name__ == "__main__":
    main() 