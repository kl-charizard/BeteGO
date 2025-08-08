from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass

import chess
import torch

from betego.device import get_device
from betego.model import PolicyValueNet, ModelConfig
from betego.mcts import MCTS, MCTSConfig
from betego.utils import policy_index_from_move


PIECE_UNICODE = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
}

LIGHT_COLOR = '#EEEED2'
DARK_COLOR  = '#769656'
HIGHLIGHT_COLOR = '#BACA44'
MOVE_COLOR = '#CC0000'


@dataclass
class GUIConfig:
    square_size: int = 72
    margin: int = 16
    sims: int = 256
    device: str = 'auto'
    model_path: str = 'checkpoints/latest.pt'
    play_as_white: bool = True


class BeteGoGUI:
    def __init__(self, root: tk.Tk, cfg: GUIConfig) -> None:
        self.root = root
        self.cfg = cfg

        self.device = get_device(cfg.device)
        
        # Build UI first so status_var is available for status updates during model load
        self.selected_square: int | None = None
        self.legal_targets_from_selected: set[int] = set()
        self.lock = threading.Lock()
        self.engine_thinking = False
        self.orientation_white_bottom = True
        self.board = chess.Board()
        self._build_ui()

        # Now load model and set up MCTS
        self.model = self._load_model(cfg.model_path)
        self.mcts = MCTS(self.model, self.device, MCTSConfig(num_simulations=cfg.sims))

        self._draw_board()
        self._maybe_engine_move()

    def _load_model(self, path: str) -> PolicyValueNet:
        net = PolicyValueNet(ModelConfig()).to(self.device)
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device)
                net.load_state_dict(ckpt["model"])
                net.eval()
                self._set_status(f"Model loaded: {path}")
            except Exception as e:
                self._set_status(f"Failed to load model at {path}; using random weights. {e}")
        else:
            self._set_status("No checkpoint found; using random weights")
        return net

    def _build_ui(self) -> None:
        self.root.title("BeteGo - GUI")
        total = self.cfg.square_size * 8

        self.canvas = tk.Canvas(self.root, width=total, height=total, bg='white', highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.bind('<Button-1>', self._on_canvas_click)

        # Controls frame
        ctrl = tk.Frame(self.root)
        ctrl.grid(row=1, column=0, sticky='ew')

        self.side_var = tk.StringVar(value='White' if self.cfg.play_as_white else 'Black')
        tk.Label(ctrl, text='Human plays:').pack(side='left', padx=4)
        tk.OptionMenu(ctrl, self.side_var, 'White', 'Black', command=lambda _: self._on_side_change()).pack(side='left')

        tk.Button(ctrl, text='New Game', command=self._new_game).pack(side='left', padx=8)
        tk.Button(ctrl, text='Flip Board', command=self._flip_board).pack(side='left', padx=4)
        tk.Button(ctrl, text='Load Checkpoint', command=self._choose_checkpoint).pack(side='left', padx=4)

        tk.Label(ctrl, text='Sims:').pack(side='left', padx=(16, 2))
        self.sims_var = tk.IntVar(value=self.cfg.sims)
        self.sims_spin = tk.Spinbox(ctrl, from_=16, to=4096, increment=16, textvariable=self.sims_var, width=6, command=self._update_sims)
        self.sims_spin.pack(side='left')

        tk.Label(ctrl, text='Device:').pack(side='left', padx=(16, 2))
        self.device_var = tk.StringVar(value=str(self.device))
        tk.Label(ctrl, textvariable=self.device_var).pack(side='left')

        self.status_var = tk.StringVar(value='Ready')
        status = tk.Label(self.root, textvariable=self.status_var, anchor='w')
        status.grid(row=2, column=0, sticky='ew', padx=4, pady=4)

        # Configure resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _update_sims(self) -> None:
        sims = int(self.sims_var.get())
        self.mcts = MCTS(self.model, self.device, MCTSConfig(num_simulations=sims))
        self._set_status(f"Updated sims to {sims}")

    def _on_side_change(self) -> None:
        self.cfg.play_as_white = (self.side_var.get() == 'White')
        self._set_status(f"Human plays {self.side_var.get()}")
        self._maybe_engine_move()

    def _choose_checkpoint(self) -> None:
        path = filedialog.askopenfilename(title='Select checkpoint', filetypes=[('PyTorch', '*.pt *.pth'), ('All', '*.*')])
        if path:
            self.cfg.model_path = path
            self.model = self._load_model(path)
            self._set_status(f"Loaded checkpoint: {os.path.basename(path)}")

    def _new_game(self) -> None:
        with self.lock:
            self.board = chess.Board()
            self.selected_square = None
            self.legal_targets_from_selected.clear()
        self._draw_board()
        self._maybe_engine_move()

    def _flip_board(self) -> None:
        self.orientation_white_bottom = not self.orientation_white_bottom
        self._draw_board()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _on_canvas_click(self, event) -> None:
        try:
            if self.engine_thinking:
                self._set_status("Please wait: engine is thinking...")
                return
            with self.lock:
                if self.board.is_game_over():
                    self._show_result()
                    return
                human_white = self.cfg.play_as_white
                if self.board.turn != human_white:
                    self._set_status("Not your turn yet")
                    return

                sq = self._coords_to_square(event.x, event.y)
                if sq is None:
                    return

                piece = self.board.piece_at(sq)
                if self.selected_square is None:
                    # Select source if it's own piece
                    if piece is not None and piece.color == human_white:
                        self.selected_square = sq
                        self.legal_targets_from_selected = self._legal_targets_from(sq)
                        self._set_status("Select a destination square")
                        self._draw_board()
                    else:
                        self._set_status("Select one of your pieces")
                else:
                    # Try to move
                    if sq == self.selected_square:
                        # Deselect
                        self.selected_square = None
                        self.legal_targets_from_selected.clear()
                        self._set_status("Selection cleared")
                        self._draw_board()
                        return

                    # Build a candidate move
                    src_piece = self.board.piece_at(self.selected_square)
                    mv = self._find_move(self.selected_square, sq)

                    # Handle promotion robustly: if a pawn moves to last rank, ask
                    if src_piece and src_piece.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                        promo = self._ask_promotion(self.board.turn)
                        if promo is None:
                            self._set_status("Promotion cancelled")
                            return
                        mv_candidate = chess.Move(self.selected_square, sq, promotion=promo)
                        if mv_candidate in self.board.legal_moves:
                            mv = mv_candidate
                        else:
                            # Fallback to any legal (should not happen)
                            self._set_status("Chosen promotion not legal; using default")

                    if mv is not None and mv in self.board.legal_moves:
                        self.board.push(mv)
                        self.selected_square = None
                        self.legal_targets_from_selected.clear()
                        self._draw_board()
                        self._set_status("Engine's turn")

                        # Engine move if game not over
                        self._maybe_engine_move()
                    else:
                        # If clicked on own piece instead, switch selection
                        if piece is not None and piece.color == human_white:
                            self.selected_square = sq
                            self.legal_targets_from_selected = self._legal_targets_from(sq)
                            self._set_status("Select a destination square")
                            self._draw_board()
                        else:
                            self._set_status("Illegal move; try another square")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _ask_promotion(self, white_to_move: bool) -> int | None:
        # Simple popup for promotion piece
        top = tk.Toplevel(self.root)
        top.title('Promotion')
        choice = {'val': None}

        def set_choice(p):
            choice['val'] = p
            top.destroy()

        pieces = [('Queen', chess.QUEEN), ('Rook', chess.ROOK), ('Bishop', chess.BISHOP), ('Knight', chess.KNIGHT)]
        for name, p in pieces:
            tk.Button(top, text=name, width=10, command=lambda pp=p: set_choice(pp)).pack(padx=10, pady=5)
        top.transient(self.root)
        top.grab_set()
        self.root.wait_window(top)
        return choice['val']

    def _legal_targets_from(self, src: int) -> set[int]:
        targets = set()
        for mv in self.board.legal_moves:
            if mv.from_square == src:
                targets.add(mv.to_square)
        return targets

    def _find_move(self, src: int, dst: int) -> chess.Move | None:
        for mv in self.board.legal_moves:
            if mv.from_square == src and mv.to_square == dst:
                return mv
        return None

    def _maybe_engine_move(self) -> None:
        with self.lock:
            if self.board.is_game_over():
                self._show_result()
                return
            human_white = self.cfg.play_as_white
            if self.board.turn == human_white:
                return
            if self.engine_thinking:
                return
            self.engine_thinking = True

        self._set_status("Engine thinking...")
        t = threading.Thread(target=self._engine_move_thread, daemon=True)
        t.start()

    def _engine_move_thread(self) -> None:
        try:
            with self.lock:
                board_copy = self.board.copy()
            visits, _ = self.mcts.run(board_copy)
            action = int(visits.argmax())
            chosen = None
            for mv in board_copy.legal_moves:
                if policy_index_from_move(board_copy, mv) == action:
                    chosen = mv
                    break
            if chosen is None:
                # Fallback: pick the most visited legal if mapping failed, else first legal
                chosen = next(iter(board_copy.legal_moves))
        except Exception as e:
            self.root.after(0, lambda: self._set_status(f"Engine error: {e}"))
            with self.lock:
                self.engine_thinking = False
            return

        def apply_move():
            with self.lock:
                if chosen in self.board.legal_moves:
                    self.board.push(chosen)
                self.engine_thinking = False
            self._draw_board()
            if self.board.is_game_over():
                self._show_result()
            else:
                self._set_status("Your move")

        self.root.after(0, apply_move)

    def _show_result(self) -> None:
        outcome = self.board.outcome()
        if outcome is None or outcome.winner is None:
            msg = f"Draw ({outcome.termination.name if outcome else 'unknown'})"
        else:
            msg = f"{'White' if outcome.winner else 'Black'} wins ({outcome.termination.name})"
        self._set_status(f"Game over: {msg}")
        messagebox.showinfo("Game Over", msg)

    def _coords_to_square(self, x: int, y: int) -> int | None:
        s = self.cfg.square_size
        file_idx = x // s
        rank_idx = y // s
        if file_idx < 0 or file_idx > 7 or rank_idx < 0 or rank_idx > 7:
            return None
        if self.orientation_white_bottom:
            file = file_idx
            rank = 7 - rank_idx
        else:
            file = 7 - file_idx
            rank = rank_idx
        return chess.square(file, rank)

    def _square_to_coords(self, square: int) -> tuple[int, int]:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        if self.orientation_white_bottom:
            file_idx = file
            rank_idx = 7 - rank
        else:
            file_idx = 7 - file
            rank_idx = rank
        return file_idx, rank_idx

    def _draw_board(self) -> None:
        self.canvas.delete('all')
        s = self.cfg.square_size
        for r in range(8):
            for f in range(8):
                x0 = f * s
                y0 = r * s
                color = LIGHT_COLOR if (r + f) % 2 == 0 else DARK_COLOR
                self.canvas.create_rectangle(x0, y0, x0 + s, y0 + s, fill=color, outline=color)

        # Highlights
        if self.selected_square is not None:
            fidx, ridx = self._square_to_coords(self.selected_square)
            x0 = fidx * s
            y0 = ridx * s
            self.canvas.create_rectangle(x0, y0, x0 + s, y0 + s, outline=HIGHLIGHT_COLOR, width=3)
            for tgt in self.legal_targets_from_selected:
                tf, tr = self._square_to_coords(tgt)
                cx = tf * s + s // 2
                cy = tr * s + s // 2
                self.canvas.create_oval(cx - 8, cy - 8, cx + 8, cy + 8, fill=HIGHLIGHT_COLOR, outline='')

        # Pieces
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece is None:
                continue
            fidx, ridx = self._square_to_coords(sq)
            x = fidx * s + s // 2
            y = ridx * s + s // 2
            char = PIECE_UNICODE.get(piece.symbol(), '?')
            fill = '#000000' if piece.color == chess.BLACK else '#000000'
            self.canvas.create_text(x, y, text=char, font=('Helvetica', s // 2), fill=fill)

        # Files/ranks labels
        for f in range(8):
            file_char = 'abcdefgh'[f]
            if not self.orientation_white_bottom:
                file_char = 'abcdefgh'[7 - f]
            x = f * s + 4
            y = 8 * s - 14
            self.canvas.create_text(x, y, text=file_char, fill='#333333', anchor='w', font=('Helvetica', 10))
        for r in range(8):
            rank_char = '12345678'[r]
            if self.orientation_white_bottom:
                rank_char = '12345678'[r]
            else:
                rank_char = '12345678'[7 - r]
            x = 8 * s - 12
            y = r * s + 2
            self.canvas.create_text(x, y, text=rank_char, fill='#333333', anchor='e', font=('Helvetica', 10))


def main() -> None:
    root = tk.Tk()
    gui = BeteGoGUI(root, GUIConfig())
    root.mainloop()


if __name__ == '__main__':
    main() 