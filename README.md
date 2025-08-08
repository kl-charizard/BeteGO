# BeteGo (AlphaZero‑style Chess with MPS)

BeteGo is a minimal AlphaZero‑style self‑play reinforcement learning system for Chess, implemented in Python with PyTorch and Apple Silicon MPS acceleration. It includes a CLI engine and a simple Tkinter GUI for playing by clicking.

## Features
- AlphaZero‑style pipeline: self‑play (MCTS) + policy‑value network + training loop
- MPS/CUDA/CPU device auto‑selection
- 4672‑dim policy head (AlphaZero move encoding)
- CLI to train and play; Tkinter GUI for point‑and‑click play

## Requirements
- Python 3.10+
- macOS on Apple Silicon for MPS (or Linux/Windows with CUDA, or CPU)

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify MPS (Apple Silicon) support:
```bash
python - << 'PY'
import torch
print('torch', torch.__version__)
print('mps is_built:', torch.backends.mps.is_built())
print('mps is_available:', torch.backends.mps.is_available())
PY
```
If MPS is not available, BeteGo falls back to CUDA or CPU automatically.

## Quickstart
- Train a short iteration (self‑play + learning):
```bash
python train.py --config configs/default.yaml --iterations 1 --device auto
```
- Play in terminal against the latest checkpoint:
```bash
python play.py --model checkpoints/latest.pt --device auto --sims 256
```
- Launch the GUI (click‑to‑move):
```bash
python gui_play.py
```

## GUI Usage (Tkinter)
- Start: `python gui_play.py`
- Controls:
  - Human plays: choose White/Black
  - New Game: reset to initial position
  - Flip Board: toggle orientation
  - Load Checkpoint: select a `.pt/.pth` file (defaults to `checkpoints/latest.pt`)
  - Sims: adjusts MCTS simulations (higher = stronger but slower)
- Play by clicking a piece, then a destination square.
- Promotion prompts a small dialog (Queen/Rook/Bishop/Knight).

## Devices
- `--device auto|mps|cuda|cpu` (default: `auto`). Auto prefers MPS, then CUDA, else CPU.

## Configuration (`configs/default.yaml`)
- `device`: preferred device selection
- `seed`: reproducibility seed
- `self_play`:
  - `games_per_iteration`: number of games per training iteration
  - `num_simulations`: MCTS simulations per move
  - `cpuct`: exploration constant
  - `dirichlet_alpha`, `dirichlet_epsilon`: root Dirichlet noise
  - `temperature_initial`, `temperature_moves`: early‑game exploration
  - `resign_threshold`: set `null` to disable resigns
- `network`:
  - `channels`, `residual_blocks`: backbone size
  - `policy_head_channels`, `value_head_channels`: head sizes
- `training`:
  - `batch_size`, `lr`, `weight_decay`, `epochs`
  - `replay_buffer_size`: samples kept
  - `checkpoint_dir`: where checkpoints are saved
  - `save_every_iterations`: save frequency

## Training
- Multi‑iteration run:
```bash
python train.py --config configs/default.yaml --iterations 50 --device auto
```
- Resume from checkpoint:
```bash
python train.py --config configs/default.yaml --iterations 50 --resume checkpoints/latest.pt
```
Notes: each iteration runs self‑play, fills the replay buffer, and performs SGD updates. The latest model is saved to `training.checkpoint_dir` (default `checkpoints/latest.pt`).

## Playing vs BeteGo (CLI)
```bash
python play.py --model checkpoints/latest.pt --device auto --sims 256
```
- You play Black by entering UCI moves (e.g., `e2e4`, `e7e8q`).
- Increase `--sims` for stronger but slower play.

## Tips and Tuning
- Faster iteration:
  - Lower `self_play.num_simulations` (e.g., 16–64)
  - Lower `self_play.games_per_iteration`
  - Use a smaller network (`network.channels`, `residual_blocks`)
- Stronger play (slower):
  - Increase network size and MCTS simulations
  - Increase `training.epochs` and `replay_buffer_size`

## Troubleshooting
- MPS not available: ensure Apple Silicon and recent PyTorch (2.3+); otherwise use `--device cpu` or `--device cuda`.
- Install issues: `pip install --upgrade pip setuptools wheel` then reinstall.
- Slow performance: reduce `num_simulations` and network size.
- OOM/memory: reduce `batch_size`, `replay_buffer_size`, or network size.

## Project Structure
- `betego/` core library
  - `device.py` device selection (MPS/CUDA/CPU)
  - `model.py` policy‑value network
  - `mcts.py` Monte Carlo Tree Search with PUCT
  - `self_play.py` self‑play trajectory generation
  - `trainer.py` training loop and checkpointing
  - `utils.py` board encoding and 4672‑action mapping
  - `chess_game.py` thin wrapper over `python-chess`
- `train.py` CLI to run training
- `play.py` CLI to play in terminal
- `gui_play.py` Tkinter GUI for click‑to‑move play
- `configs/default.yaml` hyperparameters

## License
MIT 