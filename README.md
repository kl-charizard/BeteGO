# BeteGo (AlphaZero‑style Chess with MPS)

BeteGo is a minimal AlphaZero‑style self‑play reinforcement learning system for Chess, implemented in Python with PyTorch and Apple Silicon MPS acceleration.

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

Optional: verify MPS (Apple Silicon) support
```bash
python - << 'PY'
import torch
print('torch', torch.__version__)
print('mps is_built:', torch.backends.mps.is_built())
print('mps is_available:', torch.backends.mps.is_available())
PY
```
If MPS is not available, BeteGo will fall back to CUDA or CPU automatically.

## Quickstart

- Train for a single iteration (self‑play + learning):
```bash
python train.py --config configs/default.yaml --iterations 1 --device auto
```
- Play against the latest checkpoint:
```bash
python play.py --model checkpoints/latest.pt --device auto --sims 256
```

## Devices
- `--device auto|mps|cuda|cpu` (default: `auto`).
- Auto uses MPS if available, otherwise CUDA, otherwise CPU.

## Configuration
Edit `configs/default.yaml`. Key sections:
- `device`: preferred device selection.
- `seed`: reproducibility seed.
- `self_play`:
  - `games_per_iteration`: number of games to generate per training iteration.
  - `num_simulations`: MCTS simulations per move.
  - `cpuct`: PUCT exploration constant.
  - `dirichlet_alpha`, `dirichlet_epsilon`: root noise for exploration.
  - `temperature_initial`, `temperature_moves`: temperature schedule for early moves.
  - `resign_threshold`: set `null` to disable resigns.
- `network`:
  - `channels`, `residual_blocks`: backbone size (increase for stronger models).
  - `policy_head_channels`, `value_head_channels`: head sizes.
- `training`:
  - `batch_size`, `lr`, `weight_decay`, `epochs`.
  - `replay_buffer_size`: number of recent samples kept.
  - `checkpoint_dir`: where checkpoints are saved.
  - `save_every_iterations`: save frequency.

## Training
Run multiple iterations:
```bash
python train.py --config configs/default.yaml --iterations 50 --device auto
```
Resume from a checkpoint:
```bash
python train.py --config configs/default.yaml --iterations 50 --resume checkpoints/latest.pt
```
Notes:
- Each iteration generates self‑play games, adds them to the replay buffer, and runs SGD updates.
- Checkpoints are saved to `training.checkpoint_dir` (default `checkpoints/latest.pt`).

## Playing vs BeteGo
```bash
python play.py --model checkpoints/latest.pt --device auto --sims 256
```
- You play Black by entering UCI moves (e.g., `e2e4`, `e7e8q`).
- Increase `--sims` for stronger but slower play.

## Tips and Tuning
- For faster iteration:
  - Lower `self_play.num_simulations` (e.g., 16–64).
  - Lower `self_play.games_per_iteration`.
  - Use a smaller network (`network.channels`, `residual_blocks`).
- For stronger play (slower):
  - Increase network size and MCTS simulations.
  - Increase `training.epochs` and replay buffer size.

## Troubleshooting
- MPS not available:
  - Ensure you’re on Apple Silicon and using a recent PyTorch (2.3+).
  - Use `--device cpu` or `--device cuda` as needed.
- Torch install issues:
  - Upgrade pip/wheel/setuptools and reinstall: `pip install --upgrade pip setuptools wheel`.
- Slow performance:
  - Reduce `num_simulations` and network size; ensure release builds (no debug mode).
- OOM / memory errors:
  - Reduce `batch_size`, `replay_buffer_size`, or network size.

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
- `play.py` CLI to play against the engine
- `configs/default.yaml` hyperparameters

## Notes
- This is a lean reference implementation for learning and experimentation, not an optimized production engine.
- On modest hardware, prefer smaller networks and lower simulation counts. 