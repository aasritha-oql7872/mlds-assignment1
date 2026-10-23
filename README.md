# MLDS Assignment 1: Dynamic Programming and Policy Gradients

Shuttle-dispatch MDP solved by enumeration (backward induction), value iteration and policy
iteration, plus REINFORCE with and without a learned value baseline on CartPole-v1 and Pong-v5.

Assignment part names: **Part 2** = shuttle MDP, **Policy Gradient Part 1** = plain REINFORCE,
**Policy Gradient Part 2** = REINFORCE with a baseline. Results directories are named
`pong_part1`, `pong_part2`, ... after the policy-gradient parts.

## Setup

Python 3.10 or 3.11.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Sanity check (Pong needs the ALE package, which bundles the ROMs; do not also install `gym[atari]`):

```bash
python -c "import gymnasium as gym, ale_py; gym.register_envs(ale_py); gym.make('ALE/Pong-v5').reset(seed=0); print('ok')"
```

## Reproducing every result

All commands run from the repository root. Outputs go to `results/`.

| Part | Command | Time | Outputs |
|---|---|---|---|
| 2 (a, b, c) | `python dp/run_dp.py` | seconds | `results/dp/*.png`, `summary.json` |
| 2 validation | `python dp/validate_dp.py` | seconds | `results/dp/validation.txt`, `validation_*.png` |
| PG Part 1, CartPole | `python pg/train.py --game cartpole` | minutes | `results/pg/cartpole_part1/` |
| PG Part 2, CartPole (baseline) | `python pg/train.py --game cartpole --baseline` | minutes | `results/pg/cartpole_part2/` |
| PG Part 1, Pong | `python pg/train.py --game pong` | hours | `results/pg/pong_part1/` |
| PG Part 2, Pong (baseline) | `python pg/train.py --game pong --baseline` | hours | `results/pg/pong_part2/` |
| PG validation | `python pg/validate_pg.py` (or `--quick`) | about 30 min (5 min) | `results/pg/validation/` |
| Re-plot everything | `python pg/plots.py` | seconds | all PG figures + `*_part1_vs_part2.png` |

Each `pg/train.py` run trains, rolls out 500 evaluation episodes with the trained stochastic policy, and
writes `training_curve.png` (episode reward + 100-episode moving average), `eval_histogram.png`,
`eval_stats.json` (mean, std), `train_log.csv` and the weights. Useful flags: `--episodes`, `--lr`,
`--seed`, `--max-minutes`, `--fresh` (restart), `--eval-only` (evaluate a saved policy).

Pong checkpoints every 50 episodes; re-running the same command resumes from the checkpoint.

### On the deepdish cluster

```bash
nohup bash run_deepdish.sh > logs/all.log 2>&1 &     # runs everything in the background
tail -f logs/*.log                                    # progress
```

`run_deepdish.sh` starts both Pong runs first, then the DP, CartPole and validation steps. It uses the CPU
by default (`DEVICE=cpu`): the policy is tiny and acts once per frame, so a busy shared GPU only adds
queueing delay. To use one idle GPU card instead (the assignment requires a single card), run
`nohup env DEVICE=cuda GPU=<id> bash run_deepdish.sh > logs/all.log 2>&1 &`, choosing `<id>` from `nvidia-smi`.
Pong takes roughly 10 to 15 hours per run on CPU.

## Layout

```
dp/shuttle.py        MDP model, backward induction, value iteration, policy iteration
dp/run_dp.py         Part 2 plots
dp/validate_dp.py    Part 2 validation experiments (35 checks)
pg/reinforce.py      REINFORCE agent, value baseline, Pong pre-processing, train/eval loops
pg/train.py          command-line entry point
pg/plots.py          training curves, histograms, Part 1 vs Part 2 comparison
pg/validate_pg.py    policy-gradient validation experiments (20 checks)
run_deepdish.sh      one-command runner for the cluster
```

## Hyper-parameters

| | CartPole-v1 | Pong-v5 |
|---|---|---|
| discount gamma | 0.95 | 0.99 (return restarts after each point) |
| policy net | 4-128-2 MLP | 6400-200-2 MLP on difference frames |
| optimiser | Adam, lr 0.003 | Adam, lr 0.001 |
| episodes per update | 1 | 10 |
| training episodes | 1000 | 10000 |
| baseline (PG Part 2) | value net 4-128-1, Adam lr 0.003 | value net 6400-200-1, Adam lr 0.001 |
| actions | {left, right} | {RIGHT (2), LEFT (3)} |

The assignment suggests lr 0.01; CartPole uses 0.003 because 0.01 sometimes collapses a seed
(see the learning-rate table in `results/pg/validation/validation.txt`).
