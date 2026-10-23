"""Train REINFORCE, roll out 500 evaluation episodes, and plot.

    python pg/train.py --game cartpole              # Part 1
    python pg/train.py --game cartpole --baseline   # Part 2
    python pg/train.py --game pong                  # Part 1 (hours; resumable)
    python pg/train.py --game pong --baseline       # Part 2 (hours; resumable)

Re-running a command resumes from its last checkpoint; --fresh starts over.
--eval-only skips training and evaluates the saved checkpoint.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from plots import plot_run
from reinforce import DEVICE, evaluate, load_checkpoint, make_agent, make_config, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", choices=["cartpole", "pong"], required=True)
    p.add_argument("--baseline", action="store_true", help="Part 2: learned value baseline")
    p.add_argument("--episodes", type=int, dest="n_episodes")
    p.add_argument("--max-minutes", type=float, dest="max_minutes")
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--eval-episodes", type=int, dest="eval_episodes")
    p.add_argument("--out-dir", dest="out_dir")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    overrides = {k: getattr(args, k) for k in
                 ("n_episodes", "max_minutes", "lr", "seed", "eval_episodes", "out_dir")}
    cfg = make_config(args.game, args.baseline, **overrides)
    print(f"device: {DEVICE}")

    if args.eval_only:
        agent = make_agent(cfg)
        load_checkpoint(agent, os.path.join(cfg.out_dir, "checkpoint.pt"))
    else:
        agent = train(cfg, resume=not args.fresh)
    _, stats = evaluate(cfg, agent)
    print(f"evaluation over {stats['n']} episodes: mean {stats['mean']:.2f}, std {stats['std']:.2f}")
    plot_run(cfg.out_dir)


if __name__ == "__main__":
    main()
