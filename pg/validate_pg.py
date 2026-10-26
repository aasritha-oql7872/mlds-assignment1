"""Validation experiments for the policy-gradient parts.

    python pg/validate_pg.py            # full run (~10 min on a laptop CPU)
    python pg/validate_pg.py --quick    # fewer seeds and samples (~2 min)

Every test states a prediction from RL theory or from the game, measures it,
and prints PASS/FAIL with the numbers. Log: results/pg/validation/validation.txt
"""

import argparse
import json
import os
import sys

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from plots import moving_average
from reinforce import (DEVICE, PONG_ACTIONS, Agent, Config, PongFrames, discounted_returns, evaluate,
                       make_agent, make_config, preprocess, run_episode, train)

OUT = "results/pg/validation"
LOG, RESULTS = [], []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(msg)


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def section(title):
    log(""); log("=" * 78); log(title); log("=" * 78)


class Corridor(gym.Env):
    """Positions 0..L, start at 0. Action 1 = right, 0 = left (floored at 0).
    Reaching L gives +1 and ends the episode; episodes are cut at 3L steps.
    Observation: one-hot position. Known optimum: always go right (L steps)."""

    def __init__(self, L=6):
        self.L = L
        self.observation_space = gym.spaces.Box(0, 1, (L + 1,), np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def _obs(self):
        return np.eye(self.L + 1, dtype=np.float32)[self.pos]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.t = 0, 0
        return self._obs(), {}

    def step(self, a):
        self.t += 1
        self.pos = min(self.pos + 1, self.L) if a == 1 else max(self.pos - 1, 0)
        goal = self.pos == self.L
        return self._obs(), float(goal), goal, self.t >= 3 * self.L and not goal, {}


gym.register("Corridor-v0", entry_point=Corridor)


def flat_grad(agent, obs, acts, adv):
    """REINFORCE gradient estimate (dJ/dtheta) for one episode, flattened."""
    agent.policy.zero_grad()
    agent.pg_loss(obs, acts, adv, 1).backward()
    return -torch.cat([p.grad.flatten() for p in agent.policy.parameters()])


def tensors(xs, acts, G):
    return (torch.as_tensor(xs, device=DEVICE), torch.as_tensor(acts, device=DEVICE),
            torch.as_tensor(G, device=DEVICE))


def test_returns():
    section("1. Discounted returns (edge cases)")
    G = discounted_returns(np.array([1.0, 1, 1]), 0.5)
    check("r = [1,1,1], gamma = 0.5 gives G = [1.75, 1.5, 1]", np.allclose(G, [1.75, 1.5, 1]), G.tolist())
    G = discounted_returns(np.array([5.0]), 0.95)
    check("one-step episode: G_0 = r_0", np.allclose(G, [5]), G.tolist())
    G = discounted_returns(np.array([0.0, 0, -1, 0, 1]), 0.99, reset_on_point=True)
    check("Pong: return restarts after each point", np.allclose(G, [-0.9801, -0.99, -1, 0.99, 1]),
          np.round(G, 4).tolist())
    G = discounted_returns(np.ones(500), 0.95)
    check("CartPole 500-step episode: G_0 = (1 - 0.95^500)/0.05 ~ 20", abs(G[0] - 20) < 1e-3,
          f"G_0 = {G[0]:.4f}, G_440 = {G[440]:.3f}, G_499 = {G[499]:.1f}")
    log("   The 500-step cap is treated as terminal (no bootstrapping). With gamma = 0.95 that only")
    log("   lowers G_t for the last ~60 steps of a full-length episode.")


def test_unbiased_and_variance(quick):
    section("2. Exact-gradient test: estimator is unbiased, baseline cuts variance")
    # One-step problem: 2 equally likely states, 2 actions, rewards with a common
    # offset of +10. J(theta) has a closed form, so autograd gives the exact gradient.
    torch.manual_seed(0)
    R = torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=DEVICE) + 10
    agent = Agent(Config("none", False, 0.0, 0.0, 8, 1, 0, ""), obs_dim=2, n_actions=2)
    states = torch.eye(2, device=DEVICE)
    probs = torch.softmax(agent.policy(states), -1)
    J = 0.5 * (probs * R).sum()
    exact = torch.cat([g.flatten() for g in torch.autograd.grad(J, list(agent.policy.parameters()))])
    V = (probs * R).sum(-1).detach()  # true state values = ideal baseline

    n = 4000 if quick else 20000
    s = torch.randint(0, 2, (n,), device=DEVICE)
    a = torch.distributions.Categorical(probs=probs.detach()[s]).sample()
    r = R[s, a]
    var = {}
    for name, adv in (("no baseline", r), ("baseline V(s)", r - V[s])):
        g = torch.stack([flat_grad(agent, states[s[i]][None], a[i][None], adv[i][None]) for i in range(n)])
        z = ((g.mean(0) - exact) / (g.std(0) / np.sqrt(n) + 1e-12)).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(g.mean(0), exact, 0).item()
        check(f"{name}: Monte-Carlo mean gradient = exact gradient", z < 5 and cos > 0.95,
              f"cosine {cos:.4f}, worst |z| over {len(exact)} parameters {z:.2f}")
        var[name] = g.var(0).sum().item()
    check("baseline lowers total gradient variance", var["baseline V(s)"] < var["no baseline"],
          f"{var['no baseline']:.2f} -> {var['baseline V(s)']:.3f} "
          f"({var['no baseline'] / var['baseline V(s)']:.0f}x smaller)")


def test_corridor():
    section("3. Deterministic toy task with a known optimum (Corridor, L = 6)")
    curves = {}
    for baseline in (False, True):
        cfg = Config("Corridor-v0", baseline, gamma=0.95, lr=0.01, hidden=16, episodes_per_update=1,
                     n_episodes=400, out_dir=f"{OUT}/corridor_part{2 if baseline else 1}")
        agent = train(cfg, resume=False, verbose=False)
        with torch.no_grad():
            p_right = torch.softmax(agent.policy(torch.eye(7, device=DEVICE)[:6]), -1)[:, 1].cpu().numpy()
        greedy_len = len(run_episode(gym.make(cfg.env), agent, cfg, greedy=True)[2])
        curves[baseline] = pd.read_csv(f"{cfg.out_dir}/train_log.csv").length.values
        check(f"{'baseline' if baseline else 'no baseline'}: P(right) > 0.9 everywhere, greedy path = 6 steps",
              p_right.min() > 0.9 and greedy_len == 6,
              f"P(right) = {np.round(p_right, 3).tolist()}, greedy length {greedy_len}")
    return curves


def control_run(seed, shuffle, episodes=400):
    """CartPole REINFORCE (no baseline); with shuffle=True each episode's returns are permuted,
    which keeps their values but destroys which step they belong to. Returns episode rewards."""
    cfg = make_config("cartpole", False, seed=seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent, env, ep_rewards = make_agent(cfg), gym.make(cfg.env), []
    for ep in range(episodes):
        xs, acts, rews = run_episode(env, agent, cfg, seed=ep)
        G = discounted_returns(rews, cfg.gamma)
        agent.update([(xs, acts, rng.permutation(G) if shuffle else G)])
        ep_rewards.append(rews.sum())
    return np.array(ep_rewards)


def test_negative_control():
    section("4. Negative control: shuffle each episode's returns (destroys credit assignment)")
    rewards = {shuffle: control_run(1, shuffle) for shuffle in (False, True)}
    true_r, shuf_r = rewards[False][-100:].mean(), rewards[True][-100:].mean()
    # Shuffling keeps episode-level credit (longer episodes add more positive terms),
    # so the prediction is "far slower learning that stays near random", not zero.
    check("shuffled returns learn < half as well and stay within 2x of random (~22)",
          shuf_r < 0.5 * true_r and shuf_r < 44, f"mean reward, episodes 301-400: true {true_r:.1f}, "
          f"shuffled {shuf_r:.1f}")
    return rewards


def test_variance_cartpole(quick):
    section("5. Gradient variance on CartPole at one fixed policy")
    cfg = make_config("cartpole", True, n_episodes=150, seed=3, out_dir=f"{OUT}/cartpole_150eps")
    agent = train(cfg, resume=False, verbose=False)
    env = gym.make(cfg.env)
    g_plain, g_base = [], []
    for i in range(100 if quick else 300):
        xs, acts, rews = run_episode(env, agent, cfg, seed=50_000 + i)
        obs, a, G = tensors(xs, acts, discounted_returns(rews, cfg.gamma))
        g_plain.append(flat_grad(agent, obs, a, G))
        g_base.append(flat_grad(agent, obs, a, G - agent.value(obs).squeeze(-1).detach()))
    g_plain, g_base = torch.stack(g_plain), torch.stack(g_base)
    ratio = g_plain.var(0).sum().item() / g_base.var(0).sum().item()
    check("learned baseline reduces per-episode gradient variance", ratio > 1.5,
          f"variance ratio (none / baseline) = {ratio:.1f}x over {len(g_plain)} episodes")
    # No bias: the per-episode difference sum_t grad log pi * V(s_t) has mean zero, so
    # its per-parameter z-score should exceed 2.58 for only ~1% of parameters.
    d = g_plain - g_base
    z = d.mean(0) / (d.std(0) / np.sqrt(len(d)) + 1e-12)
    frac = (z.abs() > 2.58).float().mean().item()
    check("...without biasing it: difference of the two estimates has mean zero", frac < 0.03,
          f"{100 * frac:.1f}% of {len(z)} parameters have |z| > 2.58 (expected ~1% if unbiased)")
    return agent, cfg


def test_seeds(quick):
    section("6. Robustness across seeds and learning rates (CartPole, 1000 episodes)")
    seeds = [0, 1, 2] if quick else [0, 1, 2, 3, 4]
    lrs = [0.01] if quick else [0.01, 0.003]
    table = {}
    fig, axes = plt.subplots(1, len(lrs), figsize=(6 * len(lrs), 4), squeeze=False)
    for ax, lr in zip(axes[0], lrs):
        for baseline, color in ((False, "C0"), (True, "C3")):
            last100, evals = [], []
            for seed in seeds:
                cfg = make_config("cartpole", baseline, seed=seed, lr=lr,
                                  out_dir=f"{OUT}/seeds/lr{lr}_part{2 if baseline else 1}_seed{seed}")
                agent = train(cfg, resume=False, verbose=False)
                r = pd.read_csv(f"{cfg.out_dir}/train_log.csv").reward.values
                ax.plot(moving_average(r), color=color, alpha=0.6, lw=1,
                        label=("value baseline" if baseline else "no baseline") if seed == seeds[0] else None)
                last100.append(r[-100:].mean())
                evals.append(evaluate(cfg, agent, n=100)[1]["mean"])
            key = f"lr={lr}, {'baseline' if baseline else 'no baseline'}"
            table[key] = {"train_last100_mean": float(np.mean(last100)), "eval_mean": float(np.mean(evals)),
                          "eval_per_seed": [round(e, 1) for e in evals]}
            log(f"   {key:26s} last-100 training mean {np.mean(last100):6.1f}   "
                f"eval mean {np.mean(evals):6.1f}   per seed {[round(e) for e in evals]}")
        ax.set(title=f"CartPole, {len(seeds)} seeds, lr = {lr}", xlabel="Training episode",
               ylabel="100-episode moving average")
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/cartpole_seeds.png", dpi=160); plt.close(fig)
    with open(f"{OUT}/cartpole_seeds.json", "w") as f:
        json.dump(table, f, indent=2)
    for lr in lrs:
        b, nb = table[f"lr={lr}, baseline"], table[f"lr={lr}, no baseline"]
        check(f"lr={lr}: averaged over seeds, the baseline agent learns at least as well",
              b["train_last100_mean"] >= nb["train_last100_mean"],
              f"last-100 training mean: baseline {b['train_last100_mean']:.1f} vs none {nb['train_last100_mean']:.1f}")


def test_first_steps(agent, cfg):
    section("7. First 10 steps of one CartPole episode (Part 2 agent from section 5)")
    xs, acts, rews = run_episode(gym.make(cfg.env), agent, cfg, seed=123)
    G = discounted_returns(rews, cfg.gamma)
    with torch.no_grad():
        obs = torch.as_tensor(xs, device=DEVICE)
        p_right = torch.softmax(agent.policy(obs), -1)[:, 1].cpu().numpy()
        V = agent.value(obs).squeeze(-1).cpu().numpy()
    log(f"   episode length {len(rews)}")
    log(f"   {'t':>2} {'x':>7} {'x_dot':>7} {'theta':>7} {'th_dot':>7} {'P(right)':>8} {'a':>2} {'r':>2} "
        f"{'G_t':>6} {'V(s_t)':>7} {'G-V':>6}")
    for t in range(10):
        log(f"   {t:>2} " + " ".join(f"{v:7.3f}" for v in xs[t]) +
            f" {p_right[t]:8.3f} {acts[t]:>2} {rews[t]:2.0f} {G[t]:6.2f} {V[t]:7.2f} {G[t] - V[t]:6.2f}")
    corr = np.corrcoef(xs[:, 3], p_right)[0, 1]
    check("agent pushes toward the side the pole is falling: corr(theta_dot, P(right)) > 0.5",
          corr > 0.5, f"corr = {corr:.3f} over {len(rews)} steps")


def test_pong(quick):
    section("8. Pong input pipeline and action mapping")
    env = gym.make("ALE/Pong-v5", repeat_action_probability=0.0)  # no sticky actions for this test
    first_obs, _ = env.reset(seed=0)
    for _ in range(20):
        obs = env.step(0)[0]
    x = preprocess(obs)
    check("preprocess gives an 80x80 binary float32 image", x.shape == (80, 80) and x.dtype == np.float32
          and set(np.unique(x)) <= {0.0, 1.0}, f"shape {x.shape}, values {np.unique(x).tolist()}")
    check("background erased: only paddles and ball remain (< 3% of pixels on)", 0 < x.mean() < 0.03,
          f"{100 * x.mean():.2f}% of pixels on")
    before = obs.copy(); preprocess(obs)
    check("preprocess does not modify the environment's frame", np.array_equal(before, obs))
    check("first network input of an episode is all zeros", not PongFrames()(first_obs).any())

    def paddle_row(o):  # mean row of the agent's (right-hand) paddle in the 80x80 image
        return np.flatnonzero(preprocess(o)[:, 70:72].any(1)).mean()

    moved = {}
    for name, action in zip(("RIGHT", "LEFT"), PONG_ACTIONS):
        env.reset(seed=0)
        for _ in range(15):
            env.step(0)
        start = paddle_row(env.step(0)[0])
        for _ in range(8):
            o = env.step(action)[0]
        moved[name] = (start, paddle_row(o))
    check("output 0 -> RIGHT moves the paddle up, output 1 -> LEFT moves it down",
          moved["RIGHT"][1] < moved["RIGHT"][0] and moved["LEFT"][1] > moved["LEFT"][0],
          f"row {moved['RIGHT'][0]:.0f} -> {moved['RIGHT'][1]:.0f} (RIGHT), "
          f"{moved['LEFT'][0]:.0f} -> {moved['LEFT'][1]:.0f} (LEFT)")

    # An untrained agent (uniform over RIGHT/LEFT): the reference point the trained agents must beat.
    cfg = make_config("pong", False)
    agent = make_agent(cfg)
    for p in agent.policy.parameters():
        torch.nn.init.zeros_(p)  # zero logits -> exactly 50/50
    env = gym.make(cfg.env)
    r = [run_episode(env, agent, cfg, seed=i)[2].sum() for i in range(5 if quick else 20)]
    log(f"   uniform random RIGHT/LEFT policy: {np.mean(r):.2f} +/- {np.std(r):.2f} over {len(r)} episodes")
    with open(f"{OUT}/pong_random_policy.json", "w") as f:
        json.dump({"mean": float(np.mean(r)), "std": float(np.std(r)), "n": len(r)}, f)


def save_figures(corridor, control):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for baseline, color in ((False, "C0"), (True, "C3")):
        ax.plot(moving_average(corridor[baseline], 20), color=color,
                label="value baseline" if baseline else "no baseline")
    ax.axhline(6, color="gray", ls="--", label="optimum (6 steps)")
    ax.set(xlabel="Training episode", ylabel="Episode length (20-ep avg)", title="Corridor toy task")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/corridor.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for shuffle, color in ((False, "C0"), (True, "C7")):
        ax.plot(moving_average(control[shuffle]), color=color,
                label="shuffled returns" if shuffle else "true returns")
    ax.set(xlabel="Training episode", ylabel="100-episode moving average", title="Negative control (CartPole)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/negative_control.png", dpi=160)
    plt.close("all")


def main(quick):
    os.makedirs(OUT, exist_ok=True)
    test_returns()
    test_unbiased_and_variance(quick)
    corridor = test_corridor()
    control = test_negative_control()
    agent, cfg = test_variance_cartpole(quick)
    test_seeds(quick)
    test_first_steps(agent, cfg)
    test_pong(quick)
    save_figures(corridor, control)

    section(f"SUMMARY: {sum(RESULTS)}/{len(RESULTS)} checks passed")
    with open(f"{OUT}/validation.txt", "w") as f:
        f.write("\n".join(LOG) + "\n")
    return all(RESULTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    sys.exit(0 if main(parser.parse_args().quick) else 1)
