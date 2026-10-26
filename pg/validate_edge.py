"""Edge-case checks for the policy-gradient code (CartPole, Pong, REINFORCE, baseline).

    python pg/validate_edge.py            # about 5 min on a laptop CPU
    python pg/validate_edge.py --quick    # skips the multi-seed negative control

Every check has an answer that can be predicted without running the code.
Log: results/pg/validation/edge_cases.txt
"""

import argparse
import os
import sys
import tempfile

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from reinforce import (DEVICE, PongFrames, discounted_returns, evaluate, make_agent, make_config, run_episode,
                       train)
from validate_pg import control_run

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


class Balancer:
    """Hand-written CartPole controller: push toward the side the pole is falling."""
    def act(self, x, greedy=False):
        return int(x[2] + 0.5 * x[3] > 0)


def params_vector(agent):
    return torch.cat([p.detach().flatten() for p in agent.policy.parameters()])


def test_cartpole_env():
    section("1. CartPole termination and truncation")
    env = gym.make("CartPole-v1")
    o, _ = env.reset(seed=0)
    env.unwrapped.state = np.array([2.39, 2.0, 0.0, 0.0])  # cart almost at the edge, pole upright
    o, r, terminated, truncated, _ = env.step(1)
    check("cart leaves the track (|x| > 2.4) while the pole is upright: episode terminates",
          terminated and not truncated and abs(o[0]) > 2.4 and abs(o[2]) < 0.2095,
          f"x = {o[0]:.2f}, theta = {o[2]:.3f}, terminated = {terminated}")

    o, _ = env.reset(seed=0)
    n, done = 0, False
    while not done:
        o, r, terminated, truncated, _ = env.step(0)  # always push left: the pole must fall
        n, done = n + 1, terminated or truncated
    check("pole falls (always pushing left): terminated before the limit, not truncated",
          terminated and not truncated and n < 100 and abs(o[2]) > 0.2095, f"{n} steps, theta = {o[2]:.3f}")

    cfg = make_config("cartpole", False)
    xs, acts, rews = run_episode(env, Balancer(), cfg, seed=0)
    o, _ = env.reset(seed=0)
    done = False
    while not done:
        o, _, terminated, truncated, _ = env.step(Balancer().act(o))
        done = terminated or truncated
    check("balancing for the full limit: 500 steps, ends by truncation (not a failure)",
          len(rews) == 500 and truncated and not terminated, f"{len(rews)} steps, truncated = {truncated}")
    check("the 500-step episode is treated as terminal: G_499 = 1 and G_0 = (1 - 0.95^500)/0.05",
          np.isclose(discounted_returns(rews, 0.95)[[0, 499]], [20.0, 1.0], atol=1e-3).all(),
          f"G_0 = {discounted_returns(rews, 0.95)[0]:.3f}, G_499 = {discounted_returns(rews, 0.95)[499]:.3f}")

    again = run_episode(env, Balancer(), cfg, seed=0)
    check("a new episode starts from a fresh reset: same seed gives the identical episode",
          np.array_equal(again[0], xs) and np.array_equal(again[2], rews), "inputs and rewards identical")


def test_pong_env():
    section("2. Pong frame differences, points and game end")
    env = gym.make("ALE/Pong-v5")
    obs, _ = env.reset(seed=0)
    frames = PongFrames()
    frames(obs)
    check("identical consecutive frames give an all-zero difference", not frames(obs).any(), "difference is 0")

    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    frames, saw_motion, n_changed, rewards, done = PongFrames(), False, [], [], False
    frames(obs)
    while not done:
        obs, r, terminated, truncated, _ = env.step(int(rng.choice((2, 3))))
        x = frames(obs)
        if (x > 0).any() and (x < 0).any():
            saw_motion = True
            n_changed.append(int((x != 0).sum()))
        rewards.append(r)
        done = terminated or truncated
    rewards = np.array(rewards)
    wins, losses = int((rewards > 0).sum()), int((rewards < 0).sum())
    check("a moving ball shows up as +1 where it arrived and -1 where it left, in a few pixels only",
          saw_motion and np.median(n_changed) < 64, f"median changed pixels per step: {np.median(n_changed):.0f} of 6400")
    check("scoring a point does not end the game: many points are scored within one episode",
          wins + losses > 20, f"{wins + losses} points scored in one episode ({len(rewards)} steps)")
    check("the game ends exactly when one side reaches 21 points",
          terminated and max(wins, losses) == 21 and rewards.sum() == wins - losses,
          f"wins {wins}, losses {losses}, terminated = {terminated}, score = {rewards.sum():.0f}")

    section("3. Returns around a point boundary (Pong)")
    r = np.array([0, 0, 1, 0, 0, -1.0])
    G = discounted_returns(r, 0.99, reset_on_point=True)
    want = [0.9801, 0.99, 1.0, -0.9801, -0.99, -1.0]
    check("rewards [0,0,+1,0,0,-1]: each point is its own segment", np.allclose(G, want, atol=1e-4),
          np.round(G, 4).tolist())
    r2 = r.copy(); r2[-1] = -5.0
    check("changing the second point leaves the returns of the actions before the first point unchanged",
          np.allclose(discounted_returns(r2, 0.99, True)[:3], G[:3]), "G[:3] identical")
    check("without the reset, the second point would leak into the first segment (what the rule prevents)",
          discounted_returns(r, 0.99)[0] != G[0], f"G_0 = {discounted_returns(r, 0.99)[0]:.4f} vs {G[0]:.4f}")


def test_policy_numerics():
    section("4. Action probabilities and numerical stability")
    for game, dim in (("cartpole", 4), ("pong", 6400)):
        agent = make_agent(make_config(game, False))
        p = agent.dist(torch.randn(256, dim, device=DEVICE) * (1.0 if game == "cartpole" else 0.05)).probs
        check(f"{game}: probabilities are finite, non-negative and sum to 1",
              torch.isfinite(p).all() and (p >= 0).all() and torch.allclose(p.sum(1), torch.ones(256, device=DEVICE)),
              f"min {p.min():.4f}, max row-sum error {(p.sum(1) - 1).abs().max():.1e}")

    agent = make_agent(make_config("cartpole", False))
    for q in agent.policy.parameters():
        torch.nn.init.zeros_(q)
    agent.policy[-1].bias.data[:] = torch.tensor([200.0, -200.0])  # action 0 has probability 1 - e^-400
    x, a = torch.zeros(1, 4, device=DEVICE), torch.tensor([1], device=DEVICE)  # score the e^-400 action
    loss = agent.pg_loss(x, a, torch.ones(1, device=DEVICE), 1)
    loss.backward()
    grads = torch.cat([q.grad.flatten() for q in agent.policy.parameters()])
    check("a nearly deterministic policy (rare action has P = e^-400) keeps loss and gradient finite",
          torch.isfinite(loss) and torch.isfinite(grads).all(), f"loss {loss.item():.1f}, max |grad| {grads.abs().max():.2f}")


def test_gradient_identities():
    section("5. Gradient identities for REINFORCE and the baseline")
    torch.manual_seed(0)
    xs, acts = torch.randn(30, 4, device=DEVICE), torch.randint(0, 2, (30,), device=DEVICE)

    def grad_of(agent, adv):
        agent.policy.zero_grad()
        agent.pg_loss(xs, acts, adv.to(DEVICE), 1).backward()
        return torch.cat([q.grad.flatten() for q in agent.policy.parameters()])

    agent = make_agent(make_config("cartpole", False))
    check("all returns zero: the policy gradient is exactly zero", grad_of(agent, torch.zeros(30)).abs().max() == 0, "")
    G = torch.rand(30) * 10
    agent = make_agent(make_config("cartpole", True))
    v = G.clone()  # a baseline that predicts the return exactly
    check("baseline predicts every return exactly: advantage 0, gradient exactly zero",
          grad_of(agent, G - v).abs().max() == 0, "")

    agent = make_agent(make_config("cartpole", True))
    seen = []
    original = agent.pg_loss
    agent.pg_loss = lambda obs, a, adv, n: (seen.append(adv.requires_grad), original(obs, a, adv, n))[1]
    batch = [(xs.cpu().numpy(), acts.cpu().numpy(), G.numpy())]
    value_before = torch.cat([q.detach().flatten() for q in agent.value.parameters()])
    agent.update(batch)
    check("Part 2 update: the advantage is detached, so the policy loss cannot train the value network",
          seen == [False], f"advantage requires_grad = {seen}")
    check("Part 2 update: the value network does learn (its own MSE loss)",
          not torch.equal(value_before, torch.cat([q.detach().flatten() for q in agent.value.parameters()])), "")


def test_training_and_evaluation():
    section("6. Training updates the network, evaluation does not, and nothing becomes NaN")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = make_config("cartpole", True, n_episodes=30, eval_episodes=5, out_dir=tmp)
        agent = train(cfg, verbose=False)
        after_training = params_vector(agent).clone()
        check("after 30 training episodes (with baseline) all weights are finite",
              torch.isfinite(after_training).all() and all(
                  torch.isfinite(q).all() for q in agent.value.parameters()), "no NaN or Inf in policy or value net")
        evaluate(cfg, agent, n=5)
        check("evaluation does not change the weights", torch.equal(after_training, params_vector(agent)), "weights identical")
        torch.manual_seed(cfg.seed)  # train() seeds, then builds the agent: this reproduces its initial weights
        initial = params_vector(make_agent(cfg))
        check("training does change the weights (compared with the same initialisation)",
              not torch.equal(initial, after_training), f"mean |change| = {(after_training - initial).abs().mean():.4f}")


def test_negative_control_seeds():
    section("7. Negative control repeated over seeds (shuffled returns vs true returns, CartPole)")
    rows = []
    for seed in range(2, 7):
        true_r, shuf_r = (control_run(seed, s)[-100:].mean() for s in (False, True))
        rows.append((true_r, shuf_r))
        log(f"   seed {seed}: last-100 mean reward  true returns {true_r:6.1f}   shuffled {shuf_r:6.1f}   (random play ~22)")
    ratios = [s / t for t, s in rows]
    check("on every seed the shuffled agent ends below half the true-returns agent (400 episodes each)",
          all(r < 0.5 for r in ratios), f"shuffled/true = {[round(float(r), 2) for r in ratios]}")
    log("   Note: an earlier 300-episode version of this test gave 0.53 on one seed, because the true-returns")
    log("   agent had not yet learned much by episode 300 (110). Shuffled agents stay above random (22)")
    log("   because shuffling keeps each episode's total return, so longer episodes still get more credit.")


def main(quick):
    os.makedirs(OUT, exist_ok=True)
    test_cartpole_env()
    test_pong_env()
    test_policy_numerics()
    test_gradient_identities()
    test_training_and_evaluation()
    if not quick:
        test_negative_control_seeds()
    section(f"SUMMARY: {sum(RESULTS)}/{len(RESULTS)} checks passed")
    with open(f"{OUT}/edge_cases.txt", "w") as f:
        f.write("\n".join(LOG) + "\n")
    return all(RESULTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    sys.exit(0 if main(parser.parse_args().quick) else 1)
