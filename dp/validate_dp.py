"""Validation experiments for the shuttle MDP (Part 2).

    python dp/validate_dp.py

Each test states a prediction that follows from MDP theory or from the
structure of the problem, measures it, and prints PASS/FAIL with the numbers.
The full log is written to results/dp/validation.txt.
"""

import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from shuttle import (DISPATCH, HOLD, ShuttleMDP, backward_induction, is_threshold_policy,
                     policy_evaluation, policy_iteration, threshold_of, value_iteration)

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dp")
LOG = []
RESULTS = []


def log(msg=""):
    print(msg)
    LOG.append(msg)


def check(name, ok, detail):
    RESULTS.append((name, bool(ok)))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def section(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


def simulate(mdp, policy, s0, horizon, n_runs, rng):
    """Monte Carlo estimate of the discounted cost of a stationary policy."""
    ks = np.array(list(mdp.arrivals.keys()))
    ps = np.array(list(mdp.arrivals.values()))
    s = np.full(n_runs, s0)
    total = np.zeros(n_runs)
    disc = 1.0
    for _ in range(horizon):
        a = policy[s]
        post = np.maximum(s - a * mdp.K, 0)
        total += disc * (mdp.cf * a + mdp.ch * post)
        s = np.minimum(post + rng.choice(ks, size=n_runs, p=ps), mdp.s_max)
        disc *= mdp.gamma
    return total


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    mdp = ShuttleMDP()
    V_vi, pi_vi, vi_hist = value_iteration(mdp)
    V_pi, pi_pi, _ = policy_iteration(mdp)
    V_bi, pi_bi = backward_induction(mdp, T=500)
    thr = threshold_of(pi_vi)

    # ------------------------------------------------------------------ #
    section("1. Model construction edge cases")
    rows = mdp.P.sum(axis=2)
    check("transition rows sum to 1 (incl. the s=200 boundary)",
          np.allclose(rows, 1.0), f"max |row sum - 1| = {np.max(np.abs(rows - 1)):.2e}")
    check("hold at s=200 keeps queue at 200 w.p. 1 (overflow customers balk)",
          mdp.P[HOLD, 200, 200] == 1.0, f"P[hold, 200 -> 200] = {mdp.P[HOLD, 200, 200]}")
    check("hold at s=197 hits the cap: P(200) = P(A>=3) = 0.6",
          np.isclose(mdp.P[HOLD, 197, 200], 0.6), f"P = {mdp.P[HOLD, 197, 200]:.2f}")
    check("dispatch with s < K empties the station (no negative queue)",
          np.isclose(mdp.P[DISPATCH, 7, 1:6].sum(), 1.0) and mdp.C[DISPATCH, 7] == 100,
          f"s=7 dispatch -> next state in {{1..5}}, cost {mdp.C[DISPATCH, 7]}")
    check("state 0 is never re-entered after an epoch (A >= 1)",
          np.all(mdp.P[:, :, 0] == 0), "column 0 of P is all zeros")
    check("dispatching an empty station is never optimal",
          pi_vi[0] == HOLD, f"pi*(0) = {pi_vi[0]}, Q(0,hold)-Q(0,dispatch) = "
          f"{(mdp.q_values(V_vi)[HOLD, 0] - mdp.q_values(V_vi)[DISPATCH, 0]):.2f}")

    # ------------------------------------------------------------------ #
    section("2. Agreement between the three solution methods")
    check("value iteration == policy iteration (values)",
          np.max(np.abs(V_vi - V_pi)) < 1e-6, f"max diff = {np.max(np.abs(V_vi - V_pi)):.2e}")
    check("value iteration == policy iteration (policy)",
          np.array_equal(pi_vi, pi_pi), f"threshold VI = {thr}, PI = {threshold_of(pi_pi)}")
    bound = mdp.gamma ** 500 * np.max(np.abs(V_vi))
    diff = np.max(np.abs(V_bi[0] - V_vi))
    check("finite horizon T=500 within gamma^T ||V*|| of infinite horizon",
          diff <= bound + 1e-9, f"diff = {diff:.2e} <= bound {bound:.2e}")
    log("   Horizon sweep: how fast V_0^T approaches V* (predicted ~ gamma^T):")
    for T in (1, 5, 10, 25, 50, 100, 200, 500):
        V_T, pi_T = backward_induction(mdp, T)
        log(f"     T={T:4d}  ||V_0^T - V*||_inf = {np.max(np.abs(V_T[0] - V_vi)):12.4e}"
            f"   gamma^T ||V*|| = {mdp.gamma ** T * np.max(V_vi):12.4e}"
            f"   threshold at t=0: {threshold_of(pi_T[0])}")
    log("   Near the end of the horizon the policy should become lazier (dispatching")
    log("   only pays off if enough future periods remain to recoup cf):")
    for t in (499, 498, 497, 495, 490, 480, 0):
        log(f"     t={t:3d} (periods left {500 - t:3d}): threshold = {threshold_of(pi_bi[t])}")
    check("last period: dispatch iff ch*min(s,K) > cf, i.e. never (2*15=30 < 100)",
          threshold_of(pi_bi[499]) is None, f"threshold at t=499 = {threshold_of(pi_bi[499])}")

    # ------------------------------------------------------------------ #
    section("3. Optimality certificates")
    resid = np.max(np.abs(mdp.q_values(V_vi).min(axis=0) - V_vi))
    check("Bellman residual of V* is ~0", resid < 1e-6, f"||T V* - V*||_inf = {resid:.2e}")
    V_eval = policy_evaluation(mdp, pi_vi)
    check("exact evaluation of greedy policy reproduces V*",
          np.max(np.abs(V_eval - V_vi)) < 1e-6, f"max diff = {np.max(np.abs(V_eval - V_vi)):.2e}")
    ratios = np.array(vi_hist[1:51]) / np.array(vi_hist[:50])
    check("VI is a gamma-contraction: successive error ratio <= gamma",
          np.all(ratios <= mdp.gamma + 1e-9), f"max ratio over first 50 iters = {ratios.max():.4f}")
    V_vi2, pi_vi2, _ = value_iteration(mdp, V0=rng.uniform(0, 1e5, mdp.n))
    check("VI converges to the same fixed point from a random start",
          np.max(np.abs(V_vi2 - V_vi)) < 1e-6 and np.array_equal(pi_vi2, pi_vi),
          f"max diff = {np.max(np.abs(V_vi2 - V_vi)):.2e}")
    V_pi2, pi_pi2, h2 = policy_iteration(mdp, policy0=np.ones(mdp.n, dtype=int))
    check("PI from always-dispatch converges to the same policy",
          np.array_equal(pi_pi2, pi_vi), f"{len(h2)} iterations, threshold {threshold_of(pi_pi2)}")

    # Brute-force enumeration over every threshold policy on the full problem.
    costs = []
    for t in range(mdp.n + 1):
        pol = (mdp.states >= t).astype(int)
        costs.append(policy_evaluation(mdp, pol))
    costs = np.array(costs)
    best_t = int(np.argmin(costs[:, 0]))
    dominated = np.all(costs >= V_vi - 1e-6)
    check("among all 202 threshold policies, the best is the DP threshold",
          best_t == thr and dominated, f"best threshold by enumeration = {best_t}, DP = {thr}; "
          f"V* <= every threshold policy at every state: {dominated}")
    log("   Value at s=0 of nearby threshold policies:")
    for t in range(thr - 3, thr + 4):
        log(f"     threshold {t:3d}: V(0) = {costs[t, 0]:.4f}" + ("   <- optimal" if t == thr else ""))

    # Exhaustive enumeration of ALL deterministic stationary policies on a small instance.
    small = ShuttleMDP(K=4, cf=10, ch=2, s_max=11, arrivals={1: 0.5, 2: 0.5})
    best = None
    for bits in itertools.product((0, 1), repeat=small.n):
        pol = np.array(bits)
        v = policy_evaluation(small, pol)
        if best is None or v.sum() < best[0].sum() - 1e-12:
            best = (v, pol)
    Vs, pis, _ = policy_iteration(small)
    check(f"small instance: exhaustive search over all 2^{small.n} = {2 ** small.n} policies "
          "matches PI", np.allclose(best[0], Vs) and np.array_equal(best[1], pis),
          f"max value diff = {np.max(np.abs(best[0] - Vs)):.2e}, policy {pis.tolist()}")

    # ------------------------------------------------------------------ #
    section("4. Structural properties predicted by theory")
    check("V* is non-decreasing in s (more waiting customers can never help)",
          np.all(np.diff(V_vi) >= -1e-9), f"min diff = {np.diff(V_vi).min():.4f}")
    check("optimal policy is a threshold (control-limit) policy",
          is_threshold_policy(pi_vi), f"dispatch iff s >= {thr}")
    check("for s in [thr, K], dispatch clears everyone, so V*(s) = cf + V*(0)",
          np.allclose(V_vi[thr:mdp.K + 1], mdp.cf + V_vi[0]),
          f"V*(13..15) = {np.round(V_vi[thr:mdp.K + 1], 3).tolist()}, cf + V*(0) = {mdp.cf + V_vi[0]:.3f}")
    log(f"   For large s, each extra customer is carried ~s/K periods; the slope of V* "
        f"keeps growing: dV at s=50: {np.diff(V_vi)[50]:.2f}, s=100: {np.diff(V_vi)[100]:.2f}, "
        f"s=199: {np.diff(V_vi)[199]:.2f}")

    # ------------------------------------------------------------------ #
    section("5. Monte Carlo simulation of the optimal policy vs. V*")
    for s0 in (0, 13, 50, 150):
        sims = simulate(mdp, pi_vi, s0, horizon=600, n_runs=20000, rng=rng)
        se = sims.std() / np.sqrt(len(sims))
        z = (sims.mean() - V_vi[s0]) / se
        check(f"simulated cost from s0={s0} matches V*(s0)", abs(z) < 4,
              f"MC = {sims.mean():.2f} +/- {se:.2f}, V* = {V_vi[s0]:.2f}, z = {z:+.2f}")

    # ------------------------------------------------------------------ #
    section("6. Parameter perturbations with predicted outcomes")
    m = ShuttleMDP(cf=0.0)
    _, p, _ = policy_iteration(m)
    check("cf = 0: dispatch whenever anyone is waiting", threshold_of(p) == 1,
          f"threshold = {threshold_of(p)}")
    m = ShuttleMDP(ch=0.0)
    V, p, _ = policy_iteration(m)
    check("ch = 0: never dispatch and V* = 0", threshold_of(p) is None and np.allclose(V, 0),
          f"threshold = {threshold_of(p)}, max V = {V.max():.2e}")
    m = ShuttleMDP(cf=1e7)
    V, p, _ = policy_iteration(m)
    check("cf = 1e7: never dispatch", threshold_of(p) is None, f"threshold = {threshold_of(p)}")
    m = ShuttleMDP(gamma=0.0)
    _, p, _ = policy_iteration(m)
    check("gamma = 0 (myopic): dispatch iff ch*min(s,K) > cf -> never for these costs",
          threshold_of(p) is None, f"threshold = {threshold_of(p)}")
    m = ShuttleMDP(gamma=0.0, cf=20)
    _, p, _ = policy_iteration(m)
    check("gamma = 0, cf = 20: dispatch iff 2*min(s,15) > 20 -> threshold 11",
          threshold_of(p) == 11, f"threshold = {threshold_of(p)}")
    m = ShuttleMDP(K=200)
    V, p, _ = policy_iteration(m)
    check("K = 200 (one shuttle clears everything): V* flat once dispatching",
          np.allclose(V[threshold_of(p):], V[threshold_of(p)]), f"threshold = {threshold_of(p)}")

    log("   Sweep of cf (prediction: threshold rises with cf):")
    cf_grid = [0, 10, 25, 50, 100, 150, 200, 300, 500]
    thr_cf = [threshold_of(policy_iteration(ShuttleMDP(cf=c))[1]) for c in cf_grid]
    log("     " + ", ".join(f"cf={c}:{t}" for c, t in zip(cf_grid, thr_cf)))
    check("threshold non-decreasing in cf", all(a <= b for a, b in zip(thr_cf, thr_cf[1:])), "")
    log("   Sweep of ch (prediction: threshold falls with ch):")
    ch_grid = [0.5, 1, 2, 4, 8, 16]
    thr_ch = [threshold_of(policy_iteration(ShuttleMDP(ch=c))[1]) for c in ch_grid]
    log("     " + ", ".join(f"ch={c}:{t}" for c, t in zip(ch_grid, thr_ch)))
    check("threshold non-increasing in ch", all(a >= b for a, b in zip(thr_ch, thr_ch[1:])), "")
    log("   Sweep of gamma (prediction: more patience -> willing to pay cf earlier):")
    g_grid = [0.5, 0.8, 0.9, 0.95, 0.99]
    thr_g = [threshold_of(policy_iteration(ShuttleMDP(gamma=g))[1]) for g in g_grid]
    log("     " + ", ".join(f"gamma={g}:{t}" for g, t in zip(g_grid, thr_g)))

    # Deterministic arrivals: the MDP becomes a deterministic control problem
    # whose cost under a threshold policy can be computed by hand.
    m = ShuttleMDP(arrivals={3: 1.0})
    V, p, _ = policy_iteration(m)
    t = threshold_of(p)
    s, c, d = 0, 0.0, 1.0
    for _ in range(2000):
        a = int(s >= t)
        post = max(s - a * m.K, 0)
        c += d * (m.cf * a + m.ch * post)
        s, d = min(post + 3, m.s_max), d * m.gamma
    check("deterministic arrivals (A=3): DP value equals hand-rolled trajectory cost",
          abs(c - V[0]) < 1e-6, f"threshold {t}, trajectory cost {c:.4f}, V*(0) = {V[0]:.4f}")

    # Effect of the S_MAX = 200 truncation.
    big = ShuttleMDP(s_max=400)
    Vb, pb, _ = policy_iteration(big)
    d = np.max(np.abs(Vb[:101] - V_vi[:101]))
    check("raising the cap to 400 does not change the policy or V* for s <= 100",
          threshold_of(pb) == thr and d < 1e-3, f"max diff on s<=100 = {d:.2e}")
    d_hi = np.abs(Vb[200] - V_vi[200])
    check("cap never binds under pi*: V*(200) identical with cap 200 and 400",
          d_hi < 1e-3, f"{V_vi[200]:.3f} vs {Vb[200]:.3f} (a dispatch removes 15 > max arrivals 5)")
    never = np.zeros(mdp.n, dtype=int)
    v200 = policy_evaluation(mdp, never)[200]
    v400 = policy_evaluation(big, np.zeros(big.n, dtype=int))[200]
    log("   The cap does bind for a policy that lets the queue grow: under never-dispatch,")
    log(f"   V(200) = {v200:.1f} with cap 200 (= 2*200/(1-0.95) = {2 * 200 / 0.05:.1f}, queue frozen)")
    log(f"   vs {v400:.1f} with cap 400. Truncation therefore only affects non-optimal policies.")

    # ------------------------------------------------------------------ #
    section("7. First 10 decision epochs under pi*, starting at s = 0 (seed 1)")
    r = np.random.default_rng(1)
    Q = mdp.q_values(V_vi)
    s = 0
    log(f"   {'t':>2} {'s':>4} {'Q(hold)':>9} {'Q(disp)':>9} {'a':>2} {'cost':>6} {'A':>2} {'s_next':>6}")
    for t_ in range(10):
        a = pi_vi[s]
        post = max(s - a * mdp.K, 0)
        cost = mdp.cf * a + mdp.ch * post
        A = int(r.integers(1, 6))
        nxt = min(post + A, mdp.s_max)
        log(f"   {t_:>2} {s:>4} {Q[HOLD, s]:>9.2f} {Q[DISPATCH, s]:>9.2f} {a:>2} {cost:>6.0f} {A:>2} {nxt:>6}")
        s = nxt
    log("   Reading: the queue grows by A each period and the agent holds (paying 2/customer)")
    log("   until s >= 13, where Q(dispatch) drops below Q(hold); it then pays cf=100, the")
    log("   queue resets to 0 + A, and the cycle repeats.")

    # ------------------------------------------------------------------ #
    section("8. Boundary cases with exactly predictable answers")
    def row(a, s0):  # P[a, s0, :] as {next_state: prob}
        return {int(j): round(float(p), 3) for j, p in enumerate(mdp.P[a, s0]) if p > 0}

    check("exactly K = 15 waiting: dispatch removes all 15, nobody left, cost 100",
          mdp.C[DISPATCH, 15] == 100 and row(DISPATCH, 15) == {k: 0.2 for k in range(1, 6)},
          f"cost {mdp.C[DISPATCH, 15]:.0f}, next state {row(DISPATCH, 15)}")
    check("K + 1 = 16 waiting: dispatch leaves 1 behind, which still pays ch: cost 102",
          mdp.C[DISPATCH, 16] == 102 and row(DISPATCH, 16) == {k: 0.2 for k in range(2, 7)},
          f"cost {mdp.C[DISPATCH, 16]:.0f}, next state {row(DISPATCH, 16)}")
    check("198 waiting, hold: P(199) = 0.2 (A=1), P(200) = 0.8 (A>=2 overflow accumulates)",
          row(HOLD, 198) == {199: 0.2, 200: 0.8}, f"{row(HOLD, 198)}")
    for bad, why in (({1: 0.5, 2: 0.4}, "sums to 0.9"), ({1: 1.2, 2: -0.2}, "has a negative probability")):
        try:
            ShuttleMDP(arrivals=bad)
            check(f"arrival pmf that {why} is rejected", False, "no error raised")
        except ValueError as e:
            check(f"arrival pmf that {why} is rejected", True, str(e))
    free = ShuttleMDP(cf=0.0)
    Qf = free.q_values(value_iteration(free)[0])
    check("cf = 0: at s = 0 hold and dispatch tie exactly (dispatching nobody costs nothing)",
          abs(Qf[HOLD, 0] - Qf[DISPATCH, 0]) < 1e-9, f"Q(hold) = {Qf[HOLD, 0]:.4f}, Q(dispatch) = {Qf[DISPATCH, 0]:.4f}")

    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(cf_grid, [t if t is not None else np.nan for t in thr_cf], "o-")
    axes[0].set(xlabel="dispatch cost $c_f$", ylabel="optimal threshold", title="Threshold vs $c_f$")
    axes[1].plot(ch_grid, thr_ch, "o-", color="C3")
    axes[1].set(xlabel="holding cost $c_h$", ylabel="optimal threshold", title="Threshold vs $c_h$",
                xscale="log")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "validation_sensitivity.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(thr - 10, thr + 30), costs[thr - 10:thr + 30, 0], "o-", ms=3)
    ax.axvline(thr, color="gray", ls="--", label=f"DP threshold {thr}")
    ax.set(xlabel="threshold used by policy", ylabel="V(0) of that policy",
           title="Brute-force enumeration over threshold policies")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "validation_threshold_enumeration.png"), dpi=160)
    plt.close("all")

    n_pass = sum(ok for _, ok in RESULTS)
    section(f"SUMMARY: {n_pass}/{len(RESULTS)} checks passed")
    with open(os.path.join(OUT, "validation.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
    return n_pass == len(RESULTS)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
