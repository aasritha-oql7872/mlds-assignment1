"""Solve the shuttle MDP three ways and make the Part 2 plots.

    python dp/run_dp.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from shuttle import ShuttleMDP, backward_induction, policy_iteration, threshold_of, value_iteration

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "dp")


def main():
    os.makedirs(OUT, exist_ok=True)
    mdp = ShuttleMDP()

    V_bi, pi_bi = backward_induction(mdp, T=500)
    V_vi, pi_vi, vi_hist = value_iteration(mdp)
    V_pi, pi_pi, pi_hist = policy_iteration(mdp)

    summary = {
        "threshold_enumeration_t0": threshold_of(pi_bi[0]),
        "threshold_value_iteration": threshold_of(pi_vi),
        "threshold_policy_iteration": threshold_of(pi_pi),
        "vi_iterations": len(vi_hist),
        "pi_iterations": len(pi_hist),
        "pi_changes_per_iteration": pi_hist,
        "max_abs_diff_enum_vs_vi": float(np.max(np.abs(V_bi[0] - V_vi))),
        "max_abs_diff_vi_vs_pi": float(np.max(np.abs(V_vi - V_pi))),
        "V_star_at": {str(s): round(float(V_vi[s]), 4) for s in (0, 10, 14, 15, 20, 50, 100, 200)},
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    s = mdp.states
    thr = threshold_of(pi_vi)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(s, V_bi[0], color="C0", lw=2)
    ax.set(xlabel="Customers waiting, s", ylabel=r"Optimal expected discounted cost $V_0^*(s)$",
           title="(a) Enumeration / backward induction, T = 500: value at t = 0")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "a_enumeration_value.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(s, V_vi, color="C1", lw=2)
    ax.axvline(thr, color="gray", ls="--", lw=1, label=f"dispatch threshold s = {thr}")
    ax.set(xlabel="Customers waiting, s", ylabel=r"Optimal expected discounted cost $V^*(s)$",
           title=r"(b) Value iteration, $T=\infty$, $\gamma=0.95$")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "b_value_iteration_value.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.step(s, pi_pi, where="mid", color="C2", lw=2)
    ax.set_yticks([0, 1], ["0: hold", "1: dispatch"])
    ax.axvline(thr, color="gray", ls="--", lw=1, label=f"threshold s = {thr}")
    ax.set(xlabel="Customers waiting, s", ylabel="Optimal action",
           title="(c) Policy iteration: optimal policy")
    ax.set_ylim(-0.2, 1.2); ax.legend(loc="center right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "c_policy_iteration_policy.png"), dpi=160)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.semilogy(range(1, len(vi_hist) + 1), vi_hist, color="C1")
    ax.set(xlabel="Iteration", ylabel=r"$\|V_{k+1}-V_k\|_\infty$", title="Value iteration convergence")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "vi_convergence.png"), dpi=160)
    plt.close("all")


if __name__ == "__main__":
    main()
