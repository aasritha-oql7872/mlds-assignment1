"""Shuttle dispatching MDP solved with backward induction (enumeration over a
finite horizon), value iteration and policy iteration.

MDP (cost-minimisation form):
    state      s in {0, ..., S_MAX}   customers waiting at the decision epoch
    action     a in {0, 1}             0 = hold, 1 = dispatch a shuttle
    post-decision state  s_a = max(s - a*K, 0)
    one-step cost        c(s, a) = cf * a + ch * s_a
    transition           s' = min(s_a + A, S_MAX),  A ~ Unif{1, ..., 5}
Customers that would push the queue past S_MAX are turned away (they balk).
"""

from dataclasses import dataclass, field

import numpy as np

HOLD, DISPATCH = 0, 1


@dataclass
class ShuttleMDP:
    K: int = 15
    cf: float = 100.0
    ch: float = 2.0
    gamma: float = 0.95
    s_max: int = 200
    arrivals: dict = field(default_factory=lambda: {a: 0.2 for a in range(1, 6)})

    def __post_init__(self):
        assert abs(sum(self.arrivals.values()) - 1.0) < 1e-12, "arrival pmf must sum to 1"
        self.states = np.arange(self.s_max + 1)
        self.n = len(self.states)
        self.C = np.zeros((2, self.n))          # C[a, s]
        self.P = np.zeros((2, self.n, self.n))  # P[a, s, s']
        for a in (HOLD, DISPATCH):
            post = np.maximum(self.states - a * self.K, 0)
            self.C[a] = self.cf * a + self.ch * post
            for k, p in self.arrivals.items():
                nxt = np.minimum(post + k, self.s_max)
                np.add.at(self.P[a], (self.states, nxt), p)

    def q_values(self, V):
        """Q[a, s] = c(s, a) + gamma * E[V(s')]."""
        return self.C + self.gamma * self.P @ V


def greedy(Q, prev_policy=None, tol=1e-9):
    """argmin over actions. Ties go to HOLD, or to the previous action if one is
    given (this is what stops policy iteration from cycling between equally good
    policies)."""
    policy = (Q[DISPATCH] < Q[HOLD] - tol).astype(int)
    if prev_policy is not None:
        tied = np.abs(Q[DISPATCH] - Q[HOLD]) <= tol
        policy[tied] = prev_policy[tied]
    return policy


def backward_induction(mdp, T=500, terminal=None):
    """Finite horizon: V_T = terminal (default 0), V_t = min_a Q_t.
    Returns V[t, s] for t = 0..T and the time-dependent policy pi[t, s]."""
    V = np.zeros((T + 1, mdp.n))
    if terminal is not None:
        V[T] = terminal
    pi = np.zeros((T, mdp.n), dtype=int)
    for t in range(T - 1, -1, -1):
        Q = mdp.q_values(V[t + 1])
        pi[t] = greedy(Q)
        V[t] = Q[pi[t], mdp.states]
    return V, pi


def value_iteration(mdp, eps=1e-8, max_iter=100_000, V0=None):
    """Stops when ||V_{k+1} - V_k|| < eps (1 - gamma) / (2 gamma), which
    guarantees the greedy policy is eps-optimal."""
    V = np.zeros(mdp.n) if V0 is None else V0.astype(float).copy()
    thresh = eps * (1 - mdp.gamma) / (2 * mdp.gamma) if mdp.gamma > 0 else 0.0
    history = []
    for k in range(1, max_iter + 1):
        V_new = mdp.q_values(V).min(axis=0)
        delta = np.max(np.abs(V_new - V))
        history.append(delta)
        V = V_new
        if delta <= thresh:
            break
    return V, greedy(mdp.q_values(V)), history


def policy_evaluation(mdp, policy):
    """Exact evaluation: solve (I - gamma P_pi) V = c_pi."""
    P_pi = mdp.P[policy, mdp.states]
    c_pi = mdp.C[policy, mdp.states]
    return np.linalg.solve(np.eye(mdp.n) - mdp.gamma * P_pi, c_pi)


def policy_iteration(mdp, policy0=None, max_iter=1000):
    policy = np.zeros(mdp.n, dtype=int) if policy0 is None else policy0.copy()
    history = []
    for _ in range(max_iter):
        V = policy_evaluation(mdp, policy)
        new_policy = greedy(mdp.q_values(V), prev_policy=policy)
        history.append(int((new_policy != policy).sum()))
        if np.array_equal(new_policy, policy):
            return V, policy, history
        policy = new_policy
    raise RuntimeError("policy iteration did not converge")


def threshold_of(policy):
    """Smallest s with a dispatch decision, or None if the policy never dispatches."""
    idx = np.flatnonzero(policy == DISPATCH)
    return int(idx[0]) if idx.size else None


def is_threshold_policy(policy):
    t = threshold_of(policy)
    return t is None or bool(np.all(policy[t:] == DISPATCH))
