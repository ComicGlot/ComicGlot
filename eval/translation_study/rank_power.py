"""Power of the ranking design of the translator study: 3 of 8 models per text, each with and without the guide,
ranked under a Plackett-Luce model (Gumbel noise on log-strengths). Over simulated studies it reports how often the
pooled and the per-model guide effects are found (sign test on the within-model comparisons, p < 0.05), the spread
of the estimated model strengths (Bradley-Terry on the rank-broken pairs) and how often neighbouring models come out
in the right order. Effects are on the logit scale: 0.3 is 57 to 43 within a model, 0.4 is 60 to 40, 0.8 is 69 to 31.

    python -m eval.translation_study.rank_power --out data/study2026/rank_power.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

rng = np.random.default_rng(0)
M = 8
P_SAME = 0.17  # share of texts where a model's two versions are identical (from the 302 texts)


def simulate(n_rankings, model_gap, guide_effect):
    # log-strengths: models spread evenly over model_gap * 7; the guide adds guide_effect[m]
    base = np.linspace(0, model_gap * (M - 1), M)[::-1]
    wins = np.zeros((2 * M, 2 * M))
    within = [[0, 0] for _ in range(M)]  # guide wins, plain wins (ties dropped)
    for _ in range(n_rankings):
        ms = rng.choice(M, 3, replace=False)
        items, util = [], []
        for m in ms:
            same = rng.random() < P_SAME
            g = base[m] + guide_effect[m] + rng.gumbel()
            p = g if same else base[m] + rng.gumbel()
            items += [2 * m, 2 * m + 1]
            util += [g, p]
        order = np.argsort(util)[::-1]
        ranked = [items[i] for i in order]
        u = dict(zip(items, util))
        for i, a in enumerate(ranked):
            for b in ranked[i + 1:]:
                if u[a] > u[b]:
                    wins[a, b] += 1
                else:
                    wins[a, b] += 0.5
                    wins[b, a] += 0.5
        for m in ms:
            if util[list(items).index(2 * m)] == util[list(items).index(2 * m + 1)]:
                continue
            within[m][0 if u[2 * m] > u[2 * m + 1] else 1] += 1
    # Bradley-Terry by MM on the rank-broken pairs, per model (guide and plain versions pooled)
    W = np.zeros((M, M))
    for a in range(2 * M):
        for b in range(2 * M):
            if a // 2 != b // 2:
                W[a // 2, b // 2] += wins[a, b]
    s = np.ones(M)
    for _ in range(200):
        n = W + W.T
        s = W.sum(1) / ((n / (s[:, None] + s[None, :])).sum(1))
        s /= np.exp(np.log(s).mean())
    pooled = [sum(w[0] for w in within), sum(w[1] for w in within)]
    return np.log(s), within, pooled


def run(n_rankings, model_gap, effects, reps=200):
    est, per, pool = [], np.zeros(M), 0
    for _ in range(reps):
        ls, within, pooled = simulate(n_rankings, model_gap, effects)
        est.append(ls)
        pool += binomtest(pooled[0], sum(pooled)).pvalue < 0.05
        per += [binomtest(g, g + p).pvalue < 0.05 if g + p else 0 for g, p in within]
    est = np.array(est)
    sd = est.std(0).mean()
    # how often the estimated order of neighbouring models is right
    right = (np.diff(est, axis=1) < 0).mean()
    return dict(rankings=n_rankings, sd_log_strength=round(float(sd), 3), neighbours_in_order=round(float(right), 2),
                pooled_found=round(float(pool) / reps, 2), per_model_found=np.round(per / reps, 2).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    # Mixed guide effects, as the model judges found (helps some kinds of text, hurts others); models 0.15 apart.
    mixed = np.array([0.8, 0.4, 0.4, 0.0, 0.0, -0.4, -0.4, -0.8])
    out = {"design": "3 of 8 models per text, both conditions, ranked; 302 rankings = every text once (2 raters x 5 of 10 sets)",
           "model_gap_logit": 0.15, "mixed_effects_logit": mixed.tolist(), "runs": []}
    for n in (302, 453):
        r = run(n, 0.15, mixed, a.reps)
        r["scenario"] = "mixed"
        out["runs"].append(r)
        print(r)
    r = run(302, 0.15, np.full(M, 0.3), a.reps)
    r["scenario"] = "uniform 0.3"
    out["runs"].append(r)
    print(r)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
