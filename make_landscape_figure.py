#!/usr/bin/env python3
"""
Project: Mechanistic bridges from receptors to whole-brain dynamics: promise
and limits of master-equation mean-field models (arXiv:2608.00306)
Date: 19/10/2026
Authors: YBossard + Claude Opus 5.5

make_landscape_figure.py
========================

Supplementary figure of the sigmoid benchmark (label fig:sigmoid_landscape):
landscape V(xi) = -(ln pi)/N of the exact stationary distribution
pi_K ~ C(N,K) cosh^N(Delta xi_K), for three slopes (large-N limit, Stirling).
Writes covariance_sigmoid_landscape.pdf and .png in the current folder.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.8,
                     "xtick.color": "#555550", "ytick.color": "#555550"})

def V(d, x):
    return 0.5 * (1 + x) * np.log1p(x) + 0.5 * (1 - x) * np.log1p(-x) - np.log(np.cosh(d * x))

xi = np.linspace(-0.95, 0.95, 2001)
series = [(0.5, "#2a78d6", "-", r"$|\Delta|=0.5$: one narrow valley"),
          (1.0, "#eb6834", "--", r"$|\Delta|=1$: flat bottom"),
          (1.2, "#1baf7a", "-.", r"$|\Delta|=1.2$: two valleys")]

fig, ax = plt.subplots(figsize=(4.4, 2.9))
ax.grid(True, color="#e6e6e1", linewidth=0.6)
ax.set_axisbelow(True)
for d, col, ls, lab in series:
    v = V(d, xi); v -= v.min()
    ax.plot(xi, v, color=col, ls=ls, lw=1.8, label=lab, solid_capstyle="round")
# mark the two minima of the |Delta| = 1.2 landscape
u = brentq(lambda u: u - np.tanh(1.2 * u), 1e-9, 1)
vmin = V(1.2, xi).min()
for s, txt in [(-1, r"$-u$"), (1, r"$+u$")]:
    ax.plot([s * u], [V(1.2, s * u) - vmin], "o", ms=5, color="#1baf7a", mec="white", mew=1.0, zorder=5)
    ax.annotate(txt, (s * u, 0.0), xytext=(s * u, 0.012), ha="center", fontsize=9, color="#333330")
# direct labels
ax.text(0.61, 0.134, r"$|\Delta|=0.5$", color="#333330", fontsize=9, ha="left")
ax.text(0.83, 0.108, r"$|\Delta|=1$", color="#333330", fontsize=9, ha="right")
ax.text(0.0, 0.054, r"$|\Delta|=1.2$", color="#333330", fontsize=9, ha="center")
ax.set_xlim(-0.95, 0.95)
ax.set_ylim(-0.004, 0.15)
ax.set_xlabel(r"relative deviation $\xi=(m-m_0)/m_0$")
ax.set_ylabel(r"$V(\xi)$")
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
ax.legend(loc="upper center", frameon=False, fontsize=8, ncol=1, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout()
fig.savefig("covariance_sigmoid_landscape.pdf")
fig.savefig("covariance_sigmoid_landscape.png", dpi=200)
print("u =", u)
