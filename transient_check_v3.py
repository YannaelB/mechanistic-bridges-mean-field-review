#!/usr/bin/env python3
"""
Project: Mechanistic bridges from receptors to whole-brain dynamics: promise
and limits of master-equation mean-field models (arXiv:2608.00306)
Date: 19/10/2026
Authors: YBossard + Claude Opus 5.5

transient_check_v3.py
=====================

Exact transient of the master equation after a step of the input, compared with the
covariance equations of the Supplementary:
  (A) complete second-order : A + A''c/2 + (F'^2 - 1) c + g^2 + g h          [(i), (ii), (iii)]
  (B) corrected             : A + (F'^2 - 1) c + v^2,  v = g + h/2 = T d<m>/dt  [(ii), (iii)]
  (B) without (iii)         : A + (F'^2 - 1) c + g^2                          [(ii) only]
  (C) stationary leading order: A + (F'^2 - 1) c                              [no source term]
  published                 : A + g^2 + 2 (F' - 1) c                          [El Boustani and Destexhe, 2009]
with g = F(<m>) - <m>, h = F''(<m>) c, A = F (1/T - F)/N. All are coupled to the same
second-order mean equation T d<m>/dt = g + h/2.

Transfer function (full-range sigmoid, m0 = 1/(2T)), with an input theta:
  F_theta(m) = m0 [1 + tanh(Delta (m - m0)/m0 + theta)].
Its logit is affine in m, so the initial stationary law is explicit. For t < 0, theta = theta0,
chosen so that the fixed point is m_old; at t = 0, theta -> 0 and the fixed point jumps to m0.

No sampling: the probability vector is propagated exactly, d pi/dt = (P^T - I) pi / T,
with scipy.sparse.linalg.expm_multiply.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.special import gammaln
from scipy.stats import binom
from scipy.integrate import solve_ivp

T, N = 0.005, 4000
M0 = 1.0 / (2.0 * T)
K = np.arange(N + 1)
M = K / (N * T)


def F(d, theta):
    return lambda m: M0 * (1 + np.tanh(d * (m - M0) / M0 + theta))


def derivs(d, theta, nu):
    t = np.tanh(d * (nu - M0) / M0 + theta)
    return M0 * (1 + t), d * (1 - t * t), -2 * d * d / M0 * t * (1 - t * t)


def explicit_law(d, theta):
    """logit(T F) = a + b K/N with b = 4 d, a = 2 theta - 2 d: pi_K ~ C(N,K) e^{aK} (1 + e^{a + bK/N})^N."""
    a, b = 2 * theta - 2 * d, 4 * d
    lw = gammaln(N + 1) - gammaln(K + 1) - gammaln(N - K + 1) + a * K + N * np.logaddexp(0, a + b * K / N)
    w = np.exp(lw - lw.max())
    return w / w.sum()


def transition_matrix(f):
    p = np.clip(T * f(M), 0, 1)
    rows, cols, vals = [], [], []
    for i in range(N + 1):
        mu, sd = N * p[i], np.sqrt(N * p[i] * (1 - p[i])) + 1.0
        lo, hi = max(0, int(mu - 12 * sd)), min(N, int(mu + 12 * sd) + 1)
        kk = np.arange(lo, hi + 1)
        rows.append(np.full(kk.size, i)); cols.append(kk); vals.append(binom.pmf(kk, N, p[i]))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(N + 1, N + 1))


def rhs_c(variant, d, nu, c):
    f, f1, f2 = derivs(d, 0.0, nu)
    A = f * (1 / T - f) / N
    A2 = ((1 / T - 2 * f) * f2 - 2 * f1**2) / N
    g, h = f - nu, f2 * c
    v = g + 0.5 * h
    if variant == '(A) complete':
        return A + 0.5 * A2 * c + (f1**2 - 1) * c + g * g + g * h
    if variant == '(B) corrected':
        return A + (f1**2 - 1) * c + v * v
    if variant == '(B) w/o (iii)':
        return A + (f1**2 - 1) * c + g * g
    if variant == '(C) stationary':
        return A + (f1**2 - 1) * c
    return A + g * g + 2 * (f1 - 1) * c                    # published


VARIANTS = ['(A) complete', '(B) corrected', '(B) w/o (iii)', '(C) stationary', 'published']


def run(d, m_old, t_max=8.0, n_t=161):
    xi_old = (m_old - M0) / M0
    theta0 = np.arctanh(xi_old) - d * xi_old
    pi0 = explicit_law(d, theta0)
    P = transition_matrix(F(d, 0.0))
    Q = (P.T - sp.identity(N + 1, format='csr')) / T
    ts = np.linspace(0, t_max * T, n_t)
    pis = expm_multiply(Q, pi0, start=0, stop=ts[-1], num=n_t, endpoint=True)
    ex_mean = pis @ M
    ex_var = pis @ M**2 - ex_mean**2
    sols = {}
    for k in VARIANTS:
        def f(t, y, k=k):
            nu, c = y
            f0, f1, f2 = derivs(d, 0.0, nu)
            return [(f0 - nu + 0.5 * f2 * c) / T, rhs_c(k, d, nu, c) / T]
        sols[k] = solve_ivp(f, (0, ts[-1]), [ex_mean[0], ex_var[0]], t_eval=ts, rtol=1e-10, atol=1e-12).y
    print(f"\nDelta = {d:+.2f}: input step moves the fixed point from {m_old:g} to {M0:g} Hz "
          f"(theta0 = {theta0:+.4f}). Exact variance: {ex_var[0]:.3f} at t = 0, peak {ex_var.max():.2f} Hz^2 "
          f"at t = {ts[ex_var.argmax()]/T:.2f} T, {ex_var[-1]:.3f} at t = {t_max:g} T")
    for k in VARIANTS:
        err = np.abs(sols[k][1] - ex_var)
        print(f"  {k:>15}: max |c - c_exact| = {err.max():8.3f} Hz^2 ({100*err.max()/ex_var.max():5.1f}% of the peak); "
              f"peak {sols[k][1].max():8.2f}")
    print(f"  max |c_(A) - c_(B)| over the transient = {np.abs(sols['(A) complete'][1] - sols['(B) corrected'][1]).max():.3f} Hz^2")
    print(f"  {'t/T':>5} | {'exact':>8} | " + " ".join(f"{k:>14}" for k in VARIANTS))
    for i in range(0, n_t, 20):
        print(f"  {ts[i]/T:5.1f} | {ex_var[i]:8.3f} | " + " ".join(f"{sols[k][1][i]:14.3f}" for k in VARIANTS))


if __name__ == '__main__':
    for d, m_old in [(-0.5, 60.0), (0.5, 60.0), (0.5, 95.0)]:
        run(d, m_old)
