#!/usr/bin/env python3
"""
Project: Mechanistic bridges from receptors to whole-brain dynamics: promise
and limits of master-equation mean-field models (arXiv:2608.00306)
Date: 19/10/2026
Authors: YBossard + Claude Opus 5.5

covariance_closure_tables_v3.py
===============================

Reproduces every number of the Supplementary subsection
"Benchmarks and validity domain of the covariance closures" (preprint version 3):
the affine table, the sigmoid table and the numbers quoted in the text.

No sampling is used. Every value is a closed form or an explicit finite sum.

Model (El Boustani & Destexhe 2009): one population of N neurons, time bin T.
The whole state is redrawn at rate 1/T:   K' ~ Binomial(N, T F(m)),  m' = K'/(N T).

Parameters: N = 4000, T = 5 ms and a fixed point m0 = 1/(2T) = 100 Hz, so that the
firing probability at the fixed point is T m0 = 1/2 and A^{-1} = m0 (1/T - m0)/N = 2.5 Hz^2.

Equations compared at a stationary state (fixed point nu, gain J = F'(nu)):
  (A) complete second-order     0 = A^{-1} + (1/2) A^{-1}'' c + (J^2 - 1) c   [terms (i), (ii), (iii)]
  (B) corrected = (C) stationary 0 = A^{-1} + (J^2 - 1) c                     [terms (ii), (iii)]
  published (mean-value closure) 0 = A^{-1} + 2 (J - 1) c
(at a stationary state the source term vanishes; for the transfer functions below,
F''(m0) = 0 and A^{-1}''(m0) = -2 J^2/N, so (A) gives A^{-1}/(1 - J^2 + J^2/N)).

Affine   F(m) = m0 + Delta (m - m0):
    exact = A^{-1}/(1 - Delta^2 + Delta^2/N)   (law of total variance; equal to (A))
Sigmoid  F(m) = m0 [1 + tanh(Delta (m - m0)/m0)], which spans the full range (0, 1/T)
    because m0 = 1/(2T). Its logit is affine, the chain is reversible, and the exact
    stationary law is explicit:  pi_K ~ C(N, K) cosh^N(Delta (m_K - m0)/m0).

"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import gammaln, gamma
from scipy.stats import binom
from scipy.optimize import brentq, fsolve
from scipy.linalg import solve_continuous_lyapunov

T, N = 0.005, 4000
M0 = 1.0 / (2.0 * T)                       # 100 Hz
A_INV = M0 * (1.0 / T - M0) / N            # 2.5 Hz^2

K = np.arange(N + 1)
M = K / (N * T)                            # activity grid (Hz)
XI = (M - M0) / M0                         # relative deviation, in [-1, 1]
LOG_BINOM = gammaln(N + 1) - gammaln(K + 1) - gammaln(N - K + 1)


def a_inv(f):
    return f * (1.0 / T - f) / N


# ----------------------------------------------------------------------------
# stationary solutions of the covariance equations at m0
# ----------------------------------------------------------------------------
def complete(d):
    """(A): keeps term (i); equals the exact affine variance."""
    den = 1 - d**2 + d**2 / N
    return A_INV / den if den > 0 and abs(d) < 1 else np.inf


def corrected(d):
    """(B) = (C) at a stationary state."""
    return A_INV / (1 - d**2) if abs(d) < 1 else np.inf


def published(d):
    return A_INV / (2 * (1 - d)) if d < 1 else np.inf


# ----------------------------------------------------------------------------
# exact stationary laws
# ----------------------------------------------------------------------------
def logistic_law(d):
    """Explicit stationary law of the master equation for the full-range sigmoid."""
    lw = LOG_BINOM + N * np.log(np.cosh(d * XI))
    w = np.exp(lw - lw.max())
    return w / w.sum(), lw


def moments(p, mask=None):
    if mask is not None:
        p = np.where(mask, p, 0.0)
        p = p / p.sum()
    mu = (p * M).sum()
    return mu, (p * (M - mu) ** 2).sum()


def sigmoid(d):
    return lambda m: M0 * (1.0 + np.tanh(d * (m - M0) / M0))


def affine(d):
    return lambda m: M0 + d * (m - M0)


def chain_stationary(F):
    """Independent check: solve pi P = pi for the (N+1)-state chain by sparse LU."""
    p = np.clip(T * F(M), 0.0, 1.0)
    rows, cols, vals = [], [], []
    for i in range(N + 1):
        mu, sd = N * p[i], np.sqrt(N * p[i] * (1 - p[i])) + 1.0
        lo, hi = max(0, int(mu - 40 * sd)), min(N, int(mu + 40 * sd) + 1)
        kk = np.arange(lo, hi + 1)
        pm = binom.pmf(kk, N, p[i])
        keep = pm > 1e-300
        rows.append(np.full(keep.sum(), i)); cols.append(kk[keep]); vals.append(pm[keep])
    P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N + 1, N + 1))
    A = (P.T - sp.identity(N + 1, format='csr')).tolil()
    A[N, :] = np.ones(N + 1)
    b = np.zeros(N + 1); b[N] = 1.0
    pi = spla.spsolve(A.tocsc(), b)
    pi = np.clip(pi, 0, None)
    return pi / pi.sum()


# ----------------------------------------------------------------------------
# analytical (Laplace) estimates from the explicit law
#   log pi ~ -N [ (1 - D^2) xi^2/2 + (1 + D^4) xi^4/12 + ... ]
# ----------------------------------------------------------------------------
def saturation_correction(d):
    """Relative reduction of the variance by the quartic term: (1 + D^4)/(N (1 - D^2)^2)."""
    return (1 + d**4) / (N * (1 - d**2) ** 2)


def critical_variance():
    """|D| = 1: p(xi) ~ exp(-N xi^4/6)  ->  c = m0^2 Gamma(3/4)/Gamma(1/4) sqrt(6/N)."""
    return M0**2 * gamma(0.75) / gamma(0.25) * np.sqrt(6.0 / N)


def amplitude(d):
    """|D| > 1: u* = tanh(|D| u*), a = m0 u* (stable states or period-two cycle)."""
    u = brentq(lambda u: u - np.tanh(abs(d) * u), 1e-12, 1.0)
    return M0 * u, abs(d) * (1 - u * u)


def closures_in_state(d, start):
    """D > 1: stationary second-order mean and covariance equations near a stable state."""
    def derivs(nu):
        t = np.tanh(d * (nu - M0) / M0)
        return M0 * (1 + t), d * (1 - t * t), -2 * d * d / M0 * t * (1 - t * t)
    out = {}
    for name in ['corrected', 'published']:
        def eqs(y):
            nu, c = y
            f, f1, f2 = derivs(nu)
            if name == 'corrected':
                dc = a_inv(f) + (f1**2 - 1) * c
            else:
                dc = a_inv(f) + (f - nu) ** 2 + 2 * (f1 - 1) * c
            return [f - nu + 0.5 * f2 * c, dc]
        out[name] = fsolve(eqs, [start, 2.0])
    return out


def log10_leak(d, lw):
    """log10 of the probability, per transition, of crossing m0 from the high state."""
    p = T * sigmoid(d)(M)
    hi = M > M0
    lw_hi = np.where(hi, lw, -np.inf)
    lpi = lw_hi - np.logaddexp.reduce(lw_hi[hi])
    lcross = binom.logcdf(N // 2 - (1 if N % 2 == 0 else 0), N, p)   # K' < N/2, i.e. m' < m0
    return np.logaddexp.reduce((lpi + lcross)[hi]) / np.log(10)


# ----------------------------------------------------------------------------
# quoted in the text: two populations (footnote) and small-jump model
# ----------------------------------------------------------------------------
def two_populations(kE=0.8, kI=-1.3, NE=4000, NI=1000):
    D0 = M0 * (1 / T - M0)
    J = np.array([[kE, kI], [kE, kI]])
    r = kE**2 / NE + kI**2 / NI
    s = D0 * r / (1 - (kE + kI) ** 2 + r)
    exact = np.array([(D0 - s) / NE + s, (D0 - s) / NI + s, s])
    Ainv = np.diag([D0 / NE, D0 / NI])
    C = np.linalg.solve(np.eye(4) - np.kron(J, J), Ainv.ravel()).reshape(2, 2)
    Cp = solve_continuous_lyapunov(J - np.eye(2), -Ainv)
    return exact, np.array([C[0, 0], C[1, 1], C[0, 1]]), np.array([Cp[0, 0], Cp[1, 1], Cp[0, 1]])


def fmt(x):
    return r'\infty' if np.isinf(x) else f'{x:.3f}'


def fmt_ratio(x):
    if np.isnan(x) or np.isinf(x) or x == 0:
        return '--'
    if x >= 0.00995:
        return f'{x:.3f}'
    e = int(np.floor(np.log10(x)))
    return rf'{x / 10**e:.1f}\times10^{{{e}}}'


def math(s):
    return s if s == '--' else f'${s}$'


def main():
    print(f"N = {N}, T = {T*1e3:g} ms, m0 = 1/(2T) = {M0:g} Hz, A^-1 = {A_INV:.3f} Hz^2\n")

    # ------------------------------ affine ------------------------------
    print("TABLE 1  affine F(m) = m0 + Delta (m - m0)   [exact = complete equation (A)]")
    rows1 = []
    for d in [1.5, 0.99, 0.9, 0.5, 0.0, -0.5, -0.9, -0.99, -1.5]:
        ex, co, pu = complete(d), corrected(d), published(d)
        r1 = co / ex if np.isfinite(ex) else np.nan
        r2 = pu / ex if np.isfinite(ex) else np.nan
        print(f"  {d:+5.2f}: exact {fmt(ex):>8}  corrected {fmt(co):>8}  published {fmt(pu):>8}  "
              f"ratios {fmt_ratio(r1)} {fmt_ratio(r2)}")
        lab = '$0$' if d == 0 else f'${d:+.2f}$'
        rows1.append(" & ".join([lab, math(fmt(ex)), math(fmt(co)), math(fmt(pu)),
                                 math(fmt_ratio(r1)), math(fmt_ratio(r2))]) + r" \\")
    for d in [0.9, -0.99]:
        pi = chain_stationary(affine(d))
        print(f"  check (linear system) Delta = {d:+.2f}: var {moments(pi)[1]:.4f} vs closed form {complete(d):.4f}")

    # ------------------------------ sigmoid -----------------------------
    print("\nTABLE 2  sigmoid F(m) = m0 [1 + tanh(Delta (m - m0)/m0)],  m0 = 1/(2T): explicit stationary law")
    rows2 = []
    d = 1.2
    w, lw = logistic_law(d)
    mu_w, v_w = moments(w, M > M0)
    a, gain = amplitude(d)
    cl = closures_in_state(d, M0 + a)
    (nu_c, c_c), (nu_p, c_p) = cl['corrected'], cl['published']
    print(f"  +1.20 stable states m0 +- a, a = {a:.3f} Hz (local gain {gain:.4f}): exact within-state var {v_w:.4f} "
          f"(mean {mu_w:.3f}); corrected {c_c:.4f} (mean {nu_c:.3f}); published {c_p:.4f} (mean {nu_p:.3f}); "
          f"crossing probability per transition 10^{log10_leak(d, lw):.1f}; global law var {moments(w)[1]:.1f}")
    rows2.append(" & ".join([r"$+1.20$ (each state)", math(f'{v_w:.3f}'), math(f'{c_c:.3f}'),
                             math(fmt_ratio(c_c / v_w)), math(f'{c_p:.3f}'), math(fmt_ratio(c_p / v_w))]) + r" \\")
    for d in [1.0, 0.99, 0.9, 0.5, -0.5, -0.9, -0.99, -1.0, -1.2, -2.0]:
        w, _ = logistic_law(d)
        mu, var = moments(w)
        co, pu = corrected(d), published(d)
        if abs(d) < 1:
            r = saturation_correction(d)
            note = f"linear {A_INV/(1-d*d):.4f}, relative saturation correction {r:.4f}" + (
                f" -> {A_INV/(1-d*d)*(1-r):.4f}" if r < 0.5 else " (> 1: expansion invalid)")
        elif abs(d) == 1:
            note = f"critical law {critical_variance():.3f}"
        else:
            a, _ = amplitude(d)
            note = f"period-two amplitude a = {a:.3f} Hz, a^2 = {a*a:.1f}"
        print(f"  {d:+5.2f}: exact var {var:9.4f} (mean {mu:.4f}) | corrected {fmt(co):>8} | published {fmt(pu):>8} | {note}")
        rows2.append(" & ".join([f'${d:+.2f}$', math(f'{var:.3f}'), math(fmt(co)), math(fmt_ratio(co / var)),
                                 math(fmt(pu)), math(fmt_ratio(pu / var))]) + r" \\")
    for d in [0.9, -1.2]:
        pi = chain_stationary(sigmoid(d))
        w, _ = logistic_law(d)
        print(f"  check (linear system) Delta = {d:+.2f}: max |pi_chain - pi_explicit| = {np.abs(pi - w).max():.1e}")

    # ------------------------------ text --------------------------------
    print("\nQUOTED IN THE TEXT")
    print(f"  validity criterion N (1 - D^2)^2 vs 1 + D^4: " + ", ".join(
        f"D={d}: {N*(1-d*d)**2:.2f} vs {1+d**4:.2f}" for d in [0.5, 0.9, 0.99]))
    print(f"  corrected/complete factor 1 + D^2/(N(1-D^2)) at |D| = 0.99: {1 + 0.99**2/(N*(1-0.99**2)):.4f}")
    print(f"  small-jump model (affine): c = m0/(2 N T (1 - D)); published/small-jump = 1 - T m0 = {1 - T*M0:.2f}; "
          + ", ".join(f"D={d:+g}: {M0/(2*N*T*(1-d)):.3f}" for d in [0.5, 0.0, -0.5]))
    ex2, co2, pu2 = two_populations()
    print(f"  two populations (kE=0.8, kI=-1.3, NE=4000, NI=1000), [c_EE, c_II, c_EI]: exact {np.round(ex2, 3)}, "
          f"corrected {np.round(co2, 3)}, published {np.round(pu2, 3)}, exact/published {np.round(ex2/pu2, 1)}")

    print("\n% ---- LaTeX rows, affine table ----")
    print("\n".join(rows1))
    print("% ---- LaTeX rows, sigmoid table ----")
    for i, r in enumerate(rows2):
        if r.startswith('$-0.50$'):
            print(r"\midrule")
        print(r)


if __name__ == '__main__':
    main()
