"""CLADE: coupled, covariate-adjusted effects of a frozen model's layer representations.

Notation (one row per unit, e.g. a chromatin loop):
  Y      measured phenotype (n,)              e.g. log Hi-C loop strength
  R      layer representation features (n, L) e.g. anchor-to-anchor cosine per block
  W      covariates the model also reads (n, p), e.g. span, motif scores, orientation

For each layer l the target is the partial-regression coefficient
    theta_l = E[r_Y r_T,l] / E[r_T,l^2],
with r_Y = Y - E[Y | W] and r_T,l = R_l - E[R_l | W]. The nuisance regressions are
linear projections on a fixed design built from W (an intercept is always included).
The L estimates are treated as one system: their joint covariance is the empirical
covariance of the stacked influence functions, which gives simultaneous bands, block
Wald tests and a GLS depth-trend test.
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats


def _design(W):
    W = np.asarray(W, float)
    if W.ndim == 1:
        W = W[:, None]
    has_const = np.any(np.all(np.isclose(W, W[0]), axis=0))
    return W if has_const else np.column_stack([np.ones(len(W)), W])


def residualize(M, W):
    """Residuals of each column of M after least-squares projection on the design of W."""
    X = _design(W)
    M = np.asarray(M, float)
    beta, *_ = np.linalg.lstsq(X, M, rcond=None)
    return M - X @ beta, X.shape[1]


@dataclass
class CladeFit:
    theta: np.ndarray          # (L,) per-layer effects
    se: np.ndarray             # (L,) heteroskedasticity-robust standard errors
    cov: np.ndarray            # (L, L) joint covariance of theta-hat
    corr: np.ndarray           # (L, L) cross-layer correlation of theta-hat
    se_homo: np.ndarray        # (L,) classical OLS standard errors
    df: int                    # residual degrees of freedom of the per-layer OLS fit
    n: int

    @property
    def z(self):
        return self.theta / self.se

    @property
    def t_homo(self):
        return self.theta / self.se_homo


def fit(Y, R, W):
    """Estimate all layer effects jointly."""
    Y = np.asarray(Y, float)
    R = np.asarray(R, float)
    n, L = R.shape
    rY, k = residualize(Y[:, None], W)
    rY = rY[:, 0]
    rT, _ = residualize(R, W)
    vT = (rT ** 2).mean(0)
    theta = (rY[:, None] * rT).mean(0) / vT
    eps = rY[:, None] - rT * theta                       # (n, L) per-layer residuals
    phi = rT * eps / vT                                  # influence functions
    cov = phi.T @ phi / n / n
    se = np.sqrt(np.diag(cov))
    corr = cov / np.outer(se, se)
    df = n - k - 1
    sigma2 = (eps ** 2).sum(0) / df
    se_homo = np.sqrt(sigma2 / (rT ** 2).sum(0))
    return CladeFit(theta, se, cov, corr, se_homo, df, n)


def robustness_value(t, df, q=1.0, alpha=None):
    """Omitted-variable robustness value RV_q (Cinelli and Hazlett, 2020).

    With alpha set, returns RV_{q,alpha}: the strength of confounding (common partial R^2
    with treatment and outcome) that would make the estimate statistically insignificant."""
    f = q * abs(t) / np.sqrt(df)
    if alpha is not None:
        f = f - abs(stats.t.ppf(alpha / 2, df - 1)) / np.sqrt(df - 1)
    if f <= 0:
        return 0.0
    return 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)


def partial_r2(t, df):
    return t ** 2 / (t ** 2 + df)


def spectrum(corr):
    """Eigenvalues (descending), leading-mode share, participation-ratio effective rank and
    the mean depth-lag kernel k(h) = mean_l corr[l, l+h]."""
    ev = np.sort(np.linalg.eigvalsh(corr))[::-1]
    L = corr.shape[0]
    r_eff = np.trace(corr) ** 2 / np.sum(corr ** 2)
    kernel = np.array([np.mean(np.diag(corr, h)) for h in range(L)])
    return dict(eigenvalues=ev, top_share=ev[0] / ev.sum(), r_eff=r_eff, kernel=kernel)


def block_wald(f, S):
    """Coupled Wald statistic for H0: theta_S = 0, chi-square with |S| degrees of freedom."""
    S = list(S)
    th = f.theta[S]
    V = f.cov[np.ix_(S, S)]
    W = float(th @ np.linalg.solve(V, th))
    return W, len(S), float(stats.chi2.sf(W, len(S)))


def block_wald_two(z1, z2, r):
    """Closed form for two coordinates with correlation r: the statistic splits into the
    common (z1 + z2) and contrast (z2 - z1) directions."""
    common = (z1 + z2) ** 2 / (2 * (1 + r))
    contrast = (z2 - z1) ** 2 / (2 * (1 - r))
    return common + contrast, common, contrast


def maxt_critical(corr, alpha=0.05, nsim=200000, seed=0):
    """Two-sided simultaneous critical value: the (1 - alpha) quantile of max_l |Z_l|,
    Z ~ N(0, corr), computed by Monte Carlo."""
    rng = np.random.default_rng(seed)
    Lc = np.linalg.cholesky(corr + 1e-12 * np.eye(len(corr)))
    Z = rng.standard_normal((nsim, len(corr))) @ Lc.T
    return float(np.quantile(np.abs(Z).max(1), 1 - alpha))


def gls_trend(f):
    """GLS slope of theta on layer index using the joint covariance; returns (slope, se, z)."""
    L = len(f.theta)
    X = np.column_stack([np.ones(L), np.arange(L)])
    Vi = np.linalg.inv(f.cov)
    A = np.linalg.inv(X.T @ Vi @ X)
    b = A @ X.T @ Vi @ f.theta
    se = np.sqrt(A[1, 1])
    return float(b[1]), float(se), float(b[1] / se)


def summary(f, deep_block=None, alpha=0.05):
    L = len(f.theta)
    deep_block = deep_block or [L - 2, L - 1]
    lines = ["layer   theta      SE       z      95% CI"]
    for l in range(L):
        lo, hi = f.theta[l] - 1.96 * f.se[l], f.theta[l] + 1.96 * f.se[l]
        lines.append(f"L{l:<5d} {f.theta[l]:+.4f}  {f.se[l]:.4f}  {f.z[l]:+.2f}  [{lo:+.3f}, {hi:+.3f}]")
    sp = spectrum(f.corr)
    lines.append(f"leading-mode share = {100 * sp['top_share']:.2f}%   effective rank = {sp['r_eff']:.3f}")
    lines.append("depth-lag kernel k(h) = " + " ".join(f"{v:.3f}" for v in sp["kernel"]))
    W, d, p = block_wald(f, deep_block)
    lines.append(f"coupled Wald on layers {deep_block}: chi2_{d} = {W:.3f}, P = {p:.3e}")
    lines.append(f"max-t critical value, all {L} layers = {maxt_critical(f.corr, alpha):.3f}; "
                 f"Bonferroni = {stats.norm.ppf(1 - alpha / (2 * L)):.3f}")
    s, se, z = gls_trend(f)
    lines.append(f"GLS depth-trend slope = {s:+.4f} (SE {se:.4f}, z = {z:+.2f})")
    l = L - 1
    lines.append(f"final layer: t (classical SE) = {f.t_homo[l]:.2f}, df = {f.df}, "
                 f"partial R2 = {partial_r2(f.t_homo[l], f.df):.4f}, "
                 f"RV_q=1 = {100 * robustness_value(f.t_homo[l], f.df):.1f}%, "
                 f"RV_q=1,alpha=0.05 = {100 * robustness_value(f.t_homo[l], f.df, alpha=0.05):.1f}%")
    return "\n".join(lines)
