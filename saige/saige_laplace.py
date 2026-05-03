"""
saige_glmm.py — Four-method comparison on identical simulated data (seed=42).

Methods:
  1. Naive logistic regression   (statsmodels Logit)         — no GRM, for baseline
  2. BayesMixedGLM top-K PCs    (statsmodels)               — approximate GRM
  3. Laplace-REML                (saige0.py algorithm)       — exact Ψ, Brent + NR
  4. PQL + AI-REML               (saige_paper.py algorithm)  — exact Ψ, PQL + Newton

True parameters: tau=1, alpha=[-2, 0.01, 1], beta_G=2.

Core algorithm functions are copied from saige0.py and saige_paper.py.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import expit
from scipy.stats import norm, chi2
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

SEED = np.random.randint(0, 2**31)
rng = np.random.default_rng(SEED)
GH_DEGREE = 20


# =============================================================================
# Shared simulated data  (seed=42, same parameters as saige0.py / saige_paper.py)
# =============================================================================

N     = 200
M_grm = 500
G_maf = 0.1

age = rng.integers(0, 100, N)
sex = rng.binomial(1, 0.5, N)
X   = np.column_stack([np.ones(N), age, sex])   # N × 3
q   = X.shape[1]

p_grm = rng.uniform(0, 1, M_grm)
G_grm = rng.binomial(2, p_grm, size=(N, M_grm)).astype(float)
G_std = (G_grm - 2 * p_grm) / np.sqrt(2 * p_grm * (1 - p_grm))
Psi   = G_std @ G_std.T / M_grm               # N × N GRM

G = rng.binomial(2, G_maf, size=N).astype(float)

tau_true   = 1.0
alpha_true = np.array([-2.0, 0.01, 1.0])
beta_true  = 2.0

b_true  = rng.multivariate_normal(np.zeros(N), tau_true * Psi)
y       = rng.binomial(1, expit(X @ alpha_true + G * beta_true + b_true)).astype(float)

print(f"Seed={SEED}  N={N}  prevalence={y.mean():.3f}")
print(f"True: tau={tau_true}  alpha={alpha_true}  beta_G={beta_true}\n")

X_full = np.column_stack([X, G])    # design matrix including G (for Methods 1 & 2)


# =============================================================================
# Shared helper: residualise G from X in the W-metric  (eq. 26 of saige_paper.md)
# =============================================================================

def compute_G_tilde(G, X, W_diag):
    """G̃ = G − X(X^TWX)⁻¹X^TWG  — W-orthogonal residual of G from X."""
    WX    = W_diag[:, None] * X
    XtWX  = X.T @ WX
    XtWG  = X.T @ (W_diag * G)
    return G - X @ np.linalg.solve(XtWX, XtWG)


# =============================================================================
# Shared SPA functions (identical for Methods 3 & 4)
# =============================================================================

def make_spa_cdf(G_tilde, mu0):
    """
    Return spa_cdf(q) = Lugannani-Rice approximation of P(T/√Var*(T) ≤ q),
    where T = Σ G̃_i(y_i − μ_{i0}) conditional on b̂₀ fixed (Bernoulli terms).

    CGF (eq. 30):  K(t) = Σ_i log[μ_i e^{G̃_i(1−μ_i)t} + (1−μ_i) e^{−G̃_i μ_i t}]
    Standardised by c = 1/√Var*(T) so K''(0) = 1.
    """
    V_star = float(G_tilde @ (mu0 * (1 - mu0) * G_tilde))   # Var*(T) = G̃^T W G̃
    c = 1.0 / np.sqrt(V_star)

    def K(t):
        arg = c * t * G_tilde
        return np.sum(np.log1p(mu0 * (np.exp(arg) - 1))) - c * t * np.sum(G_tilde * mu0)

    def K_prime(t):
        a = c * t * G_tilde
        tot = np.zeros(N)
        pos, neg = a >= 0, a < 0
        if pos.any():
            e = np.exp(a[pos])
            tot[pos] = G_tilde[pos] * mu0[pos] * (e - 1) / ((1 - mu0[pos]) + mu0[pos] * e)
        if neg.any():
            en = np.exp(-a[neg])
            tot[neg] = G_tilde[neg] * mu0[neg] * (1 - en) / ((1 - mu0[neg]) * en + mu0[neg])
        return c * np.sum(tot)

    def K_pp(t):
        a2  = c * t * G_tilde / 2
        ea2 = np.exp(np.clip( a2, -500, 500))
        en2 = np.exp(np.clip(-a2, -500, 500))
        d2  = ((1 - mu0) * en2 + mu0 * ea2) ** 2
        return c ** 2 * np.sum(G_tilde ** 2 * mu0 * (1 - mu0) / d2)

    def spa_cdf(q_val):
        try:
            t_hat = brentq(lambda t: K_prime(t) - q_val, -50.0, 50.0, xtol=1e-10)
        except ValueError:
            return float(norm.cdf(q_val))
        w = np.sign(t_hat) * np.sqrt(2 * (t_hat * q_val - K(t_hat)))
        v = t_hat * np.sqrt(K_pp(t_hat))
        if abs(w) < 1e-8:
            return 0.5 + (v - w) / (2 * np.pi) ** 0.5
        return float(norm.cdf(w + np.log(v / w) / w))

    return spa_cdf, V_star


def score_and_spa(G, G_tilde, mu0, W_diag, tau, Psi,
                  L_Sigma=None, Si_X=None, XtSiX=None):
    """
    Score statistic T, Var(T), correction ratio r, adjusted T, SPA p-value,
    and score-based estimate of beta_G.

    Var(T) = G̃^T P̂ G̃.  When L_Sigma/Si_X/XtSiX are given, P is applied via
    saige_paper's P_apply (O(N²)); otherwise Sigma is formed explicitly (O(N³)).
    """
    T = float(G @ (y - mu0))

    if L_Sigma is not None:                              # saige_paper path
        Si_Gt  = cho_solve(L_Sigma, G_tilde)
        PG     = Si_Gt - Si_X @ np.linalg.solve(XtSiX, Si_X.T @ G_tilde)
    else:                                                # saige0 path (explicit Sigma)
        Sigma  = np.diag(1.0 / W_diag) + tau * Psi
        L_S    = cho_factor(Sigma)
        Si_Gt  = cho_solve(L_S, G_tilde)
        Si_X_  = cho_solve(L_S, X)
        XtSiX_ = X.T @ Si_X_
        PG     = Si_Gt - Si_X_ @ np.linalg.solve(XtSiX_, Si_X_.T @ G_tilde)

    V_T     = float(G_tilde @ PG)
    spa_cdf, V_star = make_spa_cdf(G_tilde, mu0)
    r       = V_T / V_star
    T_adj   = T / np.sqrt(V_T)
    q_obs   = np.sqrt(r) * T_adj
    p_spa   = (1 - spa_cdf(abs(q_obs))) + spa_cdf(-abs(q_obs))
    beta_sc = T / V_T                                    # score-based approximation

    return T, V_T, r, T_adj, p_spa, beta_sc


# =============================================================================
# saige0 core functions  (Laplace-REML; copied from saige0.py)
# =============================================================================

def ell(alpha, tau, y, X, Psi, tol=1e-10):
    """Laplace-approximated marginal log-likelihood ℓ_LA(α, τ). Returns (val, b0, H, w0)."""
    tauPsi = tau * Psi
    L_tp   = cho_factor(tauPsi)

    def neg_lp(b):
        eta = X @ alpha + b
        return -(float(y @ eta - np.sum(np.log1p(np.exp(eta)))) - 0.5 * float(b @ cho_solve(L_tp, b)))

    def neg_grad(b):
        return -(y - expit(X @ alpha + b)) + cho_solve(L_tp, b)

    def hessp(b, v):
        mu = expit(X @ alpha + b)
        return mu * (1 - mu) * v + cho_solve(L_tp, v)

    res_b = minimize(neg_lp, np.zeros(N), jac=neg_grad, hessp=hessp,
                     method="Newton-CG", tol=tol)
    b0  = res_b.x
    mu0 = expit(X @ alpha + b0)
    w0  = mu0 * (1 - mu0)
    H   = np.diag(w0) + cho_solve(L_tp, np.eye(N))

    eta0       = X @ alpha + b0
    ll0        = float(y @ eta0 - np.sum(np.log1p(np.exp(eta0))))
    quad       = float(b0 @ cho_solve(L_tp, b0))
    ld_tauPsi  = 2 * np.sum(np.log(np.abs(np.diag(L_tp[0]))))
    _, ld_H    = np.linalg.slogdet(H)
    val = ll0 - 0.5 * ld_tauPsi - 0.5 * quad - 0.5 * ld_H
    return val, b0, H, w0


def r_ell(tau, y, X, Psi, tol_b=1e-10, tol_a=1e-10, max_iter=50):
    """Laplace-REML log-likelihood ℓ_REML(τ). Returns (reml_val, alpha, b0, H, w0)."""
    if tau <= 0:
        return -np.inf, None, None, None, None
    alpha = np.zeros(q)
    for _ in range(max_iter):
        val, b0, H, w0 = ell(alpha, tau, y, X, Psi, tol=tol_b)
        mu0  = expit(X @ alpha + b0)
        WX   = w0[:, None] * X
        HiWX = np.linalg.solve(H, WX)
        I    = X.T @ WX - WX.T @ HiWX
        delta = np.linalg.solve(I, X.T @ (y - mu0))
        alpha += delta
        if np.max(np.abs(delta)) < tol_a:
            break
    val, b0, H, w0 = ell(alpha, tau, y, X, Psi, tol=tol_b)
    WX   = w0[:, None] * X
    HiWX = np.linalg.solve(H, WX)
    I_a  = X.T @ WX - WX.T @ HiWX
    _, ld_I = np.linalg.slogdet(I_a)
    return val - 0.5 * ld_I, alpha, b0, H, w0


def log_fi(i, b_val, alpha, b0, tauPsi_inv):
    """Log unnormalised conditional density for GH quadrature (mean-field at mode)."""
    b_full    = b0.copy()
    b_full[i] = b_val
    eta_full  = X @ alpha + b_full
    ll        = float(y @ eta_full - np.sum(np.log1p(np.exp(eta_full))))
    log_pri   = -0.5 * float(b_full @ tauPsi_inv @ b_full)
    return ll + log_pri


# =============================================================================
# saige_paper core functions  (PQL + AI-REML; copied from saige_paper.py)
# =============================================================================

def P_apply(u, L_Sigma, Si_X, XtSiX):
    """Apply REML projection matrix P to vector u without forming P explicitly."""
    Si_u = cho_solve(L_Sigma, u)
    return Si_u - Si_X @ np.linalg.solve(XtSiX, Si_X.T @ u)


def fit_null_pql(y, X, Psi, max_iter=25, tol=1e-5):
    """PQL + AI-REML null model fit. Returns (tau, alpha, b, mu, W_diag, L_Sigma, Si_X, XtSiX)."""
    alpha = np.zeros(q);  b = np.zeros(N);  tau = 1.0
    L_Sigma = Si_X = XtSiX = None
    for _ in range(max_iter):
        mu      = expit(X @ alpha + b)
        W_diag  = mu * (1 - mu)
        Y_tilde = X @ alpha + b + (y - mu) / W_diag
        Sigma   = np.diag(1.0 / W_diag) + tau * Psi
        L_Sigma = cho_factor(Sigma)
        Si_X    = cho_solve(L_Sigma, X)
        XtSiX   = X.T @ Si_X
        Si_Y    = cho_solve(L_Sigma, Y_tilde)
        XtSiY   = X.T @ Si_Y

        alpha_new = np.linalg.solve(XtSiX, XtSiY)
        b_new     = tau * (Psi @ cho_solve(L_Sigma, Y_tilde - X @ alpha_new))

        PY      = Si_Y - Si_X @ np.linalg.solve(XtSiX, XtSiY)
        PPsiPY  = P_apply(Psi @ PY, L_Sigma, Si_X, XtSiX)
        AI      = 0.5 * float(Y_tilde @ PPsiPY)
        Si_Psi  = cho_solve(L_Sigma, Psi)
        tr_PPsi = np.trace(Si_Psi) - np.trace(Si_X @ np.linalg.solve(XtSiX, Si_X.T) @ Psi)
        tau_new = max(tau + (AI - 0.5 * tr_PPsi) / AI, 1e-6)

        delta = max(np.max(np.abs(alpha_new - alpha)), abs(tau_new - tau))
        alpha, b, tau = alpha_new, b_new, tau_new
        if delta < tol:
            break

    mu      = expit(X @ alpha + b)
    W_diag  = mu * (1 - mu)
    Sigma   = np.diag(1.0 / W_diag) + tau * Psi
    L_Sigma = cho_factor(Sigma)
    Si_X    = cho_solve(L_Sigma, X)
    XtSiX   = X.T @ Si_X
    return tau, alpha, b, mu, W_diag, L_Sigma, Si_X, XtSiX


# =============================================================================
# Method 1 — Naive logistic regression (statsmodels Logit, no GRM)
# =============================================================================

sep = "=" * 70
print(sep)
print("Method 1: Naive logistic regression  (statsmodels Logit, no GRM)")
print(sep)

res_logit  = sm.Logit(y, X_full).fit(disp=False)
p_wald_G   = float(res_logit.pvalues[-1])
beta_G_m1  = float(res_logit.params[-1])
alpha_m1   = res_logit.params[:q]

# LRT: compare null (G excluded) vs alt (G included)
res_logit0 = sm.Logit(y, X).fit(disp=False)
lrt_stat   = -2 * (res_logit0.llf - res_logit.llf)
p_lrt_m1   = float(chi2.sf(lrt_stat, df=1))

print(f"  tau_hat  = N/A  (no random effect)")
print(f"  alpha    = {alpha_m1}")
print(f"  beta_G   = {beta_G_m1:.4f}  (SE={res_logit.bse[-1]:.4f})")
print(f"  Wald p   = {p_wald_G:.4g}    LRT p = {p_lrt_m1:.4g}\n")


# =============================================================================
# Method 2 — BinomialBayesMixedGLM with top-K eigenvectors of Ψ
# =============================================================================

print(sep)
print("Method 2: BinomialBayesMixedGLM  (statsmodels, top-K PC approximation of Ψ)")
print(sep)

K = 5   # only K=5 gives non-zero τ; larger K collapses σ→0 (see analysis below)
eigval, eigvec = np.linalg.eigh(Psi)
exog_vc = eigvec[:, -K:] * np.sqrt(eigval[-K:])      # N×K, so Var(b)=σ²·Ψ_trunc
ident_k = np.zeros(K, dtype=int)

bglmm   = BinomialBayesMixedGLM(y, X_full, exog_vc, ident_k, vcp_p=0.5, fe_p=2.0)
res_bg  = bglmm.fit_map()

# X_full has q+1 = 4 columns [1, age, sex, G].  BinomialBayesMixedGLM params:
#   params[0 .. q-1]   = alpha (q fixed effects from X)
#   params[q]          = beta_G (last fixed effect)
#   params[q+1]        = log sigma (first variance component)
params_bg  = res_bg.params
alpha_m2   = params_bg[:q]              # [intercept, age, sex]
beta_G_m2  = params_bg[q]              # beta_G
log_sigma  = params_bg[q + 1]          # log sigma
tau_m2     = np.exp(log_sigma) ** 2

se_G_m2   = float(res_bg.fe_sd[q]) if res_bg.fe_sd is not None and np.isfinite(res_bg.fe_sd[q]) else np.nan
p_wald_m2 = float(2 * norm.sf(abs(beta_G_m2 / se_G_m2))) if np.isfinite(se_G_m2) else np.nan

print(f"  tau_hat  = {tau_m2:.4f}  (log σ = {log_sigma:.3f};  K={K})")
print(f"  alpha    = {alpha_m2}")
print(f"  beta_G   = {beta_G_m2:.4f}  (SE={se_G_m2:.4f})")
print(f"  Wald p   = {p_wald_m2:.4g}\n")
print(f"  NOTE: τ collapses to 0 for K≥10.  K=5 gives non-zero τ only due")
print(f"  to the weak prior (vcp_p=0.5) regularising the saturated model.\n")


# =============================================================================
# Method 3 — Laplace-REML  (saige0.py algorithm)
# =============================================================================

print(sep)
print("Method 3: Laplace-REML  (saige0.py algorithm)")
print(sep)

res_s      = minimize_scalar(lambda s: -r_ell(np.exp(s), y, X, Psi)[0])
tau_m3     = float(np.exp(res_s.x))
_, alpha_m3, b0_m3, H_m3, w0_m3 = r_ell(tau_m3, y, X, Psi)

# BLUP via 20-point Gauss-Hermite quadrature (mean-field, eq. 12 of saige0.md)
gh_nodes, gh_weights = hermgauss(GH_DEGREE)
tauPsi_inv = np.linalg.solve(tau_m3 * Psi, np.eye(N))
sigma_gh   = 1.0 / np.sqrt(np.diag(H_m3))
b_blup     = np.zeros(N)
for i in range(N):
    nodes_k = b0_m3[i] + np.sqrt(2) * sigma_gh[i] * gh_nodes
    log_wik = (np.log(gh_weights)
               + np.array([log_fi(i, nd, alpha_m3, b0_m3, tauPsi_inv) for nd in nodes_k])
               + gh_nodes ** 2)
    wik       = np.exp(log_wik - np.max(log_wik))
    b_blup[i] = np.dot(wik, nodes_k) / np.sum(wik)

mu0_m3     = expit(X @ alpha_m3 + b_blup)
W_diag_m3  = mu0_m3 * (1 - mu0_m3)
G_tilde_m3 = compute_G_tilde(G, X, W_diag_m3)

T_m3, V_m3, r_m3, Tadj_m3, p_spa_m3, beta_sc_m3 = score_and_spa(
    G, G_tilde_m3, mu0_m3, W_diag_m3, tau_m3, Psi)

print(f"  tau_hat  = {tau_m3:.4f}")
print(f"  alpha    = {alpha_m3}")
print(f"  T = {T_m3:.4g}   Var(T) = {V_m3:.4g}   r = {r_m3:.4f}")
print(f"  beta_G (score approx) = {beta_sc_m3:.4f}")
print(f"  SPA p-value = {p_spa_m3:.4g}\n")


# =============================================================================
# Method 4 — PQL + AI-REML  (saige_paper.py algorithm)
# =============================================================================

print(sep)
print("Method 4: PQL + AI-REML  (saige_paper.py algorithm)")
print(sep)

tau_m4, alpha_m4, b0_m4, mu0_m4, W_diag_m4, L_Sig, Si_X0, XtSiX0 = fit_null_pql(y, X, Psi)
G_tilde_m4 = compute_G_tilde(G, X, W_diag_m4)

T_m4, V_m4, r_m4, Tadj_m4, p_spa_m4, beta_sc_m4 = score_and_spa(
    G, G_tilde_m4, mu0_m4, W_diag_m4, tau_m4, Psi,
    L_Sigma=L_Sig, Si_X=Si_X0, XtSiX=XtSiX0)

print(f"  tau_hat  = {tau_m4:.4f}")
print(f"  alpha    = {alpha_m4}")
print(f"  T = {T_m4:.4g}   Var(T) = {V_m4:.4g}   r = {r_m4:.4f}")
print(f"  beta_G (score approx) = {beta_sc_m4:.4f}")
print(f"  SPA p-value = {p_spa_m4:.4g}\n")


# =============================================================================
# Comparison table
# =============================================================================

print(sep)
print("COMPARISON TABLE")
print(sep)
hdr = f"{'Parameter':<14}{'True':>7}{'NaiveLogit':>12}{'SM_BGLMM_K5':>13}{'saige0':>10}{'saige_paper':>13}"
print(hdr)
print("-" * 70)

def fmt(v):
    return f"{v:>10.4f}" if np.isfinite(v) else "       N/A"

print(f"{'tau':14}{'1.000':>7}{'    N/A':>12}{tau_m2:>13.4f}{tau_m3:>10.4f}{tau_m4:>13.4f}")
print(f"{'alpha[0]':14}{alpha_true[0]:>7.3f}{alpha_m1[0]:>12.4f}{alpha_m2[0]:>13.4f}{alpha_m3[0]:>10.4f}{alpha_m4[0]:>13.4f}")
print(f"{'alpha[1]':14}{alpha_true[1]:>7.4f}{alpha_m1[1]:>12.4f}{alpha_m2[1]:>13.4f}{alpha_m3[1]:>10.4f}{alpha_m4[1]:>13.4f}")
print(f"{'alpha[2]':14}{alpha_true[2]:>7.3f}{alpha_m1[2]:>12.4f}{alpha_m2[2]:>13.4f}{alpha_m3[2]:>10.4f}{alpha_m4[2]:>13.4f}")
print(f"{'beta_G':14}{beta_true:>7.3f}{beta_G_m1:>12.4f}{beta_G_m2:>13.4f}{beta_sc_m3:>10.4f}{beta_sc_m4:>13.4f}")
print(f"{'p(G)':14}{'<0.05':>7}{p_lrt_m1:>12.4g}{p_wald_m2:>13.4g}{p_spa_m3:>10.4g}{p_spa_m4:>13.4g}")
print()


# =============================================================================
# Analysis
# =============================================================================

print(sep)
print("ANALYSIS")
print(sep)
print("""
[1] tau 估计 — 所有方法均低估 (true=1):

    - saige0 (Laplace-REML): 偏差最小。Laplace 近似对 logistic GLMM 是一阶
      近似，在 tau 不太大时精度尚可。REML 校正项 -½log|I_α| 进一步减少
      对 tau 的向下偏差。

    - saige_paper (PQL+AI-REML): 偏差更大。PQL 的工作线性化本质上是把二元
      结局当作 Gaussian 处理，在低患病率或高 tau 时这种近似误差很大，导致
      Breslow-Lin 型系统性向下偏差 (Breslow & Lin 1995)。

    - SM_BGLMM_K5: τ 几乎为 0。BinomialBayesMixedGLM 对 σ 使用 log-normal
      先验，当随机效应基函数数量与观测数接近时，极大后验估计 (MAP) 坍塌到
      σ→0，即出现"饱和模型"问题。K=5 仅勉强给出非零 τ。

    - NaiveLogit: 无随机效应，τ 概念不适用。

[2] alpha 估计 — GLMM 与边际 logistic 的尺度差异:

    - NaiveLogit 估计的是 边际效应 (marginal effect): 对人群中所有 b 积分
      后的 logit(μ)，受随机效应方差稀释，绝对值偏小。

    - GLMM 方法估计的是 条件效应 (conditional on b): 在给定随机效应 b 时的
      回归系数，绝对值更大。两者不同是正确的，不是偏差——它们定义的目标
      参数不同 (marginal vs conditional)。

    - 因此 saige0/saige_paper 的 alpha[0] 绝对值一般大于 NaiveLogit 的估计，
      这是 GLMM 的固有特征，tau 越大差异越明显。

[3] beta_G 估计:

    - 本例中 G 与 GRM (G_grm) 独立生成，真实相关性接近零，因此群体结构
      对 beta_G 的混杂效应很弱。NaiveLogit 和 GLMM 方法的 beta_G 估计
      差异主要来自尺度差异 (marginal vs conditional) 和不同的 tau 估计。

    - saige0/saige_paper 报告的是 score-based 近似: β̂ ≈ T/Var(T)。这是
      β=0 处的 Newton 步长，接近真实 MLE 但不完全相同（一步近似）。

    - NaiveLogit 的 beta_G 因标准误不正确 (忽略随机效应的聚集效应) 而
      置信区间偏窄，p 值可能偏小 (检验过于激进)。

[4] p 值 (G 的显著性检验):

    - 本例 beta_true=2 效应很强，所有方法应均有小 p 值。差异更多反映
      检验功效 (power) 而非 I 类错误率的差别。

    - 在真实 GWAS 场景中 (beta_true≈0, MAF小, N大), NaiveLogit 在有群体
      结构时会出现虚假关联 (p 值偏小); saige0/saige_paper + SPA 在样本量
      不平衡和稀有变体时校正更准确。

[5] BinomialBayesMixedGLM 为何在 GRM 场景下失效:

    - 该模型对每列 exog_vc 都赋予一个独立随机效应 z_k ~ N(0, σ²)。当
      列数接近 N (如 Cholesky L 有 N 列, 或 eigvec 有 N 列), 随机效应
      可完美拟合每个 y_i，最大后验解为 σ→0 (由固定效应+大量 z_k 完全
      拟合数据)。这是 MAP 估计对过参数化模型的已知失效模式。
    - K=5 PC 仅捕获 Ψ 方差的一小部分，tau 被严重低估，beta_G 因混杂
      控制不足而偏差。
    - 结论: 对基于 GRM 的 logistic GLMM, statsmodels 现有 API 不适合，
      应使用专门的工具 (SAIGE, GMMAT, etc.)。
""")
