"""
Parameter estimation strategy:
  tau   -> AI-REML  (Average Information REML, one Newton step per outer iteration)
  alpha -> GLS on working vector at each PQL iteration        (eq. 12)
  b     -> BLUP formula on working vector at each PQL iteration (eq. 13)
  p     -> SPA (saddlepoint approximation, same as saige0.py)

See saige_paper.md for the mathematical derivation of each step.
Reference: Zhou et al. (2018) Nature Genetics, doi:10.1038/s41588-018-0184-y
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.special import expit  # logit^{-1}(x) = 1/(1+exp(-x))


rng = np.random.default_rng()


# =============================================================================
# Simulate data  (same setup as saige0.py)
# =============================================================================

N = 200
M_grm = 500
G_maf = 0.1

age = rng.integers(0, 100, N)
sex = rng.binomial(1, 0.5, N)
X = np.column_stack([np.ones(N), age, sex])   # N × q
q = X.shape[1]

p_grm = rng.uniform(0, 1, M_grm)
G_grm = rng.binomial(2, p_grm, size=(N, M_grm)).astype(float)
G_std = (G_grm - 2 * p_grm) / np.sqrt(2 * p_grm * (1 - p_grm))
M_qc  = G_std.shape[1]

# diag(Ψ) for PCG preconditioner — computed on-the-fly, no N×N matrix
psi_diag = np.sum(G_std**2, axis=1) / M_qc

def psi_mv(v):
    """Ψv = G_std (G_std^T v) / M_qc  — GRM-vector product, no N×N matrix formed."""
    return G_std @ (G_std.T @ v) / M_qc

Psi_sim = G_std @ G_std.T / M_qc   # only for simulating b_true below

G = rng.binomial(2, G_maf, size=N).astype(float)


# =============================================================================
# Simulate binary phenotype directly from the logistic GLMM
# =============================================================================

tau_true   = 1
alpha_true = np.array([-2, 0.01, 1])  # intercept, age, sex
b_true     = rng.multivariate_normal(np.zeros(N), tau_true * Psi_sim)
beta_true  = 2

eta_true = X @ alpha_true + G * beta_true + b_true
mu_true  = expit(eta_true)

y = rng.binomial(1, mu_true).astype(float)
print(f"True tau = {tau_true},  prevalence = {y.mean():.3f}\n")


# =============================================================================
# def pcg_sigma_inv(b, W_diag, tau) -> array
# Solve Σx = b via PCG.  Σ = diag(W⁻¹) + τΨ; never formed as N×N.
# Preconditioner: M = diag(Σ) = 1/W_diag + τ·psi_diag.
# Convergence: ||r||² ≤ tolPCG (absolute, matches SAIGE C++ criterion).
# =============================================================================

def pcg_sigma_inv(b, W_diag, tau, maxiter=500, tolPCG=1e-5):
    prec_inv = 1.0 / (1.0 / W_diag + tau * psi_diag)   # diagonal preconditioner M⁻¹
    x = np.zeros_like(b)
    r = b.copy()
    z = prec_inv * r
    p = z.copy()
    rz = float(r @ z)
    for _ in range(maxiter):
        Ap    = p / W_diag + tau * psi_mv(p)            # Σp on-the-fly, no N×N
        alpha = rz / float(p @ Ap)
        x    += alpha * p
        r    -= alpha * Ap
        if float(r @ r) <= tolPCG:
            break
        z_new  = prec_inv * r
        rz_new = float(r @ z_new)
        p      = z_new + (rz_new / rz) * p
        rz     = rz_new
    return x


# =============================================================================
# def P_apply(u, W_diag, tau, Si_X, XtSiX) -> array
# Apply the REML projection matrix P (eq. 16) to a vector u via PCG:
#     Pu = Σ⁻¹u − Σ⁻¹X (X^TΣ⁻¹X)⁻¹ X^TΣ⁻¹u
# =============================================================================

def P_apply(u, W_diag, tau, Si_X, XtSiX):
    Si_u = pcg_sigma_inv(u, W_diag, tau)
    return Si_u - Si_X @ np.linalg.solve(XtSiX, Si_X.T @ u)


# =============================================================================
# def hutchinson_tr_PPsi(W_diag, tau, Si_X, XtSiX, nrun, traceCVcutoff) -> float
# tr(PΨ) via Hutchinson's randomised estimator (eq. 19).
# Rademacher: u = Binomial(N)*2 − 1  (matches SAIGE nb(N)*2−1, seed 200).
# Adaptive: nrun starts at 30, increases by 10 while CV > traceCVcutoff=0.0025.
# =============================================================================

def hutchinson_tr_PPsi(W_diag, tau, Si_X, XtSiX, nrun=30, traceCVcutoff=0.0025, max_nrun=None):
    """
    Adaptive Hutchinson trace estimator.

    SAIGE C++ increases nrun by 10 until CV < traceCVcutoff (default 0.0025) with no hard cap.
    For large N (SAIGE's target, N≥10,000), 30 samples already achieve CV < 0.0025.
    For small N (e.g. N=200 here), CV ≈ 5 % per sample, requiring ~490 samples —
    so we cap at max_nrun (default: 10 × nrun) and accept the residual variance.
    """
    if max_nrun is None:
        max_nrun = 10 * nrun          # cap: prevents infinite loop when N is small

    rng_h    = np.random.default_rng(200)   # seed 200 matches SAIGE set_seed(200)
    samples  = []
    n_target = nrun
    while True:
        while len(samples) < n_target:
            u    = rng_h.integers(0, 2, size=N).astype(float) * 2 - 1   # Rademacher
            Si_u = pcg_sigma_inv(u, W_diag, tau)
            Pu   = Si_u - Si_X @ np.linalg.solve(XtSiX, Si_X.T @ u)
            Au   = psi_mv(u)                                              # Ψu on-the-fly
            samples.append(float(Au @ Pu))
        arr = np.array(samples)
        cv  = arr.std() / (abs(arr.mean()) + 1e-30)
        if cv <= traceCVcutoff or n_target >= max_nrun:
            break
        n_target += 10
    return float(arr.mean())


# =============================================================================
# def fit_null_pql(y, X, max_iter, tol, nrun)
#   -> (tau, alpha, b, mu, W_diag, Si_X, XtSiX)
#
# Fit null logistic GLMM via PQL + AI-REML  (eq. 23 of saige_paper.md).
# Each outer iteration:
#   (a) Linearise the logistic link at current (alpha, b) → working vector Ỹ (eq. 9)
#   (b) Update alpha via GLS on the working LMM              (eq. 12)
#   (c) Update b via BLUP on the working LMM                 (eq. 13)
#   (d) Update tau via one AI-REML Newton step               (eqs. 17, 20, 22)
# All Σ⁻¹ applications use PCG (pcg_sigma_inv); Ψ applied on-the-fly via psi_mv.
# =============================================================================

def fit_null_pql(y, X, max_iter=25, tol=1e-5, nrun=30):
    """
    PQL + AI-REML null model fit.

    Score (eq. 17):  S(τ) = ½[Ỹ^T P Ψ P Ỹ − tr(PΨ)]
    AI   (eq. 20): AI_ττ = ½ Ỹ^T P Ψ P Ỹ        (scalar, always > 0)
    Update (eq. 22): τ ← τ + S / AI = τ + 1 − tr(PΨ) / (2 AI)

    Returns (tau, alpha, b, mu, W_diag, Si_X, XtSiX).
    Σ and Ψ are never formed as N×N matrices.
    """
    N_loc, q = len(y), X.shape[1]
    alpha = np.zeros(q)
    b     = np.zeros(N_loc)
    tau   = 1.0

    Si_X  = None
    XtSiX = None

    for it in range(max_iter):

        # ── Step (a): working vector and weights ─────────────────────────────
        mu      = expit(X @ alpha + b)
        W_diag  = mu * (1 - mu)                                  # N-vector
        Y_tilde = X @ alpha + b + (y - mu) / W_diag             # (9): pseudo-response

        # ── Σ⁻¹ via PCG; Σ = W⁻¹ + τΨ never formed as N×N ──────────────────
        Si_X  = np.column_stack([pcg_sigma_inv(X[:, j], W_diag, tau) for j in range(q)])
        XtSiX = X.T @ Si_X                                      # q×q
        Si_Y  = pcg_sigma_inv(Y_tilde, W_diag, tau)
        XtSiY = X.T @ Si_Y

        # ── Step (b): GLS update α̂ = (X^TΣ⁻¹X)⁻¹ X^TΣ⁻¹Ỹ (12) ───────────
        alpha_new = np.linalg.solve(XtSiX, XtSiY)              # q×q solve

        # ── Step (c): BLUP update b̂ = τΨΣ⁻¹(Ỹ − Xα̂)  (13) ────────────────
        resid = Y_tilde - X @ alpha_new                          # Ỹ − Xα̂
        b_new = tau * psi_mv(pcg_sigma_inv(resid, W_diag, tau)) # Ψ applied on-the-fly

        # ── Step (d): AI-REML Newton step for τ  (17)(20)(22) ───────────────
        PY     = Si_Y - Si_X @ np.linalg.solve(XtSiX, XtSiY)  # PỸ
        PsiPY  = psi_mv(PY)                                      # ΨPỸ: on-the-fly
        PPsiPY = P_apply(PsiPY, W_diag, tau, Si_X, XtSiX)      # PΨPỸ

        # AI_ττ = ½ Ỹ^T P Ψ P Ỹ  (20): always positive
        AI = 0.5 * float(Y_tilde @ PPsiPY)

        # tr(PΨ) via Hutchinson estimator (eq. 19) — SAIGE always uses this
        tr_PPsi = hutchinson_tr_PPsi(W_diag, tau, Si_X, XtSiX, nrun=nrun)

        S_tau   = AI - 0.5 * tr_PPsi                            # (21): score S = AI − ½tr(PΨ)
        tau_new = max(tau + S_tau / AI, 1e-6)                   # (22): Newton, clip > 0

        # ── Convergence check ────────────────────────────────────────────────
        delta = max(np.max(np.abs(alpha_new - alpha)), abs(tau_new - tau))
        alpha, b, tau = alpha_new, b_new, tau_new
        if delta < tol:
            break

    # ── Final pass: recompute W and PCG quantities at converged parameters ───
    mu      = expit(X @ alpha + b)
    W_diag  = mu * (1 - mu)
    Si_X    = np.column_stack([pcg_sigma_inv(X[:, j], W_diag, tau) for j in range(q)])
    XtSiX   = X.T @ Si_X

    return tau, alpha, b, mu, W_diag, Si_X, XtSiX


# =============================================================================
# Step 1: τ, α, b — PQL + AI-REML  (eq. 23)
# =============================================================================

print("=== Step 1: Fit null model via PQL + AI-REML ===")
tau0, alpha0, b0, mu0, W0, Si_X0, XtSiX0 = fit_null_pql(y, X)
print(f"  tau_hat   = {tau0:.4f}  (true = {tau_true})")
print(f"  alpha_hat = {alpha0}  (true = {alpha_true})\n")


# =============================================================================
# Step 2: Score statistic T  (eqs. 24–29)
# =============================================================================

print("=== Step 2: Score statistic ===")
T = float(G @ (y - mu0))                                        # (24): raw score

# G_tilde = G − X(X^TWX)⁻¹X^TWG  (26): residualise G from X in W-metric
XtWX    = X.T @ (W0[:, None] * X)                               # q×q
XtWG    = X.T @ (W0 * G)                                        # q-vector
G_tilde = G - X @ np.linalg.solve(XtWX, XtWG)                  # N-vector

# Var(T) = G̃^T P̂ G̃  (25); apply P to G̃ implicitly via P_apply
PG_tilde    = P_apply(G_tilde, W0, tau0, Si_X0, XtSiX0)        # PCG-based
V_T         = float(G_tilde @ PG_tilde)                          # (25): Var(T)
V_T_given_b = float(G_tilde @ (W0 * G_tilde))                   # (27): Var*(T)
r           = V_T / V_T_given_b                                  # (28): correction ratio
T_adj       = T / np.sqrt(V_T)                                   # (29): standardised

p_normal = 2 * norm.sf(abs(T_adj))

print(f"  T         = {T:.4g}")
print(f"  Var(T)    = {V_T:.4g}")
print(f"  Var*(T)   = {V_T_given_b:.4g}")
print(f"  r         = {r:.4g}")
print(f"  T_adj     = {T_adj:.4g}")
print(f"  p_normal  = {p_normal:.4g}\n")


# =============================================================================
# Step 3: SPA  (eqs. 30–34)
# CGF of T = Σ_i G̃_i(y_i − μ_{i0}) conditional on b̂₀ (Bernoulli terms):
#   K(t) = Σ_i log[μ_{i0} exp(G̃_i(1−μ_{i0})t) + (1−μ_{i0}) exp(−G̃_i μ_{i0} t)]
# Standardise by c = 1/√Var*(T) so that K''(0) = 1.
# =============================================================================

print("=== Step 3: SPA ===")

c = 1.0 / np.sqrt(V_T_given_b)                                  # standardisation factor


def K(t):
    """CGF of T/√Var*(T) at saddlepoint parameter t  (eq. 30)."""
    arg       = c * t * G_tilde                                  # G̃_i · c·t
    log_terms = np.log1p(mu0 * (np.exp(arg) - 1))               # log[1 + μ(e^a − 1)]
    return np.sum(log_terms) - c * t * np.sum(G_tilde * mu0)


def K_prime(t):
    """First derivative K'(t) = E[T/√Var*(T)]  (eq. 31)."""
    a     = c * t * G_tilde
    total = np.zeros(N)
    pos   = a >= 0
    neg   = ~pos
    if pos.any():
        e          = np.exp(a[pos])
        total[pos] = G_tilde[pos] * mu0[pos] * (e - 1) / ((1 - mu0[pos]) + mu0[pos] * e)
    if neg.any():
        e_neg      = np.exp(-a[neg])
        total[neg] = G_tilde[neg] * mu0[neg] * (1 - e_neg) / ((1 - mu0[neg]) * e_neg + mu0[neg])
    return c * np.sum(total)


def K_double_prime(t):
    """Second derivative K''(t) = Var[T/√Var*(T)]  (eq. 32)."""
    a       = c * t * G_tilde
    a2      = a / 2
    exp_a2  = np.exp(np.clip( a2, -500, 500))
    exp_na2 = np.exp(np.clip(-a2, -500, 500))
    denom2  = ((1 - mu0) * exp_na2 + mu0 * exp_a2) ** 2
    numer   = G_tilde**2 * mu0 * (1 - mu0)
    return c**2 * np.sum(numer / denom2)


def spa_cdf(q_val, t_bracket=(-50.0, 50.0)):
    """
    Lugannani-Rice SPA for P(√r · T_adj < q_val)  (eq. 33).
    Solves K'(t̂) = q_val for saddlepoint t̂ via Brent; falls back to
    Gaussian if K' is monotone but does not cross q_val in the bracket.
    """
    try:
        t_hat = brentq(lambda t: K_prime(t) - q_val,
                       t_bracket[0], t_bracket[1], xtol=1e-10)
    except ValueError:
        return float(norm.cdf(q_val))
    w = np.sign(t_hat) * np.sqrt(2 * (t_hat * q_val - K(t_hat)))
    v = t_hat * np.sqrt(K_double_prime(t_hat))
    if abs(w) < 1e-8:
        return 0.5 + (v - w) / (2 * np.pi) ** 0.5
    return float(norm.cdf(w + np.log(v / w) / w))


q_obs   = np.sqrt(r) * T_adj                                    # adjusted observable (eq. 34)
p_upper = 1 - spa_cdf( abs(q_obs))                              # P(X >  |q|)
p_lower =     spa_cdf(-abs(q_obs))                              # P(X < -|q|)
p_value = p_upper + p_lower                                      # (34): two-sided p

print(f"  sqrt(r)·T_adj = {q_obs:.4g}")
print(f"  SPA p-value   = {p_value:.4g}\n")
