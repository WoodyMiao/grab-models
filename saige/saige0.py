"""
Parameter estimation strategy:
  tau   -> Laplace-REML  (corrects for fixed-effect degrees of freedom, reduces bias)
  alpha -> MLE at fixed tau_hat (conditional MLE via Newton-Raphson)
  b     -> Gauss-Hermite quadrature posterior mean (BLUP, better than MAP)
  p     -> SPA (saddlepoint approximation, same as original SAIGE)

See theory.md for the mathematical derivation of each step.
"""

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import brentq, minimize_scalar, minimize
from scipy.stats import norm
from scipy.special import expit  # logit^{-1}(x) = 1/(1+exp(-x))
from scipy.linalg import cho_factor, cho_solve

rng = np.random.default_rng()
GH_DEGREE  = 20


# =============================================================================
# Simulate data  (same setup as saige.py)
# =============================================================================

N = 200
M_grm = 500
G_maf = 0.1

age = rng.integers(0, 100, N)
sex = rng.binomial(1, 0.5, N)
X = np.column_stack([np.ones(N), age, sex])   # N x q
q = X.shape[1]

p_grm = rng.uniform(0, 1, M_grm)
G_grm = rng.binomial(2, p_grm, size=(N, M_grm)).astype(float)
G_std = (G_grm - 2 * p_grm) / np.sqrt(2 * p_grm * (1 - p_grm))
Psi   = G_std @ G_std.T / G_std.shape[1]    # N x N GRM

G = rng.binomial(2, G_maf, size=N).astype(float)


# =============================================================================
# Simulate binary phenotype directly from the logistic GLMM
# so that the true tau is well-defined and the Laplace approximation is valid.
# =============================================================================

tau_true = 1
alpha_true = np.array([-2, 0.01, 1])  # intercept, age, sex
b_true = rng.multivariate_normal(np.zeros(N), tau_true * Psi)
beta_true = 2

eta_true = X @ alpha_true + G * beta_true + b_true
mu_true = expit(eta_true)

y = rng.binomial(1, mu_true).astype(float)
print(f"True tau = {tau_true},  prevalence = {y.mean():.3f}\n")


# =============================================================================
# Three probability / likelihood functions — the mathematical core
# All accept explicit arguments (y, X, Psi) so they are self-contained
# and easy to test independently.
# =============================================================================

def log_post(b, alpha, tau, y, X, Psi):
    """
    Log joint posterior (up to constant) of b given y:  log p(b|y) ∝ log p(y|b) + log p(b).
    Equations (5) and prior log N(0, τΨ).

    Used to find the posterior mode b̃₀ (eq. 9b) via Newton-CG:
        b̃₀ = argmax_b log_post(b, alpha, tau, y, X, Psi)

    Gradient  (9a):  ∂/∂b = (y − μ) − (τΨ)⁻¹b
    Hessian   (9c):  ∂²/∂b² = −diag(μ⊙(1−μ)) − (τΨ)⁻¹   [negative definite → unique mode]
    """
    N      = len(y)
    tauPsi = tau * Psi                                         # tau=exp(s)>0, Psi PD
    eta    = X @ alpha + b
    ll     = float(y @ eta - np.sum(np.log1p(np.exp(eta))))   # (5): log p(y|b)
    L      = np.linalg.cholesky(tauPsi)
    Lb     = np.linalg.solve(L, b)
    lp     = -0.5 * float(Lb @ Lb)                            # log N(b; 0, τΨ), drop const
    return ll + lp


def ell(alpha, tau, y, X, Psi, tol=1e-10):
    """
    Laplace-approximated marginal log-likelihood ℓ_LA(α, τ)  [equation (10)].
    Marginalises over b by Taylor-expanding log p(y,b) to second order at b̃₀.

    Inner optimisation:
        b̃₀ = argmax_b  log_post(b, alpha, tau, y, X, Psi)   ← Newton-CG (eq. 9e)

    Score     (10a):  ∇_α ℓ_LA = Xᵀ(y − μ̃₀)
    Info      (10b):  −∇²_α ℓ_LA = XᵀW̃X − XᵀW̃H⁻¹W̃X   (Schur complement)

    Called by r_ell (which profiles out α) and by Step 2 (to recover α̂₀ at τ̂).
    """
    N      = len(y)
    tauPsi = tau * Psi                                         # tau=exp(s)>0, Psi PD
    L_tp   = cho_factor(tauPsi)                                # O(N³/3) once per ell call

    def neg_lp(b):                                             # −log_post: objective
        eta = X @ alpha + b
        ll  = float(y @ eta - np.sum(np.log1p(np.exp(eta))))
        Lb  = cho_solve(L_tp, b)
        return -(ll - 0.5 * float(b @ Lb))

    def neg_grad(b):                                           # −∇_b log_post = −(9a)
        mu = expit(X @ alpha + b)
        return -(y - mu) + cho_solve(L_tp, b)                  # O(N²) per call

    def hessp(b, v):                                           # H @ v (9c), O(N²) per CG call
        mu = expit(X @ alpha + b)
        return mu * (1 - mu) * v + cho_solve(L_tp, v)          # O(N²) via Cholesky

    # ── inner Newton-CG: find b̃₀ (eq. 9e) ──────────────────────────────────
    res_b = minimize(neg_lp, np.zeros(N), jac=neg_grad, hessp=hessp,
                     method='Newton-CG', tol=tol)
    b0    = res_b.x
    mu0   = expit(X @ alpha + b0)
    w0    = mu0 * (1 - mu0)
    H     = np.diag(w0) + cho_solve(L_tp, np.eye(N))          # (9c) at convergence, O(N³)

    # ── Laplace log-likelihood (10) ─────────────────────────────────────────
    eta0       = X @ alpha + b0
    ll0        = float(y @ eta0 - np.sum(np.log1p(np.exp(eta0))))
    Lb0        = cho_solve(L_tp, b0)
    quad       = float(b0 @ Lb0)                               # b₀ᵀ(τΨ)⁻¹b₀
    ld_tauPsi  = 2 * np.sum(np.log(np.abs(np.diag(L_tp[0])))) # log|τΨ|
    _, ld_H    = np.linalg.slogdet(H)                          # log|H|, O(N³)
    val = ll0 - 0.5 * ld_tauPsi - 0.5 * quad - 0.5 * ld_H    # (10)

    return val, b0, H, w0


def r_ell(tau, y, X, Psi, tol_b=1e-10, tol_a=1e-10, max_iter=50):
    """
    Laplace-REML log-likelihood ℓ_REML(τ)  [equation (11)].
    Profiles out α via hand-written NR (10c): each step calls ell() once,
    reusing b̃₀, H, w₀ for both the score (10a) and information (10b).
    Then adds the REML correction −½ log|XᵀH⁻¹X|.

    Outer optimisation:
        τ̂ = argmax_τ  r_ell(tau, y, X, Psi)   ← Brent on log τ (eq. 11a)

    Not a closed-form function of τ alone — see theory.md §4.4.
    """
    if tau <= 0:
        return -np.inf, None, None, None, None

    N, q = len(y), X.shape[1]
    alpha = np.zeros(q)

    # ── NR for α̂₀(τ) (eq. 10c): one ell() call per iteration ───────────────
    for _ in range(max_iter):
        val, b0, H, w0 = ell(alpha, tau, y, X, Psi, tol=tol_b)  # inner b̃₀ + ℓ_LA
        mu0  = expit(X @ alpha + b0)
        s    = X.T @ (y - mu0)                                   # (10a): score
        WX   = w0[:, None] * X
        HiWX = np.linalg.solve(H, WX)                            # O(N³): LU factor + N²q solve
        I    = X.T @ WX - WX.T @ HiWX                            # (10b): q×q info
        delta = np.linalg.solve(I, s)                                          # I is PD
        alpha += delta
        if np.max(np.abs(delta)) < tol_a:
            break

    # ── final ell() at converged α̂₀ ─────────────────────────────────────────
    val, b0, H, w0 = ell(alpha, tau, y, X, Psi, tol=tol_b)

    # ── REML correction −½ log|XᵀΣ⁻¹X|  (11) ────────────────────────────────
    # Σ⁻¹ = W − WH⁻¹W  (Woodbury), so XᵀΣ⁻¹X = XᵀWX − XᵀWH⁻¹WX = I_α (10b)
    WX      = w0[:, None] * X
    HiWX    = np.linalg.solve(H, WX)                             # O(N³): LU factor + N²q solve
    I_alpha = X.T @ WX - WX.T @ HiWX                            # (10b): q×q Fisher info
    _, ld_I = np.linalg.slogdet(I_alpha)
    reml_val = val - 0.5 * ld_I                                  # (11)

    W = np.diag(w0)
    return reml_val, alpha, b0, H, W


# =============================================================================
# def log_fi(i, b_val, alpha, b0, tauPsi_inv) -> float
# Log unnormalised conditional density log f_i(b_val) from (12b).
# Fixes b_{j≠i} at the joint posterior mode b0; evaluates at b_{i0} = b_val.
# =============================================================================

def log_fi(i, b_val, alpha, b0, tauPsi_inv):
    b_full    = b0.copy()
    b_full[i] = b_val                                # mean-field: j≠i fixed at mode (12b)
    eta_full  = X @ alpha + b_full
    ll        = float(y @ eta_full - np.sum(np.log1p(np.exp(eta_full))))  # from (5)
    log_pri   = -0.5 * float(b_full @ tauPsi_inv @ b_full)                # (12b)
    return ll + log_pri


# =============================================================================
# Step 1: τ — maximise r_ell over τ > 0 via Brent's method  (11), (11a)
# Reparametrise τ = exp(s) to enforce τ > 0; search s ∈ ℝ.
# scipy.optimize.minimize_scalar uses Brent by default (no derivative needed).
# =============================================================================

print("Step 1: estimating tau via Laplace-REML ...")
result = minimize_scalar(lambda s: -r_ell(np.exp(s), y, X, Psi)[0])
tau    = float(np.exp(result.x))
print(f"  tau = {tau:.4g}\n")


# =============================================================================
# Step 2: α — recover profile MLE at converged τ̂  (10a)(10b)(10c)
# r_ell already ran this inner loop; call once more to extract α̂₀, b̃₀, H, W.
# =============================================================================

print("Step 2: estimating alpha at fixed tau ...")
_, alpha, b0, H, W = r_ell(tau, y, X, Psi)
print(f"  alpha = {alpha}\n")


# =============================================================================
# Step 3: b0 — BLUP via 1D Gauss-Hermite quadrature per component
# =============================================================================

print(f"Step 3: estimating b0 via {GH_DEGREE}-point Gauss-Hermite quadrature ...")
gh_nodes, gh_weights = hermgauss(GH_DEGREE)    # (tₖ, wₖ): K-point GH rule

tauPsi     = tau * Psi
tauPsi_inv = np.linalg.solve(tauPsi, np.eye(N))
sigma      = 1.0 / np.sqrt(np.diag(H))                    # (9c): σᵢ = H_{ii}^{-1/2}
b_blup     = np.zeros(N)
for i in range(N):
    nodes_k = b0[i] + np.sqrt(2) * sigma[i] * gh_nodes    # (12c): GH quadrature nodes
    log_wik = (
        np.log(gh_weights)
        + np.array([log_fi(i, node, alpha, b0, tauPsi_inv) for node in nodes_k])
        + gh_nodes**2                                       # (12c): cancel exp(−tₖ²)
    )
    wik = np.exp(log_wik - np.max(log_wik))         # numerical stabilisation
    b_blup[i] = np.dot(wik, nodes_k) / np.sum(wik)       # (12a): posterior mean

b0 = b_blup
print(f"  b0 (first 5): {b0[:5]}\n")


# =============================================================================
# Section 4: Score statistic T
# =============================================================================

mu0 = expit(X @ alpha + b0)
T = G @ (y - mu0)

# G_tilde = G - X (X^T W_hat X)^{-1} X^T W_hat G
W_hat_diag = mu0 * (1 - mu0)
W_hat   = np.diag(W_hat_diag)
XtWX    = X.T @ W_hat @ X
XtWG    = X.T @ (W_hat_diag * G)
G_tilde = G - X @ np.linalg.solve(XtWX, XtWG)

# Note: T = G_tilde @ (y - mu0) holds iff X^T(y-mu0)=0, i.e. at the Laplace
# mode b0 with the conditional MLE alpha. After BLUP updates b0, this identity
# is only approximate, so we use T = G @ (y - mu0) as the primary definition.

Sigma_hat     = np.diag(1.0 / W_hat_diag) + tau * Psi
Sigma_hat_inv = np.linalg.solve(Sigma_hat, np.eye(N))
XtSiX         = X.T @ Sigma_hat_inv @ X
P_hat         = Sigma_hat_inv - Sigma_hat_inv @ X @ np.linalg.solve(XtSiX, X.T @ Sigma_hat_inv)

V_T         = float(G_tilde @ P_hat @ G_tilde)
V_T_given_b = float(G_tilde @ (W_hat_diag * G_tilde))
r           = V_T / V_T_given_b
T_adj       = T / np.sqrt(V_T)
p_normal = 2 * norm.sf(abs(T_adj))

print(f"--- Score statistic ---")
print(f"T = {T:.4g}")
print(f"Var(T) = {V_T:.4g}")
print(f"Var(T|b) = {V_T_given_b:.4g}")
print(f"r = {r:.4g}")
print(f"T_adj = {T_adj:.4g}")
print(f"p_norm = {p_normal:.4g}\n")


# =============================================================================
# Section 5: SPA
# =============================================================================

c = 1.0 / np.sqrt(V_T_given_b)

def K(t):
    arg       = c * t * G_tilde
    log_terms = np.log1p(mu0 * (np.exp(arg) - 1))
    return np.sum(log_terms) - c * t * np.sum(G_tilde * mu0)

def K_prime(t):
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
    a      = c * t * G_tilde
    a2     = a / 2
    exp_a2  = np.exp(np.clip( a2, -500, 500))
    exp_na2 = np.exp(np.clip(-a2, -500, 500))
    denom2  = ((1 - mu0) * exp_na2 + mu0 * exp_a2) ** 2
    numer   = G_tilde**2 * mu0 * (1 - mu0)
    return c**2 * np.sum(numer / denom2)

def spa_cdf(q_val, a=-50.0, b=50.0):
    """Lugannani-Rice SPA for P(sqrt(r)*T_adj < q_val)."""
    zeta_hat = brentq(lambda t: K_prime(t) - q_val, a, b)
    w = np.sign(zeta_hat) * np.sqrt(2 * (zeta_hat * q_val - K(zeta_hat)))
    v = zeta_hat * np.sqrt(K_double_prime(zeta_hat))
    return norm.cdf(w + np.log(v / w) / w)

q_obs   = np.sqrt(r) * T_adj
p_upper = 1 - spa_cdf( abs(q_obs))   # P(X >  |q|)
p_lower =     spa_cdf(-abs(q_obs))   # P(X < -|q|)
p_value = p_upper + p_lower

print(f"--- SPA results ---")
print(f"sqrt(r)*T_adj = {q_obs:.4g}")
print(f"SPA p-value = {p_value:.4g}\n")
