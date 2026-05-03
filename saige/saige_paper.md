# SAIGE (Paper Method: PQL + AI-REML)

In a case-control study with a sample size $N$, let

- $N \times 1$ random vector $\mathbf{y}$ represent their phenotypes;
- $N \times (1 + p)$ matrix $\mathbf{X}$ represent their $p$ covariates and a column of ones;
- $N \times 1$ vector $\mathbf{G}$ represent their genotypes coded as allele counts for a variant to be tested;
- $N \times 1$ random vector $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$, where $\mathbf{\Psi}$ is a GRM and $\tau$ is the corresponding variance component;
- $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{G}\beta + \mathbf{b}$ be the linear predictor, where $\boldsymbol{\alpha}$ is the fixed effects and $\beta$ is the genetic effect to be tested.

Suppose $(\mathbf{X}, \mathbf{G}, \mathbf{\Psi})$ are observed and treated as fixed, $(\boldsymbol{\alpha}, \beta, \tau)$ are unknown parameters; $\mathbf{y}$ and $\mathbf{b}$ are random vectors before observing the data.

For each subject, suppose

$$
y_i|b_i \sim \operatorname{Bernoulli}(\mu_i), \quad \mu_i = \operatorname{logit}^{-1}(\eta_i) \tag{1}
$$

We are interested in testing the null hypothesis $H_0: \beta=0$.

## 1. Probability functions

According to (1), we have the conditional PMF of $y_i$ given $b_i$ as

$$
p_{y_i | b_i}(y_i; b_i, \mathbf{X}_i, G_i, \boldsymbol{\alpha}, \beta) = \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}, \quad \mu_i = \frac{1}{1 + e^{-(\mathbf{X}_i\boldsymbol{\alpha} + G_i\beta + b_i)}}
$$

The conditional joint PMF of $\mathbf{y}$ given $\mathbf{b}$ is

$$
p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}; \mathbf{b}, \mathbf{X}, \mathbf{G}, \boldsymbol{\alpha}, \beta) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}, \quad \mu_i = \frac{1}{1 + e^{-(\mathbf{X}_i\boldsymbol{\alpha} + G_i\beta + b_i)}} \tag{2}
$$

The marginal PDF of $\mathbf{b}$ is given by the density of $\mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$,

$$
p_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi}) = \phi(\mathbf{b}; \mathbf{0}, \tau \mathbf{\Psi}) = \frac{1}{(2\pi)^{\frac{N}{2}} |\tau \mathbf{\Psi}|^{\frac{1}{2}}} \exp\left(-\frac{1}{2} \mathbf{b}^\top (\tau \mathbf{\Psi})^{-1} \mathbf{b}\right)
$$

The joint probability function of $(\mathbf{y},\mathbf{b})$ is

$$
p_{\mathbf{y}, \mathbf{b}}(\mathbf{y}, \mathbf{b}; \mathbf{X}, \mathbf{G}, \boldsymbol{\alpha}, \beta, \tau, \mathbf{\Psi}) = p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b})
$$

The marginal PMF of $\mathbf{y}$ is

$$
p_{\mathbf{y}}(\mathbf{y}; \mathbf{X}, \mathbf{G}, \boldsymbol{\alpha}, \beta, \tau, \mathbf{\Psi}) = \int_{\mathbb{R}^N}  p_{\mathbf{y}, \mathbf{b}}(\mathbf{y}, \mathbf{b}) \mathrm{d}\mathbf{b} = \int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b}) \mathrm{d}\mathbf{b} \tag{3}
$$

The conditional PDF of $\mathbf{b}$ given $\mathbf{y}$ is

$$
p_{\mathbf{b}|\mathbf{y}}(\mathbf{b}; \mathbf{y}, \mathbf{X}, \mathbf{G}, \boldsymbol{\alpha}, \beta, \tau, \mathbf{\Psi}) = \frac{p_{\mathbf{y}, \mathbf{b}}(\mathbf{y}, \mathbf{b})}{p_{\mathbf{y}}(\mathbf{y})} = \frac{p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b})}{\int_{\mathbb{R}^N}  p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b}) \mathrm{d}\mathbf{b}} \tag{4}
$$

## 2. Conditional likelihood function

According to (2), the conditional log-likelihood given $\mathbf{b}$ under the null ($\beta = 0$) simplifies to

$$
\ell(\boldsymbol{\alpha}; \mathbf{y}|\mathbf{b}, \mathbf{X}) = \mathbf{y}^\top \boldsymbol{\eta}_0 - \mathbf{1}^\top \log(\mathbf{1} + \exp(\boldsymbol{\eta}_0)), \quad \boldsymbol{\eta}_0 = \mathbf{X}\boldsymbol{\alpha} + \mathbf{b} \tag{5}
$$

## 3. Conditional score functions

The conditional score for $\beta$ given $\mathbf{b}$, evaluated at $\beta = 0$, is

$$
U_\beta = \mathbf{G}^\top(\mathbf{y} - \boldsymbol{\mu}_0) \tag{6}
$$

The conditional score for $\boldsymbol{\alpha}$ given $\mathbf{b}$ is

$$
U_{\boldsymbol{\alpha}} = \mathbf{X}^\top(\mathbf{y} - \boldsymbol{\mu}_0) \tag{7}
$$

where $\boldsymbol{\mu}_0 = \operatorname{logit}^{-1}(\boldsymbol{\eta}_0) = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha}_0 + \mathbf{b}_0)$.

## 4. Estimator for $(\tau_0, \boldsymbol{\alpha}_0, \mathbf{b}_0)$ under the null model: PQL + AI-REML

The SAIGE paper estimates $(\hat{\boldsymbol{\alpha}}_0, \hat{\mathbf{b}}_0, \hat{\tau}_0)$ using **Penalized Quasi-Likelihood (PQL)** with variance components updated by **Average Information REML (AI-REML)**, following the GMMAT framework (Chen et al. 2016).

The key idea is to convert the non-linear GLMM into a sequence of **working linear mixed models** (one per outer iteration) by linearising the link function at the current estimates. Each working LMM has a tractable Gaussian marginal, whose REML is then maximised with respect to $\tau$ via a single Newton step using the Average Information matrix.

### 4.1 Working vector and working linear model

At current parameter values $(\boldsymbol{\alpha}, \mathbf{b}, \tau)$, define

$$
\boldsymbol{\mu} = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha} + \mathbf{b}), \qquad
W_i = \mu_i(1 - \mu_i), \qquad \mathbf{W} = \operatorname{diag}(W_i) \tag{8}
$$

The **working vector** (pseudo-response from the IRLS algorithm) is

$$
\tilde{Y}_i = \underbrace{X_i\boldsymbol{\alpha} + b_i}_{\hat\eta_i} + \underbrace{\frac{y_i - \mu_i}{\mu_i(1-\mu_i)}}_{g'(\mu_i)(y_i-\mu_i)}, \qquad \tilde{\mathbf{Y}} = \mathbf{X}\boldsymbol{\alpha} + \mathbf{b} + \mathbf{W}^{-1}(\mathbf{y} - \boldsymbol{\mu}) \tag{9}
$$

where $g'(\mu_i) = 1/[\mu_i(1-\mu_i)]$ is the derivative of the logit link. Equation (9) corresponds to Eq. (12) of the SAIGE supplementary note.

The working vector satisfies a **working linear mixed model** (LMM) at the current estimates:

$$
\tilde{\mathbf{Y}} \approx \mathbf{X}\boldsymbol{\alpha} + \mathbf{b} + \boldsymbol{\varepsilon}, \qquad
\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau\mathbf{\Psi}), \qquad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{W}^{-1}) \tag{10}
$$

where $\mathbf{b}$ and $\boldsymbol{\varepsilon}$ are independent, and $\mathbf{W}^{-1}$ serves as the "dispersion" of the working errors (the logistic model has $\phi = 1$ so no extra scale parameter). From (10), the marginal distribution of $\tilde{\mathbf{Y}}$ is

$$
\tilde{\mathbf{Y}} \sim \mathcal{N}\!\left(\mathbf{X}\boldsymbol{\alpha},\; \boldsymbol{\Sigma}\right), \qquad \boldsymbol{\Sigma} = \mathbf{W}^{-1} + \tau\mathbf{\Psi} \tag{11}
$$

**Key insight.** At each outer iteration, $\mathbf{W}$ (and hence $\boldsymbol{\Sigma}$) is treated as fixed (evaluated at current estimates), so (11) is a standard Gaussian LMM. All subsequent steps exploit the tractable Gaussian marginal.

### 4.2 GLS estimators for $\boldsymbol{\alpha}$ and $\mathbf{b}$

Given $\tilde{\mathbf{Y}}$ and $\boldsymbol{\Sigma}$ fixed, the GLS (Best Linear Unbiased) estimator of $\boldsymbol{\alpha}$ is

$$
\hat{\boldsymbol{\alpha}} = \left(\mathbf{X}^\top \boldsymbol{\Sigma}^{-1} \mathbf{X}\right)^{-1} \mathbf{X}^\top \boldsymbol{\Sigma}^{-1} \tilde{\mathbf{Y}} \tag{12}
$$

This corresponds to Eq. (3) of the SAIGE paper. It minimises $(\tilde{\mathbf{Y}} - \mathbf{X}\boldsymbol{\alpha})^\top \boldsymbol{\Sigma}^{-1}(\tilde{\mathbf{Y}} - \mathbf{X}\boldsymbol{\alpha})$ over $\boldsymbol{\alpha}$.

The BLUP (Best Linear Unbiased Predictor) of $\mathbf{b}$ is derived from the joint distribution of $(\tilde{\mathbf{Y}}, \mathbf{b})$. By the LMM BLUP formula (Henderson equations):

$$
\hat{\mathbf{b}} = \tau\mathbf{\Psi}\,\boldsymbol{\Sigma}^{-1}\!\left(\tilde{\mathbf{Y}} - \mathbf{X}\hat{\boldsymbol{\alpha}}\right) \tag{13}
$$

This corresponds to Eq. (4) of the SAIGE paper. To verify: $\operatorname{Cov}(\mathbf{b}, \tilde{\mathbf{Y}}) = \tau\mathbf{\Psi}$, so the conditional mean of $\mathbf{b}$ given $\tilde{\mathbf{Y}}$ is $\tau\mathbf{\Psi}\boldsymbol{\Sigma}^{-1}(\tilde{\mathbf{Y}} - \mathbf{X}\boldsymbol{\alpha})$, i.e., the BLUP replaces $\boldsymbol{\alpha}$ with $\hat{\boldsymbol{\alpha}}$.

**Equivalent form via Henderson mixed model equations.** Equations (12)–(13) can be derived jointly from

$$
\begin{pmatrix}
\mathbf{X}^\top \mathbf{W} \mathbf{X} & \mathbf{X}^\top \mathbf{W} \\
\mathbf{W} \mathbf{X} & \mathbf{W} + (\tau\mathbf{\Psi})^{-1}
\end{pmatrix}
\begin{pmatrix} \hat{\boldsymbol{\alpha}} \\ \hat{\mathbf{b}} \end{pmatrix}
= \begin{pmatrix} \mathbf{X}^\top \mathbf{W}\tilde{\mathbf{Y}} \\ \mathbf{W}\tilde{\mathbf{Y}} \end{pmatrix}
\tag{14}
$$

which is the Henderson mixed model equation system. The two formulations (12)–(13) and (14) are algebraically equivalent.

### 4.3 REML log-likelihood for the working model

For the working Gaussian LMM (11), the **REML log-likelihood** (restricted to the residual space orthogonal to $\mathbf{X}$) is

$$
ql_R(\tau) = c_R - \frac{1}{2}\log|\boldsymbol{\Sigma}| - \frac{1}{2}\log|\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X}| - \frac{1}{2}\tilde{\mathbf{Y}}^\top \mathbf{P}\tilde{\mathbf{Y}} \tag{15}
$$

where $c_R$ is a constant independent of $\tau$, and the **REML projection matrix** is

$$
\mathbf{P} = \boldsymbol{\Sigma}^{-1} - \boldsymbol{\Sigma}^{-1}\mathbf{X}\left(\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X}\right)^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^{-1} \tag{16}
$$

Note that $\mathbf{P}\mathbf{X} = \mathbf{0}$ (P projects out the column space of X) and $\tilde{\mathbf{Y}}^\top\mathbf{P}\tilde{\mathbf{Y}} = (\tilde{\mathbf{Y}} - \mathbf{X}\hat{\boldsymbol{\alpha}})^\top\boldsymbol{\Sigma}^{-1}(\tilde{\mathbf{Y}} - \mathbf{X}\hat{\boldsymbol{\alpha}})$. Equation (15) corresponds to Eq. (6) of the SAIGE paper.

This is **not** the true marginal log-likelihood of $\mathbf{y}$; it is the REML of the working LMM at the current linearisation point. The outer PQL loop iterates between updating the working vector (re-linearising at new $\boldsymbol{\alpha}, \mathbf{b}$) and maximising (15) with respect to $\tau$.

### 4.4 Score function for $\tau$

Differentiating (15) with respect to $\tau$, using $\partial \boldsymbol{\Sigma}/\partial\tau = \mathbf{\Psi}$ and standard matrix identities:

$$
\frac{\partial}{\partial\tau}\!\left(-\tfrac{1}{2}\log|\boldsymbol{\Sigma}|\right) = -\tfrac{1}{2}\operatorname{tr}\!\left(\boldsymbol{\Sigma}^{-1}\mathbf{\Psi}\right)
$$

$$
\frac{\partial}{\partial\tau}\!\left(-\tfrac{1}{2}\log|\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X}|\right)
  = \tfrac{1}{2}\operatorname{tr}\!\left(\left[\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X}\right]^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{\Psi}\boldsymbol{\Sigma}^{-1}\mathbf{X}\right)
$$

$$
\frac{\partial}{\partial\tau}\!\left(-\tfrac{1}{2}\tilde{\mathbf{Y}}^\top\mathbf{P}\tilde{\mathbf{Y}}\right)
  = \tfrac{1}{2}\tilde{\mathbf{Y}}^\top\mathbf{P}\mathbf{\Psi}\mathbf{P}\tilde{\mathbf{Y}}
$$

The last identity follows from $\partial\mathbf{P}/\partial\tau = -\mathbf{P}\mathbf{\Psi}\mathbf{P}$ (derivable from $\partial\boldsymbol{\Sigma}^{-1}/\partial\tau = -\boldsymbol{\Sigma}^{-1}\mathbf{\Psi}\boldsymbol{\Sigma}^{-1}$). Combining and noting that $\operatorname{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{\Psi}) - \operatorname{tr}([\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X}]^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{\Psi}\boldsymbol{\Sigma}^{-1}\mathbf{X}) = \operatorname{tr}(\mathbf{P}\mathbf{\Psi})$, the **REML score** for $\tau$ is

$$
S(\tau) = \frac{\partial\, ql_R}{\partial\tau} = \frac{1}{2}\!\left[\tilde{\mathbf{Y}}^\top\mathbf{P}\mathbf{\Psi}\mathbf{P}\tilde{\mathbf{Y}} - \operatorname{tr}(\mathbf{P}\mathbf{\Psi})\right] \tag{17}
$$

This corresponds to Eq. (8) of the SAIGE paper.

**Explicit formula for $\operatorname{tr}(\mathbf{P}\mathbf{\Psi})$.** From (16):

$$
\operatorname{tr}(\mathbf{P}\mathbf{\Psi})
= \operatorname{tr}(\boldsymbol{\Sigma}^{-1}\mathbf{\Psi}) - \operatorname{tr}\!\left(\boldsymbol{\Sigma}^{-1}\mathbf{X}\left[\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X}\right]^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{\Psi}\right) \tag{18}
$$

SAIGE always uses Hutchinson's randomised trace estimator, starting from $R = 30$ Rademacher vectors $\mathbf{z}_r \in \{-1,+1\}^N$ and increasing $R$ by 10 until the coefficient of variation falls below `traceCVcutoff` (default 0.0025):

$$
\operatorname{tr}(\mathbf{P}\mathbf{\Psi}) \approx \frac{1}{R}\sum_{r=1}^R \mathbf{z}_r^\top \mathbf{P}\mathbf{\Psi}\mathbf{z}_r \tag{19}
$$

### 4.5 Average Information matrix

The **Average Information (AI) matrix** is obtained by averaging the observed and expected Fisher information matrices for REML. For a single variance component $\tau$, the AI reduces to

$$
\mathrm{AI}_{\tau\tau} = \frac{1}{2}\,\tilde{\mathbf{Y}}^\top \mathbf{P}\mathbf{\Psi}\mathbf{P}\tilde{\mathbf{Y}} \tag{20}
$$

This corresponds to Eq. (11) of the SAIGE paper. Comparing (20) with (17):

$$
S(\tau) = \mathrm{AI}_{\tau\tau} - \tfrac{1}{2}\operatorname{tr}(\mathbf{P}\mathbf{\Psi}) \tag{21}
$$

so $\mathrm{AI}_{\tau\tau}$ equals the quadratic term in $S(\tau)$. The AI matrix is always positive (it is a sum of squared Cholesky factors), ensuring Newton steps are always well-defined.

**Why "average information"?** The expected information (Fisher) for REML is $\frac{1}{2}\operatorname{tr}[(\mathbf{P}\mathbf{\Psi})^2]$, and the observed information is $\frac{1}{2}[2\tilde{\mathbf{Y}}^\top\mathbf{P}\mathbf{\Psi}\mathbf{P}\mathbf{\Psi}\mathbf{P}\tilde{\mathbf{Y}} - \operatorname{tr}((\mathbf{P}\mathbf{\Psi})^2)]$. The AI matrix is neither of these; instead it can be shown that $\mathrm{AI}_{\tau\tau} = \frac{1}{2}\tilde{\mathbf{Y}}^\top\mathbf{P}\mathbf{\Psi}\mathbf{P}\tilde{\mathbf{Y}}$ approximates the average of the observed and expected information when $\tilde{\mathbf{Y}}$ is near its expectation.

### 4.6 Newton-Raphson update for $\tau$ (AI-REML)

A single Newton-Raphson step for $\tau$ using the AI matrix as the Hessian approximation gives

$$
\tau^{(t+1)} = \tau^{(t)} + \mathrm{AI}_{\tau\tau}^{-1}\cdot S\!\left(\tau^{(t)}\right) \tag{22}
$$

Since $\mathrm{AI}_{\tau\tau} > 0$ always, the step direction is well-defined. We enforce $\tau > 0$ by clipping $\tau^{(t+1)} \geq \epsilon$ (e.g. $\epsilon = 10^{-6}$).

### 4.7 Alternating estimation algorithm

The full algorithm iterates between updating the working vector (PQL step) and updating $\tau$ (AI-REML step):

$$
\begin{aligned}
&\text{Initialize: } \boldsymbol{\alpha}^{(0)} = \mathbf{0},\quad \mathbf{b}^{(0)} = \mathbf{0},\quad \tau^{(0)} = 1 \\
&\text{For } t = 0, 1, 2, \ldots \\
&\quad 1.\quad \boldsymbol{\mu}^{(t)} = \operatorname{logit}^{-1}\!\left(\mathbf{X}\boldsymbol{\alpha}^{(t)} + \mathbf{b}^{(t)}\right) \\
&\quad 2.\quad \mathbf{W}^{(t)} = \operatorname{diag}\!\left(\boldsymbol{\mu}^{(t)} \odot [\mathbf{1}-\boldsymbol{\mu}^{(t)}]\right) \\
&\quad 3.\quad \tilde{\mathbf{Y}}^{(t)} = \mathbf{X}\boldsymbol{\alpha}^{(t)} + \mathbf{b}^{(t)} + [\mathbf{W}^{(t)}]^{-1}\!\left(\mathbf{y} - \boldsymbol{\mu}^{(t)}\right) \quad\text{[eq. (9)]} \\
&\quad 4.\quad \boldsymbol{\Sigma}^{(t)} = [\mathbf{W}^{(t)}]^{-1} + \tau^{(t)}\mathbf{\Psi} \quad\text{[eq. (11)]} \\
&\quad 5.\quad \text{Solve all }\boldsymbol{\Sigma}^{(t)}\mathbf{x}=\mathbf{u}\text{ via PCG (precond. }\mathbf{M}=\operatorname{diag}(\boldsymbol{\Sigma}^{(t)})\text{); }\boldsymbol{\Sigma}^{(t)}\text{ and }\mathbf{\Psi}\text{ never formed as }N\times N \\
&\quad 6.\quad \boldsymbol{\alpha}^{(t+1)} = \left[\mathbf{X}^\top[\boldsymbol{\Sigma}^{(t)}]^{-1}\mathbf{X}\right]^{-1}\mathbf{X}^\top[\boldsymbol{\Sigma}^{(t)}]^{-1}\tilde{\mathbf{Y}}^{(t)} \quad\text{[eq. (12)]} \\
&\quad 7.\quad \mathbf{b}^{(t+1)} = \tau^{(t)}\mathbf{\Psi}[\boldsymbol{\Sigma}^{(t)}]^{-1}\!\left(\tilde{\mathbf{Y}}^{(t)} - \mathbf{X}\boldsymbol{\alpha}^{(t+1)}\right) \quad\text{[eq. (13)]} \\
&\quad 8.\quad \mathbf{P}^{(t)}\tilde{\mathbf{Y}}^{(t)} = [\boldsymbol{\Sigma}^{(t)}]^{-1}\tilde{\mathbf{Y}}^{(t)} - [\boldsymbol{\Sigma}^{(t)}]^{-1}\mathbf{X}\boldsymbol{\alpha}^{(t+1)} \quad\text{[via GLS residual]} \\
&\quad 9.\quad \mathrm{AI}^{(t)} = \tfrac{1}{2}\tilde{\mathbf{Y}}^{(t)\top}\mathbf{P}^{(t)}\mathbf{\Psi}\mathbf{P}^{(t)}\tilde{\mathbf{Y}}^{(t)} \quad\text{[eq. (20)]} \\
&\quad 10.\quad \operatorname{tr}(\mathbf{P}^{(t)}\mathbf{\Psi}) \text{ via Hutchinson (19), adaptive }R\text{ (default 30, step +10)} \\
&\quad 11.\quad S^{(t)} = \mathrm{AI}^{(t)} - \tfrac{1}{2}\operatorname{tr}(\mathbf{P}^{(t)}\mathbf{\Psi}) \quad\text{[eq. (21)]} \\
&\quad 12.\quad \tau^{(t+1)} = \max\!\left(\tau^{(t)} + S^{(t)}/\mathrm{AI}^{(t)},\; \epsilon\right) \quad\text{[eq. (22)]} \\
&\quad 13.\quad \text{Stop if } \max\!\left(\|\boldsymbol{\alpha}^{(t+1)}-\boldsymbol{\alpha}^{(t)}\|_\infty,\;|\tau^{(t+1)}-\tau^{(t)}|\right) < \delta
\end{aligned} \tag{23}
$$

At convergence, set $\hat{\boldsymbol{\alpha}}_0 = \boldsymbol{\alpha}^{(t+1)}$, $\hat{\mathbf{b}}_0 = \mathbf{b}^{(t+1)}$, $\hat{\tau}_0 = \tau^{(t+1)}$, and retain $\hat{\mathbf{W}}$, $\hat{\boldsymbol{\Sigma}}$, $\hat{\mathbf{P}}$ for use in the score test.

**Computational note.** SAIGE avoids forming $N\times N$ matrices throughout:

1. **$\boldsymbol{\Sigma}^{-1}$ (step 5):** Solved by **PCG** with diagonal preconditioner $\mathbf{M} = \operatorname{diag}(\boldsymbol{\Sigma}) = \operatorname{diag}(\mathbf{W}^{-1}) + \tau\operatorname{diag}(\mathbf{\Psi})$. Each matrix-vector product $\boldsymbol{\Sigma}\mathbf{p} = \mathbf{W}^{-1}\mathbf{p} + \tau\mathbf{\Psi}\mathbf{p}$ is computed on-the-fly; $\boldsymbol{\Sigma}$ is never stored. Convergence criterion: $\|\mathbf{r}\|^2 \leq \epsilon_{\text{PCG}}$ (default $10^{-5}$), max 500 iterations.

2. **$\operatorname{tr}(\mathbf{P}\mathbf{\Psi})$ (step 10):** Estimated by Hutchinson's estimator (eq. 19). SAIGE always uses this path—never exact. Rademacher vectors are drawn as $\mathbf{u} = \text{Binomial}(N)\times 2 - 1$. Adaptive: starts with $R = 30$, increases by 10 while $\mathrm{CV} > \texttt{traceCVcutoff}$ (default 0.0025).

3. **GRM $\mathbf{\Psi}$ (steps 5, 7, 9, 10):** Never stored as $N\times N$. All products $\mathbf{\Psi}\mathbf{v}$ are computed on-the-fly as $\tilde{\mathbf{G}}\tilde{\mathbf{G}}^\top\mathbf{v}/M_{\text{QC}}$, reading standardised genotypes from the bed file in memory chunks (`memoryChunk`, default 2 GB).

## 5. Score statistic under the null model

Under the null model ($\beta = 0$), the score statistic for testing $H_0 : \beta = 0$ is

$$
T = U_\beta\big|_{\beta=0,\,\hat{\boldsymbol{\alpha}}_0,\,\hat{\mathbf{b}}_0} = \mathbf{G}^\top(\mathbf{y} - \hat{\boldsymbol{\mu}}_0), \qquad \hat{\boldsymbol{\mu}}_0 = \operatorname{logit}^{-1}(\mathbf{X}\hat{\boldsymbol{\alpha}}_0 + \hat{\mathbf{b}}_0) \tag{24}
$$

Under the null, the asymptotic variance of $T$ is

$$
\operatorname{Var}(T) = \mathbf{G}^\top \hat{\mathbf{P}} \mathbf{G} \tag{25}
$$

where $\hat{\mathbf{P}}$ is the REML projection matrix (16) evaluated at the fitted PQL null parameters.

**Residualised genotype $\tilde{\mathbf{G}}$.** To avoid repeated $N \times N$ matrix operations for each tested variant, define the **projection of $\mathbf{G}$ onto the residual space of $\mathbf{X}$ in the working metric** $\hat{\mathbf{W}}$:

$$
\tilde{\mathbf{G}} = \mathbf{G} - \mathbf{X}\left(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X}\right)^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{G} \tag{26}
$$

Then $\operatorname{Var}(T) = \tilde{\mathbf{G}}^\top\hat{\mathbf{P}}\tilde{\mathbf{G}}$ (since $\hat{\mathbf{P}}\mathbf{X} = \mathbf{0}$ implies $\hat{\mathbf{P}}\mathbf{G} = \hat{\mathbf{P}}\tilde{\mathbf{G}}$).

**Variance correction factor $r$.** The SPA is applied to the **conditional** score given $\mathbf{b}_0$ (Bernoulli terms only). The conditional variance of $T$ given $\hat{\mathbf{b}}_0$ is

$$
\operatorname{Var}^*(T) = \tilde{\mathbf{G}}^\top \hat{\mathbf{W}} \tilde{\mathbf{G}} \tag{27}
$$

Define the ratio

$$
r = \frac{\operatorname{Var}(T)}{\operatorname{Var}^*(T)} = \frac{\tilde{\mathbf{G}}^\top \hat{\mathbf{P}} \tilde{\mathbf{G}}}{\tilde{\mathbf{G}}^\top \hat{\mathbf{W}} \tilde{\mathbf{G}}} \tag{28}
$$

which captures how much the GRM-based variance correction inflates (or deflates) the variance relative to the simple logistic score. The standardised statistic is

$$
T_{\text{adj}} = \frac{T}{\sqrt{\operatorname{Var}(T)}} \tag{29}
$$

## 6. Saddlepoint approximation (SPA) for the p-value

The SPA is applied to the conditional distribution of $T = \sum_i \tilde{G}_i(y_i - \hat\mu_{i0})$ given $\hat{\mathbf{b}}_0$ fixed. Since each $y_i | \hat b_{i0}$ is Bernoulli$(\hat\mu_{i0})$, the **CGF of $T$** is exactly

$$
K(t) = \sum_{i=1}^N \log\!\left[\hat\mu_{i0}\,e^{\tilde{G}_i(1-\hat\mu_{i0})t} + (1-\hat\mu_{i0})\,e^{-\tilde{G}_i\hat\mu_{i0}t}\right] \tag{30}
$$

with derivatives

$$
K'(t) = \sum_{i=1}^N \tilde{G}_i\,\hat\mu_{i0}(t)\left[1-\hat\mu_{i0}(t)\right] \cdot \tilde{G}_i = \sum_i \tilde{G}_i^2\hat\mu_{i0}(t)(1-\hat\mu_{i0}(t)) \tag{31}
$$

$$
K''(t) = \sum_{i=1}^N \tilde{G}_i^2\,\hat\mu_{i0}(t)\left[1-\hat\mu_{i0}(t)\right] \tag{32}
$$

where $\hat\mu_{i0}(t) = \operatorname{logit}^{-1}(\mathbf{X}_i\hat{\boldsymbol{\alpha}}_0 + \hat b_{i0} + \tilde{G}_i t)$ is the tilted mean.

**Saddlepoint equation.** The saddlepoint $\hat{t}$ satisfies $K'(\hat{t}) = q_{\text{obs}}$ where $q_{\text{obs}} = \sqrt{r}\,T_{\text{adj}}$. Solve by Brent's method on $K'(t) - q_{\text{obs}} = 0$.

**Lugannani-Rice formula.** Let $\hat{w} = \operatorname{sign}(\hat{t})\sqrt{2[\hat{t}\,q_{\text{obs}} - K(\hat{t})]}$ and $\hat{u} = \hat{t}\sqrt{K''(\hat{t})}$. The upper-tail CDF approximation is

$$
\Phi_{\mathrm{SPA}}(q_{\text{obs}}) \approx \Phi(\hat{w}) + \phi(\hat{w})\!\left(\frac{1}{\hat{w}} - \frac{1}{\hat{u}}\right) \tag{33}
$$

where $\Phi$ and $\phi$ are the standard normal CDF and PDF. The two-sided $p$-value is

$$
p = \left[1 - \Phi_{\mathrm{SPA}}\!\left(|q_{\text{obs}}|\right)\right] + \Phi_{\mathrm{SPA}}\!\left(-|q_{\text{obs}}|\right) \tag{34}
$$
