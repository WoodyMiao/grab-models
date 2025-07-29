# SAIGE

We use regular characters to represent scalar variables, bold lowercase characters to represent vectors, and bold uppercase characters to represent matrices.

## Logistic mixed model for a binary trait

In a case-control study with a sample size $N$,

let $\mathbf{y} = [y_1,\ldots,y_N]^{\top}$ represent their phenotypes, where $y_i \sim \operatorname{Bernoulli}(\mu_i)$;

let an $N \times (1 + p)$ matrix $\mathbf{X}$ represent their $p$ covariates and a column of ones;

let an $N \times 1$ vector $\mathbf{g}$ represent their genotypes coded as allele counts for a variant to be tested;

let an $N \times 1$ vector $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$ represent random effects, where $\mathbf{\Psi}$ is a GRM and $\tau$ is the additive genetic variance;

let $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$ be a linear predictor, where $\boldsymbol{\alpha}$ is the fixed effects and $\beta$ is the genetic effect to be tested.

Suppose
$$
\operatorname{logit}^{-1}(\eta_i) = \mu_i = \mathbb{P}(y_i = 1 | b_i; \mathbf{x}_i, g_i, \boldsymbol{\alpha}, \beta)
$$

The PMF for $y_i$ is
$$
p(y_i | b_i; \mathbf{x}_i, g_i, \boldsymbol{\alpha}, \beta) = \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}
$$

The joint PMF for $\mathbf{y}$ conditional on $\mathbf{b}$ is
$$
p(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}
$$

The marginal PMF for $\mathbf{y}$ is:
$$
p(\mathbf{y}; \mathbf{X}, \mathbf{g}, \mathbf{\Psi}, \boldsymbol{\alpha}, \beta, \tau) = \int p(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) f_{\mathbf{b}}(\tau, \mathbf{\Psi}) \, d\mathbf{b}
$$
where $f_{\mathbf{b}}(\tau, \mathbf{\Psi})$ is the PDF of $\mathbf{b}$.

The marginal log-likelihood is:
$$
\ell(\boldsymbol{\alpha}, \beta, \tau; \mathbf{X}, \mathbf{g}, \mathbf{\Psi}, \mathbf{y}) = \log \int p(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) f_{\mathbf{b}}(\tau, \mathbf{\Psi}) \, d\mathbf{b}
$$

The score for $\beta$ is:

$$
U_\beta = \left. \frac{\partial \ell}{\partial \beta} \right|_{\beta=0} = \left. \mathbf{g}^\top (\mathbf{y} - \boldsymbol{\mu}) \right|_{\beta=0}
$$

and the Fisher information for $\beta$ is

$$
I_\beta = -\left. \mathbb{E}\left[ \frac{\partial^2 \ell}{\partial \beta^2} \right] \right|_{\beta=0} = \left. \mathbf{g}^\top \mathbf{W} \mathbf{g} \right|_{\beta=0}
$$

where $\mathbf{W} = \operatorname{diag}(\boldsymbol{\mu} \odot [\mathbf{1} - \boldsymbol{\mu}])$ and $\odot$ denotes element-wise multiplication.

## Score test statistic

Under the null hypothesis, the model becomes

$$
\text{logit}(\boldsymbol{\mu}) = \mathbf{X} \boldsymbol{\alpha} + \mathbf{b}
$$

Let $(\hat{\boldsymbol{\alpha}}, \hat{\mathbf{b}}, \hat{\tau})$ denote an estimate of $(\boldsymbol{\alpha}, \mathbf{b}, \tau)$.

Let

$$
T = \mathbf{g}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})
$$

where $\hat{\boldsymbol{\mu}} = \text{logit}^{-1}(\mathbf{X} \hat{\boldsymbol{\alpha}} + \hat{\mathbf{b}})$. Let

$$
\tilde{\mathbf{g}} = \mathbf{g} - \mathbf{X} (\mathbf{X}^\top \hat{\mathbf{W}} \mathbf{X})^{-1} \mathbf{X}^\top \hat{\mathbf{W}} \mathbf{g}
$$

where $\hat{\mathbf{W}} = \operatorname{diag}(\hat{\boldsymbol{\mu}} \odot [\mathbf{1} - \hat{\boldsymbol{\mu}}])$. Then, we have

$$
T= \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}), \quad
\mathbb{E}T = 0, \quad
\mathbb{V}T = \mathbf{g}^\top \hat{\mathbf{W}} \mathbf{g} = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}
$$

where

$$
\hat{\mathbf{P}} = \hat{\boldsymbol{\Sigma}}^{-1} - \hat{\boldsymbol{\Sigma}}^{-1} \mathbf{X} (\mathbf{X}^\top \hat{\boldsymbol{\Sigma}}^{-1} \mathbf{X})^{-1} \mathbf{X}^\top \hat{\boldsymbol{\Sigma}}^{-1},\quad
\hat{\boldsymbol{\Sigma}} = \hat{\mathbf{W}}^{-1} + \hat{\tau} \mathbf{\Psi}
$$

Then, we have

$$
\frac{T}{\sqrt{\mathbb{V}T}} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } N \to \infty
$$

That is, under the null hypothesis and as the sample size $N$ becomes large, the test statistic $T/\sqrt{\mathbb{V}T}$ asymptotically follows a standard normal distribution.

## SPA of CDF of score test statistic

To improve the accuracy of the p-value for the statistic $T/\sqrt{\mathbb{V}T}$, especially in finite samples or for rare variants, we use the saddlepoint approximation (SPA) to approximate its cumulative distribution function (CDF).

Let $K(\xi)$ denote the cumulant generating function (CGF) of $T$ under the null hypothesis:

$$
K(\xi) = \log \mathbb{E}[e^{\xi T}]
$$

Let $t$ be the observed value of $T$. The saddlepoint $\hat{\xi}$ is the solution to

$$
K'(\hat{\xi}) = t
$$

The SPA for the CDF of $T$ is given by

$$
\Pr(T \leq t) \approx \Phi(w) + \phi(w) \left( \frac{1}{w} - \frac{1}{v} \right)
$$

where

$$
w = \operatorname{sgn}(\hat{\xi}) \sqrt{2(\hat{\xi} t - K(\hat{\xi}))}
$$

$$
v = \hat{\xi} \sqrt{K''(\hat{\xi})}
$$

Here, $\Phi$ and $\phi$ are the CDF and PDF of the standard normal distribution, respectively.
