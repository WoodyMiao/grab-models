# SAIGE

We use regular characters to represent scalar variables, bold lowercase characters to represent vectors, and bold uppercase characters to represent matrices.

## Logistic mixed model for a binary trait

In a case-control study with a sample size $N$, for the $i$th individual, let $y_i \in \{0, 1\}$ represent their phenotype; let a $1 \times (1 + p)$ vector $\mathbf{x}_i$ represent their $p$ covariates and the intercept; and let $G_i \in \{0, 1, 2\}$ represent their genotype for a variant to be tested. Suppose

$$
\text{logit}(\mu_i) = \mathbf{x}_i \boldsymbol{\alpha} + G_i \beta + b_i
$$

where $\mu_i = \text{P}(y_i = 1|\mathbf{x}_i, G_i, b_i)$ is their fitted probability; $\boldsymbol{\alpha}$ is the $(1 + p) \times 1$ vector of fixed effects; $\beta$ is the genetic effect of the variant to be tested; $b_i$ is the $i$th component of vector $\mathbf{b}$, which is the $N \times 1$ vector of random effects and follows $\mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$, where $\mathbf{\Psi}$ is an $N \times N$ GRM and $\tau$ is the corresponding additive genetic variance.

Let
$$
\boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \vdots \\ \mu_N \end{bmatrix} \quad
\mathbf{y} = \begin{bmatrix} y_1 \\ \vdots \\ y_N \end{bmatrix} \quad
\mathbf{g} = \begin{bmatrix} G_1 \\ \vdots \\ G_N \end{bmatrix} \quad
\mathbf{X} = \begin{bmatrix} \mathbf{x}_1 \\ \vdots \\ \mathbf{x}_N \end{bmatrix}
$$
We write the model using vector notation as
$$
\text{logit}(\boldsymbol{\mu}) = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}
$$
where $\boldsymbol{\mu}=\mathbb{E}(\mathbf{y}|\mathbf{X}, \mathbf{g}, \mathbf{b})$ and $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$. Then, the log-likelihood function under the full model is
$$
\ell(\boldsymbol{\alpha}, \beta, \mathbf{b}, \tau; \mathbf{X}, \mathbf{g}, \mathbf{y}) = \mathbf{y}^\top \log(\boldsymbol{\mu}) + (\mathbf{1} - \mathbf{y})^\top \log(\mathbf{1} - \boldsymbol{\mu}) - \frac{1}{2\tau} \mathbf{b}^\top \mathbf{\Psi}^{-1} \mathbf{b} - \frac{N}{2} \log(2\pi \tau) - \frac{1}{2} \log|\mathbf{\Psi}|
$$
where $\boldsymbol{\mu} = \text{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b})$.

The score for $\beta$ is defined as:
$$
U_\beta = \left. \frac{\partial \ell}{\partial \beta} \right|_{\beta=0} = \mathbf{g}^\top (\mathbf{y} - \boldsymbol{\mu})
$$
where $\boldsymbol{\mu}=\text{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha} + \mathbf{b})$ is evaluated at $H_0: \beta=0$. And the Fisher information for $\beta$ is
$$
I_\beta = -\left. \mathbb{E}\left[ \frac{\partial^2 \ell}{\partial \beta^2} \right] \right|_{\beta=0} = \mathbf{g}^\top \mathbf{W} \mathbf{g}
$$
where $\mathbf{W} = \operatorname{diag}(\boldsymbol{\mu} \odot [\mathbf{1} - \boldsymbol{\mu}])$ and $\odot$ denotes element-wise multiplication.

## Score statistic

Under the null hypothesis $H_0: \beta=0$, the model becomes

$$
\text{logit}(\boldsymbol{\mu}) = \mathbf{X} \boldsymbol{\alpha} + \mathbf{b}
$$

Let $(\hat{\boldsymbol{\alpha}}, \hat{\mathbf{b}}, \hat{\tau})$ denote an estimate of $(\boldsymbol{\alpha}, \mathbf{b}, \tau)$ and let $\hat{\boldsymbol{\mu}} = \text{logit}^{-1}(\mathbf{X} \hat{\boldsymbol{\alpha}} + \hat{\mathbf{b}})$.

Let a statistic be
$$
T = \mathbf{g}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}) = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})
$$
where
$$
\tilde{\mathbf{g}} = \mathbf{g} - \mathbf{X} (\mathbf{X}^\top \hat{\mathbf{W}} \mathbf{X})^{-1} \mathbf{X}^\top \hat{\mathbf{W}} \mathbf{g},\quad \hat{\mathbf{W}} = \operatorname{diag}(\hat{\boldsymbol{\mu}} \odot [\mathbf{1} - \hat{\boldsymbol{\mu}}])
$$
is the covariate-adjusted genotype vector. The variance of $T$ is
$$
\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}},
$$
where
$$
\hat{\mathbf{P}} = \hat{\boldsymbol{\Sigma}}^{-1} - \hat{\boldsymbol{\Sigma}}^{-1} \mathbf{X} (\mathbf{X}^\top \hat{\boldsymbol{\Sigma}}^{-1} \mathbf{X})^{-1} \mathbf{X}^\top \hat{\boldsymbol{\Sigma}}^{-1},\quad
\hat{\boldsymbol{\Sigma}} = \hat{\mathbf{W}}^{-1} + \hat{\tau} \mathbf{\Psi}
$$
