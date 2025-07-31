In a case-control study with a sample size $N$, let

- $\mathbf{y} = [y_1,\ldots,y_N]^{\top}$ represent their phenotypes, where $y_i \sim \operatorname{Bernoulli}(\mu_i)$;
- $N \times (1 + p)$ matrix $\mathbf{X}$ represent their $p$ covariates and a column of ones;
- $N \times 1$ vector $\mathbf{g}$ represent their genotypes coded as allele counts for a variant to be tested;
- $N \times 1$ vector $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$ represent random effects, where $\mathbf{\Psi}$ is a GRM and $\tau$ is the variance component;
- $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$ be a linear predictor, where $\boldsymbol{\alpha}$ is the fixed effects and $\beta$ is the genetic effect to be tested.

Suppose $\mu_i = \operatorname{logit}^{-1}(\eta_i)$. The log-likelihood function for $\boldsymbol{\alpha}, \beta, \tau$ is

$$
\ell(\boldsymbol{\alpha}, \beta, \tau; \mathbf{X}, \mathbf{g}, \mathbf{\Psi}, \mathbf{y}) = \log \int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi}) \, \mathrm{d}\mathbf{b}

$$

where $p_{\mathbf{y} | \mathbf{b}}(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta)$ is the conditional PMF of $\mathbf{y}$ given $\mathbf{b}$, and $f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi})$ is the PDF of $\mathbf{b}$.

Under the null hypothesis $H_0: \beta=0$, let $(\hat{\tau}, \hat{\boldsymbol{\alpha}}, \hat{\mathbf{b}})$ be an estimate of $(\tau, \boldsymbol{\alpha}, \mathbf{b})$ and let $\hat{\boldsymbol{\eta}} = \mathbf{X} \hat{\boldsymbol{\alpha}} + \hat{\mathbf{b}}$.

The log-likelihood function for $\beta$ is

$$
\ell(\beta; \mathbf{y}, \mathbf{g}, \hat{\boldsymbol{\eta}}) = \log p_{\mathbf{y}}(\mathbf{y}; \beta, \mathbf{g}, \hat{\boldsymbol{\eta}}) = \mathbf{y}^\top \log(\boldsymbol{\mu}) + (\mathbf{1} - \mathbf{y})^\top \log(\mathbf{1} - \boldsymbol{\mu})

$$

Substituting $\boldsymbol{\mu} = [1 + \exp(-\mathbf{g}\beta - \hat{\boldsymbol{\eta}})]^{-1}$ and simplifying, we obtain

$$
\ell(\beta; \mathbf{y}, \mathbf{g}, \hat{\boldsymbol{\eta}}) = \mathbf{y}^\top (\mathbf{g}\beta + \hat{\boldsymbol{\eta}}) - \mathbf{1}^\top \log[1 + \exp(\mathbf{g}\beta + \hat{\boldsymbol{\eta}})]

$$

The score function for $\beta$ is:

$$
U_\beta(\beta) = \frac{\partial \ell}{\partial \beta} = \mathbf{g}^\top \mathbf{y} - \mathbf{g}^\top \operatorname{logit^{-1}}(\mathbf{g}\beta + \hat{\boldsymbol{\eta}}) = \mathbf{g}^\top (\mathbf{y} - \boldsymbol{\mu})
$$

The Fisher information for $\beta$ is:

$$
I_\beta(\beta) = -\mathbb{E}_{\mathbf{y}} \left[ \frac{\partial^2 \ell}{\partial\beta^2} \right]= \sum_{i=1}^N g_i^2 \mu_i (1 - \mu_i)= \mathbf{g}^\top \mathbf{W} \mathbf{g}
$$

where $\mathbf{W} = \operatorname{diag}(\boldsymbol{\mu} \odot [\mathbf{1} - \boldsymbol{\mu}])$ and $\odot$ denotes elementwise multiplication.