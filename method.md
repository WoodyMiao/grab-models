In a case-control study with a sample size $N$, let

- $\mathbf{y} = [y_1,\ldots,y_N]^{\top}$ represent their phenotypes, where $y_i \sim \operatorname{Bernoulli}(\mu_i)$;
- $N \times (1 + p)$ matrix $\mathbf{X}$ represent their $p$ covariates and a column of ones;
- $N \times 1$ vector $\mathbf{g}$ represent their genotypes coded as allele counts for a variant to be tested;
- $N \times 1$ vector $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$, where $\mathbf{\Psi}$ is a GRM and $\tau$ is the additive genetic variance;
- $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$, where $\boldsymbol{\alpha}$ is the fixed effects and $\beta$ is the genetic effect to be tested.

Suppose $\operatorname{logit}(\mu_i) = \eta_i$. 

**1. The log-likelihood function for $\boldsymbol{\alpha}, \beta, \tau$ is**

$$
\ell(\boldsymbol{\alpha}, \beta, \tau; \mathbf{X}, \mathbf{g}, \mathbf{\Psi}, \mathbf{y}) = \log \int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi}) \, \mathrm{d}\mathbf{b}
$$

where $p_{\mathbf{y} | \mathbf{b}}(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta)$ is the conditional PMF of $\mathbf{y}$ given $\mathbf{b}$, and $f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi})$ is the PDF of $\mathbf{b}$.

Under the null hypothesis $\beta = 0$, $(\tau, \boldsymbol{\alpha}, \mathbf{b})$ are estimated by $(\hat{\tau}, \hat{\boldsymbol{\alpha}}, \hat{\mathbf{b}})$.

**2. The log-likelihood function for $\beta$ is**

$$
\ell(\beta; \mathbf{y}, \mathbf{g}, \hat{\boldsymbol{\eta}}) = \mathbf{y}^\top \log(\boldsymbol{\mu}) + (\mathbf{1} - \mathbf{y})^\top \log(\mathbf{1} - \boldsymbol{\mu})
$$

where  $\hat{\boldsymbol{\eta}} = \mathbf{X} \hat{\boldsymbol{\alpha}} + \hat{\mathbf{b}}$ and $\boldsymbol{\mu} = \operatorname{logit}^{-1}(\mathbf{g}\beta + \hat{\boldsymbol{\eta}})$.

Substituting $\boldsymbol{\mu} = [1 + \exp(-\mathbf{g}\beta - \hat{\boldsymbol{\eta}})]^{-1}$ and simplifying, we obtain

$$
\ell(\beta; \mathbf{y}, \mathbf{g}, \hat{\boldsymbol{\eta}}) = \mathbf{y}^\top (\mathbf{g}\beta + \hat{\boldsymbol{\eta}}) - \mathbf{1}^\top \log[1 + \exp(\mathbf{g}\beta + \hat{\boldsymbol{\eta}})]
$$

**3. The score function for $\beta$ is**

$$
U_\beta(\beta) = \frac{\partial \ell(\beta; \mathbf{y}, \mathbf{g}, \hat{\boldsymbol{\eta}})}{\partial \beta} = \mathbf{g}^\top (\mathbf{y} - \boldsymbol{\mu})
$$

The score statistic is:
$$
T = U_\beta(0) = \mathbf{g}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})
$$

where $\hat{\boldsymbol{\mu}} = \operatorname{logit}^{-1}(\hat{\boldsymbol{\eta}})$.

**4. Express $T$ in terms of the covariate-adjusted genotype vector:**

$$
T = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})
$$

where $\tilde{\mathbf{g}} = \mathbf{g} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{g}$ is the residual from regressing $\mathbf{g}$ on $\mathbf{X}$ using the weight matrix $\hat{\mathbf{W}} = \operatorname{diag}(\hat{\boldsymbol{\mu}} \odot [\mathbf{1} - \hat{\boldsymbol{\mu}}])$.

**5. The variance of $T$ can be estimated as**
$$
\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}
$$

where $\hat{\mathbf{P}} = \hat{\boldsymbol{\Sigma}}^{-1} - \hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X}(\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}$ and $\hat{\boldsymbol{\Sigma}} = \hat{\mathbf{W}}^{-1} + \hat{\tau}\mathbf{\Psi}$.


**6. The standardized score statistic is**

$$
\frac{T}{\sqrt{\mathbb{V}(T)}} = \frac{\tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})}{\sqrt{\tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}}}
$$

which, as $N \to \infty$, converges in distribution to $\mathcal{N}(0, 1)$.

