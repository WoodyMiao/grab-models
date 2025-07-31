# Appendix

## Log-likelihood function derivation for a fixed-effects logistic model

In a case-control study with a sample size $N$, let

- $\mathbf{y}=[y_1,\ldots,y_N]^\top,\ \boldsymbol{\mu} = [\mu_1,\ldots,\mu_N]^\top,\ y_i \sim \mathrm{Bernoulli}(\mu_i)$
- $\mathbf{X}$ be an $N \times (1 + p)$ design matrix and $\mathbf{x}_i$ denote its $i$th row
- $\boldsymbol{\alpha}$ be a $(1 + p) \times 1$ vector of fixed effects

Suppose $\mu_i = \operatorname{logit}^{-1}(\mathbf{x}_i \boldsymbol{\alpha})$

We derive the log-likelihood function $\ell(\boldsymbol{\alpha}; \mathbf{y}, \mathbf{X})$:

### Joint PMF of $\mathbf{y}$

$$
p(\mathbf{y}; \mathbf{X}, \boldsymbol{\alpha}) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}, \quad \mu_i = \frac{1}{1 + \mathrm{exp}(-\mathbf{x}_i \boldsymbol{\alpha})}
$$

### Log-likelihood function for $\boldsymbol{\alpha}$

$$
\begin{aligned}
\ell(\boldsymbol{\alpha}; \mathbf{y}, \mathbf{X})
&= \log \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i} \\
&= \sum_{i=1}^N \left[ y_i \log(\mu_i) + (1 - y_i) \log(1 - \mu_i) \right] \\
&= \mathbf{y}^\top \log(\boldsymbol{\mu}) + (\mathbf{1} - \mathbf{y})^\top \log(\mathbf{1} - \boldsymbol{\mu})
\end{aligned}
$$

Substituting $\boldsymbol{\mu} = \frac{1}{1 + \exp(-\mathbf{X} \boldsymbol{\alpha})}$ yields

$$
\begin{aligned}
\log(\boldsymbol{\mu}) &= -\log(1 + \exp(-\mathbf{X} \boldsymbol{\alpha})) \\
\log(\mathbf{1} - \boldsymbol{\mu}) &= \log\left(1 - \frac{1}{1 + \exp(-\mathbf{X} \boldsymbol{\alpha})}\right) \\
&= \log\left(\frac{\exp(-\mathbf{X} \boldsymbol{\alpha})}{1 + \exp(-\mathbf{X} \boldsymbol{\alpha})}\right) \\
&= -\mathbf{X} \boldsymbol{\alpha} - \log(1 + \exp(-\mathbf{X} \boldsymbol{\alpha}))
\end{aligned}
$$

Substituting into the log-likelihood function yields
$$
\begin{aligned}
\ell(\boldsymbol{\alpha}; \mathbf{y}, \mathbf{X})
&= \mathbf{y}^\top \left[-\log(1 + \exp(-\mathbf{X} \boldsymbol{\alpha}))\right] \\
&\quad + (\mathbf{1} - \mathbf{y})^\top \left[-\mathbf{X} \boldsymbol{\alpha} - \log(1 + \exp(-\mathbf{X} \boldsymbol{\alpha}))\right] \\
&= -\mathbf{y}^\top \log(1 + \exp(-\mathbf{X} \boldsymbol{\alpha})) \\
&\quad - (\mathbf{1} - \mathbf{y})^\top \mathbf{X} \boldsymbol{\alpha} - (\mathbf{1} - \mathbf{y})^\top \log(1 + \exp(-\mathbf{X} \boldsymbol{\alpha})) \\
&= -\mathbf{1}^\top \log(1 + \exp(-\mathbf{X} \boldsymbol{\alpha})) - (\mathbf{1} - \mathbf{y})^\top \mathbf{X} \boldsymbol{\alpha} \\
&= \mathbf{y}^\top \mathbf{X}\boldsymbol{\alpha} - \mathbf{1}^\top (\mathbf{X} \boldsymbol{\alpha} + \log[1 + \exp(-\mathbf{X} \boldsymbol{\alpha})]) \\
&= \mathbf{y}^\top \mathbf{X} \boldsymbol{\alpha} - \mathbf{1}^\top \log(1 + \exp(\mathbf{X} \boldsymbol{\alpha}))
\end{aligned}
$$

## Log-likelihood function derivation for a random-effects logistic model

In a case-control study with a sample size $N$, let

- $\mathbf{y}=[y_1,\ldots,y_N]^\top,\ \boldsymbol{\mu} = [\mu_1,\ldots,\mu_N]^\top,\ y_i \sim \mathrm{Bernoulli}(\mu_i)$
- $\mathbf{Z}$ be an $N \times m$ design matrix
- $\boldsymbol{\gamma} \sim \mathcal{N}(\mathbf{0}, [\tau/m] \mathbf{I})$ be a vector of random effects
- $\mathbf{b}=[b_1,\ldots,b_N]^\top = \mathbf{Z}\boldsymbol{\gamma} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$, $\mathbf{\Psi} = (1/m)\mathbf{Z}\mathbf{Z}^\top$

Suppose $\mu_i = \operatorname{logit}^{-1}(b_i)$.
We derive the log-likelihood function $\ell(\tau; \mathbf{y}, \mathbf{\Psi})$:

### Conditional joint PMF of $\mathbf{y}$ given $\mathbf{b}$

$$
p_{\mathbf{y}|\mathbf{b}}(\mathbf{y}| \mathbf{b}) = \prod_{i=1}^N \mu_i^{y_i} (1-\mu_i)^{1-y_i}, \quad \mu_i = \frac{1}{1 + \mathrm{exp}(-b_i)}
$$

### Marginal joint PMF of $\mathbf{y}$

$$
p_{\mathbf{y}}(\mathbf{y}; \tau, \mathbf{\Psi}) = \int_{\mathbb{R}^N} p_{\mathbf{y}|\mathbf{b}}(\mathbf{y}| \mathbf{b}) \, f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi})\, \mathrm{d}\mathbf{b}
$$

where

$$
f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi}) = \frac{1}{(2\pi)^{\frac{N}{2}} |\tau \mathbf{\Psi}|^{\frac{1}{2}}} \exp\left(-\frac{1}{2} \mathbf{b}^\top (\tau \mathbf{\Psi})^{-1} \mathbf{b}\right)
$$

is the PDF of $\mathbf{b}$.

### Log-likelihood function for $\tau$

$$
\ell(\tau; \mathbf{y}, \mathbf{\Psi}) = \log \int_{\mathbb{R}^N} p_{\mathbf{y}|\mathbf{b}}(\mathbf{y}| \mathbf{b}) \, f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi})\, \mathrm{d}\mathbf{b}
$$

The integral does not have a closed-form solution.

## Log-likelihood function derivation for a logistic mixed model

In a case-control study with a sample size $N$, let

- $\mathbf{y} = [y_1,\ldots,y_N]^{\top}$ represent their phenotypes, where $y_i \sim \operatorname{Bernoulli}(\mu_i)$;
- $N \times (1 + p)$ matrix $\mathbf{X}$ represent their $p$ covariates and a column of ones;
- $N \times 1$ vector $\mathbf{g}$ represent their genotypes coded as allele counts for a variant to be tested;
- $N \times 1$ vector $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$ represent random effects, where $\mathbf{\Psi}$ is a GRM and $\tau$ is the additive genetic variance;
- $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$ be a linear predictor, where $\boldsymbol{\alpha}$ is the fixed effects and $\beta$ is the genetic effect to be tested.

Suppose $\mu_i = \operatorname{logit}^{-1}(\eta_i)$.
We derive the log-likelihood function $\ell(\boldsymbol{\alpha}, \beta, \tau; \mathbf{X}, \mathbf{g}, \mathbf{\Psi}, \mathbf{y})$:

### Conditional PMF of $\mathbf{y}$ given $\mathbf{b}$

$$
p_{\mathbf{y} | \mathbf{b}}(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}, \quad \mu_i = \frac{1}{1 + \mathrm{exp}(-\eta_i)}
$$

### Marginal PMF of $\mathbf{y}$

$$
p_{\mathbf{y}}(\mathbf{y}; \mathbf{X}, \mathbf{g}, \mathbf{\Psi}, \boldsymbol{\alpha}, \beta, \tau) = \int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi}) \, \mathrm{d}\mathbf{b}
$$

where $f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi})$ is the PDF of $\mathbf{b}$.

### Log-likelihood function for $\boldsymbol{\alpha}, \beta, \tau$

$$
\ell(\boldsymbol{\alpha}, \beta, \tau; \mathbf{X}, \mathbf{g}, \mathbf{\Psi}, \mathbf{y}) = \log \int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi}) \, \mathrm{d}\mathbf{b}
$$

The integral does not have a closed-form solution.
