# Appendix

## Log-likelihood derivation for a logistic model

In a case-control study with a sample size $N$, let

- $\mathbf{y}=[y_1,\ldots,y_N]^\top$, $y_i \sim \mathrm{Bernoulli}(\mu_i)$
- $\mathbf{X}$ is an $N \times (1 + p)$ design matrix
- $\boldsymbol{\alpha}$ is a $(1 + p) \times 1$ vector of fixed effects
- $\boldsymbol{\mu} = [\mu_1,\ldots,\mu_N]^\top = \mathbb{E}(\mathbf{y})$

Suppose $\operatorname{logit}(\boldsymbol{\mu})=\mathbf{X} \boldsymbol{\alpha}$. We derive the log-likelihood function $\ell(\boldsymbol{\alpha}; \mathbf{y}, \mathbf{X})$:

1. **Joint PMF of $\mathbf{y}$, i.e., likelihood of $\boldsymbol{\alpha}$:**
    $$
    p(\mathbf{y}; \mathbf{X}, \boldsymbol{\alpha}) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}, \quad \mu_i = \frac{1}{1 + \mathrm{exp}(-\mathbf{x}_i^\top \boldsymbol{\alpha})}
    $$

2. **Log-likelihood function:**
    $$
    \begin{align*}
    \ell(\boldsymbol{\alpha}; \mathbf{y}, \mathbf{X}) 
    &= \sum_{i=1}^N \left[ y_i \log(\mu_i) + (1 - y_i) \log(1 - \mu_i) \right] \\\\
    &= \mathbf{y}^\top \log(\boldsymbol{\mu}) + (\mathbf{1} - \mathbf{y})^\top \log(\mathbf{1} - \boldsymbol{\mu})
    \end{align*}
    $$

## Log-likelihood derivation for a logistic random effects model

In a case-control study with a sample size $N$, let

- $\mathbf{y}=[y_1,\ldots,y_N]^\top$, $y_i \sim \mathrm{Bernoulli}(\mu_i)$
- $\mathbf{Z}$ is a column-wise normalized $N \times m$ design matrix
- $\boldsymbol{\gamma} \sim \mathcal{N}(\mathbf{0}, [\tau/m] \mathbf{I})$ is a random effect vector
- $\mathbf{b} = \mathbf{Z}\boldsymbol{\gamma} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$, $\mathbf{\Psi} = (1/m)\mathbf{Z}\mathbf{Z}^\top$
- $\boldsymbol{\mu} = [\mu_1,\ldots,\mu_N]^\top = \mathbb{E}(\mathbf{y}|\mathbf{b})$

Suppose $\operatorname{logit}(\boldsymbol{\mu})=\mathbf{b}$. We derive the log-likelihood function $\ell(\tau; \mathbf{y}, \mathbf{\Psi})$:

1. **Conditional PMF of $\mathbf{y}$ given $\mathbf{b}$:**
     $$
     p_{\mathbf{y}|\mathbf{b}}(\mathbf{y}| \mathbf{b}) = \prod_{i=1}^N \mu_i^{y_i} (1-\mu_i)^{1-y_i}, \quad \mu_i = \operatorname{logit}^{-1}(b_i)
     $$

2. **Marginal PMF of $\mathbf{y}$, i.e., likelihood for $\tau$:**
     $$
     p_{\mathbf{y}}(\mathbf{y}; \tau, \mathbf{\Psi}) = \int p(\mathbf{y}|\mathbf{b})\, f_{\mathbf{b}}(\mathbf{b})\, d\mathbf{b}
     $$

    where $f_{\mathbf{b}}(\mathbf{b})$ is the PDF of $\mathbf{b}$:
    $$
     f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi}) = \frac{1}{(2\pi)^{\frac{N}{2}} |\tau \mathbf{\Psi}|^{\frac{1}{2}}} \exp\left(-\frac{1}{2} \mathbf{b}^\top (\tau \mathbf{\Psi})^{-1} \mathbf{b}\right)
     $$

3. **Log-likelihood for $\tau$:**
     $$
     \ell(\tau; \mathbf{y}, \mathbf{\Psi}) = \log \int p_{\mathbf{y}|\mathbf{b}}(\mathbf{y}|\mathbf{b})\, f_{\mathbf{b}}(\mathbf{b})\, d\mathbf{b}
     $$

## Log-likelihood derivation for a logistic mixed model

In a case-control study with a sample size $N$, let

- $\mathbf{y} = [y_1,\ldots,y_N]^{\top}$ represent their phenotypes, where $y_i \sim \operatorname{Bernoulli}(\mu_i)$;

- $N \times (1 + p)$ matrix $\mathbf{X}$ represent their $p$ covariates and a column of ones;

- $N \times 1$ vector $\mathbf{g}$ represent their genotypes coded as allele counts for a variant to be tested;

- $N \times 1$ vector $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$ represent random effects, where $\mathbf{\Psi}$ is a GRM and $\tau$ is the additive genetic variance;

- $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$ be a linear predictor, where $\boldsymbol{\alpha}$ is the fixed effects and $\beta$ is the genetic effect to be tested.

Suppose $\mathbb{E}(\mathbf{y}|\mathbf{b}) = \operatorname{logit}(\boldsymbol{\mu})=\boldsymbol{\eta}$. We derive the log-likelihood function $\ell(\boldsymbol{\alpha}, \beta, \tau; \mathbf{X}, \mathbf{g}, \mathbf{\Psi}, \mathbf{y})$;

1. **Conditional PMF of $\mathbf{y}$ given $\mathbf{b}$:**

    $$
    p_{\mathbf{y} | \mathbf{b}}(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}
    $$

2. **Marginal PMF of $\mathbf{y}$, i.e., likelihood for $\boldsymbol{\alpha}, \beta, \tau$:**

    $$
    p_{\mathbf{y}}(\mathbf{y}; \mathbf{X}, \mathbf{g}, \mathbf{\Psi}, \boldsymbol{\alpha}, \beta, \tau) = \int p(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) f_{\mathbf{b}}(\tau, \mathbf{\Psi}) \, d\mathbf{b}
    $$

    where $f_{\mathbf{b}}(\tau, \mathbf{\Psi})$ is the PDF of $\mathbf{b}$.

3. **Log-likelihood for $\boldsymbol{\alpha}, \beta, \tau$:**

    $$
    \ell(\boldsymbol{\alpha}, \beta, \tau; \mathbf{X}, \mathbf{g}, \mathbf{\Psi}, \mathbf{y}) = \log \int p(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) f_{\mathbf{b}}(\tau, \mathbf{\Psi}) \, d\mathbf{b}
    $$
