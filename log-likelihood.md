# Log-likelihood derivation for the generalized linear mixed model

Consider $N$ independent case-control samples. Let:

- $\mathbf{y}$: $N \times 1$ vector of binary phenotypes ($y_i \in \{0, 1\}$)
- $\mathbf{X}$: $N \times (1 + p)$ design matrix for fixed effects (covariates + intercept)
- $\boldsymbol{\alpha}$: $(1 + p) \times 1$ vector of fixed effect coefficients
- $\mathbf{g}$: $N \times 1$ vector of genotypes (allele counts) for the variant tested
- $\beta$: scalar genetic effect of the tested variant
- $\mathbf{Z}$: $N \times m$ matrix of normalized genotypes for $m$ markers with random effects
- $\boldsymbol{\gamma} \sim \mathcal{N}(\mathbf{0}, [\tau/m] \mathbf{I})$: $m \times 1$ vector of random effect coefficients
- $\mathbf{b} = \mathbf{Z}\boldsymbol{\gamma}$: $N \times 1$ vector of total random effects
- $\mathbf{\Psi} = \frac{1}{m}\mathbf{Z}\mathbf{Z}^\top$: $N \times N$ genetic relationship matrix (GRM)
- $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$
- $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$: $N \times 1$ linear predictor
- $\boldsymbol{\mu} = \mathbb{E}(\mathbf{y}|\mathbf{X},\mathbf{g},\mathbf{Z}) = \text{logit}^{-1}(\boldsymbol{\eta})$

We seek the log-likelihood function for parameters $(\boldsymbol{\alpha}, \beta, \tau)$ given the observed data $(\mathbf{y}, \mathbf{X}, \mathbf{g}, \mathbf{Z})$.

---

## 1. Conditional likelihood

The conditional likelihood of $\mathbf{y}$ given $\mathbf{b}$, $\boldsymbol{\alpha}$, $\beta$ is:

$$
p(\mathbf{y} | \boldsymbol{\alpha}, \beta, \mathbf{b}) = \prod_{i=1}^{N} \mu_i^{y_i}(1-\mu_i)^{1-y_i}
$$

where $\mu_i = \text{logit}^{-1}(\eta_i) = \frac{e^{\eta_i}}{1 + e^{\eta_i}}$ and $\eta_i = \mathbf{X}_{i\cdot}\boldsymbol{\alpha} + g_i \beta + b_i$.

---

## 2. Marginal likelihood

The (marginal) likelihood of $\mathbf{y}$ given $(\boldsymbol{\alpha}, \beta, \tau)$ is obtained by integrating out the random effects $\mathbf{b}$:

$$
L(\boldsymbol{\alpha}, \beta, \tau) = p(\mathbf{y} | \boldsymbol{\alpha}, \beta, \tau) = \int p(\mathbf{y} | \boldsymbol{\alpha}, \beta, \mathbf{b})\, p(\mathbf{b} | \tau)\, d\mathbf{b}
$$

Where

$$
p(\mathbf{b} | \tau) = \frac{1}{(2\pi)^{N/2} |\tau \mathbf{\Psi}|^{1/2}} \exp\left(-\frac{1}{2} \mathbf{b}^\top (\tau \mathbf{\Psi})^{-1} \mathbf{b}\right)
$$

---

## 3. Log-likelihood function

Thus, the log-likelihood function is

$$
\ell(\boldsymbol{\alpha}, \beta, \tau) = \log L(\boldsymbol{\alpha}, \beta, \tau) = \log
\left[
    \int
    \left\{
        \prod_{i=1}^{N} \left[\frac{e^{\eta_i}}{1 + e^{\eta_i}}\right]^{y_i}
        \left[\frac{1}{1 + e^{\eta_i}}\right]^{1 - y_i}
    \right\}
    \frac{1}{(2\pi)^{N/2} |\tau \mathbf{\Psi}|^{1/2}}
    \exp\left(-\frac{1}{2} \mathbf{b}^\top (\tau \mathbf{\Psi})^{-1} \mathbf{b}\right)
    d\mathbf{b}
\right]
$$

where $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$ and $\mathbf{b} \in \mathbb{R}^{N}$.

---

Alternatively, using vector notation:

$$
\ell(\boldsymbol{\alpha}, \beta, \tau) = \log
\left[
    \int
    \prod_{i=1}^{N} \left(\frac{e^{\eta_i y_i}}{1 + e^{\eta_i}}\right)
    \frac{1}{(2\pi)^{N/2} |\tau \mathbf{\Psi}|^{1/2}}
    \exp\left(-\frac{1}{2} \mathbf{b}^\top (\tau \mathbf{\Psi})^{-1} \mathbf{b}\right)
    d\mathbf{b}
\right]
$$

with $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$.

This is the exact marginal log-likelihood for the logistic mixed model under the given notation.