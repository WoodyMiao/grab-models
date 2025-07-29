# Step-by-Step Derivation of the Likelihood Function for the Logistic Mixed Model

Consider a case-control study with $N$ individuals. For the $i$th individual:
- $y_i \in \{0, 1\}$ is the binary phenotype.
- $\mathbf{x}_i$ is a $1 \times (1 + p)$ vector of covariates (including intercept).
- $g_i \in \{0, 1, 2\}$ is the genotype.
- $b_i$ is the random effect.

The model is:
$$
\text{logit}(\mu_i) = \mathbf{x}_i \boldsymbol{\alpha} + g_i \beta + b_i
$$
where $\mu_i = \mathbb{P}(y_i = 1 | \mathbf{x}_i, g_i, b_i)$.

## 1. Likelihood for a Single Observation

The conditional probability for $y_i$ is:
$$
\mathbb{P}(y_i | \mathbf{x}_i, g_i, b_i) = \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}
$$

## 2. Joint Likelihood for All Observations

Given the random effects $\mathbf{b}$, the observations $y_i$ are conditionally independent. However, the random effects $\mathbf{b}$ themselves are correlated, i.e., $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$. Therefore, the marginal likelihood does not assume independence across individuals.
$$
L(\boldsymbol{\alpha}, \beta, \mathbf{b} | \mathbf{X}, \mathbf{g}, \mathbf{y}) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}
$$

## 3. Incorporating the Random Effects

Assume $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$.
The marginal likelihood is:
$$
L(\boldsymbol{\alpha}, \beta, \tau | \mathbf{X}, \mathbf{g}, \mathbf{y}) = \int \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i} \cdot f_{\mathbf{b}}(\mathbf{b}) \, d\mathbf{b}
$$
where $f_{\mathbf{b}}(\mathbf{b})$ is the multivariate normal density:
$$
f_{\mathbf{b}}(\mathbf{b}) = \frac{1}{(2\pi \tau)^{N/2} |\mathbf{\Psi}|^{1/2}} \exp\left( -\frac{1}{2\tau} \mathbf{b}^\top \mathbf{\Psi}^{-1} \mathbf{b} \right )
$$

## 4. Log-Likelihood


## 4. Marginal Log-Likelihood

The marginal log-likelihood, integrated over the random effects $\mathbf{b}$, is:
$$
\ell(\boldsymbol{\alpha}, \beta, \tau | \mathbf{X}, \mathbf{g}, \mathbf{y}) = \log \int \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i} \cdot f_{\mathbf{b}}(\mathbf{b}) \, d\mathbf{b}
$$
where $f_{\mathbf{b}}(\mathbf{b})$ is the multivariate normal density for $\mathbf{b}$. This integral is typically approximated using numerical methods such as the Laplace approximation.

## 5. Summary

- The likelihood combines the Bernoulli likelihood for the data and the normal prior for the random effects.
- The marginal likelihood integrates over the random effects, which is typically approximated using numerical methods (e.g., Laplace approximation).

## 6. Step-by-Step Derivation of the Score Function for $\beta$

The score function for $\beta$ is the derivative of the (conditional) log-likelihood with respect to $\beta$:
$$
U_\beta = \frac{\partial \ell(\boldsymbol{\alpha}, \beta, \mathbf{b} | \mathbf{X}, \mathbf{g}, \mathbf{y})}{\partial \beta}
$$

Recall the conditional log-likelihood:
$$
\ell(\boldsymbol{\alpha}, \beta, \mathbf{b} | \mathbf{X}, \mathbf{g}, \mathbf{y}) = \sum_{i=1}^N \left[ y_i \log \mu_i + (1 - y_i) \log(1 - \mu_i) \right]
$$
where $\mu_i = \text{logit}^{-1}(\mathbf{x}_i \boldsymbol{\alpha} + g_i \beta + b_i)$.

**Step 1: Compute the derivative of $\mu_i$ with respect to $\beta$:**
$$
\frac{\partial \mu_i}{\partial \beta} = \mu_i (1 - \mu_i) g_i
$$

**Step 2: Compute the derivative of the log-likelihood:**
$$
\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^N \left[ \frac{y_i}{\mu_i} - \frac{1 - y_i}{1 - \mu_i} \right] \frac{\partial \mu_i}{\partial \beta}
$$

**Step 3: Substitute $\frac{\partial \mu_i}{\partial \beta}$:**
$$
\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^N \left[ \frac{y_i}{\mu_i} - \frac{1 - y_i}{1 - \mu_i} \right] \mu_i (1 - \mu_i) g_i
$$

**Step 4: Simplify the expression:**
$$
\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^N (y_i - \mu_i) g_i
$$

**Step 5: Write in vector notation:**
$$
U_\beta = \mathbf{g}^\top (\mathbf{y} - \boldsymbol{\mu})
$$

This is the score function for $\beta$ in the logistic mixed model, conditional on $\mathbf{b}$.

## 7. Step-by-Step Derivation of the Conditional Log-Likelihood

The conditional log-likelihood is the log-likelihood of the data given the random effects $\mathbf{b}$ and parameters $\boldsymbol{\alpha}, \beta$.

**Step 1: Write the conditional probability for each $y_i$:**
$$
\mathbb{P}(y_i | \mathbf{x}_i, g_i, b_i) = \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}
$$
where $\mu_i = \text{logit}^{-1}(\mathbf{x}_i \boldsymbol{\alpha} + g_i \beta + b_i)$.

**Step 2: Write the joint conditional likelihood for all $N$ individuals:**
$$
L(\boldsymbol{\alpha}, \beta, \mathbf{b} | \mathbf{X}, \mathbf{g}, \mathbf{y}) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}
$$

**Step 3: Take the logarithm to obtain the conditional log-likelihood:**
$$
\ell(\boldsymbol{\alpha}, \beta, \mathbf{b} | \mathbf{X}, \mathbf{g}, \mathbf{y}) = \log L(\boldsymbol{\alpha}, \beta, \mathbf{b} | \mathbf{X}, \mathbf{g}, \mathbf{y}) = \sum_{i=1}^N \left[ y_i \log \mu_i + (1 - y_i) \log(1 - \mu_i) \right]
$$

This is the conditional log-likelihood, which is used for deriving the score function and other inferential quantities, conditional on the random effects $\mathbf{b}$.
