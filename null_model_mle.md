# Appendix

## I. Derivation of log-likelihood function

Under the null hypothesis $H_0: \beta=0$, the logistic mixed model becomes

$$
\text{logit}(\mu_i) = \mathbf{x}_i \boldsymbol{\alpha} + b_i
$$

#### Step 1: Bernoulli likelihood for each observation

Given $y_i \sim \text{Bernoulli}(\mu_i)$, the likelihood for individual $i$ is:
$$
L_i = \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}
$$

#### Step 2: Log-likelihood for individual observations

Taking the logarithm:
$$
\log L_i = y_i \log(\mu_i) + (1 - y_i) \log(1 - \mu_i)
$$

#### Step 3: Joint likelihood for all observations

Assuming independence conditional on random effects:
$$
L_{\text{obs}} = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}
$$

#### Step 4: Log-likelihood for observations

$$
\log L_{\text{obs}} = \sum_{i=1}^N \left[ y_i \log(\mu_i) + (1 - y_i) \log(1 - \mu_i) \right]
$$

#### Step 5: Random effects distribution

The random effects follow $\mathbf{b} \sim \mathcal{N}(0, \tau \mathbf{\Psi})$, so:
$$
f(\mathbf{b}) = \frac{1}{(2\pi \tau)^{N/2} |\mathbf{\Psi}|^{1/2}} \exp\left(-\frac{1}{2\tau} \mathbf{b}^\top \mathbf{\Psi}^{-1} \mathbf{b}\right)
$$

#### Step 6: Log-likelihood for random effects

$$
\log f(\mathbf{b}) = -\frac{N}{2} \log(2\pi \tau) - \frac{1}{2} \log|\mathbf{\Psi}| - \frac{1}{2\tau} \mathbf{b}^\top \mathbf{\Psi}^{-1} \mathbf{b}
$$

#### Step 7: Complete log-likelihood

Combining both components:
$$
\ell(\boldsymbol{\alpha}, \tau, \mathbf{b}) = \sum_{i=1}^N \left[ y_i \log(\mu_i) + (1 - y_i) \log(1 - \mu_i) \right] - \frac{1}{2\tau} \mathbf{b}^\top \mathbf{\Psi}^{-1} \mathbf{b} - \frac{N}{2} \log(2\pi \tau) - \frac{1}{2} \log|\mathbf{\Psi}|\,
$$

Let $\boldsymbol{\mu} = [\mu_1, \ldots, \mu_N]^\top$ and $\mathbf{y} = [y_1, \ldots, y_N]^\top$. The log-likelihood function can be written as:

$$
\ell(\boldsymbol{\alpha}, \tau, \mathbf{b}) = \mathbf{y}^\top \log(\boldsymbol{\mu}) + (\mathbf{1} - \mathbf{y})^\top \log(\mathbf{1} - \boldsymbol{\mu}) - \frac{1}{2\tau} \mathbf{b}^\top \mathbf{\Psi}^{-1} \mathbf{b} - \frac{N}{2} \log(2\pi \tau) - \frac{1}{2} \log|\mathbf{\Psi}|,
$$

where $\log(\boldsymbol{\mu})$ and $\log(\mathbf{1} - \boldsymbol{\mu})$ are element-wise logarithms.
