# Score

## Step-by-Step Derivation of the Conditional Log-Likelihood

**Step 1: Write the conditional probability for each $y_i$:**
$$
\mathbb{P}(y_i=1| \mathbf{x}_i, g_i, b_i) = \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}
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


## Step-by-Step Derivation of the Score Function for $\beta$

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

