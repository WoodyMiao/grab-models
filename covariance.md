# Covariance

## Covariance between $\mathbf{y}$ and $\mathbf{b}$

Given the model setup:

- $\mathbf{y} = [y_1, \ldots, y_N]^\top$, $y_i \sim \mathrm{Bernoulli}(\mu_i)$
- $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$
- $\boldsymbol{\eta} = \mathbf{X}\boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$
- $\mu_i = \operatorname{logit}^{-1}(\eta_i)$

We want to compute $\operatorname{cov}(\mathbf{y}, \mathbf{b})$.

### Step 1: Law of Total Covariance

By the law of total covariance:
$$
\operatorname{cov}(\mathbf{y}, \mathbf{b}) = \mathbb{E}[\operatorname{cov}(\mathbf{y}, \mathbf{b} | \mathbf{b})] + \operatorname{cov}(\mathbb{E}[\mathbf{y} | \mathbf{b}], \mathbf{b})
$$

But $\mathbf{y}$ given $\mathbf{b}$ is independent of $\mathbf{b}$ (since $\mathbf{b}$ is fixed), so $\operatorname{cov}(\mathbf{y}, \mathbf{b} | \mathbf{b}) = 0$.

Thus:
$$
\operatorname{cov}(\mathbf{y}, \mathbf{b}) = \operatorname{cov}(\mathbb{E}[\mathbf{y} | \mathbf{b}], \mathbf{b})
$$

### Step 2: Compute $\mathbb{E}[\mathbf{y} | \mathbf{b}]$

Given $\mathbf{b}$, $y_i$ are independent Bernoulli with mean $\mu_i = \operatorname{logit}^{-1}(\eta_i)$, where $\eta_i = \mathbf{x}_i^\top \boldsymbol{\alpha} + g_i \beta + b_i$.

So:
$$
\mathbb{E}[\mathbf{y} | \mathbf{b}] = \boldsymbol{\mu} = [\mu_1, \ldots, \mu_N]^\top
$$

### Step 3: Covariance between $\boldsymbol{\mu}$ and $\mathbf{b}$

We have:
$$
\operatorname{cov}(\mathbf{y}, \mathbf{b}) = \operatorname{cov}(\boldsymbol{\mu}, \mathbf{b})
$$

Let us compute the $(i,j)$-th entry:
$$
\operatorname{cov}(\mu_i, b_j) = \mathbb{E}[\mu_i b_j] - \mathbb{E}[\mu_i] \mathbb{E}[b_j]
$$
But $\mathbb{E}[b_j] = 0$, so:
$$
\operatorname{cov}(\mu_i, b_j) = \mathbb{E}[\mu_i b_j]
$$

### Step 4: Compute $\mathbb{E}[\mu_i b_j]$

Recall $\mu_i = \operatorname{logit}^{-1}(\eta_i) = \frac{1}{1 + \exp(-\eta_i)}$, $\eta_i$ is linear in $b_i$.

Since $\mathbf{b}$ is multivariate normal, we can use Stein's lemma:
$$
\mathbb{E}[f(b_i) b_j] = \operatorname{cov}(b_i, b_j) \mathbb{E}[f'(b_i)]
$$
where $f(b_i) = \operatorname{logit}^{-1}(\eta_i)$, and $\operatorname{cov}(b_i, b_j) = \tau \Psi_{ij}$.

Compute $f'(b_i)$:
$$
f'(b_i) = \frac{d}{db_i} \left( \frac{1}{1 + \exp(-\eta_i)} \right ) = \mu_i (1 - \mu_i)
$$

So:
$$
\mathbb{E}[\mu_i b_j] = \tau \Psi_{ij} \mathbb{E}[\mu_i (1 - \mu_i)]
$$

### Step 5: Final Covariance Matrix

Therefore,
$$
\operatorname{cov}(y_i, b_j) = \tau \Psi_{ij} \mathbb{E}[\mu_i (1 - \mu_i)]
$$

Or in matrix form:
$$
\operatorname{cov}(\mathbf{y}, \mathbf{b}) = \tau \, \mathbb{E}[\operatorname{diag}(\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu}))] \mathbf{\Psi}
$$
where $\odot$ denotes elementwise multiplication.

### Summary

- $\operatorname{cov}(\mathbf{y}, \mathbf{b})$ is not simply $\tau \mathbf{\Psi}$, but is scaled by the expected variance of the Bernoulli means $\mu_i (1 - \mu_i)$.
- The $(i,j)$-th entry is $\tau \Psi_{ij} \mathbb{E}[\mu_i (1 - \mu_i)]$.

---

## Covariance Matrix of $\mathbf{y}$

We now derive $\operatorname{cov}(\mathbf{y})$ for the logistic mixed model.

Recall:

- $\mathbf{y} | \mathbf{b} \sim \mathrm{Bernoulli}(\boldsymbol{\mu})$ independently, with $\mu_i = \operatorname{logit}^{-1}(\eta_i)$
- $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$

By the law of total covariance:
$$
\operatorname{cov}(\mathbf{y}) = \mathbb{E}[\operatorname{cov}(\mathbf{y} | \mathbf{b})] + \operatorname{cov}(\mathbb{E}[\mathbf{y} | \mathbf{b}])
$$

### Step 1: Compute $\operatorname{cov}(\mathbf{y} | \mathbf{b})$

Given $\mathbf{b}$, $y_i$ are independent Bernoulli($\mu_i$), so:
$$
\operatorname{cov}(\mathbf{y} | \mathbf{b}) = \operatorname{diag}(\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu}))
$$

### Step 2: Compute $\mathbb{E}[\operatorname{cov}(\mathbf{y} | \mathbf{b})]$

Take expectation over $\mathbf{b}$:
$$
\mathbb{E}[\operatorname{cov}(\mathbf{y} | \mathbf{b})] = \mathbb{E}[\operatorname{diag}(\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu}))]
$$
This is a diagonal matrix with $i$-th entry $\mathbb{E}[\mu_i (1 - \mu_i)]$.

### Step 3: Compute $\operatorname{cov}(\mathbb{E}[\mathbf{y} | \mathbf{b}])$

$\mathbb{E}[\mathbf{y} | \mathbf{b}] = \boldsymbol{\mu}$, so:
$$
\operatorname{cov}(\mathbb{E}[\mathbf{y} | \mathbf{b}]) = \operatorname{cov}(\boldsymbol{\mu})
$$
The $(i,j)$-th entry is:
$$
\operatorname{cov}(\mu_i, \mu_j) = \mathbb{E}[\mu_i \mu_j] - \mathbb{E}[\mu_i] \mathbb{E}[\mu_j]
$$

### Step 4: Final Covariance Matrix

Putting together:
$$
\operatorname{cov}(\mathbf{y}) = \operatorname{diag}\left(\mathbb{E}[\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu})]\right) + \operatorname{cov}(\boldsymbol{\mu})
$$
or, entrywise:
$$
\operatorname{cov}(y_i, y_j) =
\begin{cases}
  \mathbb{E}[\mu_i (1 - \mu_i)] + \mathbb{V}(\mu_i) & i = j \\
  \operatorname{cov}(\mu_i, \mu_j) & i \neq j
\end{cases}
$$

where $\operatorname{cov}(\mu_i, \mu_j) = \mathbb{E}[\mu_i \mu_j] - \mathbb{E}[\mu_i] \mathbb{E}[\mu_j]$.

---

## Exact Expression for $\operatorname{cov}(\mathbf{y})$ (Integral Form)

The exact covariance matrix $\operatorname{cov}(\mathbf{y})$ for the logistic mixed model can be written in terms of integrals over the random effects $\mathbf{b}$:

Recall:
$$
\operatorname{cov}(\mathbf{y}) = \mathbb{E}[\operatorname{diag}(\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu}))] + \operatorname{cov}(\boldsymbol{\mu})
$$
where $\boldsymbol{\mu} = [\mu_1, \ldots, \mu_N]^\top$ with $\mu_i = \operatorname{logit}^{-1}(\eta_i)$ and $\eta_i = \mathbf{x}_i^\top \boldsymbol{\alpha} + g_i \beta + b_i$.

The $i$-th diagonal entry is:
$$
\mathbb{V}(y_i) = \mathbb{E}[\mu_i (1 - \mu_i)] + \mathbb{V}(\mu_i)
$$
The off-diagonal entries are:
$$
\operatorname{cov}(y_i, y_j) = \operatorname{cov}(\mu_i, \mu_j) = \mathbb{E}[\mu_i \mu_j] - \mathbb{E}[\mu_i] \mathbb{E}[\mu_j]
$$

Each expectation is an integral over the multivariate normal distribution of $\mathbf{b}$:
$$
\mathbb{E}[f(\mathbf{b})] = \int_{\mathbb{R}^N} f(\mathbf{b}) \, \phi(\mathbf{b}; \mathbf{0}, \tau \mathbf{\Psi}) \, d\mathbf{b}
$$
where $\phi(\mathbf{b}; \mathbf{0}, \tau \mathbf{\Psi})$ is the multivariate normal density.

Thus,
$$
\mathbb{E}[\mu_i] = \int_{\mathbb{R}^N} \operatorname{logit}^{-1}(\eta_i) \, \phi(\mathbf{b}; \mathbf{0}, \tau \mathbf{\Psi}) \, d\mathbf{b}
$$
$$
\mathbb{E}[\mu_i \mu_j] = \int_{\mathbb{R}^N} \operatorname{logit}^{-1}(\eta_i) \, \operatorname{logit}^{-1}(\eta_j) \, \phi(\mathbf{b}; \mathbf{0}, \tau \mathbf{\Psi}) \, d\mathbf{b}
$$
$$
\mathbb{E}[\mu_i (1 - \mu_i)] = \int_{\mathbb{R}^N} \operatorname{logit}^{-1}(\eta_i) [1 - \operatorname{logit}^{-1}(\eta_i)] \, \phi(\mathbf{b}; \mathbf{0}, \tau \mathbf{\Psi}) \, d\mathbf{b}
$$

In summary, the covariance matrix can be written for all $i, j$ as:
$$
\operatorname{cov}(y_i, y_j) = \int_{\mathbb{R}^N} \operatorname{logit}^{-1}(\eta_i) \operatorname{logit}^{-1}(\eta_j) \, \phi(\mathbf{b}; \mathbf{0}, \tau \mathbf{\Psi}) \, d\mathbf{b} \\
\qquad - \left[\int_{\mathbb{R}^N} \operatorname{logit}^{-1}(\eta_i) \, \phi(\mathbf{b}; \mathbf{0}, \tau \mathbf{\Psi}) \, d\mathbf{b}\right] \left[\int_{\mathbb{R}^N} \operatorname{logit}^{-1}(\eta_j) \, \phi(\mathbf{b}; \mathbf{0}, \tau \mathbf{\Psi}) \, d\mathbf{b}\right]
$$
where $\mu_i = \operatorname{logit}^{-1}(\eta_i)$ and $\eta_i = \mathbf{x}_i^\top \boldsymbol{\alpha} + g_i \beta + b_i$.

This is the exact (but implicit) form for the covariance matrix of $\mathbf{y}$ in the logistic mixed model, with all expectations written as integrals over the multivariate normal distribution of $\mathbf{b}$.

---

## When is $\operatorname{cov}(y_i, y_j) = 0$?

We have:
$$
\operatorname{cov}(y_i, y_j) = \mathbb{E}[\mu_i \mu_j] - \mathbb{E}[\mu_i] \mathbb{E}[\mu_j]
$$
where $\mu_i = \operatorname{logit}^{-1}(\eta_i)$ and $\eta_i = \mathbf{x}_i^\top \boldsymbol{\alpha} + g_i \beta + b_i$.

For $i \neq j$:
$$
\operatorname{cov}(y_i, y_j) = \mathbb{E}[\mu_i \mu_j] - \mathbb{E}[\mu_i] \mathbb{E}[\mu_j]
$$
This is zero if and only if $\mu_i$ and $\mu_j$ are independent random variables under the distribution of $\mathbf{b}$, i.e.,
$$
\mathbb{E}[\mu_i \mu_j] = \mathbb{E}[\mu_i] \mathbb{E}[\mu_j]
$$

This occurs if and only if $b_i$ and $b_j$ are independent (i.e., $\Psi_{ij} = 0$), $\eta_i$ and $\eta_j$ do not share any random or fixed effects, and then $\mu_i$ and $\mu_j$ are independent.
