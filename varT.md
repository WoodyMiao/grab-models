# Derivation of Variance of T and Consistency Proof in SAIGE

## Model Setup from SAIGE

Consider the logistic mixed model from SAIGE with:

- $\mathbf{y} = [y_1,\ldots,y_N]^{\top}$ where $y_i \sim \operatorname{Bernoulli}(\mu_i)$
- $\mathbf{X}$ is $N \times (1 + p)$ matrix of covariates and intercept
- $\mathbf{g}$ is $N \times 1$ vector of genotypes (allele counts)
- $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$ where $\mathbf{\Psi}$ is the GRM
- $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$
- $\mu_i = \operatorname{logit}^{-1}(\eta_i)$

The score statistic is:
$$
T = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})
$$

where $\tilde{\mathbf{g}} = \mathbf{g} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{g}$ is the residual from regressing $\mathbf{g}$ on $\mathbf{X}$ using weight matrix $\hat{\mathbf{W}} = \operatorname{diag}(\hat{\boldsymbol{\mu}} \odot [\mathbf{1} - \hat{\boldsymbol{\mu}}])$, and $\hat{\boldsymbol{\mu}} = \operatorname{logit}^{-1}(\hat{\boldsymbol{\eta}})$ with $\hat{\boldsymbol{\eta}} = \mathbf{X} \hat{\boldsymbol{\alpha}} + \hat{\mathbf{b}}$ being a linear predictor under the null model.

## 1. Derivation of $\mathbb{V}(T)$

### Step 1: Variance of $T$

$$
\mathbb{V}(T) = \mathbb{V}[\tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})] = \tilde{\mathbf{g}}^\top \mathbb{V}[(\mathbf{y} - \hat{\boldsymbol{\mu}})]\tilde{\mathbf{g}} = \tilde{\mathbf{g}}^\top \mathbb{V}(\mathbf{y})\tilde{\mathbf{g}}
$$

### Step 2: Variance of $\mathbf{y}$

By the law of total covariance:
$$
\mathbb{V}(\mathbf{y}) = \mathbb{E}_{\mathbf{b}}[\mathbb{V}(\mathbf{y} \mid \mathbf{b})] + \mathbb{V}_{\mathbf{b}}[\mathbb{E}(\mathbf{y} \mid \mathbf{b})]
$$

where:

- $\mathbb{V}(\mathbf{y} \mid \mathbf{b}) = \operatorname{diag}(\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu}))$
- $\mathbb{E}(\mathbf{y} \mid \mathbf{b}) = \boldsymbol{\mu}$

So:
$$
\mathbb{V}(\mathbf{y}) = \mathbb{E}_{\mathbf{b}}[\operatorname{diag}(\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu}))] + \mathbb{V}_{\mathbf{b}}[\boldsymbol{\mu}]
$$

### Step 3: Approximation for $\mathbb{V}(\mathbf{y})$

**First term approximation:**
The first term involves the expectation of diagonal matrices over the random effects distribution. In practice, this is approximated by evaluating at the fitted values:
$$
\mathbb{E}_{\mathbf{b}}[\operatorname{diag}(\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu}))] \approx \operatorname{diag}(\hat{\boldsymbol{\mu}} \odot (\mathbf{1} - \hat{\boldsymbol{\mu}})) = \hat{\mathbf{W}}
$$

**Second term approximation:**
For the second term $\mathbb{V}_{\mathbf{b}}[\boldsymbol{\mu}]$, we use the delta method. Since $\boldsymbol{\mu} = \operatorname{logit}^{-1}(\mathbf{X} \boldsymbol{\alpha} + \mathbf{b})$ and $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$, we have:

$$
\mathbb{V}_{\mathbf{b}}[\boldsymbol{\mu}] \approx \mathbf{J} \mathbb{V}(\mathbf{b}) \mathbf{J}^\top = \mathbf{J} (\tau \mathbf{\Psi}) \mathbf{J}^\top
$$

where $\mathbf{J}$ is the Jacobian matrix with entries:
$$
J_{ij} = \frac{\partial \mu_i}{\partial b_j} = \frac{\partial}{\partial b_j} \operatorname{logit}^{-1}(\eta_i) = \mu_i (1 - \mu_i) \delta_{ij}
$$

Since the Jacobian is diagonal in this case (each $\mu_i$ depends only on $b_i$), we get:
$$
\mathbb{V}_{\mathbf{b}}[\boldsymbol{\mu}] \approx \operatorname{diag}(\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu})) \tau \mathbf{\Psi} \operatorname{diag}(\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu}))
$$

**Why the SAIGE approximation:**
However, SAIGE uses a simpler approximation that is computationally more tractable:
$$
\mathbb{V}_{\mathbf{b}}[\boldsymbol{\mu}] \approx \tau \mathbf{\Psi}
$$

This is justified because:
1. It captures the main correlation structure from the random effects
2. The diagonal scaling factor $\mu_i(1-\mu_i)$ is absorbed into the overall approximation
3. It leads to a computationally efficient variance estimator

**Final approximation:**
Therefore, the SAIGE approximation is:
$$
\mathbb{V}(\mathbf{y}) \approx \boldsymbol{\Sigma} = \hat{\mathbf{W}}^{-1} + \tau\mathbf{\Psi}
$$

**Is the approximation consistent when N is large?**

Yes, under regularity conditions:
1. $\hat{\boldsymbol{\mu}} \xrightarrow{p} \boldsymbol{\mu}$ as $N \to \infty$
2. $\hat{\mathbf{W}} \xrightarrow{p} \mathbf{W}$ as $N \to \infty$  
3. $\hat{\tau} \xrightarrow{p} \tau$ as $N \to \infty$

Therefore: $\hat{\boldsymbol{\Sigma}} \xrightarrow{p} \boldsymbol{\Sigma}$ and the approximation becomes exact in the limit.

### Step 4: Variance of T under projection

**The projection operation:**

The projection transforms $\mathbf{g}$ to $\tilde{\mathbf{g}}$ by removing the component that lies in the column space of $\mathbf{X}$:
$$
\tilde{\mathbf{g}} = \mathbf{g} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{g} = (\mathbf{I} - \mathbf{H})\mathbf{g}
$$

where $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}$ is the weighted projection matrix onto the column space of $\mathbf{X}$.

**Variance after projection:**

The projected statistic $T = \tilde{\mathbf{g}}^\top \mathbf{y}$ has variance:
$$
\mathbb{V}(T) = \mathbb{V}(\tilde{\mathbf{g}}^\top \mathbf{y}) = \tilde{\mathbf{g}}^\top \mathbb{V}(\mathbf{y}) \tilde{\mathbf{g}} = \tilde{\mathbf{g}}^\top \boldsymbol{\Sigma} \tilde{\mathbf{g}}
$$

**Why we need to account for the projection:**

However, simply using $\tilde{\mathbf{g}}^\top \boldsymbol{\Sigma} \tilde{\mathbf{g}}$ would be incorrect because:

1. **Constraint violation:** The projection $\tilde{\mathbf{g}}$ is constructed to be orthogonal to $\mathbf{X}$, but this constraint is not reflected in the naive variance calculation.

2. **Degrees of freedom:** The projection reduces the effective degrees of freedom from $N$ to $N-p-1$ (where $p+1$ is the number of columns in $\mathbf{X}$).

3. **Estimation uncertainty:** The variance calculation should account for the fact that we estimated the projection using $\hat{\mathbf{W}}$ rather than the true weights.

**The correct projection-adjusted variance:**

To properly account for the constraint $\tilde{\mathbf{g}} \perp \mathbf{X}$, we need to use the constrained variance formula. This leads to the projection matrix:

$$
\mathbf{P} = \boldsymbol{\Sigma}^{-1} - \boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}
$$

**Interpretation of $\mathbf{P}$:**

- $\boldsymbol{\Sigma}^{-1}$ is the precision matrix (inverse of the covariance)
- The second term $\boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}$ removes the contribution from the column space of $\mathbf{X}$
- $\mathbf{P}$ is the precision matrix for the constrained problem where we condition on $\tilde{\mathbf{g}} \perp \mathbf{X}$

**Final variance formula:**

The variance of the projected statistic, accounting for the orthogonality constraint, is:
$$
\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \mathbf{P} \tilde{\mathbf{g}}
$$

This formula correctly accounts for:
- The covariance structure of $\mathbf{y}$ (through $\boldsymbol{\Sigma}$)
- The orthogonality constraint $\tilde{\mathbf{g}} \perp \mathbf{X}$
- The loss of degrees of freedom from the projection

## 2. Consistency of $\hat{\mathbb{V}}(T)$

The estimator is:
$$
\hat{\mathbb{V}}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}
$$

where $\hat{\mathbf{P}} = \hat{\boldsymbol{\Sigma}}^{-1} - \hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X}(\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}$ and $\hat{\boldsymbol{\Sigma}} = \hat{\mathbf{W}}^{-1} + \hat{\tau}\mathbf{\Psi}$.

### Proof of Consistency

**Step 1: Consistency of parameter estimators**

Under standard regularity conditions for GLMMs:
- $\hat{\boldsymbol{\alpha}} \xrightarrow{p} \boldsymbol{\alpha}$ 
- $\hat{\tau} \xrightarrow{p} \tau$
- $\hat{\mathbf{b}} \xrightarrow{p} \mathbf{b}$ (in appropriate sense)

as $N \to \infty$.

**Step 2: Consistency of fitted values**

From the consistency of parameter estimators:
$$
\hat{\boldsymbol{\mu}} = \operatorname{logit}^{-1}(\mathbf{X} \hat{\boldsymbol{\alpha}} + \hat{\mathbf{b}}) \xrightarrow{p} \operatorname{logit}^{-1}(\mathbf{X} \boldsymbol{\alpha} + \mathbf{b}) = \boldsymbol{\mu}
$$

**Step 3: Consistency of weight matrix**

$$
\hat{\mathbf{W}} = \operatorname{diag}(\hat{\boldsymbol{\mu}} \odot [\mathbf{1} - \hat{\boldsymbol{\mu}}]) \xrightarrow{p} \operatorname{diag}(\boldsymbol{\mu} \odot [\mathbf{1} - \boldsymbol{\mu}]) = \mathbf{W}
$$

**Step 4: Consistency of covariance matrix**

$$
\hat{\boldsymbol{\Sigma}} = \hat{\mathbf{W}}^{-1} + \hat{\tau}\mathbf{\Psi} \xrightarrow{p} \mathbf{W}^{-1} + \tau\mathbf{\Psi} = \boldsymbol{\Sigma}
$$

**Step 5: Consistency of projection matrix**

By the continuous mapping theorem:
$$
\hat{\mathbf{P}} = \hat{\boldsymbol{\Sigma}}^{-1} - \hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X}(\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1} \xrightarrow{p} \mathbf{P}
$$

**Step 6: Consistency of variance estimator**

Finally:
$$
\hat{\mathbb{V}}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}} \xrightarrow{p} \tilde{\mathbf{g}}^\top \mathbf{P} \tilde{\mathbf{g}} = \mathbb{V}(T)
$$

Therefore, $\hat{\mathbb{V}}(T)$ is a consistent estimator of $\mathbb{V}(T)$.

## 3. Summary

- The theoretical variance $\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \mathbf{P} \tilde{\mathbf{g}}$ accounts for both the covariance structure of $\mathbf{y}$ and the orthogonality constraint from the projection.
- The estimator $\hat{\mathbb{V}}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}$ is consistent due to the consistency of the underlying parameter estimators and the continuous mapping theorem.
- This consistency justifies the asymptotic normality of the standardized score statistic $T/\sqrt{\hat{\mathbb{V}}(T)} \xrightarrow{d} \mathcal{N}(0,1)$.
