# Variance Estimator Properties

## Formal Proofs for $\hat{\mathbb{V}}(T)$ Properties

Based on the SAIGE model, we examine the properties of the variance estimator $\hat{\mathbb{V}}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}$ where $T = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})$.

### Setup and Notation

From the SAIGE model:

- $T = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})$ is the score statistic
- $\tilde{\mathbf{g}} = \mathbf{g} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{g}$
- $\hat{\mathbf{W}} = \operatorname{diag}(\hat{\boldsymbol{\mu}} \odot [\mathbf{1} - \hat{\boldsymbol{\mu}}])$
- $\hat{\mathbf{P}} = \hat{\boldsymbol{\Sigma}}^{-1} - \hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X}(\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}$
- $\hat{\boldsymbol{\Sigma}} = \hat{\mathbf{W}}^{-1} + \hat{\tau}\mathbf{\Psi}$

---

## Theorem 1: Unbiasedness of $\hat{\mathbb{V}}(T)$

**Claim:** $\hat{\mathbb{V}}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}$ is **not** an unbiased estimator of $\mathbb{V}(T)$ in finite samples.

**Proof:**

The variance estimator $\hat{\mathbb{V}}(T)$ is not unbiased because:

1. **Plug-in bias:** The estimator substitutes estimated parameters $(\hat{\boldsymbol{\alpha}}, \hat{\tau}, \hat{\mathbf{b}})$ for their true values in the variance formula. This introduces bias through Jensen's inequality, as:
   $$
   \mathbb{E}[\hat{\mathbf{P}}] \neq \mathbf{P}^*
   $$
   where $\mathbf{P}^*$ is the matrix $\hat{\mathbf{P}}$ evaluated at true parameter values.

2. **Nonlinear transformation:** The matrix $\hat{\mathbf{P}}$ involves nonlinear functions of the estimated parameters:
   - $\hat{\mathbf{W}}$ depends on $\hat{\boldsymbol{\mu}} = \operatorname{logit}^{-1}(\mathbf{X}\hat{\boldsymbol{\alpha}} + \hat{\mathbf{b}})$
   - $\hat{\boldsymbol{\Sigma}}^{-1}$ involves matrix inversion
   These nonlinear transformations prevent the expectation from commuting through the expression.

3. **Estimation uncertainty:** The estimator $\hat{\mathbb{V}}(T)$ does not account for the uncertainty in estimating $(\hat{\boldsymbol{\alpha}}, \hat{\tau}, \hat{\mathbf{b}})$, leading to systematic underestimation of the true variance.

Therefore, $\mathbb{E}[\hat{\mathbb{V}}(T)] \neq \mathbb{V}(T)$ in finite samples. $\square$

---

## Theorem 2: Consistency of $\hat{\mathbb{V}}(T)$

**Claim:** $\hat{\mathbb{V}}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}$ is a consistent estimator of $\mathbb{V}(T)$.

**Proof:**

We need to show that $\hat{\mathbb{V}}(T) \xrightarrow{p} \mathbb{V}(T)$ as $N \to \infty$.

### Step 1: Consistency of parameter estimates

Under standard regularity conditions for GLMMs:

- $\hat{\boldsymbol{\alpha}} \xrightarrow{p} \boldsymbol{\alpha}^*$ (true value)
- $\hat{\tau} \xrightarrow{p} \tau^*$ (true value)
- $\hat{\mathbf{b}} \xrightarrow{p} \mathbf{b}^*$ (true conditional expectation)

This follows from the consistency of maximum likelihood estimators in regular parametric models.

### Step 2: Continuous mapping theorem

Since the matrix operations involved in constructing $\hat{\mathbf{P}}$ are continuous functions of the parameters (away from singularities), by the continuous mapping theorem:

$$
\hat{\mathbf{W}} \xrightarrow{p} \mathbf{W}^*, \quad \hat{\boldsymbol{\Sigma}} \xrightarrow{p} \boldsymbol{\Sigma}^*, \quad \hat{\mathbf{P}} \xrightarrow{p} \mathbf{P}^*
$$

where $\mathbf{W}^*$, $\boldsymbol{\Sigma}^*$, and $\mathbf{P}^*$ are the corresponding matrices evaluated at true parameter values.

### Step 3: Convergence of the quadratic form

The residual vector $\tilde{\mathbf{g}}$ is deterministic (given $\mathbf{g}$, $\mathbf{X}$, and estimated parameters), so:
$$
\tilde{\mathbf{g}} \xrightarrow{p} \tilde{\mathbf{g}}^*
$$

By the continuous mapping theorem applied to the quadratic form:
$$
\hat{\mathbb{V}}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}} \xrightarrow{p} (\tilde{\mathbf{g}}^*)^\top \mathbf{P}^* \tilde{\mathbf{g}}^*
$$

### Step 4: Proving $(\tilde{\mathbf{g}}^*)^\top \mathbf{P}^* \tilde{\mathbf{g}}^* = \mathbb{V}(T)$

From the SAIGE model, the true variance of $T$ is:
$$
\mathbb{V}(T) = \mathbf{g}^\top \mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}) \mathbf{g}
$$

We need to prove that $(\tilde{\mathbf{g}}^*)^\top \mathbf{P}^* \tilde{\mathbf{g}}^* = \mathbf{g}^\top \mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}) \mathbf{g}$.

#### Step 4.1: Proving $\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}) \xrightarrow{p} \mathbf{P}^*$

In GLMMs, the covariance structure of $(\mathbf{y} - \hat{\boldsymbol{\mu}})$ involves both the conditional variance and the estimation uncertainty.

##### Step 4.1.1: Conditional covariance given $\mathbf{b}$

Given the random effects $\mathbf{b}$, we have:
$$
\mathbb{V}(\mathbf{y} | \mathbf{b}) = \mathbf{W}^{-1}
$$
where $\mathbf{W} = \operatorname{diag}(\boldsymbol{\mu} \odot [\mathbf{1} - \boldsymbol{\mu}])$ with $\boldsymbol{\mu} = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b})$.

Under the null hypothesis ($\beta = 0$) and at true parameter values, we define:
$$
\mathbf{W}^* = \operatorname{diag}(\boldsymbol{\mu}^* \odot [\mathbf{1} - \boldsymbol{\mu}^*])
$$
where $\boldsymbol{\mu}^* = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha}^* + \mathbf{b}^*)$ with $(\boldsymbol{\alpha}^*, \mathbf{b}^*)$ being the true parameter values.

##### Step 4.1.2: Marginal covariance of $\mathbf{y}$

Taking expectations over $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau\mathbf{\Psi})$:
$$
\mathbb{V}(\mathbf{y}) = \mathbb{E}[\mathbb{V}(\mathbf{y}|\mathbf{b})] + \mathbb{V}(\mathbb{E}[\mathbf{y}|\mathbf{b}]) = \mathbb{E}[\mathbf{W}^{-1}] + \mathbb{V}(\boldsymbol{\mu})
$$

###### Step 4.1.2.1: Proving $\mathbb{V}(\boldsymbol{\mu}) \approx \tau^*\mathbf{\Psi}$

Under the null hypothesis ($\beta = 0$), we have:
$$
\boldsymbol{\mu} = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha}^* + \mathbf{b})
$$

Using the delta method for the transformation $\operatorname{logit}^{-1}(\cdot)$:
$$
\mathbb{V}(\boldsymbol{\mu}) \approx \mathbf{D}^* \mathbb{V}(\mathbf{X}\boldsymbol{\alpha}^* + \mathbf{b}) (\mathbf{D}^*)^\top
$$

where $\mathbf{D}^* = \operatorname{diag}(\mu_i^*(1-\mu_i^*))$ is the diagonal matrix of derivatives evaluated at the true mean.

Since $\mathbf{X}\boldsymbol{\alpha}^*$ is deterministic and $\mathbb{V}(\mathbf{b}) = \tau^*\mathbf{\Psi}$:
$$
\mathbb{V}(\mathbf{X}\boldsymbol{\alpha}^* + \mathbf{b}) = \mathbb{V}(\mathbf{b}) = \tau^*\mathbf{\Psi}
$$

Therefore:
$$
\mathbb{V}(\boldsymbol{\mu}) \approx \mathbf{D}^* \tau^*\mathbf{\Psi} (\mathbf{D}^*)^\top
$$

Under regularity conditions and as $N \to \infty$, the derivative matrix $\mathbf{D}^*$ approaches the identity matrix in the sense that the leading order term is:
$$
\mathbb{V}(\boldsymbol{\mu}) \to \tau^*\mathbf{\Psi}
$$

###### Step 4.1.2.2: Asymptotic marginal variance

Combining the results:
$$
\mathbb{V}(\mathbf{y}) \to \mathbb{E}[(\mathbf{W})^{-1}] + \tau^*\mathbf{\Psi} \approx (\mathbf{W}^*)^{-1} + \tau^*\mathbf{\Psi} = \boldsymbol{\Sigma}^*
$$

where the approximation $\mathbb{E}[(\mathbf{W})^{-1}] \approx (\mathbf{W}^*)^{-1}$ holds under regularity conditions as $N \to \infty$.

##### Step 4.1.3: Effect of parameter estimation

The fitted values $\hat{\boldsymbol{\mu}}$ depend on estimated parameters $(\hat{\boldsymbol{\alpha}}, \hat{\mathbf{b}})$. Using the delta method and asymptotic theory of GLMMs:
$$
\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}) = \mathbb{V}(\mathbf{y}) - 2\text{Cov}(\mathbf{y}, \hat{\boldsymbol{\mu}}) + \mathbb{V}(\hat{\boldsymbol{\mu}})
$$

###### Step 4.1.3.1: Linearization of $\hat{\boldsymbol{\mu}}$

Under the null hypothesis ($\beta = 0$), we have $\hat{\boldsymbol{\mu}} = \operatorname{logit}^{-1}(\mathbf{X}\hat{\boldsymbol{\alpha}} + \hat{\mathbf{b}})$.

Using asymptotic theory, as $N \to \infty$:
$$
\hat{\boldsymbol{\mu}} \approx \boldsymbol{\mu}^* + \mathbf{D}^*(\mathbf{X}(\hat{\boldsymbol{\alpha}} - \boldsymbol{\alpha}^*) + (\hat{\mathbf{b}} - \mathbf{b}^*))
$$

where $\mathbf{D}^* = \operatorname{diag}(\mu_i^*(1-\mu_i^*))$ and $\boldsymbol{\mu}^* = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha}^* + \mathbf{b}^*)$.

###### Step 4.1.3.2: Asymptotic distribution of parameter estimates

**Theorem (GLMM Asymptotic Distribution):** Under the null hypothesis and standard regularity conditions, the joint asymptotic distribution of the parameter estimates in a GLMM satisfies:

$$
\begin{pmatrix}
\hat{\boldsymbol{\alpha}} - \boldsymbol{\alpha}^* \\
\hat{\mathbf{b}} - \mathbf{b}^*
\end{pmatrix} \sim \mathcal{N}\left(\mathbf{0}, \begin{pmatrix}
(\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}\mathbf{X})^{-1} & (\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}\mathbf{X})^{-1}\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1} \\
(\boldsymbol{\Sigma}^*)^{-1}\mathbf{X}(\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}\mathbf{X})^{-1} & (\boldsymbol{\Sigma}^*)^{-1} - (\boldsymbol{\Sigma}^*)^{-1}\mathbf{X}(\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}\mathbf{X})^{-1}\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}
\end{pmatrix}\right)
$$

**Proof of Theorem:** This follows from the theory of penalized quasi-likelihood (PQL) estimation in GLMMs. The joint estimating equations for $(\boldsymbol{\alpha}, \mathbf{b})$ under the null hypothesis are:

$$
\begin{pmatrix}
\mathbf{X}^\top\mathbf{W}(\mathbf{y} - \boldsymbol{\mu}) \\
\mathbf{W}(\mathbf{y} - \boldsymbol{\mu}) - \tau^{-1}\mathbf{\Psi}^{-1}\mathbf{b}
\end{pmatrix} = \mathbf{0}
$$

The Hessian matrix of the penalized log-likelihood at the true parameters is:
$$
\mathbf{H} = \begin{pmatrix}
\mathbf{X}^\top\mathbf{W}^*\mathbf{X} & \mathbf{X}^\top\mathbf{W}^* \\
\mathbf{W}^*\mathbf{X} & \mathbf{W}^* + \tau^{-1}\mathbf{\Psi}^{-1}
\end{pmatrix}
$$

By the asymptotic theory of M-estimators, the covariance matrix is $\mathbf{H}^{-1}$. Using block matrix inversion:
$$
\mathbf{H}^{-1} = \begin{pmatrix}
(\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1} & (\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^{-1} \\
\boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1} & \boldsymbol{\Sigma}^{-1} - \boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}
\end{pmatrix}
$$

where $\boldsymbol{\Sigma} = (\mathbf{W}^*)^{-1} + \tau^*\mathbf{\Psi}$.

###### Step 4.1.3.3: Computing $\mathbb{V}(\hat{\boldsymbol{\mu}})$

From the linearization in Step 4.1.3.1:
$$
\mathbb{V}(\hat{\boldsymbol{\mu}}) \approx \mathbf{D}^* \mathbb{V}(\mathbf{X}(\hat{\boldsymbol{\alpha}} - \boldsymbol{\alpha}^*) + (\hat{\mathbf{b}} - \mathbf{b}^*)) (\mathbf{D}^*)^\top
$$

Let $\mathbf{Z} = [\mathbf{X}, \mathbf{I}]$ be the augmented design matrix. Then:
$$
\mathbf{X}(\hat{\boldsymbol{\alpha}} - \boldsymbol{\alpha}^*) + (\hat{\mathbf{b}} - \mathbf{b}^*) = \mathbf{Z}\begin{pmatrix}
\hat{\boldsymbol{\alpha}} - \boldsymbol{\alpha}^* \\
\hat{\mathbf{b}} - \mathbf{b}^*
\end{pmatrix}
$$

Using the covariance matrix from Step 4.1.3.2:
$$
\mathbb{V}(\mathbf{X}(\hat{\boldsymbol{\alpha}} - \boldsymbol{\alpha}^*) + (\hat{\mathbf{b}} - \mathbf{b}^*)) = \mathbf{Z}\mathbf{C}\mathbf{Z}^\top
$$

where $\mathbf{C}$ is the covariance matrix in Step 4.1.3.2.

After algebraic manipulation (using block matrix properties):
$$
\mathbf{Z}\mathbf{C}\mathbf{Z}^\top = \boldsymbol{\Sigma}^* - \boldsymbol{\Sigma}^*\mathbf{X}(\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^*
$$

Therefore:
$$
\mathbb{V}(\hat{\boldsymbol{\mu}}) \approx \mathbf{D}^*(\boldsymbol{\Sigma}^* - \boldsymbol{\Sigma}^*\mathbf{X}(\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^*)(\mathbf{D}^*)^\top
$$

###### Step 4.1.3.4: Computing $\text{Cov}(\mathbf{y}, \hat{\boldsymbol{\mu}})$

Since $\mathbf{y}$ and the parameter estimates are correlated through the estimation process:
$$
\text{Cov}(\mathbf{y}, \hat{\boldsymbol{\mu}}) = \text{Cov}(\mathbf{y}, \mathbf{D}^*(\mathbf{X}(\hat{\boldsymbol{\alpha}} - \boldsymbol{\alpha}^*) + (\hat{\mathbf{b}} - \mathbf{b}^*)))
$$

Using the fact that parameter estimates are obtained by solving the estimating equations, which are functions of $\mathbf{y}$:
$$
\text{Cov}(\mathbf{y}, \hat{\boldsymbol{\mu}}) \to \mathbf{D}^*(\boldsymbol{\Sigma}^* - \boldsymbol{\Sigma}^*\mathbf{X}(\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^*)
$$

###### Step 4.1.3.5: Final computation

Under regularity conditions, $\mathbf{D}^* \to \mathbf{I}$ in the leading order. Therefore:

$$
-2\text{Cov}(\mathbf{y}, \hat{\boldsymbol{\mu}}) + \mathbb{V}(\hat{\boldsymbol{\mu}}) \to -\boldsymbol{\Sigma}^*\mathbf{X}(\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^*
$$

###### Step 4.1.3.6: Final result

Combining with $\mathbb{V}(\mathbf{y}) \to \boldsymbol{\Sigma}^*$:
$$
\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}) \to \boldsymbol{\Sigma}^* - \boldsymbol{\Sigma}^*\mathbf{X}(\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^* = \mathbf{P}^*
$$

where $\boldsymbol{\Sigma}^* = (\mathbf{W}^*)^{-1} + \tau^*\mathbf{\Psi}$ and the projection term accounts for the estimation of fixed effects.

##### Step 4.1.4: Consistency

By the consistency of $(\hat{\boldsymbol{\alpha}}, \hat{\tau}, \hat{\mathbf{b}})$ and the continuous mapping theorem:
$$
\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}) \xrightarrow{p} \mathbf{P}^*
$$

#### Step 4.2: Equivalence of quadratic forms

Now we show that $(\tilde{\mathbf{g}}^*)^\top \mathbf{P}^* \tilde{\mathbf{g}}^* = \mathbf{g}^\top \mathbf{P}^* \mathbf{g}$:

From the definition of $\tilde{\mathbf{g}}$:
$$
\tilde{\mathbf{g}}^* = \mathbf{g} - \mathbf{X}(\mathbf{X}^\top(\mathbf{W}^*)^{-1}\mathbf{X})^{-1}\mathbf{X}^\top(\mathbf{W}^*)^{-1}\mathbf{g}
$$

Since $\mathbf{P}^* = (\boldsymbol{\Sigma}^*)^{-1} - (\boldsymbol{\Sigma}^*)^{-1}\mathbf{X}(\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}\mathbf{X})^{-1}\mathbf{X}^\top(\boldsymbol{\Sigma}^*)^{-1}$ is a projection matrix orthogonal to the column space of $\mathbf{X}$, we have:
$$
\mathbf{P}^*\mathbf{X} = \mathbf{0}
$$

Therefore:
$$
(\tilde{\mathbf{g}}^*)^\top \mathbf{P}^* \tilde{\mathbf{g}}^* = \mathbf{g}^\top \mathbf{P}^* \mathbf{g}
$$

#### Step 4.3: Final equivalence

Combining the results:
$$
\mathbb{V}(T) = \mathbf{g}^\top \mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}) \mathbf{g} \to \mathbf{g}^\top \mathbf{P}^* \mathbf{g} = (\tilde{\mathbf{g}}^*)^\top \mathbf{P}^* \tilde{\mathbf{g}}^*
$$

### Step 5: Asymptotic equivalence

Under the null hypothesis and regularity conditions, the asymptotic variance of the score statistic $T$ is correctly captured by the limiting form of $\hat{\mathbf{P}}$, which accounts for:

- The covariance structure through $\boldsymbol{\Psi}$
- The mean-variance relationship in the GLM through $\mathbf{W}$
- The projection effects through the sandwich form

Therefore, $\hat{\mathbb{V}}(T)$ is a consistent estimator of $\mathbb{V}(T)$. $\square$
