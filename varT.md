# Is $\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}$ an unbiased estimator of $\operatorname{Var}(T)$?

## 1. Definition

Let $T = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})$, where $\tilde{\mathbf{g}}$ is the covariate-adjusted genotype vector and $\hat{\mathbf{P}}$ is a projection/variance matrix depending on the model fit.

The variance estimator is
$$
\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}
$$

## 2. Is $\mathbb{V}(T)$ unbiased for $\operatorname{Var}(T)$?

### a. Theoretical variance of $T$


#### Step-by-step derivation of $\operatorname{Cov}(\mathbf{y})$ in the mixed model

Consider the logistic mixed model:
$$
\mathbf{y} \mid \mathbf{b} \sim \operatorname{Bernoulli}(\boldsymbol{\mu}), \quad \boldsymbol{\mu} = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b})
$$
where $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$.

The marginal mean is
$$
\mathbb{E}(\mathbf{y}) = \mathbb{E}_{\mathbf{b}}[\mathbb{E}(\mathbf{y} \mid \mathbf{b})] = \mathbb{E}_{\mathbf{b}}[\boldsymbol{\mu}]
$$

The marginal covariance is given by the law of total covariance:
$$
\operatorname{Cov}(\mathbf{y}) = \mathbb{E}_{\mathbf{b}}[\operatorname{Cov}(\mathbf{y} \mid \mathbf{b})] + \operatorname{Cov}_{\mathbf{b}}[\mathbb{E}(\mathbf{y} \mid \mathbf{b})]
$$
where
- $\operatorname{Cov}(\mathbf{y} \mid \mathbf{b}) = \operatorname{diag}(\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu}))$
- $\mathbb{E}(\mathbf{y} \mid \mathbf{b}) = \boldsymbol{\mu}$

So,
$$
\operatorname{Cov}(\mathbf{y}) = \mathbb{E}_{\mathbf{b}}[\operatorname{diag}(\boldsymbol{\mu} \odot (\mathbf{1} - \boldsymbol{\mu}))] + \operatorname{Cov}_{\mathbf{b}}[\boldsymbol{\mu}]
$$

In practice, $\hat{\boldsymbol{\Sigma}}$ is a plug-in estimator for this marginal covariance, using fitted values and estimated variance components.


For $T = \tilde{\mathbf{g}}^\top \mathbf{y}$,
$$
\operatorname{Var}(T) = \tilde{\mathbf{g}}^\top \operatorname{Cov}(\mathbf{y}) \tilde{\mathbf{g}}
$$

However, when $\tilde{\mathbf{g}}$ is computed as a residual from a regression, and $\hat{\mathbf{P}}$ is the projection matrix orthogonal to $\mathbf{X}$, the estimator $\mathbb{V}(T)$ is designed to account for the loss of degrees of freedom and the structure of the model.

### b. Unbiasedness in the linear model


#### Step-by-step derivation of unbiasedness in the linear model

Suppose $\mathbf{y} \sim \mathcal{N}(\mathbf{X}\boldsymbol{\alpha}, \sigma^2 \mathbf{I})$ and $\tilde{\mathbf{g}}$ is orthogonal to $\mathbf{X}$.

1. $T = \tilde{\mathbf{g}}^\top \mathbf{y}$ is a linear combination of $\mathbf{y}$.
2. $\operatorname{Var}(T) = \operatorname{Var}(\tilde{\mathbf{g}}^\top \mathbf{y}) = \tilde{\mathbf{g}}^\top \operatorname{Cov}(\mathbf{y}) \tilde{\mathbf{g}} = \sigma^2 \|\tilde{\mathbf{g}}\|^2$.
3. The unbiased estimator of $\sigma^2$ is
   $$
   \hat{\sigma}^2 = \frac{\|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\alpha}}\|^2}{N - p - 1}
   $$
   where $\hat{\boldsymbol{\alpha}}$ is the least squares estimator.
4. The variance estimator for $T$ is
   $$
   \mathbb{V}(T) = \hat{\sigma}^2 \|\tilde{\mathbf{g}}\|^2
   $$
5. To show unbiasedness:
   $$
   \mathbb{E}[\mathbb{V}(T)] = \mathbb{E}[\hat{\sigma}^2] \|\tilde{\mathbf{g}}\|^2 = \sigma^2 \|\tilde{\mathbf{g}}\|^2 = \operatorname{Var}(T)
   $$
   since $\mathbb{E}[\hat{\sigma}^2] = \sigma^2$.

Therefore, $\mathbb{V}(T)$ is an unbiased estimator of $\operatorname{Var}(T)$ in the classical linear model.

### c. In the generalized linear mixed model (GLMM)

In the GLMM or logistic mixed model, $\hat{\mathbf{P}}$ is a plug-in estimator using fitted values and estimated variance components. It is generally a consistent estimator, but not exactly unbiased in finite samples due to the use of estimated parameters and the nonlinearity of the model.

## 3. Step-by-step proof in the linear model

Suppose $\mathbf{y} \sim \mathcal{N}(\mathbf{X}\boldsymbol{\alpha}, \sigma^2 \mathbf{I})$ and $\tilde{\mathbf{g}}$ is orthogonal to $\mathbf{X}$.

- $T = \tilde{\mathbf{g}}^\top \mathbf{y}$
- $\operatorname{Var}(T) = \sigma^2 \|\tilde{\mathbf{g}}\|^2$
- The unbiased estimator of $\sigma^2$ is $\hat{\sigma}^2 = \frac{\|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\alpha}}\|^2}{N - p - 1}$
- Then $\mathbb{V}(T) = \hat{\sigma}^2 \|\tilde{\mathbf{g}}\|^2$ is unbiased for $\operatorname{Var}(T)$

## 4. In the mixed model


#### Step-by-step derivation of consistency in the mixed model

1. In the mixed model, $T = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})$ and $\operatorname{Var}(T) = \tilde{\mathbf{g}}^\top \operatorname{Cov}(\mathbf{y}) \tilde{\mathbf{g}}$.
2. The estimator $\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}$ uses plug-in estimates for the covariance structure, i.e., $\hat{\mathbf{P}}$ is computed from estimated parameters $(\hat{\boldsymbol{\alpha}}, \hat{\tau}, \hat{\mathbf{b}})$ and fitted values $\hat{\boldsymbol{\mu}}$.
3. As $N \to \infty$, the maximum likelihood (or REML) estimators $\hat{\boldsymbol{\alpha}}, \hat{\tau}, \hat{\mathbf{b}}$ are consistent, i.e., they converge in probability to the true parameter values.
4. Therefore, $\hat{\boldsymbol{\mu}} \to \boldsymbol{\mu}$ and $\hat{\mathbf{P}} \to \mathbf{P}$ in probability, where $\mathbf{P}$ is the true variance structure for $T$.
5. By the continuous mapping theorem, $\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}} \to \tilde{\mathbf{g}}^\top \mathbf{P} \tilde{\mathbf{g}} = \operatorname{Var}(T)$ in probability as $N \to \infty$.
6. Thus, $\mathbb{V}(T)$ is a consistent estimator for $\operatorname{Var}(T)$ in the mixed model.

In summary, consistency follows from the consistency of the parameter estimators and the plug-in principle for variance estimation.

## 5. Conclusion

- $\mathbb{V}(T)$ is an estimator of $\operatorname{Var}(T)$.
- It is unbiased in the classical linear model with known variance.
- In the mixed model, it is generally consistent but not exactly unbiased in finite samples.
