# SAIGE

In a case-control study with a sample size $N$, let

- $N \times 1$ random vector $\mathbf{y}$ represent their phenotypes;
- $N \times (1 + p)$ matrix $\mathbf{X}$ represent their $p$ covariates and a column of ones;
- $N \times 1$ vector $\mathbf{g}$ represent their genotypes coded as allele counts for a variant to be tested;
- $N \times 1$ random vector $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$, where $\mathbf{\Psi}$ is a GRM and $\tau$ is the corresponding variance component;
- $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$ be a linear predictor, where $\boldsymbol{\alpha}$ is the fixed effects and $\beta$ is the genetic effect to be tested.

Suppose $(\mathbf{X}, \mathbf{g}, \mathbf{\Psi})$ are observed and treated as fixed, $(\boldsymbol{\alpha}, \beta, \tau)$ are unknown parameters; $\mathbf{y}$ and $\mathbf{b}$ are random vectors before observing the data.

For each subject, suppose

$$
y_i|b_i \sim \operatorname{Bernoulli}(\mu_i), \quad \mu_i = \operatorname{logit}^{-1}(\eta_i) \tag{1}
$$

We are interested in testing the null hypothesis $H_0: \beta=0$.

## 1. Probability functions

According to (1), we have the conditional PMF of $y_i$ given $b_i$ as

$$
p_{y_i | b_i}(y_i; b_i, \mathbf{x}_i, g_i, \boldsymbol{\alpha}, \beta) = \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}, \quad \mu_i = \frac{1}{1 + e^{-(\mathbf{x}_i\boldsymbol{\alpha} + g_i\beta + b_i)}}
$$

The conditional joint PMF of $\mathbf{y}$ given $\mathbf{b}$ is

$$
p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}; \mathbf{b}, \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}, \quad \mu_i = \frac{1}{1 + e^{-(\mathbf{x}_i\boldsymbol{\alpha} + g_i\beta + b_i)}} \tag{2}
$$

The marginal PDF of $\mathbf{b}$ is given by the density of $\mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$,

$$
p_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi}) = \phi(\mathbf{b}; \mathbf{0}, \tau \mathbf{\Psi}) = \frac{1}{(2\pi)^{\frac{N}{2}} |\tau \mathbf{\Psi}|^{\frac{1}{2}}} \exp\left(-\frac{1}{2} \mathbf{b}^\top (\tau \mathbf{\Psi})^{-1} \mathbf{b}\right)
$$

The joint probability function of $(\mathbf{y},\mathbf{b})$ is

$$
p_{\mathbf{y}, \mathbf{b}}(\mathbf{y}, \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta, \tau, \mathbf{\Psi}) = p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b})
$$

The marginal PMF of $\mathbf{y}$ is

$$
p_{\mathbf{y}}(\mathbf{y}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta, \tau, \mathbf{\Psi}) = \int_{\mathbb{R}^N}  p_{\mathbf{y}, \mathbf{b}}(\mathbf{y}, \mathbf{b}) \mathrm{d}\mathbf{b} = \int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b}) \mathrm{d}\mathbf{b} \tag{3}
$$

The conditional PDF of $\mathbf{b}$ given $\mathbf{y}$ is

$$
p_{\mathbf{b}|\mathbf{y}}(\mathbf{b}; \mathbf{y}, \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta, \tau, \mathbf{\Psi}) = \frac{p_{\mathbf{y}, \mathbf{b}}(\mathbf{y}, \mathbf{b})}{p_{\mathbf{y}}(\mathbf{y})} = \frac{p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b})}{\int_{\mathbb{R}^N}  p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b}) \mathrm{d}\mathbf{b}} \tag{4}
$$

## 2. Conditional likelihood function

According to (2), the conditional likelihood function given $\mathbf{b}$ is

$$
\begin{aligned}
\ell(\boldsymbol{\alpha, \beta}; \mathbf{y}|\mathbf{b}, \mathbf{X}, \mathbf{g})
&= \log p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}; \mathbf{b}, \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) \\
&= \log \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i} \\
&= \sum_{i=1}^N \left[ y_i \log(\mu_i) + (1 - y_i) \log(1 - \mu_i) \right] \\
&= \mathbf{y}^\top \log(\boldsymbol{\mu}) + (\mathbf{1} - \mathbf{y})^\top \log(\mathbf{1} - \boldsymbol{\mu})
\end{aligned}
$$

Substituting $\mu_i = \operatorname{logit}^{-1}(\eta_i)$ yields

$$
\begin{aligned}
\log(\boldsymbol{\mu}) &= -\log(1 + \exp(-\boldsymbol{\eta})) \\
\log(\mathbf{1} - \boldsymbol{\mu}) &= \log\left(1 - \frac{1}{1 + \exp(-\boldsymbol{\eta})}\right) \\
&= \log\left(\frac{\exp(-\boldsymbol{\eta})}{1 + \exp(-\boldsymbol{\eta})}\right) \\
&= -\boldsymbol{\eta} - \log(1 + \exp(-\boldsymbol{\eta}))
\end{aligned}
$$

Substituting into the log-likelihood function yields
$$
\begin{aligned}
\ell(\boldsymbol{\alpha, \beta}; \mathbf{y}|\mathbf{b}, \mathbf{X}, \mathbf{g})
&= \mathbf{y}^\top \left[-\log(1 + \exp(-\boldsymbol{\eta}))\right] \\
&\quad + (\mathbf{1} - \mathbf{y})^\top \left[-\boldsymbol{\eta} - \log(1 + \exp(-\boldsymbol{\eta}))\right] \\
&= -\mathbf{y}^\top \log(1 + \exp(-\boldsymbol{\eta})) \\
&\quad - (\mathbf{1} - \mathbf{y})^\top \boldsymbol{\eta} - (\mathbf{1} - \mathbf{y})^\top \log(1 + \exp(-\boldsymbol{\eta})) \\
&= -\mathbf{1}^\top \log(1 + \exp(-\boldsymbol{\eta})) - (\mathbf{1} - \mathbf{y})^\top \boldsymbol{\eta} \\
&= \mathbf{y}^\top \boldsymbol{\eta} - \mathbf{1}^\top (\boldsymbol{\eta} + \log[1 + \exp(-\boldsymbol{\eta})]) \\
&= \mathbf{y}^\top \boldsymbol{\eta} - \mathbf{1}^\top \log(1 + \exp(\boldsymbol{\eta}))
\end{aligned}
\tag{5}
$$

## 3. Conditional score functions

According to (5), the conditional score function for $\beta$ given $\mathbf{b}$ is

$$
U_\beta(\beta; \boldsymbol{\alpha}, \mathbf{y}, \mathbf{b}, \mathbf{X}, \mathbf{g}) = \frac{\partial \ell(\boldsymbol{\alpha}, \beta; \mathbf{y}, \mathbf{b}, \mathbf{X}, \mathbf{g})}{\partial \beta} = \frac{\partial}{\partial \beta} \left[ \mathbf{y}^\top \boldsymbol{\eta} - \mathbf{1}^\top \log(1 + \exp(\boldsymbol{\eta}) \right]
$$

The partial derivative for the first term is

$$
\frac{\partial}{\partial \beta} \left[ \mathbf{y}^\top \boldsymbol{\eta} \right] = \frac{\partial}{\partial \beta} \left[ \mathbf{y}^\top (\mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}) \right] = \mathbf{y}^\top \mathbf{g}
$$

For the second term is

$$
\frac{\partial}{\partial \beta} \left[ \mathbf{1}^\top \log(1 + \exp(\boldsymbol{\eta})) \right] = \frac{\partial}{\partial \beta} \sum_{i=1}^N \log(1 + \exp({\eta}_i))
$$

Since

$$
\frac{\partial}{\partial \beta} \log(1 + \exp(\eta_i))  = \frac{\exp(\eta_i)}{1 + \exp(\eta_i)}g_i = g_i\operatorname{logit}^{-1}(\eta_i)
$$

We have

$$
\frac{\partial}{\partial \beta} \left[ \mathbf{1}^\top \log(1 + \exp(\boldsymbol{\eta})) \right] = \sum_{i=1}^N g_i\operatorname{logit}^{-1}(\eta_i) = \mathbf{g}^\top \operatorname{logit}^{-1}(\boldsymbol{\eta})
$$

Combine terms, we get

$$
U_\beta(\beta) =  \mathbf{y}^\top \mathbf{g} - \mathbf{g}^\top \operatorname{logit}^{-1}(\boldsymbol{\eta}) = \mathbf{g}^\top (\mathbf{y} - \boldsymbol{\mu}) \tag{6}
$$

Similarly, the conditional score function for $\boldsymbol{\alpha}$ given $\mathbf{b}$ is

$$
U_{\boldsymbol{\alpha}}(\boldsymbol{\alpha}; \beta, \mathbf{y}, \mathbf{b}, \mathbf{X}, \mathbf{g}) = \mathbf{X}^\top (\mathbf{y} - \boldsymbol{\mu}) \tag{7}
$$

## 4. Estimate of $(\tau, \boldsymbol{\alpha}, \mathbf{b})$ under null model

Under the null model, i.e. $\beta=0$, according to (3), the marginal log-likelihood function for $(\boldsymbol{\alpha}, \tau)$, marginalizing over $\mathbf{b}$, is

$$
\ell(\boldsymbol{\alpha}, \tau; \mathbf{X}, \mathbf{\Psi}, \mathbf{y}) = \log p_{\mathbf{y}}(\mathbf{y}; \mathbf{X}, \mathbf{\Psi}, \boldsymbol{\alpha}, \tau) = \log \int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) \phi(\mathbf{b}) \mathrm{d}\mathbf{b}
$$

where

$$
p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}, \quad \mu_i = \frac{1}{1 + e^{-(\mathbf{x}_i\boldsymbol{\alpha} + b_i)}}
$$

Let
$$
(\hat{\boldsymbol{\alpha}}, \hat{\tau}) = \arg\max_{\boldsymbol{\alpha}, \tau}\ \ell(\boldsymbol{\alpha}, \tau; \mathbf{X}, \mathbf{\Psi}, \mathbf{y}) \tag{8}
$$

and

$$
\hat{\mathbf{b}} = \mathbb{E}[\mathbf{b} \mid \mathbf{y}, \hat{\boldsymbol{\alpha}}, \hat{\tau}] = \int_{\mathbb{R}^N} \mathbf{b} \frac{p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) \phi(\mathbf{b})}{\int_{\mathbb{R}^N}  p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) \phi(\mathbf{b}) \mathrm{d}\mathbf{b}} \mathrm{d}\mathbf{b} \tag{9}
$$

be an estimate of $(\tau, \boldsymbol{\alpha}, \mathbf{b})$ under the null model.

## 5. Score statistic

Let the score statistic for $\beta$ be
$$
T = U_\beta(\beta=0, \mathbf{y}, \hat{\mathbf{b}}, \mathbf{X}, \mathbf{g}, \hat{\boldsymbol{\alpha}}) = \mathbf{g}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}) \tag{10}
$$

where $\hat{\boldsymbol{\mu}} = \operatorname{logit}^{-1}(\mathbf{X}\hat{\boldsymbol{\alpha}} + \hat{\mathbf{b}})$.

Since $(\mathbf{X}, \mathbf{g}, \mathbf{\Psi})$ are treated as fixed, prior to observing $\mathbf{y}$, the estimates $\hat{\boldsymbol{\alpha}}$, $\hat{\mathbf{b}}$, and $\hat{\boldsymbol{\mu}}$ defined in (8) and (9) are functions of the random vector $\mathbf{y}$ and hence are themselves random. The variance of $T$ is

$$
\mathbb{V}(T) = \mathbf{g}^\top \mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}) \mathbf{g}
$$

Let

$$
\tilde{\mathbf{g}} = \mathbf{g} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{g}, \quad \hat{\mathbf{W}} = \operatorname{diag}(\hat{\boldsymbol{\mu}} \odot [\mathbf{1} - \hat{\boldsymbol{\mu}}])
$$

which is the residual vector from regressing $\mathbf{g}$ on $\mathbf{X}$ using the weight matrix $\hat{\mathbf{W}}$. Then, upon substituting $\mathbf{g}$ in (8), we get

$$
\begin{aligned}
T
&= \left[ \tilde{\mathbf{g}} + \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{g} \right]^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}) \\
&= \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}) + \left[ (\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{g} \right]^\top \mathbf{X}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}) \\
\end{aligned}
$$

According to (7), $\mathbf{X}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}) = U_{\boldsymbol{\alpha}}(\hat{\boldsymbol{\alpha}}, \beta=0, \mathbf{y}, \hat{\mathbf{b}}, \mathbf{X})$ is the conditional score function for $\boldsymbol{\alpha}$, evaluated at its MLE under the null model with the same dataset. By definition of the MLE, this score function equals the zero vector since the MLE solves the score equations. Thus, we have

$$
T = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}) \tag{11}
$$

Let

$$
\hat{\mathbb{V}}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}} \tag{12}
$$

where $\hat{\mathbf{P}} = \hat{\boldsymbol{\Sigma}}^{-1} - \hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X}(\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}$ and $\hat{\boldsymbol{\Sigma}} = \hat{\mathbf{W}}^{-1} + \hat{\tau}\mathbf{\Psi}$, be an estimator of the variance of $T$. It is a consistent estimator (proof required).

The standardized score statistic is

$$
\frac{T}{\sqrt{\hat{\mathbb{V}}(T)}} = \frac{\tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})}{\sqrt{\tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}}}
$$

which, as $N \to \infty$, converges in distribution to $\mathcal{N}(0, 1)$.
