# SAIGE

In a case-control study with a sample size $N$, let

- $N \times 1$ random vector $\mathbf{y}$ represent their phenotypes;
- $N \times (1 + p)$ matrix $\mathbf{X}$ represent their $p$ covariates and a column of ones;
- $N \times 1$ vector $\mathbf{G}$ represent their genotypes coded as allele counts for a variant to be tested;
- $N \times 1$ random vector $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$, where $\mathbf{\Psi}$ is a GRM and $\tau$ is the corresponding variance component;
- $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{G}\beta + \mathbf{b}$ be the linear predictor, where $\boldsymbol{\alpha}$ is the fixed effects and $\beta$ is the genetic effect to be tested.

Suppose $(\mathbf{X}, \mathbf{G}, \mathbf{\Psi})$ are observed and treated as fixed, $(\boldsymbol{\alpha}, \beta, \tau)$ are unknown parameters; $\mathbf{y}$ and $\mathbf{b}$ are random vectors before observing the data.

For each subject, suppose

$$
y_i|b_i \sim \operatorname{Bernoulli}(\mu_i), \quad \mu_i = \operatorname{logit}^{-1}(\eta_i) \tag{1}
$$

We are interested in testing the null hypothesis $H_0: \beta=0$.

## 1. Probability functions

According to (1), we have the conditional PMF of $y_i$ given $b_i$ as

$$
p_{y_i | b_i}(y_i; b_i, \mathbf{X}_i, G_i, \boldsymbol{\alpha}, \beta) = \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}, \quad \mu_i = \frac{1}{1 + e^{-(\mathbf{X}_i\boldsymbol{\alpha} + G_i\beta + b_i)}}
$$

The conditional joint PMF of $\mathbf{y}$ given $\mathbf{b}$ is

$$
p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}; \mathbf{b}, \mathbf{X}, \mathbf{G}, \boldsymbol{\alpha}, \beta) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1 - y_i}, \quad \mu_i = \frac{1}{1 + e^{-(\mathbf{X}_i\boldsymbol{\alpha} + G_i\beta + b_i)}} \tag{2}
$$

The marginal PDF of $\mathbf{b}$ is given by the density of $\mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$,

$$
p_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi}) = \phi(\mathbf{b}; \mathbf{0}, \tau \mathbf{\Psi}) = \frac{1}{(2\pi)^{\frac{N}{2}} |\tau \mathbf{\Psi}|^{\frac{1}{2}}} \exp\left(-\frac{1}{2} \mathbf{b}^\top (\tau \mathbf{\Psi})^{-1} \mathbf{b}\right)
$$

The joint probability function of $(\mathbf{y},\mathbf{b})$ is

$$
p_{\mathbf{y}, \mathbf{b}}(\mathbf{y}, \mathbf{b}; \mathbf{X}, \mathbf{G}, \boldsymbol{\alpha}, \beta, \tau, \mathbf{\Psi}) = p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b})
$$

The marginal PMF of $\mathbf{y}$ is

$$
p_{\mathbf{y}}(\mathbf{y}; \mathbf{X}, \mathbf{G}, \boldsymbol{\alpha}, \beta, \tau, \mathbf{\Psi}) = \int_{\mathbb{R}^N}  p_{\mathbf{y}, \mathbf{b}}(\mathbf{y}, \mathbf{b}) \mathrm{d}\mathbf{b} = \int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b}) \mathrm{d}\mathbf{b} \tag{3}
$$

The conditional PDF of $\mathbf{b}$ given $\mathbf{y}$ is

$$
p_{\mathbf{b}|\mathbf{y}}(\mathbf{b}; \mathbf{y}, \mathbf{X}, \mathbf{G}, \boldsymbol{\alpha}, \beta, \tau, \mathbf{\Psi}) = \frac{p_{\mathbf{y}, \mathbf{b}}(\mathbf{y}, \mathbf{b})}{p_{\mathbf{y}}(\mathbf{y})} = \frac{p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b})}{\int_{\mathbb{R}^N}  p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) p_{\mathbf{b}}(\mathbf{b}) \mathrm{d}\mathbf{b}} \tag{4}
$$

## 2. Conditional likelihood function

According to (2), the conditional likelihood function given $\mathbf{b}$ is

$$
\begin{aligned}
\ell(\boldsymbol{\alpha, \beta}; \mathbf{y}|\mathbf{b}, \mathbf{X}, \mathbf{G})
&= \log p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}; \mathbf{b}, \mathbf{X}, \mathbf{G}, \boldsymbol{\alpha}, \beta) \\
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
\ell(\boldsymbol{\alpha, \beta}; \mathbf{y}|\mathbf{b}, \mathbf{X}, \mathbf{G})
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
U_\beta(\beta; \boldsymbol{\alpha}, \mathbf{y}, \mathbf{b}, \mathbf{X}, \mathbf{G}) = \frac{\partial \ell(\boldsymbol{\alpha}, \beta; \mathbf{y}, \mathbf{b}, \mathbf{X}, \mathbf{G})}{\partial \beta} = \frac{\partial}{\partial \beta} \left[ \mathbf{y}^\top \boldsymbol{\eta} - \mathbf{1}^\top \log(1 + \exp(\boldsymbol{\eta}) \right]
$$

The partial derivative for the first term is

$$
\frac{\partial}{\partial \beta} \left[ \mathbf{y}^\top \boldsymbol{\eta} \right] = \frac{\partial}{\partial \beta} \left[ \mathbf{y}^\top (\mathbf{X} \boldsymbol{\alpha} + \mathbf{G}\beta + \mathbf{b}) \right] = \mathbf{y}^\top \mathbf{G}
$$

For the second term is

$$
\frac{\partial}{\partial \beta} \left[ \mathbf{1}^\top \log(1 + \exp(\boldsymbol{\eta})) \right] = \frac{\partial}{\partial \beta} \sum_{i=1}^N \log(1 + \exp({\eta}_i))
$$

Since

$$
\frac{\partial}{\partial \beta} \log(1 + \exp(\eta_i))  = \frac{\exp(\eta_i)}{1 + \exp(\eta_i)}G_i = G_i\operatorname{logit}^{-1}(\eta_i)
$$

We have

$$
\frac{\partial}{\partial \beta} \left[ \mathbf{1}^\top \log(1 + \exp(\boldsymbol{\eta})) \right] = \sum_{i=1}^N G_i\operatorname{logit}^{-1}(\eta_i) = \mathbf{G}^\top \operatorname{logit}^{-1}(\boldsymbol{\eta})
$$

Combine terms, we get

$$
U_\beta(\beta) =  \mathbf{y}^\top \mathbf{G} - \mathbf{G}^\top \operatorname{logit}^{-1}(\boldsymbol{\eta}) = \mathbf{G}^\top (\mathbf{y} - \boldsymbol{\mu}) \tag{6}
$$

Similarly, the conditional score function for $\boldsymbol{\alpha}$ given $\mathbf{b}$ is

$$
U_{\boldsymbol{\alpha}}(\boldsymbol{\alpha}; \beta, \mathbf{y}, \mathbf{b}, \mathbf{X}, \mathbf{G}) = \mathbf{X}^\top (\mathbf{y} - \boldsymbol{\mu}) \tag{7}
$$

## 4. Estimate of $(\tau, \boldsymbol{\alpha}, \mathbf{b})$ under null model

Under the null model, i.e. $\beta=0$, let $\boldsymbol{\eta}_0 = \mathbf{X} \boldsymbol{\alpha}_0 + \mathbf{b}_0$ be the linear predictor, where $\boldsymbol{\alpha}_0$ is the vector of fixed effects and $\mathbf{b}_0 \sim \mathcal{N}(\mathbf{0}, \tau_0 \mathbf{\Psi})$, and let $\boldsymbol{\mu}_0 = \operatorname{logit}^{-1}(\boldsymbol{\eta}_0)$.

According to (3), the marginal log-likelihood function for $(\boldsymbol{\alpha}_0, \tau_0)$, marginalizing over $\mathbf{b}_0$, is

$$
\ell(\boldsymbol{\alpha}_0, \tau_0; \mathbf{X}, \mathbf{\Psi}, \mathbf{y}) = \log p_{\mathbf{y}}(\mathbf{y}; \mathbf{X}, \mathbf{\Psi}, \boldsymbol{\alpha}_0, \tau_0) = \log \int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) \phi(\mathbf{b}_0) \mathrm{d}\mathbf{b}_0
$$

where

$$
p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) = \prod_{i=1}^N \mu_{i0}^{y_i} (1 - \mu_{i0})^{1 - y_i}, \quad \mu_{i0} = \frac{1}{1 + e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + b_{i0})}}
$$

Let
$$
(\hat{\boldsymbol{\alpha}_0}, \hat{\tau}_0) = \arg\max_{\boldsymbol{\alpha}_0, \tau_0}\ \ell(\boldsymbol{\alpha}_0, \tau_0; \mathbf{X}, \mathbf{\Psi}, \mathbf{y}) \tag{8}
$$

and according to (4),

$$
\hat{\mathbf{b}}_0 = \mathbb{E}[\mathbf{b}_0 \mid \mathbf{y}, \hat{\boldsymbol{\alpha}}_0, \hat{\tau}_0] = \int_{\mathbb{R}^N} \mathbf{b}_0 \frac{p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) \phi(\mathbf{b}_0)}{\int_{\mathbb{R}^N}  p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) \phi(\mathbf{b}_0) \mathrm{d}\mathbf{b}_0} \mathrm{d}\mathbf{b}_0 \tag{9}
$$

be an estimate of $(\tau_0, \boldsymbol{\alpha}_0, \mathbf{b}_0)$ under the null model.

## 5. Score statistic

Let the score statistic for $\beta$ be
$$
T = U_\beta(\beta=0, \hat{\boldsymbol{\alpha}}_0, \mathbf{y}, \hat{\mathbf{b}}_0, \mathbf{X}, \mathbf{G}) = \mathbf{G}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) \tag{10}
$$

where $\hat{\boldsymbol{\mu}}_0 = \operatorname{logit}^{-1}(\mathbf{X}\hat{\boldsymbol{\alpha}}_0 + \hat{\mathbf{b}}_0)$. Since $\hat{\boldsymbol{\alpha}}_0$ and $\hat{\mathbf{b}}_0$ defined in (8) and (9) are functions of sample $\mathbf{y}$, they are random vectors and are determined after $\mathbf{y}$ is observed. As $N$ becomes large, $\hat{\boldsymbol{\alpha}}_0$ converges to $\boldsymbol{\alpha}_0$, while $\hat{\mathbf{b}}_0 = \mathbb{E}[\mathbf{b}_0 \mid \mathbf{y}, \hat{\boldsymbol{\alpha}}_0, \hat{\tau}_0]$ continues to vary across different samples. When $N$ is large, there is $\hat{\boldsymbol{\alpha}}_0 \approx \boldsymbol{\alpha}_0$; conditioning on $\mathbf{b}_0$, there is $\hat{\mathbf{b}}_0 = \mathbf{b}_0$ and $\hat{\boldsymbol{\mu}}_0 \approx \boldsymbol{\mu}_0$, and thus we have

- $\mathbb{V}(\hat{\boldsymbol{\mu}}_0|\mathbf{b}_0) \approx \mathbb{V}(\boldsymbol{\mu}_0|\mathbf{b}_0)=0$
- $\mathbb{E}(\hat{\boldsymbol{\mu}}_0|\mathbf{b}_0) \approx \mathbb{E}(\boldsymbol{\mu}_0|\mathbf{b}_0)=\boldsymbol{\mu}_0$
- $\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0|\mathbf{b}_0) \approx \mathbb{V}(\mathbf{y}|\mathbf{b}_0) = \operatorname{diag}(\boldsymbol{\mu}_0 \odot [\mathbf{1} - \boldsymbol{\mu}_0])$
- $\mathbb{E}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0|\mathbf{b}_0) \approx \mathbb{E}(\mathbf{y} |\mathbf{b}_0) - \mathbb{E}(\boldsymbol{\mu}_0|\mathbf{b}_0) =\boldsymbol{0}$

Therefore,

$$
\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0) = \mathbb{E}_{\mathbf{b}_0}[\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0|\mathbf{b}_0)] + \mathbb{V}_{\mathbf{b}_0}[\mathbb{E}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0|\mathbf{b}_0)] \approx \mathbb{E}_{\mathbf{b}_0}(\mathbf{W})
$$

where $\mathbf{W} = \operatorname{diag}(\boldsymbol{\mu}_0 \odot [\mathbf{1} - \boldsymbol{\mu}_0])$, with the $i$th element in the dianonal being

$$
W_{ii} = \frac{e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + b_{i0})}}{(1 + e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + b_{i0})})^2},\quad b_{i0} \sim N(0,\tau\mathbf{\Psi}_{ii})
$$

Therefore, $\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0)$ is a diagonal matrix, with the $i$th element in the dianonal being

$$
\mathbb{E}_{b_{i0}}(W_{ii}) = \int_{-\infty}^{\infty} \frac{e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + b_{i0})}}{(1 + e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + b_{i0})})^2} \cdot \frac{1}{\sqrt{2\pi \tau \mathbf{\Psi}_{ii}}} \exp\left(-\frac{b_{i0}^2}{2\tau \mathbf{\Psi}_{ii}}\right) \mathrm{d}b_{i0}
$$

Since $W_{ii}$ is a convex function of $b_{i0}$, by Jensen's inequality, we have

$$
\mathbb{E}_{b_{i0}}(W_{ii}) \geq \frac{e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + \mathbb{E}b_{i0})}}{(1 + e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + \mathbb{E}b_{i0})})^2}
$$

Let

- $\hat{\mathbf{W}} = \operatorname{diag}(\hat{\boldsymbol{\mu}}_0 \odot [\mathbf{1} - \hat{\boldsymbol{\mu}}_0])$
- $\hat{\boldsymbol{\Sigma}} = \hat{\mathbf{W}}^{-1} + \hat{\tau}\mathbf{\Psi}$
- $\hat{\mathbf{P}} = \hat{\boldsymbol{\Sigma}}^{-1} - \hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X}(\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}$
- $\operatorname{Var}(T) = \mathbf{G}^\top \hat{\mathbf{P}} \mathbf{G}$
- $\operatorname{Var}(T)^* = \mathbf{G}^\top \hat{\mathbf{W}} \mathbf{G}$

where $\operatorname{Var}(T)$ is a consistent estimator for $\mathbb{V}(T)$, and $\operatorname{Var}(T)^*$ is a consistent estimator for $\mathbb{V}(T|\mathbf{b}_0)$ (proof needed).

Let

$$
\tilde{\mathbf{G}} = \mathbf{G} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{G}
$$

which is the residual vector from regressing $\mathbf{G}$ on $\mathbf{X}$ using the weight matrix $\hat{\mathbf{W}}$. Then, upon substituting $\mathbf{G}$ in (10), we get

$$
\begin{aligned}
T
&= \left[ \tilde{\mathbf{G}} + \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{G} \right]^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) \\
&= \tilde{\mathbf{G}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) + \left[ (\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{G} \right]^\top \mathbf{X}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) \\
\end{aligned}
$$

According to (7), $\mathbf{X}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) = U_{\boldsymbol{\alpha}}(\hat{\boldsymbol{\alpha}}_0, \beta=0, \mathbf{y}, \hat{\mathbf{b}}_0, \mathbf{X}, \mathbf{G})$ is the conditional score function for $\boldsymbol{\alpha}$ evaluated at its MLE under the null model. By definition of the MLE, this score function equals the zero vector since the MLE solves the score equations. Thus, we have

$$
T = \mathbf{G}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) = \tilde{\mathbf{G}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) \tag{11}
$$

And there is

$$
\operatorname{Var}(T) = \mathbf{G}^\top \hat{\mathbf{P}} \mathbf{G} = \tilde{\mathbf{G}}^\top \hat{\mathbf{P}} \tilde{\mathbf{G}}
$$

$$
\operatorname{Var}(T)^* = \mathbf{G}^\top \hat{\mathbf{W}} \mathbf{G} = \tilde{\mathbf{G}}^\top \hat{\mathbf{W}} \tilde{\mathbf{G}}
$$

According to Zhou et al., 2018, the ratio between the two variance estimators

$$ r = \frac{\operatorname{Var}(T)}{\operatorname{Var}(T)^*} = \frac{\tilde{\mathbf{G}}^\top \hat{\mathbf{P}} \tilde{\mathbf{G}}}{\tilde{\mathbf{G}}^\top \hat{\mathbf{W}} \tilde{\mathbf{G}}}
$$

remains nearly constant across all tested variants.

Let the variance-adjusted test statistic be

$$
T_\mathrm{adj}=\frac{T}{\sqrt{\operatorname{Var}(T)}} = \frac{\tilde{\mathbf{G}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0)}{\sqrt{\tilde{\mathbf{G}}^\top \hat{\mathbf{P}} \tilde{\mathbf{G}}}}
$$

$$
T_\mathrm{adj}^*=\frac{T}{\sqrt{\operatorname{Var}(T)^*}} = \frac{\tilde{\mathbf{G}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0)}{\sqrt{\tilde{\mathbf{G}}^\top \hat{\mathbf{W}} \tilde{\mathbf{G}}}}
$$

## 6. SPA for CDF

### 6.1 Conditional CGF of $T_\mathrm{adj}^*$ given $\mathbf{b}_0$

For $y_i|b_{i0} \sim \text{Bernoulli}(\hat{\mu}_{i0})$, the MGF is

$$
M_{y_i}(t) = \mathbb{E}[e^{ty_i}] = (1 - \hat{\mu}_{i0}) + \hat{\mu}_{i0} e^t = 1 - \hat{\mu}_{i0} + \hat{\mu}_{i0} e^t
$$

The CGF is

$$
K_{y_i}(t) = \log M_{y_i}(t) = \log(1 - \hat{\mu}_{i0} + \hat{\mu}_{i0} e^t)
$$

For the centered variable $(y_i - \hat{\mu}_{i0})$, the CGF is
$$
K_{y_i - \hat{\mu}_{i0}}(t) = K_{y_i}(t) - t\hat{\mu}_{i0} = \log(1 - \hat{\mu}_{i0} + \hat{\mu}_{i0} e^t) - t\hat{\mu}_{i0}
$$

Given $\mathbf{b}_0$, we consider

$$
T_\mathrm{adj}^* = \sum_{i=1}^N w_i (y_i - \hat{\mu}_{i0}),\quad w_i = c\tilde{G}_i, \quad c = \frac{1}{\sqrt{\operatorname{Var}(T)^*}}
$$

as weighted sum of independent Bernoulli variables. Its CGF is
$$
\begin{aligned}
K(t)
&= \sum_{i=1}^N K_{y_i - \hat{\mu}_{i0}}(w_i t) \\
&= \sum_{i=1}^N \left[\log(1 - \hat{\mu}_{i0} + \hat{\mu}_{i0} e^{w_i t}) - w_i t \hat{\mu}_{i0}\right] \\
&= \sum_{i=1}^N \log(1 - \hat{\mu}_i + \hat{\mu}_i e^{ct\tilde{G}_i}) - ct \sum_{i=1}^N \tilde{G}_i \hat{\mu}_i
\end{aligned}
$$

### 6.2 Derivatives of the conditional CGF of $T_\mathrm{adj}^*$ given $\mathbf{b}_0$

The first derivative of $K(t)$ is

$$
\begin{aligned}
K'(t)
&= c \sum_{i=1}^N \left[\frac{\tilde{G}_i \hat{\mu}_i e^{ct\tilde{G}_i}}{1 - \hat{\mu}_i + \hat{\mu}_i e^{ct\tilde{G}_i}} - \tilde{G}_i \hat{\mu}_i\right] \\
&= c \sum_{i=1}^N \tilde{G}_i \hat{\mu}_i \left[\frac{e^{ct\tilde{G}_i} - 1}{1 - \hat{\mu}_i + \hat{\mu}_i e^{ct\tilde{G}_i}}\right]
\end{aligned}
$$

The second derivative of $K(t)$ is

$$
K''(t) = c^2 \sum_{i=1}^N \frac{\tilde{G}_i^2 \hat{\mu}_i (1 - \hat{\mu}_i) e^{ct\tilde{G}_i}}{(1 - \hat{\mu}_i + \hat{\mu}_i e^{ct\tilde{G}_i})^2}
$$

The first cumulant (mean) of $T_\mathrm{adj}^*$ is

$$
\kappa_1 = K'(0) = c \sum_{i=1}^N \tilde{G}_i \hat{\mu}_i \left[\frac{e^0 - 1}{1 - \hat{\mu}_i + \hat{\mu}_i e^0}\right] = 0
$$

The second cumulant (variance) of $T_\mathrm{adj}^*$ is

$$
\kappa_2 = K''(0) = c^2 \sum_{i=1}^N \tilde{G}_i^2 \hat{\mu}_i (1 - \hat{\mu}_i) =  c^2\tilde{\mathbf{G}}^\top \hat{\mathbf{W}} \tilde{\mathbf{G}} = 1
$$

### 6.3 SPA for CDF of $T_\mathrm{adj}$

According to Lugannani and Rice (1980), the SPA for CDF of $T_\mathrm{adj}$ is

$$
P(T_\mathrm{adj} < q) \approx \Phi\left\{w + \frac{1}{w} \log\left(\frac{v}{w}\right)\right\}
$$

where:

- $\hat{\zeta}$ is the solution of $K'(\hat{\zeta}) = q$
- $w = \operatorname{sgn}(\hat{\zeta})\sqrt{2[\hat{\zeta}q - K(\hat{\zeta})]}$
- $v = \hat{\zeta}\sqrt{K''(\hat{\zeta})}$

The relationship between the marginal distribution of $T_\mathrm{adj}$ and the conditional distribution of $T_\mathrm{adj}^*$ given $\mathbf{b}_0$ needs to be explored.
