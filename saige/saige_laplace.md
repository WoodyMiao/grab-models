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

Given $\mu_i = \operatorname{logit}^{-1}(\eta_i)$ yields

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
U_\beta(\beta; \boldsymbol{\alpha}, \mathbf{y}, \mathbf{b}, \mathbf{X}, \mathbf{G}) = \frac{\partial \ell(\boldsymbol{\alpha}, \beta; \mathbf{y}, \mathbf{b}, \mathbf{X}, \mathbf{G})}{\partial \beta} = \frac{\partial}{\partial \beta} \left[ \mathbf{y}^\top \boldsymbol{\eta} - \mathbf{1}^\top \log(1 + \exp(\boldsymbol{\eta})) \right]
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

## 4. Estimator for $(\tau_0, \boldsymbol{\alpha}_0, \mathbf{b}_0)$ under null model

Under the null model, i.e. $\beta=0$, let $\boldsymbol{\eta}_0 = \mathbf{X} \boldsymbol{\alpha}_0 + \mathbf{b}_0$ be the linear predictor, where $\boldsymbol{\alpha}_0$ is the vector of fixed effects and $\mathbf{b}_0 \sim \mathcal{N}(\mathbf{0}, \tau_0 \mathbf{\Psi})$, and let $\boldsymbol{\mu}_0 = \operatorname{logit}^{-1}(\boldsymbol{\eta}_0)$.

According to (3), the marginal log-likelihood function for $(\boldsymbol{\alpha}_0, \tau_0)$, marginalizing over $\mathbf{b}_0$, is

$$
\ell(\boldsymbol{\alpha}_0, \tau_0; \mathbf{X}, \mathbf{\Psi}, \mathbf{y}) = \log \int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) \phi(\mathbf{b}_0;\mathbf{0},\tau_0\mathbf{\Psi})\, \mathrm{d}\mathbf{b}_0 \tag{8}
$$

where

$$
p_{\mathbf{y} | \mathbf{b}}(\mathbf{y}) = \prod_{i=1}^N \mu_{i0}^{y_i} (1 - \mu_{i0})^{1 - y_i}, \quad \mu_{i0} = \frac{1}{1 + e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + b_{i0})}}
$$

Since this integral is intractable, we use the **Laplace approximation**: Taylor-expand $\log p_{\mathbf{y},\mathbf{b}_0}$ to second order around the posterior mode $\tilde{\mathbf{b}}_0$, yielding a Gaussian integral in closed form.

### 4.1 Posterior mode of $\mathbf{b}_0$

**Posterior PDF.** Under the null model, according to (4), the posterior PDF of $\mathbf{b}_0$ given $\mathbf{y}$ is

$$
p_{\mathbf{b}_0|\mathbf{y}}(\mathbf{b}_0; \mathbf{y}, \mathbf{X}, \boldsymbol{\alpha}_0, \tau_0, \mathbf{\Psi}) \propto p_{\mathbf{y}|\mathbf{b}_0}(\mathbf{y};\mathbf{b}_0,\mathbf{X},\boldsymbol{\alpha}_0)\cdot\phi(\mathbf{b}_0;\mathbf{0},\tau_0\mathbf{\Psi})
$$

**Posterior mode** of $\mathbf{b}_0$ is defined as

$$
\tilde{\mathbf{b}}_0 = \operatorname*{argmax}_{\mathbf{b}_0} p_{\mathbf{y}|\mathbf{b}_0}(\mathbf{y};\mathbf{b}_0,\mathbf{X},\boldsymbol{\alpha}_0)\cdot\phi(\mathbf{b}_0;\mathbf{0},\tau_0\mathbf{\Psi})
$$

**Log-posterior PDF.** Substituting the explicit form from (5) and the Gaussian density:

$$
\begin{aligned}
\log p_{\mathbf{b}_0|\mathbf{y}}(\mathbf{b}_0;\boldsymbol{\alpha}_0,\tau_0;\mathbf{y},\mathbf{X},\mathbf{\Psi})
  &= \ell(\boldsymbol{\alpha}_0;\mathbf{y}|\mathbf{b}_0,\mathbf{X}) + \log\phi(\mathbf{b}_0;\mathbf{0},\tau_0\mathbf{\Psi}) + \mathrm{C} \\
  &= \mathbf{y}^\top(\mathbf{X} \boldsymbol{\alpha}_0 + \mathbf{b}_0) - \mathbf{1}^\top\log(\mathbf{1}+\exp[\mathbf{X} \boldsymbol{\alpha}_0 + \mathbf{b}_0]) \\
  &\quad - \frac{N}{2}\log(2\pi) - \frac{1}{2}\log|\tau_0\mathbf{\Psi}| - \frac{1}{2}\mathbf{b}_0^\top(\tau_0\mathbf{\Psi})^{-1}\mathbf{b}_0 + \mathrm{C} \tag{9}
\end{aligned}
$$

**Gradient of the log-posterior.** Differentiating with respect to $\mathbf{b}_0$, using $\partial\boldsymbol{\eta}_0/\partial\mathbf{b}_0 = \mathbf{I}$ and $\partial\log(1+e^{\eta_i})/\partial b_{i0} = \mu_{i0}$:

$$
\nabla_{\mathbf{b}_0}\log p_{\mathbf{b}_0|\mathbf{y}} = \mathbf{y} - \boldsymbol{\mu}_0 - (\tau_0\mathbf{\Psi})^{-1}\mathbf{b}_0 \tag{9a}
$$

**First-order condition for $\tilde{\mathbf{b}}_0$.** Setting (9a) to zero at $\mathbf{b}_0 = \tilde{\mathbf{b}}_0$:

$$
\mathbf{y} - \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha}_0 + \tilde{\mathbf{b}}_0) - (\tau_0\mathbf{\Psi})^{-1}\tilde{\mathbf{b}}_0 = \mathbf{0} \tag{9b}
$$

**Hessian of the log-posterior.** Differentiating (9a) again with respect to $\mathbf{b}_0$, using $\partial\mu_{i0}/\partial b_{i0} = \mu_{i0}(1-\mu_{i0})$:

$$
\nabla^2_{\mathbf{b}_0}\log p_{\mathbf{b}_0|\mathbf{y}}
= -\operatorname{diag}\!\left(\boldsymbol{\mu}_0 \odot [\mathbf{1} - \boldsymbol{\mu}_0]\right) - (\tau_0\mathbf{\Psi})^{-1}
$$

This is negative definite everywhere (both terms are negative definite), so the log-posterior is strictly concave and the mode is unique. The Hessian of $-\log p_{\mathbf{b}_0|\mathbf{y}}$ evaluated at $\tilde{\mathbf{b}}_0$ is the positive-definite matrix:

$$
\mathbf{H} = \operatorname{diag}\!\left(\tilde{\boldsymbol{\mu}}_0 \odot [\mathbf{1} - \tilde{\boldsymbol{\mu}}_0]\right) + (\tau_0\mathbf{\Psi})^{-1} \tag{9c}
$$

where $\tilde{\boldsymbol{\mu}}_0 = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha}_0 + \tilde{\mathbf{b}}_0)$.

**Numerical computation of $\tilde{\mathbf{b}}_0$ by Newton-Raphson.** Because the log-posterior is strictly concave, Newton-Raphson converges globally to the unique mode. At each iteration, approximate the log-posterior to second order around the current estimate $\mathbf{b}^{(t)}$ and solve for the step that sets the linearised gradient to zero:

$$
\mathbf{g}^{(t)} - \mathbf{H}^{(t)}\boldsymbol{\delta}^{(t)} = \mathbf{0}
\quad\Longrightarrow\quad
\boldsymbol{\delta}^{(t)} = \left[\mathbf{H}^{(t)}\right]^{-1}\mathbf{g}^{(t)} \tag{9d}
$$

where $\mathbf{g}^{(t)} = \mathbf{y}-\boldsymbol{\mu}^{(t)} - (\tau_0\mathbf{\Psi})^{-1}\mathbf{b}^{(t)}$ follows (9a) evaluated at $\mathbf{b}^{(t)}$, and $\mathbf{H}^{(t)} = \mathbf{W}^{(t)} + (\tau_0\mathbf{\Psi})^{-1}$ follows (9c) evaluated at $\mathbf{b}^{(t)}$.

Initialise $\mathbf{b}^{(0)} = \mathbf{0}$ and iterate:

$$
\begin{aligned}
&\text{For } t = 0, 1, 2, \ldots \\
&\quad 1.\quad \boldsymbol{\mu}^{(t)} = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha}_0 + \mathbf{b}^{(t)}) \\
&\quad 2.\quad \mathbf{g}^{(t)} = (\mathbf{y} - \boldsymbol{\mu}^{(t)}) - (\tau_0\mathbf{\Psi})^{-1}\mathbf{b}^{(t)} \\
&\quad 3.\quad \mathbf{W}^{(t)} = \operatorname{diag}\!\left(\boldsymbol{\mu}^{(t)} \odot [\mathbf{1} - \boldsymbol{\mu}^{(t)}]\right) \\
&\quad 4.\quad \mathbf{H}^{(t)} = \mathbf{W}^{(t)} + (\tau_0\mathbf{\Psi})^{-1} \\
&\quad 5.\quad \text{Solve } \mathbf{H}^{(t)}\boldsymbol{\delta}^{(t)} = \mathbf{g}^{(t)} \text{ for } \boldsymbol{\delta}^{(t)} \\
&\quad 6.\quad \mathbf{b}^{(t+1)} = \mathbf{b}^{(t)} + \boldsymbol{\delta}^{(t)} \\
&\quad 7.\quad \text{Stop if } \|\boldsymbol{\delta}^{(t)}\|_\infty < \epsilon
\end{aligned} \tag{9e}
$$

At convergence, set $\tilde{\mathbf{b}}_0 = \mathbf{b}^{(t+1)}$ and $\mathbf{H} = \mathbf{H}^{(t)}$. The per-iteration cost is dominated by the $N\times N$ linear solve in step 5, which costs $O(N^3)$ naively.

### 4.2 Laplace-approximated marginal log-likelihood for $(\boldsymbol{\alpha}_0, \tau_0)$

**Step 1: Write the marginal log-likelihood as a log-integral of the joint.** Let

$$
h(\mathbf{b}_0) = \log p_{\mathbf{y},\mathbf{b}_0}(\mathbf{y},\mathbf{b}_0) = \ell(\boldsymbol{\alpha}_0;\mathbf{y}|\mathbf{b}_0) + \log\phi(\mathbf{b}_0;\mathbf{0},\tau_0\mathbf{\Psi})
$$

so that

$$
\ell(\boldsymbol{\alpha}_0,\tau_0) = \log\int_{\mathbb{R}^N} e^{h(\mathbf{b}_0)}\,\mathrm{d}\mathbf{b}_0
$$

**Step 2: Taylor-expand $h$ around the posterior mode $\tilde{\mathbf{b}}_0$.** Because $\tilde{\mathbf{b}}_0$ is the maximum of $h$, the first-order term vanishes and the expansion to second order is

$$
h(\mathbf{b}_0) \approx h(\tilde{\mathbf{b}}_0) - \frac{1}{2}(\mathbf{b}_0 - \tilde{\mathbf{b}}_0)^\top \mathbf{H}\,(\mathbf{b}_0 - \tilde{\mathbf{b}}_0)
$$

where $\mathbf{H} = -\nabla^2_{\mathbf{b}_0} h\big|_{\tilde{\mathbf{b}}_0}$ is positive definite.

**Step 3: Evaluate the Gaussian integral.** Substituting the approximation:

$$
\int_{\mathbb{R}^N} e^{h(\mathbf{b}_0)}\,\mathrm{d}\mathbf{b}_0
\approx e^{h(\tilde{\mathbf{b}}_0)} \int_{\mathbb{R}^N} \exp\!\left[-\tfrac{1}{2}(\mathbf{b}_0-\tilde{\mathbf{b}}_0)^\top\mathbf{H}(\mathbf{b}_0-\tilde{\mathbf{b}}_0)\right]\mathrm{d}\mathbf{b}_0
= e^{h(\tilde{\mathbf{b}}_0)}\,(2\pi)^{N/2}\,|\mathbf{H}|^{-1/2}
$$

**Step 4: Take the log and expand $h(\tilde{\mathbf{b}}_0)$.** Taking the log of both sides:

$$
\ell(\boldsymbol{\alpha}_0,\tau_0)
\approx h(\tilde{\mathbf{b}}_0) + \frac{N}{2}\log(2\pi) - \frac{1}{2}\log|\mathbf{H}|
$$

Expanding $h(\tilde{\mathbf{b}}_0) = \ell(\boldsymbol{\alpha}_0;\mathbf{y}|\tilde{\mathbf{b}}_0) + \log\phi(\tilde{\mathbf{b}}_0;\mathbf{0},\tau_0\mathbf{\Psi})$ and using

$$
\log\phi(\tilde{\mathbf{b}}_0;\mathbf{0},\tau_0\mathbf{\Psi})
= -\frac{N}{2}\log(2\pi) - \frac{1}{2}\log|\tau_0\mathbf{\Psi}| - \frac{1}{2}\tilde{\mathbf{b}}_0^\top(\tau_0\mathbf{\Psi})^{-1}\tilde{\mathbf{b}}_0
$$

the $\pm\tfrac{N}{2}\log(2\pi)$ terms cancel, leaving:

$$
\begin{aligned}
\ell_{\mathrm{LA}}(\boldsymbol{\alpha}_0, \tau_0; \tilde{\mathbf{b}}_0,\mathbf{y},\mathbf{X},\mathbf{\Psi})
  &= \underbrace{\mathbf{y}^\top(\mathbf{X}\boldsymbol{\alpha}_0 + \tilde{\mathbf{b}}_0) - \mathbf{1}^\top\log(\mathbf{1}+\exp[\mathbf{X}\boldsymbol{\alpha}_0 + \tilde{\mathbf{b}}_0])}_{\ell(\boldsymbol{\alpha}_0;\,\mathbf{y}\,|\,\tilde{\mathbf{b}}_0)\text{ — conditional log-lik at mode}} \\[4pt]
  &\quad - \underbrace{\frac{1}{2}\log|\tau_0\mathbf{\Psi}| - \frac{1}{2}\tilde{\mathbf{b}}_0^\top(\tau_0\mathbf{\Psi})^{-1}\tilde{\mathbf{b}}_0}_{\log\phi(\tilde{\mathbf{b}}_0;\mathbf{0},\tau_0\mathbf{\Psi})\text{ — log-prior at mode (up to constant)}} \\[4pt]
  &\quad - \underbrace{\frac{1}{2}\log|\mathbf{H}|}_{\text{Laplace volume correction}}
\end{aligned} \tag{10}
$$

### 4.3 Estimation of $\boldsymbol{\alpha}_0$: conditional MLE

With $\tau_0 = \hat{\tau}_0$ fixed, the log-likelihood for $\boldsymbol{\alpha}_0$ is the Laplace marginal log-likelihood at $\hat{\tau}_0$:

$$
\begin{aligned}
\ell(\boldsymbol{\alpha}_0; \hat{\tau}_0,\tilde{\mathbf{b}}_0,\mathbf{y},\mathbf{X},\mathbf{\Psi})
  &= \ell_{\mathrm{LA}}(\boldsymbol{\alpha}_0, \hat{\tau}_0) \\[4pt]
  &= \mathbf{y}^\top\tilde{\boldsymbol{\eta}}_0 - \mathbf{1}^\top\log(\mathbf{1}+e^{\tilde{\boldsymbol{\eta}}_0}) \\[4pt]
  &\quad - \frac{1}{2}\log|\hat{\tau}_0\mathbf{\Psi}| - \frac{1}{2}\tilde{\mathbf{b}}_0^\top(\hat{\tau}_0\mathbf{\Psi})^{-1}\tilde{\mathbf{b}}_0 \\[4pt]
  &\quad - \frac{1}{2}\log|\mathbf{H}(\boldsymbol{\alpha}_0,\hat{\tau}_0)|
\end{aligned}
$$

where $\tilde{\boldsymbol{\eta}}_0 = \mathbf{X}\boldsymbol{\alpha}_0 + \tilde{\mathbf{b}}_0(\boldsymbol{\alpha}_0,\hat{\tau}_0)$ depends on $\boldsymbol{\alpha}_0$ through the posterior mode. The ML estimator is

$$
\hat{\boldsymbol{\alpha}}_0 = \operatorname*{argmax}_{\boldsymbol{\alpha}_0}\, \ell_{\mathrm{LA}}(\boldsymbol{\alpha}_0, \hat{\tau}_0)
$$

solved by Newton-Raphson. By analogy with the conditional score (7) for $\boldsymbol{\alpha}$, the marginalised score (10a) and observed information (10b) are the Schur complement of the joint Hessian in $(\boldsymbol{\alpha}_0,\mathbf{b}_0)$:

$$
\nabla_{\boldsymbol{\alpha}_0}\ell_{\mathrm{LA}} = \mathbf{X}^\top(\mathbf{y} - \tilde{\boldsymbol{\mu}}_0) \tag{10a}
$$

$$
-\nabla^2_{\boldsymbol{\alpha}_0}\ell_{\mathrm{LA}} = \underbrace{\mathbf{X}^\top\tilde{\mathbf{W}}\mathbf{X}}_{\text{naive info}} - \underbrace{\mathbf{X}^\top\tilde{\mathbf{W}}\mathbf{H}^{-1}\tilde{\mathbf{W}}\mathbf{X}}_{\text{correction for } \mathbf{b}_0\text{ uncertainty}} \tag{10b}
$$

where $\tilde{\mathbf{W}} = \operatorname{diag}(\tilde{\boldsymbol{\mu}}_0\odot[\mathbf{1}-\tilde{\boldsymbol{\mu}}_0])$ and $\mathbf{H}$ is from (9c).

**Numerical computation of $\hat{\boldsymbol{\alpha}}_0$ by Newton-Raphson.** Starting from the intercept-only logistic GLM estimate $\boldsymbol{\alpha}_0^{(0)}$ (the $\tau_0\to 0$ limit), iterate:

$$
\begin{aligned}
&\text{For } t = 0, 1, 2, \ldots \\
&\quad 1.\quad (\tilde{\mathbf{b}}_0^{(t)}, \mathbf{H}^{(t)}) = \texttt{b\_tilde}(\boldsymbol{\alpha}_0^{(t)}, \hat{\tau}_0) \quad \text{[inner NR via (9e)]} \\
&\quad 2.\quad \tilde{\boldsymbol{\mu}}_0^{(t)} = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha}_0^{(t)} + \tilde{\mathbf{b}}_0^{(t)}) \\
&\quad 3.\quad \tilde{\mathbf{W}}^{(t)} = \operatorname{diag}\!\left(\tilde{\boldsymbol{\mu}}_0^{(t)} \odot [\mathbf{1} - \tilde{\boldsymbol{\mu}}_0^{(t)}]\right) \\
&\quad 4.\quad \mathbf{s}^{(t)} = \mathbf{X}^\top(\mathbf{y} - \tilde{\boldsymbol{\mu}}_0^{(t)}) \\
&\quad 5.\quad \mathcal{I}^{(t)} = \mathbf{X}^\top\tilde{\mathbf{W}}^{(t)}\mathbf{X} - \mathbf{X}^\top\tilde{\mathbf{W}}^{(t)}[\mathbf{H}^{(t)}]^{-1}\tilde{\mathbf{W}}^{(t)}\mathbf{X} \\
&\quad 6.\quad \text{Solve } \mathcal{I}^{(t)}\boldsymbol{\delta}^{(t)} = \mathbf{s}^{(t)} \text{ for } \boldsymbol{\delta}^{(t)} \\
&\quad 7.\quad \boldsymbol{\alpha}_0^{(t+1)} = \boldsymbol{\alpha}_0^{(t)} + \boldsymbol{\delta}^{(t)} \\
&\quad 8.\quad \text{Stop if } \|\boldsymbol{\delta}^{(t)}\|_\infty < \epsilon
\end{aligned} \tag{10c}
$$

At convergence, set $\hat{\boldsymbol{\alpha}}_0 = \boldsymbol{\alpha}_0^{(t+1)}$ and $\tilde{\mathbf{W}} = \tilde{\mathbf{W}}^{(t)}$, $\mathbf{H} = \mathbf{H}^{(t)}$.

### 4.4 Estimation of $\tau_0$: Laplace-REML

Let $\hat{\boldsymbol{\alpha}}_0 = \hat{\boldsymbol{\alpha}}_0(\tau_0) = \operatorname*{argmax}_{\boldsymbol{\alpha}_0}\ell_{\mathrm{LA}}(\boldsymbol{\alpha}_0,\tau_0)$ be the profile MLE at fixed $\tau_0$, and let $\tilde{\mathbf{b}}_0 = \tilde{\mathbf{b}}_0(\hat{\boldsymbol{\alpha}}_0(\tau_0),\tau_0)$ be the corresponding posterior mode. Let

$$
\qquad \tilde{\mathbf{W}} = \tilde{\mathbf{W}}(\hat{\boldsymbol{\alpha}}_0(\tau_0),\tilde{\mathbf{b}}_0[\hat{\boldsymbol{\alpha}}_0(\tau_0),\tau_0]) = \operatorname{diag}(\operatorname{logit}^{-1}(\mathbf{X}\hat{\boldsymbol{\alpha}}_0 + \tilde{\mathbf{b}}_0) \odot [\mathbf{1} - \operatorname{logit}^{-1}(\mathbf{X}\hat{\boldsymbol{\alpha}}_0 + \tilde{\mathbf{b}}_0)])
$$

The REML correction $-\tfrac{1}{2}\log|\mathbf{X}^\top[\tilde{\mathbf{W}}+(\tau_0\mathbf{\Psi})^{-1}]^{-1}\mathbf{X}|$ is the log-determinant of the observed information for $\boldsymbol{\alpha}_0$ under the Laplace approximation, which integrates out the uncertainty in $\hat{\boldsymbol{\alpha}}_0$ and removes the $q$ degrees-of-freedom bias in $\hat{\tau}_0$. The Laplace-REML log-likelihood is

$$
\begin{aligned}
\ell_{\mathrm{REML}}(\tau_0)
  &= \mathbf{y}^\top(\mathbf{X}\hat{\boldsymbol{\alpha}}_0 + \tilde{\mathbf{b}}_0) - \mathbf{1}^\top\log\!\left(\mathbf{1}+e^{\mathbf{X}\hat{\boldsymbol{\alpha}}_0 + \tilde{\mathbf{b}}_0}\right) \\[4pt]
  &\quad - \frac{1}{2}\log|\tau_0\mathbf{\Psi}| - \frac{1}{2}\tilde{\mathbf{b}}_0^\top(\tau_0\mathbf{\Psi})^{-1}\tilde{\mathbf{b}}_0 \\[4pt]
  &\quad - \frac{1}{2}\log\!\left|\tilde{\mathbf{W}} + (\tau_0\mathbf{\Psi})^{-1}\right| \\[4pt]
  &\quad - \frac{1}{2}\log\!\left|\mathbf{X}^\top\!\left[\tilde{\mathbf{W}} + (\tau_0\mathbf{\Psi})^{-1}\right]^{-1}\!\mathbf{X}\right|
\end{aligned} \tag{11}
$$

Note: $\ell_{\mathrm{REML}}(\tau_0)$ in (11) is an explicit formula once $\hat{\boldsymbol{\alpha}}_0(\tau_0)$ and $\tilde{\mathbf{b}}_0(\tau_0)$ are in hand — but computing those profile quantities at any given $\tau_0$ requires running the two inner Newton-Raphson loops to convergence. It is therefore **not** a closed-form function of $\tau_0$ alone. The REML estimator for $\tau_0$ is

$$
\hat{\tau}_0 = \operatorname*{argmax}_{\tau_0 > 0}\, \ell_{\mathrm{REML}}(\tau_0)
$$

computed numerically by Brent's method applied to $-\ell_{\mathrm{REML}}(e^s)$ over $s\in\mathbb{R}$ (log-scale search ensures $\tau_0 > 0$).

**Numerical computation of $\hat{\tau}_0$ by Brent's method.** Let $f(s) = -\ell_{\mathrm{REML}}(e^s)$ be the scalar objective (no derivative required).

**Bracketing.** Find $a < s_0 < b$ with $f(a) > f(s_0)$ and $f(b) > f(s_0)$, so the minimum of $f$ lies in $(a,b)$. Start from $s_0 = 0$ ($\tau_0 = 1$) and step outward by 1 until both sides are established.

**Brent iteration.** Maintain bracket $[a,b]$, best point $x$ (current minimiser of $f$), second-best $w$, previous second-best $v$, and size $e$ of the previous accepted parabolic step ($e=0$ initially):

$$
\begin{aligned}
&\text{Initialize: } x = w = v = s_0,\quad f_x = f_w = f_v = f(s_0),\quad e = 0 \\
&\text{For } t = 0, 1, 2, \ldots \\
&\quad 1.\quad m = \tfrac{1}{2}(a + b) \\
&\quad 2.\quad \text{Stop if } b - a < \epsilon_s(1 + |x|) \\
&\quad 3.\quad \text{If } |e| > 0, \text{ attempt parabolic interpolation through } (v,f_v),(w,f_w),(x,f_x): \\
&\qquad\quad \delta_{\mathrm{para}} = -\frac{1}{2}\,\frac{(x-w)^2(f_x-f_v)-(x-v)^2(f_x-f_w)}{(x-w)(f_x-f_v)-(x-v)(f_x-f_w)} \\
&\qquad\quad \text{Accept if } x + \delta_{\mathrm{para}} \in (a,b) \text{ and } |\delta_{\mathrm{para}}| < \tfrac{1}{2}|e|;\text{ set } \delta = \delta_{\mathrm{para}},\;e \leftarrow \delta \\
&\quad 4.\quad \text{If parabola rejected, golden-section into the longer sub-interval:} \\
&\qquad\quad e \leftarrow \begin{cases}a - x & x \geq m \\ b - x & x < m\end{cases}, \qquad \delta = (1 - \phi^{-1})\,e \approx 0.382\,e \\
&\quad 5.\quad u = x + \delta, \quad f_u = f(u) \quad \text{[one evaluation of } -\ell_{\mathrm{REML}}(e^u)\text{, via alpha\_hat}(e^u)\text{]} \\
&\quad 6.\quad \text{Update bracket:} \\
&\qquad\quad \text{if } f_u \leq f_x\text{: if } u < x \text{ then } b \leftarrow x \text{ else } a \leftarrow x \\
&\qquad\quad \text{else: if } u < x \text{ then } a \leftarrow u \text{ else } b \leftarrow u \\
&\quad 7.\quad \text{Update best/second-best:} \\
&\qquad\quad \text{if } f_u \leq f_x\text{:}\quad (v,f_v) \leftarrow (w,f_w);\;(w,f_w) \leftarrow (x,f_x);\;(x,f_x) \leftarrow (u,f_u) \\
&\qquad\quad \text{elif } f_u \leq f_w \text{ or } w = x\text{:}\quad (v,f_v) \leftarrow (w,f_w);\;(w,f_w) \leftarrow (u,f_u) \\
&\qquad\quad \text{else:}\quad (v,f_v) \leftarrow (u,f_u)
\end{aligned} \tag{11a}
$$

At convergence: $\hat{s} = x$, $\hat{\tau}_0 = e^{\hat{s}}$.

### 4.5 BLUP of $\mathbf{b}_0$: Gauss-Hermite quadrature

The BLUP of $\mathbf{b}_0$ is the posterior mean given (4), evaluated at $(\hat{\boldsymbol{\alpha}}_0, \hat{\tau}_0)$:

$$
\hat{\mathbf{b}}_0 = \mathbb{E}[\mathbf{b}_0 \mid \mathbf{y}, \hat{\boldsymbol{\alpha}}_0, \hat{\tau}_0]
= \frac{\displaystyle\int_{\mathbb{R}^N} \mathbf{b}_0\, p_{\mathbf{y} | \mathbf{b}}(\mathbf{y})\,\phi(\mathbf{b}_0;\mathbf{0},\hat{\tau}_0\mathbf{\Psi})\,\mathrm{d}\mathbf{b}_0}{\displaystyle\int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y})\,\phi(\mathbf{b}_0;\mathbf{0},\hat{\tau}_0\mathbf{\Psi})\,\mathrm{d}\mathbf{b}_0} \tag{12}
$$

Direct $N$-dimensional quadrature is infeasible. Instead, for each component $i$ we apply a **1D mean-field approximation**: fix $b_{j0} = \tilde{b}_{j0}$ (joint posterior mode) for all $j\neq i$, and integrate over $b_{i0}$ alone using the unnormalised conditional density

$$
f_i(b_{i0}) \propto \exp\!\Big[\ell(\hat{\boldsymbol{\alpha}}_0;\mathbf{y}|\mathbf{b}^{(i)}) + \log\phi(\mathbf{b}^{(i)};\mathbf{0},\hat{\tau}_0\mathbf{\Psi})\Big]
$$

where $\mathbf{b}^{(i)}$ denotes $\tilde{\mathbf{b}}_0$ with its $i$th entry replaced by $b_{i0}$. Expanding and dropping terms constant in $b_{i0}$:

$$
\log f_i(b_{i0}) \propto \sum_{j=1}^N \left[y_j\eta_j^{(i)} - \log(1+e^{\eta_j^{(i)}})\right] - \frac{1}{2}(\mathbf{b}^{(i)})^\top(\hat{\tau}_0\mathbf{\Psi})^{-1}\mathbf{b}^{(i)} \tag{12b}
$$

where $\eta_j^{(i)} = \mathbf{X}_j\hat{\boldsymbol{\alpha}}_0 + b_j^{(i)}$.

To apply $K$-point **Gauss-Hermite quadrature**, centre and scale at the marginal mode with $\sigma_i = H_{ii}^{-1/2}$ (from the diagonal of $\mathbf{H}$). Substitute $b_{i0} = \tilde{b}_{i0} + \sqrt{2}\,\sigma_i\, t$, so the Gaussian factor $\phi(b_{i0})$ contributes $e^{-t^2}$ (absorbed into the GH weights). The $K$ quadrature nodes and unnormalised weights are

$$
b_{i0}^{(k)} = \tilde{b}_{i0} + \sqrt{2}\,\sigma_i\, t_k, \qquad
\tilde{w}_{ik} = w_k \cdot \exp\!\left[\log f_i\!\left(b_{i0}^{(k)}\right) + t_k^2\right], \quad k = 1,\ldots,K \tag{12c}
$$

where $(t_k, w_k)$ are the standard Gauss-Hermite nodes and weights satisfying $\int f(t)e^{-t^2}\mathrm{d}t \approx \sum_k w_k f(t_k)$. The $t_k^2$ term in the exponent cancels the $e^{-t^2}$ factor so that the weights only encode $f_i$. The posterior mean for component $i$ is then approximated as

$$
\hat{b}_{i0} \approx \frac{\displaystyle\sum_{k=1}^K \tilde{w}_{ik}\, b_{i0}^{(k)}}{\displaystyle\sum_{k=1}^K \tilde{w}_{ik}} \tag{12d}
$$

Let

$$
\hat{\boldsymbol{\mu}}_0 = \operatorname{logit}^{-1}(\mathbf{X}\hat{\boldsymbol{\alpha}}_0 + \hat{\mathbf{b}}_0) \tag{13}
$$

### 4.6 Algorithm summary: pseudocode

The three estimates $\hat{\tau}_0$, $\hat{\boldsymbol{\alpha}}_0$, $\hat{\mathbf{b}}_0$ are computed in a nested three-level structure. Every evaluation of $\ell_{\mathrm{REML}}(\tau_0)$ requires profiling out $\boldsymbol{\alpha}_0$ via inner Newton-Raphson using (10a) and (10b), which in turn calls $\tilde{\mathbf{b}}_0(\boldsymbol{\alpha}_0,\tau_0)$ at each step via inner-inner Newton-Raphson using (9a), (9c), (9d), (9e). Although (11) is an explicit formula once the profile quantities are in hand, it is **not** a closed-form function of $\tau_0$ alone — computing it at any $\tau_0$ requires running both inner loops to convergence. The formula-to-level mapping is:

| Level | Task | Formulas |
|-------|------|----------|
| Inner-inner | NR for $\tilde{\mathbf{b}}_0(\boldsymbol{\alpha}_0,\tau_0)$, solving (9b) | (9a) $\mathbf{g}^{(t)}$, (9c) $\mathbf{H}^{(t)}$, (9d) Newton step, (9e) convergence |
| Inner | NR for $\hat{\boldsymbol{\alpha}}_0(\tau_0) = \operatorname{argmax}_{\boldsymbol{\alpha}_0}\ell_{\mathrm{LA}}$ | (10) objective, (10a) score, (10b) observed information |
| Outer | Brent for $\hat{\tau}_0 = \operatorname{argmax}_{\tau_0}\ell_{\mathrm{REML}}$ | (11), (11a) |
| Post-estimation | GH quadrature for $\hat{\mathbf{b}}_0$ | (12b) log $f_i$, (12c) nodes/weights, (12d) mean |
| Post-estimation | $\hat{\boldsymbol{\mu}}_0$ | (13) |

```pseudocode
# ----------------------------------------------------------------
# def b_tilde(𝛂₀, τ₀) → (𝐛₀, 𝐇)
# Posterior mode of 𝐛₀ at fixed (𝛂₀, τ₀), solving FOC (9b) via NR.
# NR uses gradient (9a) and its derivative (9c) — the Hessian.
# scipy: scipy.optimize.minimize(
#   fun=lambda b: -(y @ (X@α₀+b) - np.log1p(np.exp(X@α₀+b)).sum()
#                  - 0.5*b @ solve(τ₀*Ψ, b)),
#   x0=np.zeros(N), jac=neg_grad_9a, hess=hess_9c, method='Newton-CG'
# )
# ----------------------------------------------------------------
def b_tilde(𝛂₀, τ₀):
    𝐛₀⁽⁰⁾ = 𝟎
    while True:
        𝛍⁽ᵗ⁾ = logit⁻¹(𝐗𝛂₀ + 𝐛₀⁽ᵗ⁾)
        𝐠⁽ᵗ⁾ = (𝐲 − 𝛍⁽ᵗ⁾) − (τ₀𝚿)⁻¹𝐛₀⁽ᵗ⁾  # (9a)
        𝐖⁽ᵗ⁾ = diag(𝛍⁽ᵗ⁾ ⊙ [𝟏 − 𝛍⁽ᵗ⁾])
        𝐇⁽ᵗ⁾ = 𝐖⁽ᵗ⁾ + (τ₀𝚿)⁻¹            # (9c)
        𝛅⁽ᵗ⁾ = solve(𝐇⁽ᵗ⁾, 𝐠⁽ᵗ⁾)          # (9d)
        𝐛₀⁽ᵗ⁺¹⁾ = 𝐛₀⁽ᵗ⁾ + 𝛅⁽ᵗ⁾
        if ‖𝛅⁽ᵗ⁾‖∞ < ε:
            return 𝐛₀⁽ᵗ⁺¹⁾, 𝐇⁽ᵗ⁾

# ----------------------------------------------------------------
# def alpha_hat(τ₀) → (𝛂₀, 𝐛₀, 𝐇, 𝐖)
# Profile MLE of 𝛂₀ at fixed τ₀ via NR on ℓ_LA(𝛂₀, τ₀).
# Init: intercept-only logistic GLM (τ₀→0 limit).
# scipy: scipy.optimize.minimize(
#   fun=lambda alpha: -ell_LA(alpha, τ₀),
#   x0=logistic_glm(y, X), jac=neg_score_10a, hess=obs_info_10b, method='Newton-CG'
# )
# ----------------------------------------------------------------
def alpha_hat(τ₀):
    𝛂₀ = logistic_glm(𝐲, 𝐗)
    while True:
        (𝐛₀, 𝐇) = b_tilde(𝛂₀, τ₀)      # (9e)
        𝛍₀ = logit⁻¹(𝐗𝛂₀ + 𝐛₀)
        𝐖  = diag(𝛍₀ ⊙ [𝟏 − 𝛍₀])
        𝐬  = 𝐗ᵀ(𝐲 − 𝛍₀)                 # (10a)
        ℐ  = 𝐗ᵀ𝐖𝐗 − 𝐗ᵀ𝐖𝐇⁻¹𝐖𝐗         # (10b)
        𝛅  = solve(ℐ, 𝐬)
        𝛂₀ = 𝛂₀ + 𝛅
        if ‖𝛅‖∞ < ε:
            return 𝛂₀, 𝐛₀, 𝐇, 𝐖

# ----------------------------------------------------------------
# def tau_hat() → (τ₀, 𝛂₀, 𝐛₀, 𝐇, 𝐖)
# REML estimator for τ₀: Brent's method on s = log τ₀ ∈ ℝ.
# Each f(s) evaluation calls alpha_hat(exp(s)) (two inner NR loops).
# scipy: scipy.optimize.minimize_scalar(
#   fun=lambda s: -ell_reml(exp(s)), method='brent'
# )
# ----------------------------------------------------------------
def tau_hat():
    def f(s):                                           # (11): −ℓ_REML(exp(s))
        (𝛂₀, 𝐛₀, 𝐇, 𝐖) = alpha_hat(exp(s))
        η₀ = 𝐗𝛂₀ + 𝐛₀
        return -(𝐲ᵀη₀ − 𝟏ᵀlog(𝟏+exp(η₀))
               − ½log|exp(s)·𝚿| − ½𝐛₀ᵀ(exp(s)·𝚿)⁻¹𝐛₀
               − ½log|𝐇| − ½log|𝐗ᵀ𝐇⁻¹𝐗|)

    # Bracketing: start at s₀=0, step outward by 1 until f(a) > f(s₀) and f(b) > f(s₀)
    s₀ = 0.0;  f₀ = f(s₀)
    a = s₀ − 1;  while f(a) ≤ f₀: a −= 1
    b = s₀ + 1;  while f(b) ≤ f₀: b += 1

    # Brent's method — (11a)
    x = w = v = s₀;  fx = fw = fv = f₀;  e = 0
    while True:
        m = ½(a + b)
        if b − a < ε_s·(1 + |x|): break                        # (11a, step 2)
        δ_para = −½·[(x−w)²(fx−fv)−(x−v)²(fx−fw)] /           # (11a, step 3)
                     [(x−w)(fx−fv)−(x−v)(fx−fw)]
        if |e| > 0 and x+δ_para ∈ (a,b) and |δ_para| < ½|e|:  # accept parabola
            δ = δ_para;  e = δ
        else:                                                    # golden section (11a, step 4)
            e = (a−x) if x ≥ m else (b−x)
            δ = (1−φ⁻¹)·e   # φ = golden ratio ≈ 1.618
        u = x + δ;  fu = f(u)                                   # (11a, step 5)
        if fu ≤ fx:                                              # (11a, step 6)
            if u < x: b ← x  else: a ← x
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu           # (11a, step 7)
        else:
            if u < x: a ← u  else: b ← u
            if fu ≤ fw or w == x: v, fv, w, fw = w, fw, u, fu
            else: v, fv = u, fu

    return exp(x)

# --- Step 1: τ₀ ---
τ₀ = tau_hat()

# --- Step 2: 𝛂₀ ---
𝛂₀, 𝐛₀, 𝐇, 𝐖 = alpha_hat(τ₀)

# --- Step 3: 𝐛₀ ---
def log_fi(i, b_val):                               # (12b)
    𝐛⁽ⁱ⁾ = 𝐛₀.copy();  𝐛⁽ⁱ⁾[i] ← b_val              # mean-field: j≠i fixed at mode
    η⁽ⁱ⁾ = 𝐗𝛂₀ + 𝐛⁽ⁱ⁾                               # ηⱼ⁽ⁱ⁾ = 𝐗ⱼ𝛂₀ + bⱼ⁽ⁱ⁾
    return  𝐲ᵀη⁽ⁱ⁾ − 𝟏ᵀlog(𝟏+exp(η⁽ⁱ⁾))             # cond. log-lik, from (5)
          − ½𝐛⁽ⁱ⁾ᵀ(τ₀𝚿)⁻¹𝐛⁽ⁱ⁾                      # log-prior quadratic form

(tₖ, wₖ) = hermgauss(K)                             # standard K-point GH rule
σᵢ = 𝐇ᵢᵢ^{-1/2}  for i=1,…,N                       # (9c): σᵢ at joint mode
b_blup = 𝟎
for i = 1,…,N:
    nodes_k = 𝐛₀[i] + √2·σᵢ·tₖ,  k=1,…,K           # (12c): GH quadrature nodes
    log_wik = log(wₖ) + log_fi(i, nodes_k) + tₖ²    # (12c): unnorm. log-weights
    wik     = exp(log_wik − max_k log_wik)         # numerical stabilisation
    b_blup[i] = (Σₖ wik·nodes_k) / (Σₖ wik)         # (12d): posterior mean
𝐛₀ = b_blup                                        # update MAP → BLUP

# --- Step 4: 𝛍₀ ---
𝛍₀ = logit⁻¹(𝐗𝛂₀ + 𝐛₀)                            # (13)
```

## 5. Score statistic

### 5.1 Score statistic and its variance

Let the score statistic for $\beta$ be

$$
T = U_\beta(\beta=0, \hat{\boldsymbol{\alpha}}_0, \mathbf{y}, \hat{\mathbf{b}}_0, \mathbf{X}, \mathbf{G}) = \mathbf{G}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) \tag{14}
$$

As $N \to \infty$,

- $\hat{\boldsymbol{\alpha}}_0 \to \boldsymbol{\alpha}_0$
- $\mathbb{V}(\hat{\boldsymbol{\mu}}_0|\mathbf{b}_0) \to 0$
- $\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0|\mathbf{b}_0) \to \mathbb{V}(\mathbf{y}|\mathbf{b}_0) = \operatorname{diag}(\boldsymbol{\mu}_0 \odot [\mathbf{1} - \boldsymbol{\mu}_0])$
- $\mathbb{E}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0|\mathbf{b}_0) = \boldsymbol{\mu}_0 - \mathbb{E}(\hat{\boldsymbol{\mu}}_0|\mathbf{b}_0) \to \boldsymbol{0}$

Therefore,

$$
\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0) = \mathbb{E}_{\mathbf{b}_0}[\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0|\mathbf{b}_0)] + \mathbb{V}_{\mathbf{b}_0}[\mathbb{E}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0|\mathbf{b}_0)] \approx \mathbb{E}_{\mathbf{b}_0}(\mathbf{W})
$$

where $\mathbf{W} = \operatorname{diag}(\boldsymbol{\mu}_0 \odot [\mathbf{1} - \boldsymbol{\mu}_0])$, with the $i$th element in the diagonal being

$$
W_{ii} = \frac{e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + b_{i0})}}{(1 + e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + b_{i0})})^2},\quad b_{i0} \sim N(0,\tau\mathbf{\Psi}_{ii})
$$

Therefore, $\mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0)$ is a diagonal matrix, with the $i$th element in the diagonal being

$$
\mathbb{E}_{b_{i0}}(W_{ii}) = \int_{-\infty}^{\infty} \frac{e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + b_{i0})}}{(1 + e^{-(\mathbf{X}_i\boldsymbol{\alpha}_0 + b_{i0})})^2} \cdot \frac{1}{\sqrt{2\pi \tau \mathbf{\Psi}_{ii}}} \exp\left(-\frac{b_{i0}^2}{2\tau \mathbf{\Psi}_{ii}}\right) \mathrm{d}b_{i0}
$$

Then, according to (14), we have

$$
\mathbb{V}(T) = \mathbf{G}^\top \mathbb{V}(\mathbf{y} - \hat{\boldsymbol{\mu}}_0) \mathbf{G} \approx \mathbf{G}^\top \mathbb{E}_\mathbf{b_0}(\mathbf{W}) \mathbf{G} \tag{15}
$$

### 5.2 Covariate-adjusted genotype vector

Let

$$
\tilde{\mathbf{G}} = \mathbf{G} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{G}
$$

which is the residual vector from regressing $\mathbf{G}$ on $\mathbf{X}$ using the weight matrix $\hat{\mathbf{W}}$. Then, upon substituting $\mathbf{G}$ in (14), we get

$$
\begin{aligned}
T
&= \left[ \tilde{\mathbf{G}} + \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{G} \right]^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) \\
&= \tilde{\mathbf{G}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) + \left[ (\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{G} \right]^\top \mathbf{X}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) \\
\end{aligned}
$$

According to (7), $\mathbf{X}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) = U_{\boldsymbol{\alpha}}(\hat{\boldsymbol{\alpha}}_0, \beta=0, \mathbf{y}, \hat{\mathbf{b}}_0, \mathbf{X}, \mathbf{G})$ is the conditional score function for $\boldsymbol{\alpha}$ evaluated at its MLE under the null model. By definition of the MLE, this score function equals the zero vector since the MLE solves the score equations. Thus, we have

$$
T = \mathbf{G}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) = \tilde{\mathbf{G}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0) \tag{16}
$$

### 5.3 Variance-adjusted score statistic

Let

- $\hat{\mathbf{W}} = \operatorname{diag}(\hat{\boldsymbol{\mu}}_0 \odot [\mathbf{1} - \hat{\boldsymbol{\mu}}_0])$
- $\hat{\boldsymbol{\Sigma}} = \hat{\mathbf{W}}^{-1} + \hat{\tau}\mathbf{\Psi}$
- $\hat{\mathbf{P}} = \hat{\boldsymbol{\Sigma}}^{-1} - \hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X}(\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}$

Let

- $\hat{\mathbb{V}}(T) = \tilde{\mathbf{G}}^\top \hat{\mathbf{P}} \tilde{\mathbf{G}}$
- $\hat{\mathbb{V}}(T|\mathbf{b}_0) = \tilde{\mathbf{G}}^\top \hat{\mathbf{W}} \tilde{\mathbf{G}}$

be estimators for $\mathbb{V}(T)$ and $\mathbb{V}(T|\mathbf{b}_0)$. According to Zhou et al. (2018), the ratio between the two variance estimators

$$ r = \frac{\tilde{\mathbf{G}}^\top \hat{\mathbf{P}} \tilde{\mathbf{G}}}{\tilde{\mathbf{G}}^\top \hat{\mathbf{W}} \tilde{\mathbf{G}}}
$$

remains nearly constant across all tested variants.

Let the variance-adjusted test statistic be

$$
T_\mathrm{adj}=\frac{T}{\sqrt{\hat{\mathbb{V}}(T)}} = \frac{\tilde{\mathbf{G}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0)}{\sqrt{\tilde{\mathbf{G}}^\top \hat{\mathbf{P}} \tilde{\mathbf{G}}}} = \frac{\tilde{\mathbf{G}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}_0)}{\sqrt{r\tilde{\mathbf{G}}^\top \hat{\mathbf{W}} \tilde{\mathbf{G}}}} \tag{17}
$$

## 6. SPA for CDF of $T_\mathrm{adj}$

### 6.1 Conditional CGF of $\sqrt{r}T_\mathrm{adj}$ given $\mathbf{b}_0$ fixed at $\hat{\mathbf{b}}_0$

For $y_i \mid b_{i0}=\hat{b}_{i0} \sim \text{Bernoulli}(\hat{\mu}_{i0})$, the MGF is

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

Given $\mathbf{b}_0 = \hat{\mathbf{b}}_0$, we consider

$$
\sqrt{r}T_\mathrm{adj} = \sum_{i=1}^N w_i (y_i - \hat{\mu}_{i0}),\quad w_i = c\tilde{G}_i, \quad c = \frac{1}{\sqrt{\hat{\mathbb{V}}(T|\mathbf{b}_0)}}
$$

as weighted sum of independent Bernoulli variables. Its CGF is

$$
\begin{aligned}
K(t)
&= \sum_{i=1}^N K_{y_i - \hat{\mu}_{i0}}(w_i t) \\
&= \sum_{i=1}^N \left[\log(1 - \hat{\mu}_{i0} + \hat{\mu}_{i0} e^{w_i t}) - w_i t \hat{\mu}_{i0}\right] \\
&= \sum_{i=1}^N \log(1 - \hat{\mu}_{i0} + \hat{\mu}_{i0} e^{ct\tilde{G}_i}) - ct \sum_{i=1}^N \tilde{G}_i \hat{\mu}_{i0}
\end{aligned} \tag{18}
$$

### 6.2 Derivatives of the conditional CGF

The first derivative of $K(t)$ is

$$
\begin{aligned}
K'(t)
&= c \sum_{i=1}^N \left[\frac{\tilde{G}_i \hat{\mu}_{i0} e^{ct\tilde{G}_i}}{1 - \hat{\mu}_{i0} + \hat{\mu}_{i0} e^{ct\tilde{G}_i}} - \tilde{G}_i \hat{\mu}_{i0}\right] \\
&= c \sum_{i=1}^N \tilde{G}_i \hat{\mu}_{i0} \left[\frac{e^{ct\tilde{G}_i} - 1}{1 - \hat{\mu}_{i0} + \hat{\mu}_{i0} e^{ct\tilde{G}_i}}\right]
\end{aligned} \tag{19}
$$

The second derivative of $K(t)$ is

$$
K''(t) = c^2 \sum_{i=1}^N \frac{\tilde{G}_i^2 \hat{\mu}_{i0} (1 - \hat{\mu}_{i0}) e^{ct\tilde{G}_i}}{(1 - \hat{\mu}_{i0} + \hat{\mu}_{i0} e^{ct\tilde{G}_i})^2} \tag{20}
$$

The first cumulant (mean) of $\sqrt{r}T_\mathrm{adj} \mid \mathbf{b}_0 = \hat{\mathbf{b}}_0$ is

$$
\kappa_1 = K'(0) = c \sum_{i=1}^N \tilde{G}_i \hat{\mu}_{i0} \left[\frac{e^0 - 1}{1 - \hat{\mu}_{i0} + \hat{\mu}_{i0} e^0}\right] = 0
$$

The second cumulant (variance) of $\sqrt{r}T_\mathrm{adj} \mid \mathbf{b}_0 = \hat{\mathbf{b}}_0$ is

$$
\kappa_2 = K''(0) = c^2 \sum_{i=1}^N \tilde{G}_i^2 \hat{\mu}_{i0} (1 - \hat{\mu}_{i0}) =  c^2\tilde{\mathbf{G}}^\top \hat{\mathbf{W}} \tilde{\mathbf{G}} = 1
$$

Therefore, $\mathbb{V}(\sqrt{r}T_\mathrm{adj} \mid \mathbf{b}_0 = \hat{\mathbf{b}}_0) = 1$.

### 6.3 Approximated SPA for CDF of $T_\mathrm{adj}$ by SAIGE

According to Zhou et al. (2018), the CDF of $T_\mathrm{adj}$ is well-approximated by the CDF of $\sqrt{r}T_\mathrm{adj}$ conditional on $\mathbf{b}_0 = \hat{\mathbf{b}}_0$.

According to Lugannani and Rice (1980), and equations (18)-(20), the SPA for CDF of $T_\mathrm{adj}$ is

$$
P(T_\mathrm{adj} < q) \approx \Phi\left\{w + \frac{1}{w} \log\left(\frac{v}{w}\right)\right\} \tag{21}
$$

where:

- $\hat{\zeta}$ is the solution of $K'(\hat{\zeta}) = q$
- $w = \operatorname{sgn}(\hat{\zeta})\sqrt{2[\hat{\zeta}q - K(\hat{\zeta})]}$
- $v = \hat{\zeta}\sqrt{K''(\hat{\zeta})}$
