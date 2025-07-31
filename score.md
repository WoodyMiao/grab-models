# Appendix

## Derivation of the score function for $\beta$

Given the likelihood function

$$
\ell(\beta; \mathbf{y}, \mathbf{g}, \hat{\boldsymbol{\eta}}) = \mathbf{y}^\top (\mathbf{g}\beta + \hat{\boldsymbol{\eta}}) - \mathbf{1}^\top \log[1 + \exp(\mathbf{g}\beta + \hat{\boldsymbol{\eta}})]
$$

We want to derive an explicit expression for

$$
U_\beta(\beta) = \frac{\partial \ell}{\partial \beta}
$$

The partial derivative for the first term is

$$
\frac{\partial}{\partial \beta} \left[ \mathbf{y}^\top (\mathbf{g}\beta + \hat{\boldsymbol{\eta}}) \right] = \mathbf{y}^\top \mathbf{g}
$$

For the second term is

$$
\frac{\partial}{\partial \beta} \left[ -\mathbf{1}^\top \log(1 + \exp(\mathbf{g}\beta + \hat{\boldsymbol{\eta}})) \right] = -\frac{\partial}{\partial \beta} \sum_{i=1}^N \log(1 + \exp(g_i\beta + \hat{\eta}_i))
$$

Since

$$
\frac{\partial}{\partial \beta} \log(1 + \exp(g_i\beta + \hat{\eta}_i))  = \frac{\exp(g_i\beta + \hat{\eta}_i)}{1 + \exp(g_i\beta + \hat{\eta}_i)}g_i = g_i\operatorname{logit^{-1}}(g_i\beta + \hat{\eta}_i)
$$

We have

$$
\frac{\partial}{\partial \beta} \left[ -\mathbf{1}^\top \log(1 + \exp(\mathbf{g}\beta + \hat{\boldsymbol{\eta}})) \right] = -\sum_{i=1}^N g_i\operatorname{logit^{-1}}(g_i\beta + \hat{\eta}_i) = \mathbf{g}^\top \operatorname{logit^{-1}}(\mathbf{g}\beta + \hat{\boldsymbol{\eta}})
$$

Combine terms, we get

$$
U_\beta(\beta) = \frac{\partial \ell}{\partial \beta} = \mathbf{y}^\top \mathbf{g} - \mathbf{g}^\top \operatorname{logit^{-1}}(\mathbf{g}\beta + \hat{\boldsymbol{\eta}}) = \mathbf{g}^\top (\mathbf{y} - \boldsymbol{\mu})
$$

## Derivation of the Fisher information for $\beta$

Given the likelihood function

$$
\ell(\beta; \mathbf{y}, \mathbf{g}, \hat{\boldsymbol{\eta}}) = \mathbf{y}^\top (\mathbf{g}\beta + \hat{\boldsymbol{\eta}}) - \mathbf{1}^\top \log[1 + \exp(\mathbf{g}\beta + \hat{\boldsymbol{\eta}})]
$$

and its first-order partial derivative with respect to $\beta$

$$
U_\beta(\beta) = \frac{\partial \ell}{\partial \beta}  = \mathbf{g}^\top (\mathbf{y} - \boldsymbol{\mu})
$$

We want to derive an explicit expression for

$$
I_\beta(\beta) = -\mathbb{E}_{\mathbf{y}} \left[ \frac{\partial^2 \ell}{\partial\beta^2} \right]
$$

Take the partial derivative of $U_\beta(\beta)$ with respect to $\beta$:

$$
\frac{\partial^2 \ell}{\partial \beta^2} = -\frac{\partial}{\partial \beta} (\mathbf{g}^\top \boldsymbol{\mu}) = -\sum_{i=1}^N \frac{\partial g_i\mu_i}{\partial \beta} = - \sum_{i=1}^N g_i \frac{\partial \mu_i}{\partial \beta}
$$

Since

$$
\frac{\partial \mu_i}{\partial \beta} = \frac{\partial}{\partial \beta} \frac{1}{1 + \exp(-g_i\beta - \hat{\eta}_i)} =  \mu_i (1 - \mu_i) g_i
$$

We have,

$$
\frac{\partial^2 \ell}{\partial \beta^2} = -\sum_{i=1}^N g_i^2 \mu_i (1 - \mu_i)
$$

Because the second derivative does not involve $\mathbf{y}$, the Fisher information reduces to the second derivative itself:

$$
I_\beta(\beta) = -\mathbb{E}_{\mathbf{y}} \left[ \frac{\partial^2 \ell}{\partial\beta^2} \right]= \sum_{i=1}^N g_i^2 \mu_i (1 - \mu_i)= \mathbf{g}^\top \mathbf{W} \mathbf{g}
$$

where $\mathbf{W} = \operatorname{diag}(\boldsymbol{\mu} \odot [\mathbf{1} - \boldsymbol{\mu}])$ and $\odot$ denotes elementwise multiplication.
