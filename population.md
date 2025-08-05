# SAIGE

## Statistical model

Consider a binary trait in a population. Let

- $y$, following a Bernoulli distribution, be the phenotype;
- $\mathbf{x}$ be the $1 \times (1 + p)$ random vector representing the intercept and selected $p$ covariates;
- $g$, following a binomial distribution, be the genotype for a genetic locus of interest;
- $\mathbf{z}$ be a $1 \times m$ random vector of element-wise standardized genotypes for $m$ other SNPs.

Let $\eta = \mathbf{x}\boldsymbol{\alpha} + g\beta + \mathbf{z}\mathbf{u}$ be a linear predictor, where

- $\boldsymbol{\alpha}$ is the fixed effects corresponding to $\mathbf{x}$;
- $\beta$ be the fixed effect corresponding to $g$;
- $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, [\tau/m]\mathbf{I})$ is the random effects corresponding to $\mathbf{z}$; $\tau$ is the fixed variance component.

Suppose $y|\eta \sim \operatorname{Bernoulli}(\operatorname{logit}^{-1}(\eta))$

## Sampling distribution

When a subject $(y_i, \mathbf{x}_i, g_i, \mathbf{z}_i)$ is sampled from the population, the values of the random variables are observed. When the $m$ other SNPs are chosen, the corresponding random effect sizes $\mathbf{u}$ are realized.

Let $(\mathbf{y}, \mathbf{X}, \mathbf{g}, \mathbf{Z})$ be a sample of $(y, \mathbf{x}, g, \mathbf{z})$, and let $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{Z}\mathbf{u}$.
For the $i$th subject, we have $y_i|\eta_i \sim \operatorname{Bernoulli}(\operatorname{logit}^{-1}(\eta_i))$

