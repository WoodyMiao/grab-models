# Score

## Model Specification

In a case-control study with a sample size $N$, let:

- $\mathbf{y} = [y_1,\ldots,y_N]^{\top},\ y_i \sim \operatorname{Bernoulli}(\mu_i)$ represent their phenotypes;
- $N \times (1 + p)$ matrix $\mathbf{X}$ represent $p$ covariates and a column of ones;
- $N \times 1$ vector $\mathbf{g}$ represent their genotypes coded as allele counts for a variant to be tested;
- $N \times 1$ vector $\mathbf{b} \sim \mathcal{N}(\mathbf{0}, \tau \mathbf{\Psi})$ represent random effects, where $\mathbf{\Psi}$ is a GRM and $\tau$ is the variance component;
- $\boldsymbol{\eta} = \mathbf{X} \boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b}$ be a linear predictor, where $\boldsymbol{\alpha}$ is the fixed effects, and $\beta$ is the genetic effect to be tested.

Suppose $\mu_i = \operatorname{logit}^{-1}(\eta_i)$. Under the null hypothesis $H_0: \beta=0$, let $(\hat{\tau}, \hat{\boldsymbol{\alpha}}, \hat{\mathbf{b}})$ be an estimate of $(\tau, \boldsymbol{\alpha}, \mathbf{b})$ and let $\hat{\boldsymbol{\eta}} = \mathbf{X} \hat{\boldsymbol{\alpha}} + \hat{\mathbf{b}}$.

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
\frac{\partial}{\partial \beta} \log(1 + \exp(g_i\beta + \hat{\eta}_i))  = \frac{\exp(g_i\beta + \hat{\eta}_i)}{1 + \exp(g_i\beta + \hat{\eta}_i)}g_i = g_i\operatorname{logit}^{-1}(g_i\beta + \hat{\eta}_i)
$$

We have

$$
\frac{\partial}{\partial \beta} \left[ -\mathbf{1}^\top \log(1 + \exp(\mathbf{g}\beta + \hat{\boldsymbol{\eta}})) \right] = -\sum_{i=1}^N g_i\operatorname{logit}^{-1}(g_i\beta + \hat{\eta}_i) = \mathbf{g}^\top \operatorname{logit}^{-1}(\mathbf{g}\beta + \hat{\boldsymbol{\eta}})
$$

Combine terms, we get

$$
U_\beta(\beta) = \frac{\partial \ell}{\partial \beta} = \mathbf{y}^\top \mathbf{g} - \mathbf{g}^\top \operatorname{logit}^{-1}(\mathbf{g}\beta + \hat{\boldsymbol{\eta}}) = \mathbf{g}^\top (\mathbf{y} - \boldsymbol{\mu})
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

## Derivation of the Score Test Statistic and its Asymptotic Distribution

Under the null hypothesis $H_0: \beta = 0$, we want to test whether there is significant genetic effect. The score test is based on the score function evaluated at the null hypothesis.

### Step 1: Evaluate the Score Function at the Null

Under $H_0: \beta = 0$, let $(\hat{\boldsymbol{\alpha}}, \hat{\mathbf{b}}, \hat{\tau})$ be the maximum likelihood estimates of $(\boldsymbol{\alpha}, \mathbf{b}, \tau)$ under the null model. Define $\hat{\boldsymbol{\eta}} = \mathbf{X}\hat{\boldsymbol{\alpha}} + \hat{\mathbf{b}}$ and $\hat{\boldsymbol{\mu}} = \operatorname{logit}^{-1}(\hat{\boldsymbol{\eta}})$.

The score statistic is:
$$
T = U_\beta(0) = \left. \frac{\partial \ell}{\partial \beta} \right|_{\beta=0} = \mathbf{g}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})
$$

**Note:** $T$ is the raw score statistic. The **score test statistic** is the standardized version $T/\sqrt{\mathbb{V}(T)}$, which follows a standard normal distribution asymptotically.

### Step 2: Find the Variance of the Score Statistic

**Why we cannot use Fisher information directly:**

Even though we have obtained MLEs $(\hat{\boldsymbol{\alpha}}, \hat{\mathbf{b}}, \hat{\tau})$ under the null model, the variance of the score statistic in mixed models is more complex than the simple Fisher information. This is because:

1. The random effects $\mathbf{b}$ introduce correlations between observations
2. The parameters $(\boldsymbol{\alpha}, \mathbf{b}, \tau)$ are estimated jointly, creating dependencies
3. The genotype vector $\mathbf{g}$ may be correlated with the design matrix $\mathbf{X}$

The correct variance must account for the uncertainty in estimating the nuisance parameters and the correlation structure induced by the random effects.

**Derivation of the residual genotype vector:**

The key insight is that we need to orthogonalize $\mathbf{g}$ with respect to the fixed effects design matrix $\mathbf{X}$ using the appropriate weight matrix.

Let $\tilde{\mathbf{g}}$ be the residual from regressing $\mathbf{g}$ on $\mathbf{X}$ using the weight matrix $\hat{\mathbf{W}}$:
$$
\tilde{\mathbf{g}} = \mathbf{g} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{g}
$$

where $\hat{\mathbf{W}} = \operatorname{diag}(\hat{\boldsymbol{\mu}} \odot [\mathbf{1} - \hat{\boldsymbol{\mu}}])$ is the estimated weight matrix.

**Properties of the residual genotype vector:**
1. $\tilde{\mathbf{g}}$ is orthogonal to the column space of $\mathbf{X}$ with respect to the weight matrix $\hat{\mathbf{W}}$:
   $$\mathbf{X}^\top\hat{\mathbf{W}}\tilde{\mathbf{g}} = \mathbf{0}$$
2. This orthogonalization removes the confounding between genetic effects and fixed effects

**Why use this specific form:**
The residual vector $\tilde{\mathbf{g}}$ ensures that:
$$
T = \mathbf{g}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}) = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})
$$

This equality holds because $(\mathbf{y} - \hat{\boldsymbol{\mu}})$ is orthogonal to the column space of $\mathbf{X}$ at the MLE under the null model.

The variance of $T$ under the null hypothesis is:
$$
\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}
$$

where $\hat{\mathbf{P}} = \hat{\boldsymbol{\Sigma}}^{-1} - \hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X}(\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\boldsymbol{\Sigma}}^{-1}$ and $\hat{\boldsymbol{\Sigma}} = \hat{\mathbf{W}}^{-1} + \hat{\tau}\mathbf{\Psi}$.

**Derivation of the variance formula:**

The matrix $\hat{\mathbf{P}}$ is the projection matrix onto the orthogonal complement of the column space of $\mathbf{X}$ with respect to the metric $\hat{\boldsymbol{\Sigma}}^{-1}$. It accounts for:
1. The covariance structure of the observations ($\hat{\boldsymbol{\Sigma}}$)
2. The uncertainty from estimating the fixed effects ($\boldsymbol{\alpha}$)
3. The correlation induced by random effects ($\hat{\tau}\mathbf{\Psi}$)

### Step 3: Asymptotic Distribution

**Proof of asymptotic normality:**

Under standard regularity conditions and the null hypothesis $H_0: \beta = 0$, we can show that the standardized score statistic follows a standard normal distribution asymptotically.

**Key steps in the proof:**

1. **Central Limit Theorem:** Since $T = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})$ is a weighted sum of the residuals $(\mathbf{y} - \hat{\boldsymbol{\mu}})$, and under the null hypothesis $\mathbb{E}[T] = 0$, the CLT applies as $N \to \infty$.

2. **Consistency of variance estimator:** Under regularity conditions, $\hat{\mathbf{P}} \to \mathbf{P}$ in probability, where $\mathbf{P}$ is the true projection matrix. This ensures $\mathbb{V}(T)$ is consistently estimated.

3. **Martingale property:** The score function has the martingale property under the null hypothesis, which ensures the CLT conditions are satisfied.

4. **Lindeberg condition:** For the CLT to apply, we need the contributions from individual observations to become negligible as $N$ increases. This is satisfied under standard regularity conditions for logistic mixed models.

**Formal statement:**
Under the null hypothesis and standard regularity conditions, as the sample size $N \to \infty$:

$$
\frac{T}{\sqrt{\mathbb{V}(T)}} = \frac{\tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})}{\sqrt{\tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

**Standardized Score Statistic:**
$$
\frac{T}{\sqrt{\mathbb{V}(T)}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

**Squared Score Statistic:**
By the continuous mapping theorem, since the square of a standard normal random variable follows a chi-squared distribution with 1 degree of freedom:
$$
\frac{T^2}{\mathbb{V}(T)} \xrightarrow{d} \chi^2_1
$$

### Step 4: Score Test

The score test rejects $H_0: \beta = 0$ when:
$$
\frac{T^2}{\mathbb{V}(T)} > \chi^2_{1,\alpha}
$$

where $\chi^2_{1,\alpha}$ is the $(1-\alpha)$-quantile of the chi-squared distribution with 1 degree of freedom.

Alternatively, using the standard normal distribution:
$$
\left| \frac{T}{\sqrt{\mathbb{V}(T)}} \right| > z_{\alpha/2}
$$

where $z_{\alpha/2}$ is the $(1-\alpha/2)$-quantile of the standard normal distribution.

### Step 5: P-value Calculation

The p-value can be calculated as:
$$
p = 2\left[1 - \Phi\left(\left|\frac{T}{\sqrt{\mathbb{V}(T)}}\right|\right)\right]
$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution.

## Summary of Key Points

**Q1: Is $T/\sqrt{\mathbb{V}(T)}$ the score test statistic?**
Yes, exactly. $T = \mathbf{g}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})$ is the raw score statistic (the score function evaluated at the null). The **score test statistic** is the standardized version $T/\sqrt{\mathbb{V}(T)}$, which asymptotically follows $\mathcal{N}(0,1)$.

**Q2: Why not use Fisher information directly?**
Even after obtaining MLEs under the null, we cannot use Fisher information directly because:
- Random effects create correlations between observations
- Joint estimation of $(\boldsymbol{\alpha}, \mathbf{b}, \tau)$ introduces parameter dependencies  
- The variance must account for uncertainty in nuisance parameter estimation
- The correlation structure $\hat{\tau}\mathbf{\Psi}$ affects the variance calculation

**Q3: Why the residual genotype vector $\tilde{\mathbf{g}}$?**
The residual genotype vector $\tilde{\mathbf{g}} = \mathbf{g} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{g}$ removes confounding with fixed effects by orthogonalizing $\mathbf{g}$ against $\mathbf{X}$. This ensures the test is specifically for the genetic effect $\beta$, not confounded by fixed effects. The asymptotic normality follows from the Central Limit Theorem applied to the weighted sum of residuals, with the variance properly accounting for the mixed model correlation structure.

## Fixed vs. Random Variables and Correlation Structure

**What is treated as fixed vs. random:**

- **Fixed (non-random):** $\mathbf{X}$, $\mathbf{g}$, $\mathbf{\Psi}$, and the estimated parameters $(\hat{\boldsymbol{\alpha}}, \hat{\mathbf{b}}, \hat{\tau})$
- **Random:** Only $\mathbf{y} = [y_1, \ldots, y_N]^\top$ where $y_i \sim \operatorname{Bernoulli}(\mu_i)$

**Are the $y_i$ correlated?**

Yes, the observations $y_i$ are **marginally correlated** due to the random effects structure, even though they are conditionally independent given $\mathbf{b}$.

**Marginal correlation structure:**

**Step-by-step derivation of $\operatorname{Cov}(\mathbf{y})$:**

To find the marginal covariance of $\mathbf{y}$, we use the law of total covariance:
$$
\operatorname{Cov}(\mathbf{y}) = \mathbb{E}[\operatorname{Cov}(\mathbf{y}|\mathbf{b})] + \operatorname{Cov}(\mathbb{E}[\mathbf{y}|\mathbf{b}])
$$

**Step 1: Conditional covariance given $\mathbf{b}$**
Given $\mathbf{b}$, the $y_i$ are independent Bernoulli random variables, so:
$$
\operatorname{Cov}(\mathbf{y}|\mathbf{b}) = \operatorname{diag}(\boldsymbol{\mu} \odot [\mathbf{1} - \boldsymbol{\mu}]) = \mathbf{W}
$$
where $\boldsymbol{\mu} = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b})$.

Therefore: $\mathbb{E}[\operatorname{Cov}(\mathbf{y}|\mathbf{b})] = \mathbb{E}[\mathbf{W}] \approx \mathbf{W}$ (using $\hat{\boldsymbol{\mu}}$).

**Step 2: Covariance of conditional expectations**
The conditional expectation is:
$$
\mathbb{E}[\mathbf{y}|\mathbf{b}] = \boldsymbol{\mu} = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha} + \mathbf{g}\beta + \mathbf{b})
$$

Under the null hypothesis ($\beta = 0$):
$$
\mathbb{E}[\mathbf{y}|\mathbf{b}] = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha} + \mathbf{b})
$$

For small random effects, we can use the delta method approximation:
$$
\frac{\partial}{\partial \mathbf{b}} \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha} + \mathbf{b}) = \operatorname{diag}(\boldsymbol{\mu} \odot [\mathbf{1} - \boldsymbol{\mu}]) = \mathbf{W}
$$

Therefore:
$$
\operatorname{Cov}(\mathbb{E}[\mathbf{y}|\mathbf{b}]) \approx \mathbf{W} \operatorname{Cov}(\mathbf{b}) \mathbf{W} = \mathbf{W} (\tau \mathbf{\Psi}) \mathbf{W} = \tau \mathbf{W} \mathbf{\Psi} \mathbf{W}
$$

**Step 3: Total marginal covariance**
Combining both terms:
$$
\operatorname{Cov}(\mathbf{y}) = \mathbf{W} + \tau \mathbf{W} \mathbf{\Psi} \mathbf{W}
$$

**Alternative form accounting for fixed effects:**
When we account for the uncertainty in fixed effects estimation, the effective covariance becomes:
$$
\operatorname{Cov}(\mathbf{y}) = \mathbf{W} + \mathbf{W}^{1/2} \mathbf{P}_{\mathbf{X}}^{\perp} \tau \mathbf{\Psi} \mathbf{P}_{\mathbf{X}}^{\perp} \mathbf{W}^{1/2}
$$

where $\mathbf{P}_{\mathbf{X}}^{\perp} = \mathbf{I} - \mathbf{X}(\mathbf{X}^\top\mathbf{W}\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{W}$ projects onto the orthogonal complement of $\mathbf{X}$.

**Can we use $\mathbb{V}(T) = \mathbf{g}^\top \operatorname{Cov}(\mathbf{y}) \mathbf{g}$?**

**Short answer: No, not directly.**

The issue is that $T = \mathbf{g}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})$, not $T = \mathbf{g}^\top \mathbf{y}$. The estimated mean $\hat{\boldsymbol{\mu}}$ introduces additional complexity because:

1. $\hat{\boldsymbol{\mu}}$ depends on the estimated parameters $(\hat{\boldsymbol{\alpha}}, \hat{\mathbf{b}}, \hat{\tau})$
2. The residuals $(\mathbf{y} - \hat{\boldsymbol{\mu}})$ have a different covariance structure than $\mathbf{y}$ **in mixed models**
3. We need to account for the estimation uncertainty in the nuisance parameters

**Important distinction: Mixed model vs. Fixed-effects model**

**For a pure fixed-effects logistic model (no random effects):**

If we had only $\boldsymbol{\mu} = \operatorname{logit}^{-1}(\mathbf{X}\boldsymbol{\alpha} + \mathbf{g}\beta)$ with no $\mathbf{b}$, then:

1. **The $y_i$ would be independent** (no correlation from random effects)
2. **$\operatorname{Cov}(\mathbf{y}) = \mathbf{W}$** (diagonal matrix)
3. **The residuals $(\mathbf{y} - \hat{\boldsymbol{\mu}})$ would have the same covariance structure** as $\mathbf{y}$ asymptotically
4. **We could use Fisher information** $\mathbb{V}(T) \approx \tilde{\mathbf{g}}^\top \mathbf{W} \tilde{\mathbf{g}}$ for the variance

**For mixed models (with random effects $\mathbf{b}$):**

1. **The $y_i$ are marginally correlated** due to $\mathbf{b}$
2. **$\operatorname{Cov}(\mathbf{y}) = \mathbf{W} + \tau \mathbf{W} \mathbf{\Psi} \mathbf{W}$** (non-diagonal)
3. **The residuals $(\mathbf{y} - \hat{\boldsymbol{\mu}})$ have different covariance** because $\hat{\mathbf{b}}$ introduces additional correlations
4. **Fisher information is insufficient** - we need the complex formula $\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}$

**Why the difference?**

In mixed models, $\hat{\mathbf{b}}$ creates **additional dependence** beyond what exists in fixed-effects models. Each $\hat{b}_i$ depends on the entire data vector $\mathbf{y}$ through the genetic relatedness matrix $\mathbf{\Psi}$, creating complex correlations that Fisher information doesn't capture.

The correct variance formula $\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}$ incorporates these corrections, which are necessary specifically for mixed models.

**Key insight about conditional vs. marginal independence:**

You've identified a crucial point! You're asking: "If $\mathbf{b}$ is given by $\hat{\mathbf{b}}$ (fixed), why can't we say $y_i$ are independent?"

**The answer is subtle but fundamental:**

## The Likelihood vs. Score Test Distinction

**1. In the likelihood derivation:**
$$
\ell(\boldsymbol{\alpha}, \beta, \tau; \mathbf{X}, \mathbf{g}, \mathbf{\Psi}, \mathbf{y}) = \log \int_{\mathbb{R}^N} p_{\mathbf{y} | \mathbf{b}}(\mathbf{y} | \mathbf{b}; \mathbf{X}, \mathbf{g}, \boldsymbol{\alpha}, \beta) f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi}) \, \mathrm{d}\mathbf{b}
$$

Here, **$\mathbf{b}$ is treated as a random variable** with distribution $f_{\mathbf{b}}(\mathbf{b}; \tau,\mathbf{\Psi})$. We integrate over all possible values of $\mathbf{b}$. Within the integrand, $p_{\mathbf{y} | \mathbf{b}}(\mathbf{y} | \mathbf{b})$ treats $\mathbf{b}$ as given (conditional), so the $y_i$ are independent conditional on any fixed value of $\mathbf{b}$.

**2. In the score test:**
We use **estimated values** $\hat{\mathbf{b}}$ that were obtained by maximizing the likelihood above. Now $\hat{\mathbf{b}}$ is a **function of the data** $\mathbf{y}$, creating a circular dependency.

## Simple Principle for Independence

**Simple Principle:** The $y_i$ are independent **if and only if** all the parameters they depend on are either:
1. **Known constants**, or 
2. **True random variables with known distributions** (not estimated from the data)

**Applied to our cases:**

**Case 1 (Likelihood):** ✅ Independence holds conditionally
- $\mathbf{b}$ is a true random variable with known distribution
- Conditional on any value of $\mathbf{b}$, the $y_i$ are independent
- We integrate over the distribution of $\mathbf{b}$

**Case 2 (Score test):** ❌ Independence fails
- $\hat{\mathbf{b}}$ is estimated from the same data $\mathbf{y}$ we're analyzing
- This creates dependence: $\hat{\mathbf{b}} = \hat{\mathbf{b}}(\mathbf{y})$
- The $y_i$ are no longer independent because they all contribute to determining $\hat{\mathbf{b}}$

## Mathematical Illustration

**In the likelihood:** For any fixed $\mathbf{b}$:
$$
p_{\mathbf{y} | \mathbf{b}}(\mathbf{y} | \mathbf{b}) = \prod_{i=1}^N p_{y_i | b_i}(y_i | b_i) = \prod_{i=1}^N \mu_i^{y_i}(1-\mu_i)^{1-y_i}
$$

**In the score test:** With $\hat{\mathbf{b}} = \hat{\mathbf{b}}(\mathbf{y})$:
$$
p(\mathbf{y} | \hat{\mathbf{b}}(\mathbf{y})) \neq \prod_{i=1}^N p(y_i | \hat{b}_i(\mathbf{y}))
$$
The product form breaks down because each $\hat{b}_i$ depends on all of $\mathbf{y}$, not just $y_i$.

**The base difference:**
- **Likelihood:** $\mathbf{b}$ is random, $\mathbf{y}$ is observed
- **Score test:** $\mathbf{y}$ is random, $\hat{\mathbf{b}}$ is a function of $\mathbf{y}$

This circular dependency is exactly why we need the complex variance formula that accounts for both the correlation structure and the estimation uncertainty, rather than simply treating the observations as independent.

**Correlation coefficients:**

The correlation between $y_i$ and $y_j$ (for $i \neq j$) is approximately:
$$
\operatorname{Corr}(y_i, y_j) \approx \frac{\tau \Psi_{ij}}{\sqrt{[\mu_i(1-\mu_i) + \tau \Psi_{ii}][\mu_j(1-\mu_j) + \tau \Psi_{jj}]}}
$$

This correlation:
- Is positive when individuals $i$ and $j$ are genetically related ($\Psi_{ij} > 0$)
- Increases with the variance component $\tau$
- Decreases as the Bernoulli variances $\mu_i(1-\mu_i)$ increase

**Why this matters for the score test:**

The correlation structure explains why we need the complex variance formula $\mathbb{V}(T) = \tilde{\mathbf{g}}^\top \hat{\mathbf{P}} \tilde{\mathbf{g}}$ rather than simple Fisher information. The matrix $\hat{\mathbf{P}}$ properly accounts for this correlation structure when computing the variance of the score statistic.
