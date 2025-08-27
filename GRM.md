# Empirical GRM

## Kinship coefficient and inbreeding coefficient

Definitions:

- The kinship coefficient for a pair of individuals $i$ and $j$ is the probability that a random allele selected from $i$ and a random allele selected from $j$ at a locus are IBD.
- The inbreeding coefficient for individual $i$ is the probability that $i$’s two alleles at a locus are IBD.
- There is no absolute measure for IBD, and which alleles are considered to be identical copies of an ancestral allele is relative to some choice of previous reference point in time.

Consider

- at time $t_0$, there was an ancestral homogeneous population;
- at time $t_K$, there were $K$ distinct subpopulations derived from the ancestral population;
- at time $t_N$, there is a current structured population descended from all $K$ subpopulations.

Let

- $\psi_{ij}$ be the kinship coefficient for a pair of individuals $i$ and $j$ when the reference population is the ancestral population;
- $\phi_{ij}$ be the kinship coefficient for a pair of individuals $i$ and $j$ when the reference population is composed of the K subpopulations;
- $F_i$ be the inbreeding coefficient for individual $i$ when the reference population is the ancestral population;
- $f_i$ be the inbreeding coefficient for individual $i$ when the reference population is composed of the K subpopulations;
- $k_{ij}^{(2)}$, $k_{ij}^{(1)}$, and $k_{ij}^{(0)}$ be the probability that i and j share 2, 1, or 0 alleles IBD at a locus, respectively, when the reference population is composed of the K subpopulations.

There is

- $F_i = 2\psi_{ii} -1$
- $f_i = 2\phi_{ii}-1$

Let

- $M_{ij}$ be the set of most recent common ancestors of individuals $i$ and $j$ with respect to the $K$ subpopulations;
- $n_{im}$ be the number of generations from a common ancestor $m \in M_{ij}$ to individual $i$, plus 1;
- $\phi_{ij|m} = (1/2)^{(n_{im}+n_{jm}-1)} (1 + f_m)$ be the contribution to the kinship between $i$ and $j$ through $m$.

The kinship coefficient for individuals $i$ and $j$ can be written as

$$
\phi_{ij} = \sum_{m \in M_{ij}} \phi_{ij|m} = \sum_{m \in M_{ij}} \left[ \left(\frac{1}{2}\right)^{(n_{im}+n_{jm}-1)} (1 + f_m) \right] \tag{1}
$$

## Empirical GRM estimator

Let

- $N$ be a set of individuals sampled from the current structured population;
- $S$ be a set of autosomal SNPs that all individuals in $N$ have genotype data;
- $p_s$ be the reference allele frequency for SNP $s \in S$ in the ancestral population;
- $g_{is} \sim \operatorname{Binomial}(2, p_s) $ be the number of reference alleles that individual $i \in N$ has at SNP $s \in S$
- $\mathbf{\Psi}$ be an empirical GRM, with its $[i, j]$th element being $2\hat{\psi}_{ij}$, where

$$
\hat{\psi}_{ij} = \frac{1}{|S|} \sum_{s \in S} \frac{(g_{is} - 2\hat{p}_s)(g_{js} - 2\hat{p}_s)}{4\hat{p}_s(1 - \hat{p}_s)},\quad \hat{p}_s = \frac{1}{2|N|} \sum_{i \in N} g_{is} \tag{2}
$$

## Allele frequency in subpopulation

Let

- $\mathbf{a}_i=(a_i^1,\ldots,a_i^K)^\top$ be the ancestry vector for individual $i \in N$, where $a_i^k \in [0, 1]$ is the proportion of ancestry across the autosomes for $i$ from subpopulation $k \in \{1,\ldots,K\}$, with $\mathbf{a}_i^\top\mathbf{1} = 1$;
- $\mathbf{p}_s=(p_s^1,\ldots,p_s^K)^\top$ be the random vector of reference allele frequencies for SNP $s \in S$, where $p_s^k \in [0, 1]$ is the reference allele frequency in subpopulation $k \in \{1,\ldots,K\}$.

Suppose

- $\mathbb{E}(\mathbf{p}_s)=p_s\mathbf{1}$;
- $\mathbb{V}(\mathbf{p}_s) = p_s(1-p_s)\mathbf{\Theta}_K$ for each $s \in S$, where the $[k,k']$th element of $\mathbf{\Theta}_K$ is the correlation of a random allele from subpopulation $k$ and a random allele from subpopulation $k'$ relative to the total population.

## Individual-specific allele frequency

Let

$$
\mu_{is} \coloneqq \mathbf{a}_i^\top \mathbf{p}_s \tag{3}
$$

be the individual-specific allele frequency for individual $i$ at SNP $s$, which is $i$'s expected allele frequency at SNP $s$ conditional on $i$'s ancestral background. Therefore,

$$
\mathbb{E}[\mu_{is}] = \mathbb{E}[\mathbf{a}_i^\top \mathbf{p}_s] = \mathbf{a}_i^\top \mathbb{E}[\mathbf{p}_s] = (\mathbf{a}_i^\top \mathbf{1}) p_s = p_s \tag{4}
$$

$$
\begin{aligned}
\mathbb{E}[\mu_{is}\mu_{js}]
&= \mathbb{E}[\mathbf{a}_i^\top \mathbf{p}_s \mathbf{p}_s^\top \mathbf{a}_j] \\
&= \mathbf{a}_i^\top \mathbb{E}[\mathbf{p}_s \mathbf{p}_s^\top]\mathbf{a}_j \\
&= \mathbf{a}_i^\top (\mathbb{E}[\mathbf{p}_s]\mathbb{E}[\mathbf{p}_s^\top] + \mathbb{V}[\mathbf{p}_s])\mathbf{a}_j \\
&= \mathbf{a}_i^\top [(p_s)^2\mathbf{1}\mathbf{1}^\top + p_s(1-p_s)\mathbf{\Theta}_K]\mathbf{a}_j \\
&= (p_s)^2\mathbf{a}_i^\top\mathbf{1}\mathbf{1}^\top\mathbf{a}_j + p_s(1-p_s)\mathbf{a}_i^\top\mathbf{\Theta}_K\mathbf{a}_j \\
&= (p_s)^2 + p_s(1-p_s)\mathbf{a}_i^\top\mathbf{\Theta}_K\mathbf{a}_j \\
&= (p_s)^2 + p_s(1-p_s)\theta_{ij}
\end{aligned} \tag{5}
$$

$$
\operatorname{Cov}[\mu_{is},\mu_{js}] = \mathbb{E}[\mu_{is}\mu_{js}] - \mathbb{E}[\mu_{is}]\mathbb{E}[\mu_{js}] = p_s(1-p_s)\theta_{ij}
$$

where $\theta_{ij} \coloneqq \mathbf{a}_i^\top \mathbf{\Theta}_K \mathbf{a}_j$ is the co-ancestry coefficient due to population structure for a pair of individuals $i$ and $j$.

Let

- $\mathbf{A}=(\mathbf{a}_1, \ldots, \mathbf{a}_{|N|})$
- $\boldsymbol{\mu}_s=\mathbf{A}^\top\mathbf{p}_s$

We have

- $\mathbb{E}[\boldsymbol{\mu}_s] = p_s\mathbf{1}$
- $\mathbb{V}[\boldsymbol{\mu}_s] = p_s(1-p_s)\mathbf{A}^\top\mathbf{\Theta}_K\mathbf{A}$
- $\theta_{ij}$ is the $[i,j]$th element of $\mathbf{A}^\top\mathbf{\Theta}_K\mathbf{A}$

## Expectation of the product of genotype values

Let the random variable $x_{is_r}$ be the indicator that individual $i$'s allele $r \in \{1,2\}$ at SNP $s$ is the reference allele. We have

$$
\begin{aligned}
\mathbb{E}[g_{is}g_{js}]
&= \mathbb{E}_{\mathbf{p}_s}(\mathbb{E}[g_{is}g_{js}|\mathbf{p}_s]) \\
&= \mathbb{E}_{\mathbf{p}_s}(\mathbb{E}[(x_{is_1} + x_{is_2})(x_{js_1} + x_{js_2})|\mathbf{p}_s]) \\
&= 4\mathbb{E}_{\mathbf{p}_s}(\mathbb{E}[x_{is_r}x_{js_{r'}}|\mathbf{p}_s])
\end{aligned}
$$

Since $x_{is_r}$ and $x_{js_{r'}}$ may be IBD through any $m \in M_{ij}$ or not IBD, by the law of total expectation, we have

$$
\begin{aligned}
\mathbb{E}[x_{is_r}x_{js_r}|\mathbf{p}_s]
&= \sum_{m\in M_{ij}}\mathbb{E}(x_{is_r}x_{js_{r'}}|\mathbf{p}_s, \text{IBD through}\ m)\operatorname{Pr}(\text{IBD through}\ m) \\
&\phantom{=} + \mathbb{E}(x_{is_r}x_{js_{r'}}|\mathbf{p}_s, \text{not IBD})\operatorname{Pr}(\text{not IBD}) \\
&= \sum_{m\in M_{ij}}\mathbb{E}(x_{ms_{r''}}|\mathbf{p}_s, \text{IBD through}\ m)\phi_{ij|m} \\
&\phantom{=} + \mathbb{E}(x_{is_r}|\mathbf{p}_s)\mathbb{E}(x_{js_{r'}}|\mathbf{p}_s)(1-\sum_{m \in M_{ij}} \phi_{ij|m}) \\
&= \sum_{m\in M_{ij}}\mu_{ms}\phi_{ij|m} + \mu_{is}\mu_{js}-\mu_{is}\mu_{js}\sum_{m \in M_{ij}} \phi_{ij|m} \\
&= \sum_{m\in M_{ij}}\phi_{ij|m}(\mu_{ms} - \mu_{is}\mu_{js}) + \mu_{is}\mu_{js}
\end{aligned}
$$

Taking the expectation with respect to $\mathbf{p}_s$ and applying equations (4), (5), and (1), we obtain the unconditional expectation:

$$
\begin{aligned}
\mathbb{E}_{\mathbf{p}_s}(\mathbb{E}[x_{is_r}x_{js_{r'}}|\mathbf{p}_s])
&= \sum_{m\in M_{ij}}\phi_{ij|m}(\mathbb{E}_{\mathbf{p}_s}[\mu_{ms}] - \mathbb{E}_{\mathbf{p}_s}[\mu_{is}\mu_{js}]) + \mathbb{E}_{\mathbf{p}_s}[\mu_{is}\mu_{js}] \\
&= \sum_{m\in M_{ij}}\phi_{ij|m}\left[p_s - (p_s)^2 - p_s(1-p_s)\theta_{ij}\right] + (p_s)^2 + p_s(1-p_s)\theta_{ij} \\
&= p_s(1 - p_s)(1-\theta_{ij}) \sum_{m\in M_{ij}}\phi_{ij|m}  + (p_s)^2 + p_s(1-p_s)\theta_{ij} \\
&= p_s(1 - p_s)(1-\theta_{ij}) \phi_{ij} + (p_s)^2 + p_s(1-p_s)\theta_{ij} \\
&= p_s(1 - p_s) (\phi_{ij} + \theta_{ij} - \theta_{ij}\phi_{ij}) + (p_s)^2 \\
\end{aligned}
$$

Thus,

$$
\mathbb{E}[g_{is}g_{js}] = 4p_s(1 - p_s) (\phi_{ij} + \theta_{ij} - \theta_{ij}\phi_{ij}) + 4(p_s)^2 \tag{6}
$$

## Asymptotic behavior of the empirical GRM estimator

As $|S| \rightarrow \infty$ and $|N| \rightarrow \infty$, by plugging equation (6) into the empirical GRM estimator from equation (2), we have

$$
\begin{aligned}
\hat{\psi}_{ij}
&= \frac{1}{|S|} \sum_{s \in S} \frac{(g_{is} - 2\hat{p}_s)(g_{js} - 2\hat{p}_s)}{4\hat{p}_s(1 - \hat{p}_s)} \rightarrow \mathbb{E}\left[\frac{(g_{is} - 2p_s)(g_{js} - 2p_s)}{4p_s(1 - p_s)} \right] \\
&= \frac{\mathbb{E}[g_{is}g_{js}] - 2p_s\mathbb{E}[g_{is}] - 2p_s\mathbb{E}[g_{js}] + 4(p_s)^2}{4p_s(1 - p_s)} \\
&= \frac{4p_s(1 - p_s) (\phi_{ij} + \theta_{ij} - \theta_{ij}\phi_{ij}) + 4(p_s)^2 - 4(p_s)^2 - 4(p_s)^2 + 4(p_s)^2}{4p_s(1 - p_s)} \\
&= \phi_{ij} + \theta_{ij} - \theta_{ij}\phi_{ij} \\
\end{aligned}
$$

The empirical genetic relationship between individuals $i$ and $j$ can be decomposed into three components: their kinship coefficient with respect to the $K$ subpopulations ($\phi_{ij}$), the co-ancestry coefficient reflecting population structure among the $K$ subpopulations ($\theta_{ij}$), and an interaction term between kinship and population structure.

## Estimator for individual-specific allele frequency

Assume that the top $D$ PCs reflect the population structure in this sample, and let $\mathbf{V} = [\mathbf{V}^1, \ldots, \mathbf{V}^D]$ be an $|N| \times D$ matrix whose column vectors correspond to the top $D$ PCs. Let $\mathbf{g}_s$ be a length $|N|$ vector of genotype values for all sampled individuals at SNP $s$, and consider the linear regression model

$$
\mathbb{E}[\mathbf{g}_s \mid \mathbf{V}] = \mathbf{1}\beta_0 + \mathbf{V}\boldsymbol{\beta},
$$


Assume the expectation of $\mathbf{g}_s$ conditional on $\mathbf{V}$ is equivalent to the expectation of $\mathbf{g}_s$ conditional on the true ancestries of the sampled individuals, i.e., 

$$
\mathbb{E}[\mathbf{g}_s \mid \mathbf{V}] = \mathbb{E}[\mathbf{g}_s \mid \mathbf{A}^\top\mathbf{p}_s] = 2\boldsymbol{\mu}_s
$$ 

Therefore, the fitted values from this linear regression model can be used to predict individual-specific allele frequencies from the PCs, and our proposed estimator for $\mu_{is}$ at each SNP $s \in S$ is

$$
\hat{\mu}_{is} = \frac{1}{2} \widehat{\mathbb{E}}\left[ g_{is} \mid V_i^1, \ldots, V_i^D \right] = \frac{1}{2} \left( \hat{\beta}_0 + \sum_{d=1}^D \hat{\beta}_d V_i^d \right)
$$

where $V_i^d$ is the coordinate for individual $i$ along the $d^\text{th}$ PC, $\mathbf{V}_d$, with $d \in \{1, \ldots, D\}$. Because each PC has mean $0$, $(1/2)\hat{\beta}_0$ is equal to the sample average allele frequency at SNP $s$, which can be interpreted as an estimate of $p_s$, the population allele frequency, and each $\hat{\beta}_d$ can be viewed as a measure of deviation in allele frequency from the sample average due to the ancestry component represented by $\mathbf{V}_d$.

## PC-Relate estimator for $\phi_{ij}$ and its asymptotic properties

The PC-Relate estimator of the kinship coefficient $\phi_{ij}$ for individuals $i$ and $j$ is

$$
\hat{\phi}_{ij} =
\frac{
    \sum_{s \in S_{ij}} (g_{is} - 2\hat{\mu}_{is})(g_{js} - 2\hat{\mu}_{js})
}{
    4 \sum_{s \in S_{ij}} \left[ \hat{\mu}_{is}(1 - \hat{\mu}_{is}) \hat{\mu}_{js}(1 - \hat{\mu}_{js}) \right]^{1/2}
}
$$

Because $\mu_{js}$ is a fixed quantity conditional on $\mathbf{p}_s$, it can easily be seen that $\mathbb{E}[g_{is}\mu_{js}] = \mathbb{E}[\mu_{js}\mathbb{E}[g_{is} \mid \mathbf{p}_s]] = 2\mathbb{E}[\mu_{is}\mu_{js}]$. The expectation of the denominator of the PC-Relate kinship coefficient estimator is not straightforward to calculate, but we can define it to be $\mathbb{E}[[\mu_{is}(1-\mu_{is})\mu_{js}(1-\mu_{js})]^{1/2}] \equiv p_s(1-p_s) [1-d_\phi(i,j)]$, where $d_\phi(i,j)$ is some function of $\mathbf{a}_i$, $\mathbf{a}_j$, and $\mathbf{\Theta}_K$. Therefore, by plugging the appropriate expectations in for Equation 4, we obtain

$$
\hat{\phi}_{ij} \rightarrow \frac{
    \mathbb{E}[g_{is}\delta_{js}] - 4\mathbb{E}[\mu_{is}\mu_{js}]
}{
    4\mathbb{E}\left[ [\mu_{is}(1-\mu_{is})\mu_{js}(1-\mu_{js})]^{1/2} \right]
}
$$

$$
= \phi_{ij} - b_\phi(i, j),
$$

where the bias term is given by the function

$$
b_\phi(i, j) \equiv \sum_{m \in M_{ij}} \phi_{ij|m} \left( \frac{\theta_{mm} - d_\phi(i, j)}{1 - d_\phi(i, j)} \right)
$$
