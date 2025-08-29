# Empirical GRM

## Kinship coefficient and inbreeding coefficient

Definitions:

- The kinship coefficient for a pair of individuals $i$ and $j$ is the probability that a random allele selected from $i$ and a random allele selected from $j$ at a locus are IBD.
- The inbreeding coefficient for individual $i$ is the probability that $i$’s two alleles at a locus are IBD.
- There is no absolute measure for IBD, and which alleles are considered to be identical copies of an ancestral allele is relative to some choice of previous reference point in time.

Consider

- at time $t_O$, there was one ancestral homogeneous population;
- at time $t_K$, there were $K$ distinct subpopulations derived from the ancestral population;
- at time $t_N$, there is a current structured population descended from all $K$ subpopulations.

Let

- $\psi_{ij}$ be the kinship coefficient for a pair of individuals $i$ and $j$ when the reference population is the ancestral population;
- $\phi_{ij}$ be the kinship coefficient for a pair of individuals $i$ and $j$ when the reference population is composed of the K subpopulations;
- $F_i$ be the inbreeding coefficient for individual $i$ when the reference population is the ancestral population;
- $f_i$ be the inbreeding coefficient for individual $i$ when the reference population is composed of the K subpopulations;
- $k_{ij}^{(2)}$, $k_{ij}^{(1)}$, and $k_{ij}^{(0)}$ be the probability that $i$ and $j$ share 2, 1, or 0 alleles IBD at a locus, respectively, when the reference population is composed of the K subpopulations.

We have

- $F_i = 2\psi_{ii} -1$
- $f_i = 2\phi_{ii}-1$

Let

- $M_{ij}^O$ be the set of most recent common ancestors of individuals $i$ and $j$ with respect to the ancestral populations;
- $M_{ij}^K$ be the set of most recent common ancestors of individuals $i$ and $j$ with respect to the $K$ subpopulations;
- $n_{im}$ be the number of generations from an ancestor $m$ to an individual $i$;

We have

$$
\psi_{ij} = \sum_{m \in M_{ij}^O} \psi_{ij|m} = \sum_{m \in M_{ij}^O} \left[ \left(\frac{1}{2}\right)^{(n_{im}+n_{jm}+1)} (1 + F_m) \right] \tag{1}
$$

$$
\phi_{ij} = \sum_{m \in M_{ij}^K} \phi_{ij|m} = \sum_{m \in M_{ij}^K} \left[ \left(\frac{1}{2}\right)^{(n_{im}+n_{jm}+1)} (1 + f_m) \right] \tag{2}
$$

## Consistent estimator for $\psi_ {ij}$

Let

- $N$ be a set of individuals sampled from the current structured population;
- $S$ be a set of autosomal SNPs that all individuals in the current population have genotype data;
- $p_s$ be the reference allele frequency for SNP $s \in S$ in the ancestral population;
- $g_{is} \sim \operatorname{Binomial}(2, p_s) $ be the number of reference alleles that individual $i \in N$ has at SNP $s \in S$;
- $x_{is_r} \sim \operatorname{Bernoulli}(p_s)$ indicate whether individual $i$'s allele $r$ at SNP $s$ is the reference allele;
- $I_{ij|m}^O \sim \operatorname{Bernoulli}(\psi_{ij|m})$ indicate whether $x_{is_r}$ and $x_{js_r}$ are IBD through ancestor $m \in M_{ij}^O$;
- $I_{ij|m}^K \sim \operatorname{Bernoulli}(\phi_{ij|m})$ indicate whether $x_{is_r}$ and $x_{js_r}$ are IBD through ancestor $m \in M_{ij}^K$.

Let an estimator for $\psi_{ij}$ be

$$
\hat{\psi}_{ij} = \frac{1}{|S|} \sum_{s \in S} \frac{(g_{is} - 2\hat{p}_s)(g_{js} - 2\hat{p}_s)}{4\hat{p}_s(1 - \hat{p}_s)}, \quad \hat{p}_s = \frac{1}{2|N|} \sum_{i \in N} g_{is}  \tag{3}
$$

As $|S| \rightarrow \infty$ and $|N| \rightarrow \infty$, it holds that

$$
\begin{aligned}
\hat{\psi}_{ij}
&\rightarrow \mathbb{E}\left[\frac{(g_{is} - 2p_s)(g_{js} - 2p_s)}{4p_s(1 - p_s)} \right] \\
&= \frac{\mathbb{E}[g_{is}g_{js}] - 2p_s\mathbb{E}[g_{is}] - 2p_s\mathbb{E}[g_{js}] + 4(p_s)^2}{4p_s(1 - p_s)} \\
&= \frac{4p_s(1 - p_s)\psi_{ij}  + 4(p_s)^2 - 4(p_s)^2 - 4(p_s)^2 + 4(p_s)^2}{4p_s(1 - p_s)} \\
&= \psi_{ij} \\
\end{aligned}
$$

where

$$
\begin{aligned}
\mathbb{E}[g_{is}g_{js}]
&= 4\mathbb{E}[x_{is_r}x_{js_r}] \\
&= 4\sum_{m\in M_{ij}^O}\mathbb{E}(x_{is_r}x_{js_r}| I_{ij|m}^O = 1)\operatorname{Pr}(I_{ij|m}^O = 1) \\
&\phantom{=} + 4\mathbb{E}(x_{is_r}x_{js_r}| I_{ij|m}^O = 0\ \forall m \in M_{ij}^O)\operatorname{Pr}(I_{ij|m}^O = 0\ \forall m \in M_{ij}^O) \\
&= 4\sum_{m\in M_{ij}^O}\mathbb{E}(x_{ms_r}| I_{ij|m}^O = 1)\psi_{ij|m} + 4\mathbb{E}(x_{is_r})\mathbb{E}(x_{js_r})(1-\psi_{ij}) \\
&= 4\sum_{m\in M_{ij}^O}p_s\psi_{ij|m} + 4(p_s)^2 (1-\psi_{ij}) \\
&= 4p_s\psi_{ij} + 4(p_s)^2 - 4(p_s)^2\psi_{ij} \\
&= 4p_s(1-p_s)\psi_{ij} + 4(p_s)^2
\end{aligned}
$$

## Individual-specific allele frequency and its moments

Let

- $\mathbf{a}_i=(a_i^1,\ldots,a_i^K)^\top$ be the ancestry vector for individual $i \in N$, where $a_i^k \in [0, 1]$ is the proportion of ancestry across the autosomes for $i$ from subpopulation $k \in \{1,\ldots,K\}$, with $\mathbf{a}_i^\top\mathbf{1} = 1$; $\mathbf{A}=(\mathbf{a}_1, \ldots, \mathbf{a}_{|N|})$;
- $\mathbf{p}_s=(p_s^1,\ldots,p_s^K)^\top$ be the random vector of reference allele frequencies for SNP $s \in S$, where $p_s^k \in [0, 1]$ is the reference allele frequency in subpopulation $k \in \{1,\ldots,K\}$;
- $\mu_{is} = \mathbf{a}_i^\top \mathbf{p}_s$ be the individual-specific allele frequency for individual $i$ at SNP $s$, which is $i$'s expected allele frequency at SNP $s$ conditional on $i$'s ancestral background; $\boldsymbol{\mu}_s=\mathbf{A}^\top\mathbf{p}_s$.

Suppose that the $K$ subpopulations have not experienced natural selection, then $\mathbb{E}[\mathbf{p}_s] = p_s\mathbf{1}$ and $\mathbb{V}[p_s^k] = p_s(1 - p_s) / n_e^k$, where $n_e^k$ is the effective population size of subpopulation $k$. Let

- $\mathbf{\Theta}_K = \mathbb{V}(\mathbf{p}_s) / [p_s(1-p_s)]$, where the $[k,k']$th element reflects the correlation of a random allele from subpopulation $k$ and a random allele from subpopulation $k'$ relative to the total population;
- $\theta_{ij} = \mathbf{a}_i^\top \mathbf{\Theta}_K \mathbf{a}_j = (\mathbf{A}^\top\mathbf{\Theta}_K\mathbf{A})_{ij}$ be the co-ancestry coefficient due to population structure for a pair of individuals $i$ and $j$.

We have

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

and

- $\operatorname{Cov}[\mu_{is},\mu_{js}] = \mathbb{E}[\mu_{is}\mu_{js}] - \mathbb{E}[\mu_{is}]\mathbb{E}[\mu_{js}] = p_s(1-p_s)\theta_{ij}$;
- $\mathbb{E}[\boldsymbol{\mu}_s] = p_s\mathbf{1}$;
- $\mathbb{V}[\boldsymbol{\mu}_s] = p_s(1-p_s)\mathbf{A}^\top\mathbf{\Theta}_K\mathbf{A}$.

## Asymptotic property of $\hat{\psi}_{ij}$ in terms of $\phi_{ij}$ and $\theta_{ij}$

The conditional expectation of the product of two random alleles from individuals $i$ and $j$ is given by

$$
\begin{aligned}
\mathbb{E}[x_{is_r}x_{js_r}|\mathbf{p}_s]
&= \sum_{m\in M_{ij}^K}\mathbb{E}(x_{is_r}x_{js_r}|\mathbf{p}_s, I_{ij|m}^K = 1)\operatorname{Pr}(I_{ij|m}^K = 1) \\
&\phantom{=} + \mathbb{E}(x_{is_r}x_{js_r}|\mathbf{p}_s, I_{ij|m}^K = 0\ \forall m \in M_{ij}^K)\operatorname{Pr}(I_{ij|m}^K = 0\ \forall m \in M_{ij}^K) \\
&= \sum_{m\in M_{ij}^K}\mathbb{E}(x_{ms_r}|\mathbf{p}_s, I_{ij|m}^K = 1)\phi_{ij|m} \\
&\phantom{=} + \mathbb{E}(x_{is_r}|\mathbf{p}_s)\mathbb{E}(x_{js_r}|\mathbf{p}_s)(1-\sum_{m \in M_{ij}^K} \phi_{ij|m}) \\
&= \sum_{m\in M_{ij}^K}\mu_{ms}\phi_{ij|m} + \mu_{is}\mu_{js}-\mu_{is}\mu_{js}\sum_{m \in M_{ij}^K} \phi_{ij|m} \\
&= \sum_{m\in M_{ij}^K}\phi_{ij|m}(\mu_{ms} - \mu_{is}\mu_{js}) + \mu_{is}\mu_{js}
\end{aligned}
$$

Taking the expectation with respect to $\mathbf{p}_s$ and applying equations (4), (5), and (2), we obtain the unconditional expectation of the product of two random genotypic values:

$$
\begin{aligned}
\mathbb{E}[g_{is}g_{js}]
&= 4\mathbb{E}[x_{is_r}x_{js_r}] \\
&= 4\mathbb{E}_{\mathbf{p}_s}(\mathbb{E}[x_{is_r}x_{js_r}|\mathbf{p}_s]) \\
&= 4\sum_{m\in M_{ij}^K}\phi_{ij|m}(\mathbb{E}_{\mathbf{p}_s}[\mu_{ms}] - \mathbb{E}_{\mathbf{p}_s}[\mu_{is}\mu_{js}]) + 4\mathbb{E}_{\mathbf{p}_s}[\mu_{is}\mu_{js}] \\
&= 4\sum_{m\in M_{ij}^K}\phi_{ij|m}\left[p_s - (p_s)^2 - p_s(1-p_s)\theta_{ij}\right] + 4(p_s)^2 + 4p_s(1-p_s)\theta_{ij} \\
&= 4p_s(1 - p_s)(1-\theta_{ij}) \sum_{m\in M_{ij}^K}\phi_{ij|m}  + 4(p_s)^2 + 4p_s(1-p_s)\theta_{ij} \\
&= 4p_s(1 - p_s)(1-\theta_{ij}) \phi_{ij} + 4(p_s)^2 + 4p_s(1-p_s)\theta_{ij} \\
&= 4p_s(1 - p_s) (\phi_{ij} + \theta_{ij} - \theta_{ij}\phi_{ij}) + 4(p_s)^2 \\
\end{aligned}
$$

As $|S| \rightarrow \infty$ and $|N| \rightarrow \infty$, it follows from equation (3) that

$$
\begin{aligned}
\hat{\psi}_{ij}
&= \frac{1}{|S|} \sum_{s \in S} \frac{(g_{is} - 2\hat{p}_s)(g_{js} - 2\hat{p}_s)}{4\hat{p}_s(1 - \hat{p}_s)} \rightarrow \mathbb{E}\left[\frac{(g_{is} - 2p_s)(g_{js} - 2p_s)}{4p_s(1 - p_s)} \right] \\
&= \frac{\mathbb{E}[g_{is}g_{js}] - 2p_s\mathbb{E}[g_{is}] - 2p_s\mathbb{E}[g_{js}] + 4(p_s)^2}{4p_s(1 - p_s)} \\
&= \frac{4p_s(1 - p_s) (\phi_{ij} + \theta_{ij} - \theta_{ij}\phi_{ij}) + 4(p_s)^2 - 4(p_s)^2 - 4(p_s)^2 + 4(p_s)^2}{4p_s(1 - p_s)} \\
&= \phi_{ij} + \theta_{ij} - \theta_{ij}\phi_{ij} \\
\end{aligned}
$$

which indicates that $\hat{\psi}_{ij}$ can be decomposed into three components:

- $\phi_{ij}$, the kinship coefficient with respect to the $K$ subpopulations;
- $\theta_{ij}$, the co-ancestry coefficient reflecting population structure among the $K$ subpopulations;
- $\theta_{ij}\phi_{ij}$, an interaction term.

## Estimator for individual-specific allele frequencies

Let

- $\tilde{N}\subseteq N$ be a subset of unrelated individuals, where $\forall i, j \in \tilde{N},\ \phi_{ij} = 0$;
- $\mathbf{X}$ be an $|\tilde{N}| \times |S|$ matrix of per-SNP standardized genotype data for individuals in $\tilde{N}$ and SNPs in $S$;
- $\mathbf{X}^\top \mathbf{X} / |\tilde{N}|= \mathbf{U} \mathbf{\Lambda} \mathbf{U}^\top$ be an eigendecomposition, where $\mathbf{U}=(\mathbf{u}_1, \ldots, \mathbf{u}_D, \ldots, \mathbf{u}_{|S|} )$, and $\mathbf{\Lambda}=\mathrm{diag}(\lambda_1, \ldots, \lambda_D, \ldots, \lambda_{|S|})$ with $\lambda_1 \geq \ldots \geq \lambda_D >0$ and $\lambda_s = 0$ for $s > D$.

Let

- $\mathbf{G}=(\mathbf{g}_1, \ldots, \mathbf{g}_{|S|})$ be the genotype matrix for individuals in $N$ and SNPs in $S$;
- $\mathbf{V} = (\mathbf{v}_1, \ldots, \mathbf{v}_D)$, where $\mathbf{v}_d = \mathbf{G}\mathbf{u}_d$ is the coordinates of individuals in $N$ on the eigenvector $\mathbf{u}_d$;

Suppose

- $\mathbb{E}[\mathbf{g}_s \mid \mathbf{V}] = \mathbb{E}[\mathbf{g}_s \mid \mathbf{A}^\top\mathbf{p}_s] = 2\boldsymbol{\mu}_s$;
- a linear regression model $\boldsymbol{\mu}_s = (\mathbf{1}, \mathbf{V})\boldsymbol{\beta}_s$, where $\boldsymbol{\beta}_s$ is the vector of regression coefficients;
- a logistic regression model $\operatorname{logit}(\boldsymbol{\mu}_s)=(\mathbf{1}, \mathbf{V})\boldsymbol{\gamma}_s$, where $\boldsymbol{\gamma}_s$ is the vector of regression coefficients.

Let

- $\hat{\boldsymbol{\beta}}_s$ be an estimator for $\boldsymbol{\beta}_s$;
- $\hat{\boldsymbol{\mu}}_s^{\beta} = (\mathbf{1}, \mathbf{V})\hat{\boldsymbol{\beta}}_s$ be an estimator for $\boldsymbol{\mu}_s$;
- $\hat{\boldsymbol{\gamma}}_s$ be an estimator for or $\boldsymbol{\gamma}_s$;
- $\hat{\boldsymbol{\mu}}_s^{\gamma} = \operatorname{logit}^{-1}\left((\mathbf{1}, \mathbf{V})\hat{\boldsymbol{\gamma}}_s\right)$ be another estimator for $\boldsymbol{\mu}_s$.
