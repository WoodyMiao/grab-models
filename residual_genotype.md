# Score statistic

**1. Proof $T = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})$**

The weighted least squares solution of $\mathbb{E}(\mathbf{g}) = \mathbf{X}\boldsymbol{\gamma}$ with weight $\hat{\mathbf{W}}$ is
$$
\hat{\boldsymbol{\gamma}} = (\mathbf{X}^\top \hat{\mathbf{W}} \mathbf{X})^{-1} \mathbf{X}^\top \hat{\mathbf{W}} \mathbf{g}
$$

So the residual genotype vector is
$$
\tilde{\mathbf{g}} = \mathbf{g} - \mathbf{X}\hat{\boldsymbol{\gamma}} = \mathbf{g} - \mathbf{X}(\mathbf{X}^\top \hat{\mathbf{W}} \mathbf{X})^{-1} \mathbf{X}^\top \hat{\mathbf{W}} \mathbf{g}
$$

So,
$$
T = \mathbf{g}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}) = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}) + \hat{\boldsymbol{\gamma}}^\top \mathbf{X}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})
$$

Since $\mathbf{X}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})$ is the score function for $\boldsymbol{\alpha}$ evaluated at the MLE, we have

$$
\mathbf{X}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}}) = \mathbf{0}
$$

Therefore,
$$
T = \tilde{\mathbf{g}}^\top (\mathbf{y} - \hat{\boldsymbol{\mu}})
$$

**2. Proof $\tilde{\mathbf{g}}^\top \mathbf{X}=0$**

$$
\begin{aligned}
  \tilde{\mathbf{g}}^\top \mathbf{X}
  &= [\mathbf{g} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{g}]^{\top} \mathbf{X} \\
  &= \mathbf{g}^{\top}[\mathbf{I} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}]^{\top}\mathbf{X} \\
  &= \mathbf{g}^{\top}[\mathbf{X} - \mathbf{X}(\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X})^{-1}\mathbf{X}^\top\hat{\mathbf{W}}\mathbf{X}] \\
  &= \mathbf{g}^{\top}[\mathbf{X} - \mathbf{X}] \\
  &= \mathbf{0}
\end{aligned}
$$
