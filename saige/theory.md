# SAIGE: 两种估计路径的理论推导

本文从边际对数似然函数出发，推导两种参数估计路径：
**理论最优路径**（统计性质最好，计算代价极高）和
**实用近似路径**（SAIGE 软件的实际做法）。

---

## 0. 边际对数似然函数

模型设定：

$$
y_i \mid b_i \sim \operatorname{Bernoulli}(\mu_i), \quad \mu_i = \operatorname{logit}^{-1}(\mathbf{X}_i \boldsymbol{\alpha} + b_i)
$$

$$
\mathbf{b} \sim \mathcal{N}(\mathbf{0},\ \tau \boldsymbol{\Psi})
$$

其中 $\boldsymbol{\alpha}$ 是固定效应，$\tau$ 是方差分量，$\boldsymbol{\Psi}$ 是 GRM。

对随机效应 $\mathbf{b}$ 积分，得到 $\mathbf{y}$ 的**边际对数似然**：

$$
\boxed{
\ell(\boldsymbol{\alpha}, \tau)
= \log \int_{\mathbb{R}^N} p(\mathbf{y} \mid \mathbf{b};\, \boldsymbol{\alpha})\; \phi(\mathbf{b};\, \mathbf{0},\, \tau\boldsymbol{\Psi})\; \mathrm{d}\mathbf{b}
}
\tag{ML}
$$

这个 $N$ 维积分对 logistic 模型**没有解析解**，是所有近似方法的起点。

---

## 1. Laplace 近似

对被积函数取对数，在其众数（后验众数）$\tilde{\mathbf{b}}$ 处做二阶 Taylor 展开：

$$
\log p(\mathbf{y} \mid \mathbf{b}) + \log \phi(\mathbf{b})
\approx
\underbrace{\log p(\mathbf{y} \mid \tilde{\mathbf{b}}) + \log \phi(\tilde{\mathbf{b}})}_{f(\tilde{\mathbf{b}})} - \frac{1}{2} (\mathbf{b} - \tilde{\mathbf{b}})^\top \mathbf{H} (\mathbf{b} - \tilde{\mathbf{b}})
$$

其中 $\mathbf{H} = \mathbf{W}(\tilde{\mathbf{b}}) + (\tau\boldsymbol{\Psi})^{-1}$ 是负对数后验在 $\tilde{\mathbf{b}}$ 处的 Hessian，
$\mathbf{W} = \operatorname{diag}(\mu_i(1-\mu_i))$。

对上述二次型积分，得到 Laplace 近似边际对数似然：

$$
\ell_{\mathrm{LA}}(\boldsymbol{\alpha}, \tau)
= \underbrace{\log p(\mathbf{y} \mid \tilde{\mathbf{b}})}_{\text{条件对数似然}} + \underbrace{\log \phi(\tilde{\mathbf{b}};\, \mathbf{0},\, \tau\boldsymbol{\Psi})}_{\text{对数先验}} + \underbrace{\frac{N}{2}\log(2\pi) - \frac{1}{2}\log|\mathbf{H}|}_{\text{Laplace 体积校正}}
\tag{LA}
$$

误差阶为 $O(N^{-1})$（相对于 $\ell$ 的误差）。

---

## 2. REML 校正

### 先对 $\boldsymbol{\alpha}$ 积分，再对 $\tau$ 求导

LMM下的REML对数似然函数为：

$$
\ell_{\mathrm{REML}}(\tau)
= \ell_{\mathrm{ML}}(\hat{\boldsymbol{\alpha}}(\tau),\, \tau) - \frac{1}{2} \log \left| \mathbf{X}^\top \boldsymbol{\Sigma}^{-1}(\tau) \mathbf{X} \right|
\tag{REML-LMM}
$$

其中 $\boldsymbol{\Sigma}(\tau) = \tau\boldsymbol{\Psi} + \mathbf{W}^{-1}$，$\hat{\boldsymbol{\alpha}}(\tau)$ 是在 $\tau$ 固定时的 MLE。

修正项 $-\frac{1}{2}\log|\mathbf{X}^\top \boldsymbol{\Sigma}^{-1}\mathbf{X}|$ 扣除了固定效应占用的信息量（$q$ 个参数）。

### GLMM 下的 Laplace-REML

将上述修正直接作用在 Laplace 近似上：

$$
\ell_{\mathrm{LA\text{-}REML}}(\tau)
= \ell_{\mathrm{LA}}(\hat{\boldsymbol{\alpha}}(\tau),\, \tau) - \frac{1}{2} \log \left| \mathbf{X}^\top \mathbf{H}^{-1}(\tau) \mathbf{X} \right|
\tag{LA-REML}
$$

其中 $\mathbf{H}^{-1}$ 是后验协方差（Laplace 近似），$\mathbf{X}^\top \mathbf{H}^{-1} \mathbf{X}$ 是 $\boldsymbol{\alpha}$ 的有效信息矩阵。

**$\hat{\tau}_{\mathrm{REML}}$ 的渐近性质**：$N \to \infty$ 时无偏（一致），有限样本下偏差为 $O(N^{-2})$（ML 偏差为 $O(N^{-1})$）。

---

## 3. 两种估计路径

### 路径 A：理论最优路径

$$
\tau \xrightarrow{\text{(LA-REML)}}
\hat{\tau}
\xrightarrow{\text{固定}\hat{\tau}}
\hat{\boldsymbol{\alpha}} = \operatorname{argmax}_{\boldsymbol{\alpha}} \ell_{\mathrm{LA}}(\boldsymbol{\alpha}, \hat{\tau})
\xrightarrow{\text{数值积分}}
\hat{\mathbf{b}} = E[\mathbf{b} \mid \mathbf{y}, \hat{\boldsymbol{\alpha}}, \hat{\tau}]
$$

**步骤 A1：Laplace-REML 估计 $\hat{\tau}$**

最大化 (LA-REML)，对标量 $\tau$ 做一维优化（Brent 方法）：

$$
\hat{\tau} = \operatorname{argmax}_{\tau > 0}\; \ell_{\mathrm{LA\text{-}REML}}(\tau)
$$

每步需要：(i) 内层 NR 求 $\tilde{\mathbf{b}}$；(ii) 内层 NR 求 $\hat{\boldsymbol{\alpha}}(\tau)$；(iii) 计算 $\det(\mathbf{H})$。

**步骤 A2：固定 $\hat{\tau}$，NR 求 $\hat{\boldsymbol{\alpha}}$**

$$
\hat{\boldsymbol{\alpha}} = \operatorname{argmax}_{\boldsymbol{\alpha}}\; \ell_{\mathrm{LA}}(\boldsymbol{\alpha}, \hat{\tau})
$$

渐近无偏，有限样本偏差 $O(N^{-1})$。

**步骤 A3：Gauss-Hermite 求积计算后验均值 $\hat{\mathbf{b}}$**

精确后验均值：
$$
\hat{b}_i = E[b_i \mid \mathbf{y}, \hat{\boldsymbol{\alpha}}, \hat{\tau}]
= \frac{\int b_i\; p(\mathbf{y} \mid \mathbf{b})\; \phi(\mathbf{b})\; \mathrm{d}\mathbf{b}}
       {\int p(\mathbf{y} \mid \mathbf{b})\; \phi(\mathbf{b})\; \mathrm{d}\mathbf{b}}
$$

$N$ 维积分无法直接计算。实现中用**均值场近似**：固定 $b_{-i} = \tilde{b}_{-i}$，对第 $i$ 个分量做一维 Gauss-Hermite 积分。

换元 $b_i = \tilde{b}_i + \sqrt{2}\,\sigma_i\,t$（其中 $\sigma_i = 1/\sqrt{H_{ii}}$ 为边际后验标准差）：

$$
\hat{b}_i \approx \frac{\sum_{k=1}^{K} w_k\; b_i^{(k)}\; e^{f(b_i^{(k)}) + t_k^2}}
                       {\sum_{k=1}^{K} w_k\; e^{f(b_i^{(k)}) + t_k^2}}
$$

其中 $(t_k, w_k)$ 是 $K$ 点 Gauss-Hermite 节点和权重，$f(b_i) = \log p(\mathbf{y} \mid \mathbf{b}) + \log\phi(\mathbf{b})$。

$\hat{b}_i$ 是 BLUP（最佳线性无偏预测量），满足 $E[\hat{\mathbf{b}} - \mathbf{b}] = \mathbf{0}$（对随机效应 $\mathbf{b}$ 的随机性取期望）。

**计算代价**：$N \times K$ 次 log-lik 计算（例如 $N=10^5, K=20$ 时代价极高）。

---

### 路径 B：实用近似路径（SAIGE 实际做法）

$$
\tau \xrightarrow{\text{AI-REML}}
\hat{\tau}
\xrightarrow{\text{联合 NR}}
(\hat{\boldsymbol{\alpha}},\, \hat{\mathbf{b}}_{\mathrm{MAP}})
\xrightarrow{\text{SPA}}
p\text{-value}
$$

**步骤 B1：AI-REML 估计 $\hat{\tau}$**

SAIGE 使用**平均信息（Average Information，AI）**算法，它结合了 Fisher scoring 和 Newton-Raphson：

$$
\tau^{(t+1)} = \tau^{(t)} + \left( \frac{1}{2} \mathbf{y}^\top \mathbf{P} \boldsymbol{\Psi} \mathbf{P} \boldsymbol{\Psi} \mathbf{P} \mathbf{y} \right)^{-1}
\cdot \frac{1}{2}\left( \mathbf{y}^\top \mathbf{P} \boldsymbol{\Psi} \mathbf{P} \mathbf{y} - \operatorname{tr}(\mathbf{P}\boldsymbol{\Psi}) \right)
$$

其中 $\mathbf{P} = \boldsymbol{\Sigma}^{-1} - \boldsymbol{\Sigma}^{-1}\mathbf{X}(\mathbf{X}^\top \boldsymbol{\Sigma}^{-1}\mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\Sigma}^{-1}$ 是投影矩阵，$\boldsymbol{\Sigma} = \tau\boldsymbol{\Psi} + \mathbf{W}^{-1}$。

每步用**共轭梯度法**计算 $\mathbf{P}\mathbf{y}$（避免显式求逆 $N\times N$ 矩阵），时间复杂度 $O(N^2)$。

**步骤 B2：后验众数（MAP）替代后验均值**

$$
(\hat{\boldsymbol{\alpha}},\, \hat{\mathbf{b}}) = \operatorname{argmax}_{(\boldsymbol{\alpha}, \mathbf{b})}\; \left[ \log p(\mathbf{y} \mid \mathbf{b};\, \boldsymbol{\alpha}) + \log\phi(\mathbf{b};\, \mathbf{0},\, \hat{\tau}\boldsymbol{\Psi}) \right]
$$

用联合 Newton-Raphson（Schur complement block update）求解，确保 $\mathbf{X}^\top(\mathbf{y} - \hat{\boldsymbol{\mu}}) = \mathbf{0}$（固定效应得分方程成立）。

MAP 比后验均值偏向 $\mathbf{0}$（因后验轻微不对称），但当 $N$ 大时偏差很小。

**步骤 B3：SPA p 值**

对每个 SNP 做**鞍点近似（SPA）**：

$$
p = 2\left(1 - \Phi\!\left(w + \frac{1}{w}\ln\frac{v}{w}\right)\right)
$$

其中 $\hat{\zeta}$ 解方程 $K'(\hat{\zeta}) = q_{\mathrm{obs}}$，$w = \operatorname{sgn}(\hat{\zeta})\sqrt{2(\hat{\zeta} q_{\mathrm{obs}} - K(\hat{\zeta}))}$，
$v = \hat{\zeta}\sqrt{K''(\hat{\zeta})}$。

SPA 比正态近似在稀有变异（$p < 0.01$）下精确 1-2 个数量级。

---

## 4. 两种路径对比

| 项目 | 路径 A（理论最优） | 路径 B（SAIGE 实际） |
|---|---|---|
| $\tau$ 估计方法 | Laplace-REML，一维 Brent | AI-REML，Newton 迭代 |
| $\tau$ 偏差 | $O(N^{-2})$，渐近无偏 | $O(N^{-2})$，渐近无偏（同量级） |
| $\boldsymbol{\alpha}$ 估计方法 | 条件 MLE，NR | 联合 MAP，NR |
| $\boldsymbol{\alpha}$ 偏差 | $O(N^{-1})$ | $O(N^{-1})$（同量级） |
| $\mathbf{b}$ 估计方法 | GH 数值积分（后验均值） | 后验众数（MAP） |
| $\mathbf{b}$ 偏差 | 接近无偏（BLUP） | 有限样本有偏（MAP 偏向 0） |
| $p$ 值方法 | SPA | SPA |
| 时间复杂度（每次迭代） | $O(N^2 K)$ | $O(N^2)$ |
| $N = 10^5$ 下可行性 | 否（GH 积分太慢） | 是（SAIGE 实际支持） |

---

## 5. 无偏性

- **$\hat{\tau}$**：REML 在 LMM 中无偏，GLMM 中 Laplace-REML 渐近无偏（$O(N^{-2})$ 偏差）。
- **$\hat{\boldsymbol{\alpha}}$**：MLE 渐近无偏（$O(N^{-1})$ 偏差）；Firth 惩罚消除一阶偏差（用于稀有事件）。
- **$\hat{\mathbf{b}}$**：后验均值（BLUP）是最优线性无偏预测量；完整 MCMC 可做到，但不可扩展。
- **$p$ 值**：SPA 比正态近似精度高，但仍依赖 $\hat{\boldsymbol{\alpha}}, \hat{\tau}$ 的估计误差。

实际软件（SAIGE、GCTA、lme4）均采用 REML + MAP 的组合，接受 $\mathbf{b}$ 的有限样本偏差，换取计算可扩展性。
