## Formulation

动作-状态离散马尔科夫决策过程 $\mathcal{M}(\mathcal{S},\mathcal{A},P,r,\gamma)$

对于动作价值函数 $Q(s, a):\mathcal{S}\times\mathcal{A}\to\mathbb{R}_+$, 定义动作价值向量 $\mathbf{Q}\in\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+$ 是一个长为 $|\mathcal{S}||\mathcal{A}|$ 的向量，$\mathbf{Q}_{(s,a)}=Q(s,a)$.

转移矩阵 $\mathbf{P}\in\mathbb{R}^{|\mathcal{S}||\mathcal{A}|\times|\mathcal{S}|}_+$, $\mathbf{P}_{(s,a),s^\prime} = P(s^\prime|s,a)$.

奖励向量 $\mathbf{r}\in\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+$, $\mathbf{r}_{(s,a)}=r(s,a)$

先定义算子 $V:\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+\to\mathbb{R}^{|\mathcal{S}|}_+$

$$
(V[\mathbf{Q}])_{s} = \max_{a\in\mathcal{A}}\mathbf{Q}_{(s,a)}
$$

于是 Bellman 最优算子 $\mathcal{T}:\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+\to\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+$ 定义为

$$
\mathcal{T}[\mathbf{Q}] = \mathbf{r} + \gamma \mathbf{P}V[\mathbf{Q}]
$$

定义权重矩阵 $\color{orange}\mathbf{w}\in\mathbb{R}^{|\mathcal{S}||\mathcal{A}|\times |\mathcal{S}||\mathcal{A}|}$, $\mathbf{w}=\textbf{diag}(w_1,\;w_2\dotsb,w_{|\mathcal{S}||\mathcal{A}|})$

则 mask Bellman 算子 $\mathcal{T}_{\mathbf{w}}:\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+\to\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+$

$$
\mathcal{T}_{\mathbf{w}}[\mathbf{Q}] = \mathbf{r} + \gamma \mathbf{P}V[\mathbf{w}\mathbf{Q}]
$$

## Assumption

- $|\mathbf{r}|_\infty\le R$ where $R$ 是一个常数
- $1-\delta\le\mathbf{w}_{(s,a)}\le 1$ where $\delta$ 是一个不太大的正数
- new
  - 可以改成 $0< w \le 1$，证明的时候再引入 $\varepsilon = \min_{s,a} w(s,a)$
  - 可以再想想如果 w 逐渐衰减 到 1（趋向于均匀分布）

## 算子 $\mathcal{T}_{\mathbf{w}}$ 压缩性

$\forall \mathbf{Q}_1,\; \mathbf{Q}_2 \in\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+$

$$
\begin{aligned}
    \left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}_1]-\mathcal{T}_{\mathbf{w}}[\mathbf{Q}_2]\right|_\infty
    &=\gamma \left|\mathbf{P}(V[\mathbf{w}\mathbf{Q}_1] - V[\mathbf{w}\mathbf{Q}_2])\right|_\infty\\
    &\le\gamma\left|\mathbf{P}\right|_\infty\left|V[\mathbf{w}\mathbf{Q}_1] - V[\mathbf{w}\mathbf{Q}_2]\right|_\infty\\
    &\le \gamma\left|V[\mathbf{w}\mathbf{Q}_1] - V[\mathbf{w}\mathbf{Q}_2]\right|_\infty\\
    &=\gamma\max_{s\in\mathcal{S}}\left|(V[\mathbf{w}\mathbf{Q}_1])_s - (V[\mathbf{w}\mathbf{Q}_2])_s\right|_\infty\\
    &=\gamma\max_{s\in\mathcal{S}}\left|\max_{a_1\in\mathcal{A}}\mathbf{w}_{(s,a_1)}(\mathbf{Q}_1)_{(s,a_1)} - \max_{a_2\in\mathcal{A}}\mathbf{w}_{(s,a_2)}(\mathbf{Q}_2)_{(s,a_2)}\right|\\
\end{aligned}
$$

$\forall s \in\mathcal{S}$，取定 $s$ 后，不妨设

$$
\max_{a_1\in\mathcal{A}}\mathbf{w}_{(s,a_1)}(\mathbf{Q}_1)_{(s,a_1)} \ge \max_{a_2\in\mathcal{A}}\mathbf{w}_{(s,a_2)}(\mathbf{Q}_2)_{(s,a_2)}
$$

并且令 $a_1=\arg\max_{a_1\in\mathcal{A}}\mathbf{w}_{(s,a_1)}(\mathbf{Q}_1)_{(s,a_1)}$, $a_2=\arg\max_{a_2\in\mathcal{A}}\mathbf{w}_{(s,a_2)}(\mathbf{Q}_2)_{(s,a_2)}$

$$
\begin{aligned}
    &\left|\max_{a_1\in\mathcal{A}}\mathbf{w}_{(s,a_1)}(\mathbf{Q}_1)_{(s,a_1)} - \max_{a_2\in\mathcal{A}}\mathbf{w}_{(s,a_2)}(\mathbf{Q}_2)_{(s,a_2)}\right|\\
    &=\mathbf{w}_{(s,a_1)}(\mathbf{Q}_1)_{(s,a_1)} - \mathbf{w}_{(s,a_2)}(\mathbf{Q}_2)_{(s,a_2)}\\
    &\le \mathbf{w}_{(s,a_1)}(\mathbf{Q}_1)_{(s,a_1)} - \mathbf{w}_{(s,a_1)}(\mathbf{Q}_2)_{(s,a_1)}\\
    &= \mathbf{w}_{(s,a_1)} \left((\mathbf{Q}_1)_{(s,a_1)} - (\mathbf{Q}_2)_{(s,a_1)}\right)\\
    &\le \mathbf{w}_{(s,a_1)} |\left((\mathbf{Q}_1)_{(s,a_1)} - (\mathbf{Q}_2)_{(s,a_1)}\right)|\\
    &\le \max_{a\in\mathcal{A}} \left|(\mathbf{Q}_1)_{(s,a)} - (\mathbf{Q}_2)_{(s,a)}\right|\\
\end{aligned}
$$

故

$$
\begin{aligned}
    \left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}_1]-\mathcal{T}_{\mathbf{w}}[\mathbf{Q}_2]\right|_\infty
    &\le\gamma\max_{s\in\mathcal{S}}\left|\max_{a_1\in\mathcal{A}}\mathbf{w}_{(s,a_1)}(\mathbf{Q}_1)_{(s,a_1)} - \max_{a_2\in\mathcal{A}}\mathbf{w}_{(s,a_2)}(\mathbf{Q}_2)_{(s,a_2)}\right|\\
    &\le\gamma\max_{s\in\mathcal{S}}\max_{a\in\mathcal{A}} \left|(\mathbf{Q}_1)_{(s,a)} - (\mathbf{Q}_2)_{(s,a)}\right|\\
    &=\gamma \left|\mathbf{Q}_1 - \mathbf{Q}_2\right|_{\infty}
\end{aligned}
$$

## 算子 $\mathcal{T}_{\mathbf{w}}$ 不动点性质

设 $\mathbf{Q}^{\mathbf{w}}\in\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+$ 是 $\mathcal{T}_{\mathbf{w}}$ 不动点, $\mathbf{Q}^{*}\in\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+$ 是 $\mathcal{T}$ 不动点, i.e.

$$
\mathbf{Q}^{\mathbf{w}} = \mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{\mathbf{w}}]
$$

$$
\mathbf{Q}^{*} = \mathcal{T}[\mathbf{Q}^{*}]
$$

则有

$$
\begin{aligned}
    \left|\mathbf{Q}^{\mathbf{w}}-\mathbf{Q}^{*}\right|_\infty
    &= \left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{\mathbf{w}}]-\mathcal{T}[\mathbf{Q}^{*}]\right|_\infty\\
    &= \left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{\mathbf{w}}] - \mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{*}] + \mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{*}] - \mathcal{T}[\mathbf{Q}^{*}]\right|_\infty\\
    &\le \left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{\mathbf{w}}] - \mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{*}] \right|_{\infty} + \left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{*}] - \mathcal{T}[\mathbf{Q}^{*}]\right|_\infty\\
    &\le \gamma\left|\mathbf{Q}^{\mathbf{w}}-\mathbf{Q}^{*}\right|_\infty + \left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{*}] - \mathcal{T}[\mathbf{Q}^{*}]\right|_\infty\\
\end{aligned}
$$

$$
\Rightarrow \left|\mathbf{Q}^{\mathbf{w}}-\mathbf{Q}^{*}\right|_\infty\le\frac{1}{1-\gamma}\left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{*}] - \mathcal{T}[\mathbf{Q}^{*}]\right|_\infty
$$

下面考察 $\left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{*}] - \mathcal{T}[\mathbf{Q}^{*}]\right|_\infty$

$$
\begin{aligned}
    \left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{*}] - \mathcal{T}[\mathbf{Q}^{*}]\right|_\infty
    &=\gamma \left|\mathbf{P}V[\mathbf{Q}^{*}] - \mathbf{P}V[\mathbf{w}\mathbf{Q}^{*}]\right|_\infty\\
    &=\gamma \left|\mathbf{P}(V[\mathbf{Q}^{*}] - V[\mathbf{w}\mathbf{Q}^{*}])\right|_\infty\\
    &\le\gamma \left|\mathbf{P}\right|_\infty\left|V[\mathbf{Q}^{*}] - V[\mathbf{w}\mathbf{Q}^{*}]\right|_\infty\\
    &\le\gamma \left|V[\mathbf{Q}^{*}] - V[\mathbf{w}\mathbf{Q}^{*}]\right|_\infty\\
    &=\gamma \max_{s\in\mathcal{S}}\left|(V[\mathbf{Q}^{*}])_s - (V[\mathbf{w}\mathbf{Q}^{*}])_s\right|\\
\end{aligned}
$$

取定 $s\in\mathcal{S}$，对于这个固定的 $s$，不妨设 $a_1=\arg\max_{a\in\mathcal{A}}\mathbf{Q}^*_{(s,a)}$, $a_2=\arg\max_{a\in\mathcal{A}}\mathbf{w}_{(s,a)}\mathbf{Q}^*_{(s,a)}$. 由于 $\max\mathbf{w}\le 1$, 所以有

$$
(V[\mathbf{Q}^{*}])_s \ge (V[\mathbf{w}\mathbf{Q}^{*}])_s
$$

假设 $1 - \min\mathbf{w} \le \delta$

$$
(1 - \delta)\mathbf{Q}^*_{(s,a_1)}\le(\min\mathbf{w})\mathbf{Q}^*_{(s,a_1)}\le \mathbf{w}_{(s,a_1)}\mathbf{Q}^*_{(s,a_1)} \le \mathbf{w}_{(s,a_2)}\mathbf{Q}^*_{(s,a_2)}
$$

$$
\Rightarrow \mathbf{Q}^*_{(s,a_1)}\le \frac{\mathbf{w}_{(s,a_2)}}{1 - \delta}\mathbf{Q}^*_{(s,a_2)}
$$

于是

$$
\begin{aligned}
    |(V[\mathbf{Q}^{*}])_s - (V[\mathbf{w}\mathbf{Q}^{*}])_s|
    &= \mathbf{Q}^{*}_{(s,a_1)} - \mathbf{w}_{(s,a_2)}\mathbf{Q}^*_{(s,a_2)}\\
    &\le \frac{\mathbf{w}_{(s,a_2)}}{1 - \delta}\mathbf{Q}^*_{(s,a_2)} - \mathbf{w}_{(s,a_2)}\mathbf{Q}^*_{(s,a_2)}\\
    &\le \left(\frac{1}{1 - \delta} - 1\right)\mathbf{w}_{(s,a_2)}\mathbf{Q}^*_{(s,a_2)}\\
    &= \frac{\delta}{1 - \delta}\mathbf{w}_{(s,a_2)}\mathbf{Q}^*_{(s,a_2)}\\
    &= \frac{\delta}{1 - \delta}\frac{R}{1-\gamma}\\
\end{aligned}
$$

从而

$$
\begin{aligned}
    \left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}^{*}] - \mathcal{T}[\mathbf{Q}^{*}]\right|_\infty
    &\le\frac{\delta}{1 - \delta}\frac{\gamma R}{1-\gamma}\\
\end{aligned}
$$

最后

$$
\left|\mathbf{Q}^{\mathbf{w}}-\mathbf{Q}^{*}\right|_\infty\le\frac{\gamma R \delta}{(1-\delta)(1-\gamma)^2}
$$