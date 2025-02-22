# 和 PD-MARL 对齐的离散的版本

## Formulation

动作-状态离散马尔科夫决策过程 $\mathcal{M}(\mathcal{S},\mathcal{A},P,r,\gamma)$

$$
\mathcal{T}_{w} Q(s, a) := r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[Q\left(s^\prime, \arg\max_{a^\prime\in\mathcal{A}}w(s', a^\prime)Q(s^\prime, a^\prime)\right)\right]
$$

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

定义权重矩阵 $\color{orange}\mathbf{w}\in\mathbb{R}^{|\mathcal{S}||\mathcal{A}|\times |\mathcal{S}||\mathcal{A}|}$, 定义算子 $V_\mathbf{w}:\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+\to\mathbb{R}^{|\mathcal{S}|}_+$:

$$
(V_\mathbf{w}[\mathbf{Q}])_s = \mathbf{Q}_{(s, a^\prime)}\;\text{ where }\; a^\prime = \arg\max_{a\in\mathcal{A}} (\mathbf{W}\mathbf{Q})_{(s,a)}
$$

则 mask Bellman 算子 $\mathcal{T}_{\mathbf{w}}:\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+\to\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+$

$$
\mathcal{T}_{\mathbf{w}}[\mathbf{Q}] = \mathbf{r} + \gamma \mathbf{P}V_{\mathbf{w}}[\mathbf{Q}]
$$

## 算子压缩性证明

$\forall \mathbf{Q}_1,\; \mathbf{Q}_2 \in\mathbb{R}^{|\mathcal{S}||\mathcal{A}|}_+$

$$
\begin{aligned}
    \left|\mathcal{T}_{\mathbf{w}}[\mathbf{Q}_1]-\mathcal{T}_{\mathbf{w}}[\mathbf{Q}_2]\right|_\infty
    &=\gamma \left|\mathbf{P}(V_{\mathbf{w}}[\mathbf{Q}_1] - V_{\mathbf{w}}[\mathbf{Q}_2])\right|_\infty\\
    &\le\gamma\left|\mathbf{P}\right|_\infty\left|V_{\mathbf{w}}[\mathbf{Q}_1] - V_{\mathbf{w}}[\mathbf{Q}_2]\right|_\infty\\
    &\le \gamma\left|V_{\mathbf{w}}[\mathbf{Q}_1] - V_{\mathbf{w}}[\mathbf{Q}_2]\right|_\infty
\end{aligned}
$$

$\forall s \in\mathcal{S}$，取定 $s$ 后，不妨设 $(V_\mathbf{w}[\mathbf{Q}_1])_s\ge (V_\mathbf{w}[\mathbf{Q}_2])_s$, $a_1=\arg\max_{a\in\mathcal{A}}(\mathbf{W}\mathbf{Q}_1)_{(s,a)}$， $a_2=\arg\max_{a\in\mathcal{A}}(\mathbf{W}\mathbf{Q}_2)_{(s,a)}$. 那么有

$$
(\mathbf{W}\mathbf{Q}_2)_{(s,a_2)} \ge (\mathbf{W}\mathbf{Q}_2)_{(s,a_1)}
$$

$$
||
$$
