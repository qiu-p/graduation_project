# 和 PD-MARL 对齐的版本

$$
\mathcal{T}_{w} Q(s, a) := r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[Q\left(s^\prime, \arg\max_{a^\prime\in\mathcal{A}}w(s', a^\prime)Q(s^\prime, a^\prime)\right)\right]
$$

## 算子压缩性证明

$\forall Q_1,\; Q_2\in \mathcal{F}$

$$
\begin{aligned}
    &|\mathcal{T}_{w} Q_1(s, a) - \mathcal{T}_{w} Q_2(s, a)|\\
    {\color{grey}{(\text{def. of } \mathcal{T}_w)}}&= \gamma\left|\mathbb{E}_{s^\prime}\left[Q_1\left(s^\prime, \arg\max_{a_1\in\mathcal{A}}w(s^\prime, a_1)Q_1(s^\prime, a_1)\right) - Q_2\left(s^\prime, \arg\max_{a_2\in\mathcal{A}}w(s^\prime, a_2)Q_2(s^\prime, a_2)\right)\right]\right|\\
    {\color{grey}{(|\mathbb{E}\cdot|\le\mathbb{E}|\cdot|)}}&\le \gamma \mathbb{E}_{s^\prime}\left|Q_1\left(s^\prime, \arg\max_{a_1\in\mathcal{A}}w(s^\prime, a_1)Q_1(s^\prime, a_1)\right) - Q_2\left(s^\prime, \arg\max_{a_2\in\mathcal{A}}w(s^\prime, a_2)Q_2(s^\prime, a_2)\right)\right|\\
\end{aligned}
$$

取定一个 $s^\prime\in\mathcal{S}$，不妨假设

$$
Q_1\left(s^\prime, \arg\max_{a_1\in\mathcal{A}}w(s^\prime, a_1)Q_1(s^\prime, a_1)\right) \ge Q_2\left(s^\prime, \arg\max_{a_2\in\mathcal{A}}w(s^\prime, a_2)Q_2(s^\prime, a_2)\right)
$$

，并且设 $a_1^\prime = \arg\max_{a_1\in\mathcal{A}}w(s^\prime, a_1)Q_1(s^\prime, a_1)$。则

$$
\begin{aligned}
    &\left|Q_1\left(s^\prime, \arg\max_{a_1\in\mathcal{A}}w(s^\prime, a_1)Q_1(s^\prime, a_1)\right) - Q_2\left(s^\prime, \arg\max_{a_2\in\mathcal{A}}w(s^\prime, a_2)Q_2(s^\prime, a_2)\right)\right|\\
    &=Q_1\left(s^\prime, a_1^\prime\right) - Q_2\left(s^\prime, \arg\max_{a_2\in\mathcal{A}}w(s^\prime, a_2)Q_2(s^\prime, a_2)\right)\\
    {\color{grey}{\text{def. of } \sup_{a_2\in\mathcal{A}}\cdot}}&\le Q_1(s^\prime, a_1^\prime) - Q_2(s^\prime, a_2^\prime)\\
    {\color{grey}{\cdot\le|\cdot|}}&\le |Q_1(s^\prime, a_1^\prime) - Q_2(s^\prime, a_2^\prime)|\\
    {\color{grey}{\cdot\le\sup_{\cdot\cdot}\cdot}}&\le \sup_{s^{\prime\prime}\in\mathcal{S},a^{\prime\prime}\in\mathcal{A}} |Q_1(s^{\prime\prime}, a^{\prime\prime}) - Q_2(s^{\prime\prime}, a^{\prime\prime})|
\end{aligned}
$$


注意 $\sup_{s^{\prime\prime}\in\mathcal{S},a^{\prime\prime}\in\mathcal{A}} |Q_1(s^{\prime\prime}, a^{\prime\prime}) - Q_2(s^{\prime\prime}, a^{\prime\prime})|$ 是和 $s^\prime$ 无关的常数。故

$$
\begin{aligned}
    &|\mathcal{T}_{w} Q_1(s, a) - \mathcal{T}_{w} Q_2(s, a)|\\
    &\le\gamma\mathbb{E}_{s^\prime}\sup_{s^{\prime\prime}\in\mathcal{S},a^{\prime\prime}\in\mathcal{A}} |Q_1(s^{\prime\prime}, a^{\prime\prime}) - Q_2(s^{\prime\prime}, a^{\prime\prime})|\\
    &=\gamma\sup_{s^{\prime\prime}\in\mathcal{S},a^{\prime\prime}\in\mathcal{A}} |Q_1(s^{\prime\prime}, a^{\prime\prime}) - Q_2(s^{\prime\prime}, a^{\prime\prime})|\\
\end{aligned}
$$

上式对任意 $(s, a)$ 成立，故 $\gamma\sup_{s^{\prime\prime}\in\mathcal{S},a^{\prime\prime}\in\mathcal{A}} |Q_1(s^{\prime\prime}, a^{\prime\prime}) - Q_2(s^{\prime\prime}, a^{\prime\prime})|$ 是 $|\mathcal{T}_{w} Q_1(s, a) - \mathcal{T}_{w} Q_2(s, a)|$ 的一个上界，因此大于等于它的上确界：

$$
\sup_{s\in\mathcal{S},a\in\mathcal{A}}|\mathcal{T}_{w} Q_1(s, a) - \mathcal{T}_{w} Q_2(s, a)|\le \gamma\sup_{s^{\prime\prime}\in\mathcal{S},a^{\prime\prime}\in\mathcal{A}} |Q_1(s^{\prime\prime}, a^{\prime\prime}) - Q_2(s^{\prime\prime}, a^{\prime\prime})|
$$

## $\mathcal{T}_w$ 算子不动点的性质

- 思路：
  - 极端情况下，$w(s,a)===1$ 那么 $Q_w$ 就是和 $Q^*$ 一样的。
  - 所以要是想让这俩足够接近，应当假设 $|w(s, a) - 1| < \delta$，并且 $\delta$ 是一个比较小的数。
  - 这样就可以做小量近似了。但是可能涉及求导，可能需要 Lipschitz 连续？

设 $Q_w\in\mathcal{F}$ 是 $\mathcal{T}_w$ 的不动点，$Q^*$ 是最优 Bellman 算子 $\mathcal{T}$，i.e.

$$
Q^*(s, a) = (\mathcal{T} Q^*)(s, a) = r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[\max_{a^\prime\in\mathcal{A}}Q^*(s^\prime, a^\prime)\right]
$$

$$
Q_w(s, a) = (\mathcal{T}_w Q_w)(s, a) = r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[Q_w\left(s^\prime, \arg\max_{a^\prime\in\mathcal{A}}w(s', a^\prime)Q_w(s^\prime, a^\prime)\right)\right]
$$

于是有

$$
\begin{aligned}
    &|Q_w(s, a) - Q^*(s, a)|\\
    &=|(\mathcal{T}_w Q_w)(s, a) - (\mathcal{T} Q^*)(s, a)|\\
    &=|(\mathcal{T}_w Q_w)(s, a) - (\mathcal{T}_w Q^*)(s, a) + (\mathcal{T}_w Q^*)(s, a) - (\mathcal{T} Q^*)(s, a)|\\
    &\le |(\mathcal{T}_w Q_w)(s, a) - (\mathcal{T}_w Q^*)(s, a)| + |(\mathcal{T}_w Q^*)(s, a) - (\mathcal{T} Q^*)(s, a)|\\
    &\le \gamma \sup_{s^\prime\in\mathcal{S},a^\prime\in\mathcal{A}}|Q_w(s^\prime, a^\prime) - Q^*(s^\prime, a^\prime)| + |(\mathcal{T}_w Q^*)(s, a) - (\mathcal{T} Q^*)(s, a)|
\end{aligned}
$$

其中

$$
\begin{aligned}
    &|(\mathcal{T}_w Q^*)(s, a) - (\mathcal{T} Q^*)(s, a)|\\
    &=\gamma\left|\mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[\max_{a_1\in\mathcal{A}}Q^*(s^\prime, a_1) - Q^*\left(s^\prime, \arg\max_{a_2\in\mathcal{A}}w(s', a_2)Q^*(s^\prime, a_2)\right)\right]\right|\\
    &\le\gamma\mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left|\max_{a_1\in\mathcal{A}}Q^*(s^\prime, a_1) - Q^*\left(s^\prime, \arg\max_{a_2\in\mathcal{A}}w(s', a_2)Q^*(s^\prime, a_2)\right)\right|\\
\end{aligned}
$$

下面来考察 $\displaystyle Q^*\left(s^\prime, \arg\max_{a_2\in\mathcal{A}}w(s', a_2)Q^*(s^\prime, a_2)\right)$ 的性质。令 $s_1^\prime = \arg\max_{a_1\in\mathcal{A}}Q^*(s^\prime, a_1)$, $a_2^\prime = \arg\max_{a_2\in\mathcal{A}}w(s', a_2)Q^*(s^\prime, a_2)$

假设 $\color{red}{w\le w(s, a)\le W}$，由 $\max$ 定义有

$$
\begin{aligned}
    W Q^*(s^\prime, a_2^\prime) \ge w(s^\prime, a_2^\prime)Q^*(s^\prime, a_2^\prime)\ge w(s^\prime, a_1^\prime)Q^*(s^\prime, a_1^\prime)\ge w Q^*(s^\prime, a_1^\prime)\\
    \Rightarrow Q^*(s^\prime, a_1^\prime)\le\frac{W}{w}Q^*(s^\prime, a_2^\prime)
\end{aligned}
$$

于是有

$$
\begin{aligned}
    &\left|\max_{a_1\in\mathcal{A}}Q^*(s^\prime, a_1) - Q^*\left(s^\prime, \arg\max_{a_2\in\mathcal{A}}w(s', a_2)Q^*(s^\prime, a_2)\right)\right|\\
    &=Q^*(s^\prime, a_1^\prime) - Q^*(s^\prime, a_2^\prime)\\
    &\le\left(\frac{W}{w} - 1\right)Q^*(s^\prime, a_2^\prime)\\
    &=\frac{W-w}{w}Q^*(s^\prime, a_2^\prime)\\
    &\le\frac{W-w}{w}\frac{R}{1-\gamma}
\end{aligned}
$$

注意上式最后是一个和 $s^\prime$ 无关的常数。于是有

$$
\begin{aligned}
    &|(\mathcal{T}_w Q^*)(s, a) - (\mathcal{T} Q^*)(s, a)|\\
    &\le\gamma\mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left|\max_{a_1\in\mathcal{A}}Q^*(s^\prime, a_1) - Q^*\left(s^\prime, \arg\max_{a_2\in\mathcal{A}}w(s', a_2)Q^*(s^\prime, a_2)\right)\right|\\
    &\le\frac{(W - w)\gamma R}{w(1-\gamma)}
\end{aligned}
$$

从而

$$
\begin{aligned}
    &|Q_w(s, a) - Q^*(s, a)|\\
    &\le \gamma \sup_{s^\prime\in\mathcal{S},a^\prime\in\mathcal{A}}|Q_w(s^\prime, a^\prime) - Q^*(s^\prime, a^\prime)| + |(\mathcal{T}_w Q^*)(s, a) - (\mathcal{T} Q^*)(s, a)|\\
    &\le \gamma \sup_{s^\prime\in\mathcal{S},a^\prime\in\mathcal{A}}|Q_w(s^\prime, a^\prime) - Q^*(s^\prime, a^\prime)| + \frac{(W-w)\gamma R}{w(1-\gamma)}\\
\end{aligned}
$$

注意到上式的右侧是一个和 $(s,a)$ 无关的常数，因此是 $|Q_w(s, a) - Q^*(s, a)|$ 的一个上界，所以大于它的上确界：

$$
\begin{aligned}
    &\sup_{s\in\mathcal{S},a\in\mathcal{A}}|Q_w(s, a) - Q^*(s, a)|
    &\le \gamma \sup_{s^\prime\in\mathcal{S},a^\prime\in\mathcal{A}}|Q_w(s^\prime, a^\prime) - Q^*(s^\prime, a^\prime)| + \frac{(W- w )\gamma R}{w(1-\gamma)}\\
\end{aligned}
$$

$$
\Rightarrow \sup_{s\in\mathcal{S},a\in\mathcal{A}}|Q_w(s, a) - Q^*(s, a)|\le\frac{(W-w)\gamma R}{w(1-\gamma)^2}
$$