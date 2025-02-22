MDP $\mathcal{M}(\mathcal{S}, \mathcal{A}, P, \gamma, r)$

策略 $\pi:\mathcal{S}\to\mathcal{A}$

策略 $\pi$ 的价值函数

$$
Q^{\pi}(s, a) := \mathbb{E}_{
    \displaystyle \stackrel{\displaystyle s_{i+1}\sim P(\cdot|s_i, a_i)}{a_{i+1}\sim\pi(\cdot|s_t)} 
}\left[\sum_{i=0}^{+\infty}\gamma^i r(s_i, a_i)\bigg|s_0=s,\;a_0=a\right]
$$


最优 Bellman 算子 $\mathcal{T}$ 定义为

$$
\mathcal{T} Q(s, a) := r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[\max_{a^\prime\in\mathcal{A}}Q(s^\prime, a^\prime)\right]\;\forall Q\in\mathcal{F}
$$

现在定义 masked Bellman 算子 $\mathcal{T}$ 

$$
\color{red}\mathcal{T}_{w} Q(s, a) := r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[\max_{a^\prime\in\mathcal{A}}w(s', a^\prime)Q(s^\prime, a^\prime)\right]\;\forall Q\in\mathcal{F}
$$

$$
\begin{aligned}
    &\mathcal{T}_{w} Q(s, a) := r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[\max_{a^\prime\in\mathcal{A}}w(s', a^\prime)Q(s^\prime, a^\prime)\right]\\
    &=r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[\max_{a^\prime\in\mathcal{A}}w(s', a^\prime)\mathbb{E}_{...}\sum_{...}\gamma^i r(s_i,a_i)|s\right]\\
\end{aligned}
$$

- $w$ 中应该是 $s$ 还是 $s^\prime$？
  - 应该是 $s^\prime$
  - 因为只有在涉及到 target q 计算的时候才会有 max，从而才会有 mask 的存在


大致需要以下 step
- 定义度量
- 证明空间的完备性
- 证明算子的压缩性

一些疑问：
- 结果是否要和原始的 Q* 差不多？
  - 按理说不应该是和原来的一样，不如为什么要这样设计呢？
- 怎么刻画 T_w 更好优化 power 这个命题？

**需要证明的是 $Q^*$ 和 新的不动点是比较接近的。**

（因为假设的 $r$ 没有问题，所以理论 $Q^*$ 是很好的，不应该差很远）

假设：
1. 可以假设确定性环境

2. 可以假设 表格形式的 S， A

3. w 是 0-1

## 定义度规
函数空间 $\mathcal{F}$ 表示所有 $\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ 的函数的集合。定义柯西度量 $d$

$$
d(Q_1, Q_2):=\sup_{s\in\mathcal{S},\;a\in\mathcal{A}} |Q_1(s, a) - Q_2(s, a)|
$$

## 空间完备性证明

> 证明思路：完备性的定义：任意柯西列收敛
> 
> 任意取一个柯西列，构造收敛到的函数。构造方法是拉回到实数域上去证明。

度量空间 $(d, \mathcal{F})$. 考虑柯西列 $\{Q_n\}\sub\mathcal{F}$. 根据定义有，$\forall \epsilon > 0,\;\exist N\in\mathbb{N}_+$ s.t. $\forall m, n > N$

$$
d(Q_m, Q_n) = \sup_{s\in\mathcal{S},\;a\in\mathcal{A}} |Q_m(s, a) - Q_n(s, a)| < \epsilon
$$

下面证明这个柯西列是收敛的。任意给定输入 $(s_0,\; a_0)\in\mathcal{S}\times\mathcal{A}$，则对一个实数域上的数列 $\{q_n\}=\{Q_n(s_0, a_0)\}\sub\mathbb{R}$. 首先证明这个数列是柯西列。$\forall\epsilon>0,$

$$
\begin{aligned}
    |q_m - q_n| &= |Q_m(s_0, a_0) - Q_n(s_0, a_0)|\\
    &\le \sup_{s\in\mathcal{S},\;a\in\mathcal{A}} |Q_m(s, a) - Q_n(s, a)|\\
    &<\epsilon
\end{aligned}
$$

只需要取 $N$ 为上面柯西列定义中的那个 $N$ 即可。于是 $\{q_n\}$ 是 $\mathbb{R}$ 上的柯西列，由于 $(\mathbb{R}, |\cdot-\cdot|)$ 是完备的，所以 $\{q_n\}$ 收敛。设 $\displaystyle q=\lim_{n\to\infty} q_n$.

构造函数 $Q\in\mathcal{F}$，且在 $(s_0, a_0)$ 上的取值定义为 $Q(s_0, a_0):= q$. 下面证明柯西列 $\{Q_n\}$ 收敛于 $Q$。

$\forall \epsilon>0$

$\forall (a_0, s_0) \in\mathcal{A}\times\mathcal{S}$, 

$$
\begin{aligned}
    |Q_n(s_0, a_0) - Q(s_0, a_0)| < \epsilon ?
\end{aligned}
$$

遇到的难题：需要有一个对任何 $(s_0, a_0)$ 都成立的只和 $\epsilon$ 相关的 $N$，应该是需要加一个一致连续的假设？

## 算子压缩性证明

$\forall Q_1,\; Q_2\in \mathcal{F}$

$$
\begin{aligned}
    |\mathcal{T}_{w} Q_1(s, a) - \mathcal{T}_{w} Q_2(s, a)|\\
    {\color{grey}{(\text{def. of } \mathcal{T}_w)}}&= \gamma\left|\mathbb{E}_{s^\prime}\left[\max_{a_1}w(s^\prime, a_1)Q_1(s^\prime, a_1) - \max_{a_2}w(s^\prime, a_2)Q_2(s^\prime, a_2)\right]\right|\\
    {\color{grey}{(|\mathbb{E}\cdot|\le\mathbb{E}|\cdot|)}}&\le \gamma \mathbb{E}_{s^\prime}\left|\max_{a_1}w(s^\prime, a_1)Q_1(s^\prime, a_1) - \max_{a_2}w(s^\prime, a_2)Q_2(s^\prime, a_2)\right|\\
\end{aligned}
$$

取定一个 $s^\prime$, 不妨设 $\max_{a_1}w(s^\prime, a_1)Q_1(s^\prime, a_1) \ge \max_{a_2}w(s^\prime, a_2)Q_2(s^\prime, a_2)$.
假设 $a_1$ 是使得 $w(s^\prime, \cdot)Q_1(s^\prime, \cdot)$ 取得最大的 $a_1$

$$
\begin{aligned}
    \left|\max_{a_1}w(s^\prime, a_1)Q_1(s^\prime, a_1) - \max_{a_2}w(s^\prime, a_2)Q_2(s^\prime, a_2)\right|\\
    {\color{grey}{(\text{left}\ge \text{right})}}&=
    \max_{a_1}w(s^\prime, a_1)Q_1(s^\prime, a_1) - \max_{a_2}w(s^\prime, a_2)Q_2(s^\prime, a_2)\\
    {\color{grey}{(\text{def. of } a_1)}}&=
    w(s^\prime, a_1)Q_1(s^\prime, a_1) - \max_{a_2}w(s^\prime, a_2)Q_2(s^\prime, a_2)\\
    {\color{grey}{(\text{def. of } \max_{a_2}\cdot)}}&\le w(s^\prime, a_1)Q_1(s^\prime, a_1) - w(s^\prime, a_1)Q_2(s^\prime, a_1)\\
    &=w(s^\prime, a_1)\left[Q_1(s^\prime, a_1)-Q_2(s^\prime, a_1)\right]\\
    {\color{grey}{|w(s^\prime, a_1)|\le 1}}&\le |Q_1(s^\prime, a_1) - Q_2(s^\prime, a_1)|\\
    {\color{grey}{\text{def. of } \sup_{a^{\prime\prime}, s^{\prime\prime}}\cdot}}&\le \sup_{s^{\prime\prime}, a^{\prime\prime}}|Q_1(s^{\prime\prime}, a^{\prime\prime}) - Q_2(s^{\prime\prime}, a^{\prime\prime})|
\end{aligned}
$$

$$
\begin{aligned}
    |\mathcal{T}_{w} Q_1(s, a) - \mathcal{T}_{w} Q_2(s, a)|\\
    &\le \gamma\mathbb{E}_{s^\prime}\left[\sup_{s, a}|Q_1(s, a) - Q_2(s, a)|\right]\\
    &=\gamma \sup_{s^{\prime\prime}, a^{\prime\prime}}|Q_1(s^{\prime\prime}, a^{\prime\prime}) - Q_2(s^{\prime\prime}, a^{\prime\prime})|
\end{aligned}
$$

这对任意 $(s, a)$ 成立。且注意到 $\sup_{s^{\prime\prime}, a^{\prime\prime}}|Q_1(s^{\prime\prime}, a^{\prime\prime}) - Q_2(s^{\prime\prime}, a^{\prime\prime})|$ 是不依赖于 $(s, a)$ 的常数，因此是函数 $|\mathcal{T}_{w} Q_1(s, a) - \mathcal{T}_{w} Q_2(s, a)|$ 的一个上界，因此大于它的上确界，即

$$
\sup_{s, a}|\mathcal{T}_{w} Q_1(s, a) - \mathcal{T}_{w} Q_2(s, a)| \le \gamma \sup_{s^{\prime\prime}, a^{\prime\prime}}|Q_1(s^{\prime\prime}, a^{\prime\prime}) - Q_2(s^{\prime\prime}, a^{\prime\prime})|
$$

所以这个算子是压缩映射。

## $\mathcal{T}_w$ 算子不动点的性质

- 假设 $Q\ge 0$
- 假设 $w(s, a)\le \Omega$

设 $Q_w\in\mathcal{F}$ 是 $\mathcal{T}_w$ 的不动点，$Q^*$ 是最优 Bellman 算子 $\mathcal{T}$，i.e.

$$
Q_w(s, a) = (\mathcal{T}_w Q_w)(s, a) = r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[\max_{a^\prime\in\mathcal{A}}w(s', a^\prime)Q_w(s^\prime, a^\prime)\right]
$$

$$
Q^*(s, a) = (\mathcal{T} Q^*)(s, a) = r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[\max_{a^\prime\in\mathcal{A}}Q^*(s^\prime, a^\prime)\right]
$$

$$
\begin{aligned}
    \left|Q_w(s, a) - Q^*(s, a)\right|\\
    &= \left|(\mathcal{T}_w Q_w)(s, a) - (\mathcal{T} Q^*)(s, a)\right|\\
    &=\gamma \left|\mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[\max_{a_1\in\mathcal{A}}w(s', a_1)Q_w(s^\prime, a_1) - \max_{a_2\in\mathcal{A}}Q^*(s^\prime, a_2)\right]\right|\\
    &\le\gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left|\max_{a_1\in\mathcal{A}}w(s', a_1)Q_w(s^\prime, a_1) - \max_{a_2\in\mathcal{A}}Q^*(s^\prime, a_2)\right|\\
\end{aligned}
$$

现在取定一个 $s^\prime$，不妨假设 $\max_{a_1\in\mathcal{A}}w(s', a_1)Q_w(s^\prime, a_1)\ge \max_{a_2\in\mathcal{A}}Q^*(s^\prime, a_2)$，并且假设 $a_1^\prime$ 就是使得 $w(s', a_1)Q_w(s^\prime, a_1)$ 取到最大的动作，$a_2^\prime$ 就是使得 $Q^*(s^\prime, a_2)$ 取得最大的动作。那么有 $\color{red}{\text{不能这样假设，因为不是对称的}}$。

$$
\begin{aligned}
    \left|\max_{a_1\in\mathcal{A}}w(s', a_1)Q_w(s^\prime, a_1) - \max_{a_2\in\mathcal{A}}Q^*(s^\prime, a_2)\right|\\
    &= \max_{a_1\in\mathcal{A}}w(s', a_1)Q_w(s^\prime, a_1) - \max_{a_2\in\mathcal{A}}Q^*(s^\prime, a_2)\\
    &= w(s', a_1^\prime)Q_w(s^\prime, a_1^\prime) - Q^*(s^\prime, a_2^\prime)\\
    &= w(s', a_1^\prime)Q_w(s^\prime, a_1^\prime) - w(s^\prime, a_1^\prime)Q^*(s^\prime, a_1^\prime) + w(s^\prime, a_1^\prime)Q^*(s^\prime, a_1^\prime) - Q^*(s^\prime, a_2^\prime)\\
    &\le w(s', a_1^\prime)Q_w(s^\prime, a_1^\prime) - w(s^\prime, a_1^\prime)Q^*(s^\prime, a_1^\prime) + w(s^\prime, a_1^\prime)Q^*(s^\prime, a_2^\prime) - Q^*(s^\prime, a_2^\prime)\\
    &=w(s^\prime, a_1^\prime)\left[Q_w(s^\prime, a_1^\prime) - Q^*(s^\prime, a_1^\prime)\right] - (1 - w(s^\prime, a_1^\prime)) Q^*(s^\prime, a^\prime_2)\\
    &\le \Omega \left|Q_w(s^\prime, a_1^\prime) - Q^*(s^\prime, a_1^\prime)\right| + (1 - \Omega) Q^*(s^\prime, a_2^\prime)\\
    &\le \Omega \sup_{s^{\prime\prime}, a^{\prime\prime}}|Q_w(s^{\prime\prime},a^{\prime\prime}) - Q^*(s^{\prime\prime},a^{\prime\prime})| + \frac{1 - \Omega}{1 - \gamma}R
\end{aligned}
$$

主意到 $Q^*(s, a)\le\frac{R}{1-\gamma}$. 上式的最后一项是一个常数，因此

$$
\begin{aligned}
    \left|Q_w(s, a) - Q^*(s, a)\right|
    &\le\gamma \Omega \sup_{s^{\prime\prime}, a^{\prime\prime}}|Q_w(s^{\prime\prime},a^{\prime\prime}) - Q^*(s^{\prime\prime},a^{\prime\prime})| + \frac{\gamma(1 - \Omega)}{1 - \gamma}R
\end{aligned}
$$

进一步的，式子两边同时对 $(s, a)$ 取 sup 得（右边是常数）

$$
\begin{aligned}
    \sup_{s^{\prime\prime}, a^{\prime\prime}}|Q_w(s^{\prime\prime},a^{\prime\prime}) - Q^*(s^{\prime\prime},a^{\prime\prime})|
    &\le\gamma \Omega \sup_{s^{\prime\prime}, a^{\prime\prime}}|Q_w(s^{\prime\prime},a^{\prime\prime}) - Q^*(s^{\prime\prime},a^{\prime\prime})| + \frac{\gamma(1 - \Omega)}{1 - \gamma}R\\
    \Rightarrow \sup_{s^{\prime\prime}, a^{\prime\prime}} |Q_w(s^{\prime\prime},a^{\prime\prime}) - Q^*(s^{\prime\prime},a^{\prime\prime})| \le \frac{\gamma(1 - \Omega)}{(1 - \gamma)(1 - \gamma\Omega)} R
\end{aligned}
$$


##

1. 把 w * Q 视作 q 值，更新的时候也把 w 带着走

$$
\mathcal{T}_{w} Q(s, a) := r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[\max_{a^\prime\in\mathcal{A}}w(s', a^\prime)Q(s^\prime, a^\prime)\right]
$$

2. w * Q 只用于选择动作，实际上仍然使用 q 值做迭代

$$
\mathcal{T}_{w} Q(s, a) := r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[Q\left(s^\prime, \max_{a^\prime\in\mathcal{A}}w(s', a^\prime)Q(s^\prime, a^\prime)\right)\right]
$$

# 和 PD-MARL 对齐的版本

$$
\mathcal{T}_{w} Q(s, a) := r(s, a) + \gamma \mathbb{E}_{s^\prime\sim P(\cdot|s, a)}\left[Q\left(s^\prime, \arg\max_{a^\prime\in\mathcal{A}}w(s', a^\prime)Q(s^\prime, a^\prime)\right)\right]
$$
