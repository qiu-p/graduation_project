## Step 0. Notation

集合 $\mathcal{F}$ 为所有 Full adder 的集合, $\mathcal{H}$ 为所有 Half adder 的集合。

定义 stage $i$ column $j$ 为 slice $_{i,j}$ . Slice $_{i,j}$ 中的全加器的集合 $\mathcal{F}_{i,j}$ ，半加器的集合为 $\mathcal{H}_{i,j}$ .

记最大阶段数量为 $S$ , 记列数为 $N$

## Step 1: Linear Switching Propagation Power Model
原始情况下， $y = f(x_1, x_2, \dotsb, x_n)$ 的传播计算为

$$
T[y] = \sum_{i=1}^n \mathbb{E}\left[\frac{\partial f}{\partial x_i}\right] T[x_i]
$$

由于压缩树的<font color=orange> 周期性结构</font>，因此 <font color=orange> 假设输入的占空比 $P(x_i=1)$ 是一个常数。 </font>

从而对于全加器 $f\in\mathcal{F}$

$$
T_f[s] = {\color{orange}\nu_s[a]} T[a] + {\color{orange}\nu_s[b]} T[b] + {\color{orange}\nu_s[c]} T[c]
$$

$$
T_f[c] = {\color{orange}\nu_c[a]} T[a] + {\color{orange}\nu_c[b]} T[b] + {\color{orange}\nu_c[c]} T[c]
$$

对于半加器 $h\in\mathcal{H}$

$$
T_h[s] = {\color{orange}\mu_s[a]} T[a] + {\color{orange}\mu_s[b]} T[b]
$$

$$
T_h[c] = {\color{orange}\mu_c[a]} T[a] + {\color{orange}\mu_c[b]} T[b]
$$

而对于 power 的计算，有

$$
P_f = {\color{orange}\beta[a]} T[a] + {\color{orange}\beta[b]} T[b] + {\color{orange}\beta[c]} T[c] + {\color{orange}\beta}
$$

$$
P_h = {\color{orange}\alpha[a]} T[a] + {\color{orange}\alpha[b]} T[b] + {\color{orange}\alpha}
$$

则总体的 power 为

$$
P = \sum_{f\in\mathcal{F}} P_f + \sum_{h\in\mathcal{H}} P_h
$$

从而这是关于 $T[\cdot]$ 的线性函数，且目前所有和 $T$ 相关的约束条件都是线性约束。

在这个模型中共有 17 个需要学习的参数，通过 Ceres/Scipy 得到。

## Step 2: Mix Interger Programing

假设 slice$_{i,j}$ 的输入有 $m$ 个 pp. 那么记这个 $m$ 个输入的切换频率为 $\mathbf{T}_{i,j}\in\mathbb{R}^m$ ，排序后为 $\widetilde{\mathbf{T}}_{ij}\in\mathbb{R}^m$ ，变换矩阵为 $\mathbf{Z}_{i,j}\in\mathbb{R}^{m\times m}$ . 则

$$
\widetilde{\mathbf{T}}_{ij} = \mathbf{Z}_{i,j} {\mathbf{T}}_{ij}
$$

并且

$$
\sum_{k} (\mathbf{Z}_{i,j})_{(k,\cdot)} = 1
$$

$$
\sum_{l} (\mathbf{Z}_{i,j})_{(\cdot,l)} = 1
$$

$$
(\mathbf{Z}_{i,j})_{(k,l)}=0,1
$$

转化为线性约束，则为 对于一个大常数 $Z\gg 1$

$$
(\widetilde{\mathbf{T}}_{ij})_{k} - ({\mathbf{T}}_{ij})_{l} \le Z [1 - (\mathbf{Z}_{i,j})_{(k,l)}]
$$

$$
({\mathbf{T}}_{ij})_{k} - (\widetilde{\mathbf{T}}_{ij})_{l} \le Z [1 - (\mathbf{Z}_{i,j})_{(k,l)}]
$$

根据代码中的规则，输出的排线顺序是 3:2s, 2:2 s, res, 3:2c, 2:2c

于是就有

$$
\kappa_{i,j} = pp_{i,j+1} - 2 f_{i,j+1} - h_{i,j+1}
$$

- 3:2

$$
\begin{aligned}
    ({\mathbf{T}}_{i+1,j})_{k} = {\color{orange}\nu_s[a]} (\widetilde{\mathbf{T}}_{ij})_{3k} + {\color{orange}\nu_s[b]} (\widetilde{\mathbf{T}}_{ij})_{3k+1} + {\color{orange}\nu_s[c]} (\widetilde{\mathbf{T}}_{ij})_{3k+2}\;
    && 0\le k < f_{i,j}
\end{aligned}
$$

$$
\begin{aligned}
    ({\mathbf{T}}_{i+1,j+1})_{k+\kappa_{i,j}} = {\color{orange}\nu_c[a]} (\widetilde{\mathbf{T}}_{ij})_{3k} + {\color{orange}\nu_c[b]} (\widetilde{\mathbf{T}}_{ij})_{3k+1} + {\color{orange}\nu_c[c]} (\widetilde{\mathbf{T}}_{ij})_{3k+2}
    && 0\le k < f_{i,j}
\end{aligned}
$$

$$
(P_f){i,j} = \sum_{k=0}^{f_{i,j} - 1}\left[
    {\color{orange}\beta[a]} (\widetilde{\mathbf{T}}_{ij})_{3k} + {\color{orange}\beta[b]} (\widetilde{\mathbf{T}}_{ij})_{3k+1} + {\color{orange}\beta[c]} (\widetilde{\mathbf{T}}_{ij})_{3k+2} + {\color{orange}\beta}
\right]
$$

- 2:2

$$
\begin{aligned}
    ({\mathbf{T}}_{i+1,j})_{k+f_{i,j}} = {\color{orange}\mu_s[a]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k} + {\color{orange}\mu_s[b]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k + 1}\;
    && 0\le k < h_{i,j}
\end{aligned}
$$

$$
\begin{aligned}
    ({\mathbf{T}}_{i+1,j+1})_{k+f_{i,j}+\kappa_{i,j}} = {\color{orange}\mu_c[a]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k} + {\color{orange}\mu_c[b]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k + 1}
    && 0\le k < h_{i,j}
\end{aligned}
$$

$$
(P_h){i,j} = \sum_{k=0}^{f_{i,j} - 1}\left[
    {\color{orange}\alpha[a]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k} + {\color{orange}\alpha[b]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k + 1} + {\color{orange}\alpha}
\right]
$$

- remain

$$
\begin{aligned}
    ({\mathbf{T}}_{i+1,j})_{k+f_{i,j}+h_{i,j}} = (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j} + 2h_{i,j} + k}
    && 0\le k < pp_{i,j}-3f_{i,j}-2h_{i,j}
\end{aligned}
$$

- pp 约束

$$
pp_{i + 1,j} = pp_{i,j} - 2 f_{i,j} - h_{i,j} + f_{i,j-1} + h_{i,j-1}
$$

- 初始 $T$ 约束

$$
\mathbf{T}_{0,j} = \mathbf{1}
$$

- 其他约束和 GOMIL 一样

- 优化目标

$$
\min P = \sum_{i,j}[(P_f)_{i,j} + (P_h)_{i,j}]
$$

注意，当参数 $\color{orange} \alpha[\cdot]$ 和 $\color{orange} \beta[\cdot]$ 取 $0$ 时，问题退化为 GOMIL 中和面积优化形式一样的问题。

## Step 3 形式化数学问题

$$
\begin{align}
    \min \sum_{i=0}^{S-1} \sum_{j=0}^{N-1}\left[(P_f)_{i,j} + (P_h)_{i,j}\right],
\end{align}
$$

$$
\begin{align}
    \textbf{s.t.} && (P_f)_{i,j} = \sum_{k=0}^{f_{i,j} - 1}\left[
    {\color{orange}\beta[a]} (\widetilde{\mathbf{T}}_{ij})_{3k} + {\color{orange}\beta[b]} (\widetilde{\mathbf{T}}_{ij})_{3k+1} + {\color{orange}\beta[c]} (\widetilde{\mathbf{T}}_{ij})_{3k+2} + {\color{orange}\beta}
\right],

&& 0\le i < S,\;0\le j < N,
\end{align}
$$

$$
\begin{align}
(P_h)_{i,j} = \sum_{k=0}^{f_{i,j} - 1}\left[
    {\color{orange}\alpha[a]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k} + {\color{orange}\alpha[b]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k + 1} + {\color{orange}\alpha}
\right],
&& 0\le i < S,\;0\le j < N,
\end{align}
$$

$$
\begin{align}
    ({\mathbf{T}}_{i+1,j})_{k} = {\color{orange}\nu_s[a]} (\widetilde{\mathbf{T}}_{ij})_{3k} + {\color{orange}\nu_s[b]} (\widetilde{\mathbf{T}}_{ij})_{3k+1} + {\color{orange}\nu_s[c]} (\widetilde{\mathbf{T}}_{ij})_{3k+2},\;
    && 0\le k < f_{i,j},\;0\le i < S-1,\;0 \le j < N,
\end{align}
$$

$$
\begin{align}
    \kappa_{i,j} := pp_{i,j+1} - 2 f_{i,j+1} - h_{i,j+1},
    && 0\le i < S-1,\;0\le j < N - 1,
\end{align}
$$

$$
\begin{align}
    ({\mathbf{T}}_{i+1,j+1})_{k+\kappa_{i,j}} = {\color{orange}\nu_c[a]} (\widetilde{\mathbf{T}}_{ij})_{3k} + {\color{orange}\nu_c[b]} (\widetilde{\mathbf{T}}_{ij})_{3k+1} + {\color{orange}\nu_c[c]} (\widetilde{\mathbf{T}}_{ij})_{3k+2},
    && 0\le k < f_{i,j},\;0\le i < S-1,\;0\le j < N - 1,
\end{align}
$$

$$
\begin{align}
    ({\mathbf{T}}_{i+1,j})_{k+f_{i,j}} = {\color{orange}\mu_s[a]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k} + {\color{orange}\mu_s[b]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k + 1}\;
    && 0\le k < h_{i,j},\;0\le i < S-1,\;0\le j < N,
\end{align}
$$

$$
\begin{align}
    ({\mathbf{T}}_{i+1,j+1})_{k+f_{i,j}+\kappa_{i,j}} = {\color{orange}\mu_c[a]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k} + {\color{orange}\mu_c[b]} (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j}+2k + 1}
    && 0\le k < h_{i,j},\;0\le i < S-1,\;0\le j < N - 1,
\end{align}
$$


$$
\begin{align}
    ({\mathbf{T}}_{i+1,j})_{k+f_{i,j}+h_{i,j}} = (\widetilde{\mathbf{T}}_{ij})_{3f_{i,j} + 2h_{i,j} + k},
    && 0\le k < pp_{i,j} - 3f_{i,j} - 2h_{i,j}
\end{align}
$$

$$
\begin{align}
    (\widetilde{\mathbf{T}}_{ij})_{k} - ({\mathbf{T}}_{ij})_{l} \le Z [1 - (\mathbf{Z}_{i,j})_{(k,l)}],
    && 0\le i < S,\;0\le j < N,\; 0\le k,l < pp_{i,j},
\end{align}
$$

$$
\begin{align}
    ({\mathbf{T}}_{ij})_{k} - (\widetilde{\mathbf{T}}_{ij})_{l} \le Z [1 - (\mathbf{Z}_{i,j})_{(k,l)}],
    && 0\le i < S,\;0\le j < N,\; 0\le k,l < pp_{i,j},
\end{align}
$$

$$
\begin{align}
    (\mathbf{T}_{0,j})_{k} = 1.0,
    && 0 \le j < N - 1,\; 0\le k < pp_{0, j},
\end{align}
$$

$$
\begin{align}
    ({\mathbf{T}}_{ij})_{k}\ge 0,\;(\widetilde{\mathbf{T}}_{ij})_{k}\ge 0,
    && 0\le i < S,\;0\le j < N,\; 0\le k < pp_{i,j},
\end{align}
$$

$$
\begin{align}
    (\mathbf{Z}_{i,j})_{(k,l)}\in\{0,1\},
    && 0\le i < S,\;0\le j < N,\; 0\le k,l < pp_{i,j},
\end{align}
$$

$$
\begin{align}
    \sum_{k=0}^{pp_{i,j}-1} (\mathbf{Z}_{i,j})_{(k,l)} = 1,
    && 0\le i < S,\;0\le j < N,\; 0\le l < pp_{i,j},
\end{align}
$$

$$
\begin{align}
    \sum_{l=0}^{pp_{i,j}-1} (\mathbf{Z}_{i,j})_{(\cdot,l)} = 1
    && 0\le i < S,\;0\le j < N,\; 0\le l < pp_{i,j}.
\end{align}
$$

$$
\begin{align}
    pp_{i+1,j} = pp_{i,j} - 2 f_{i,j} - h_{i,j} + f_{i,j-1} + h_{i,j-1},
    && 0\le i < S-1,\; 1\le j < N,
\end{align}
$$

$$
\begin{align}
    pp_{i+1,0} = pp_{i,0} - 2 f_{i,0} - h_{i,0},
    && 0\le i < S-1,
\end{align}
$$