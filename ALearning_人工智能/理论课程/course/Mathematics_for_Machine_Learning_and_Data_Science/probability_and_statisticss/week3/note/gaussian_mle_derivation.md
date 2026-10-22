在课程视频中，你已经对高斯总体 (Gaussian population) 的均值 (mean) 和方差 (variance) 的最大似然估计 (Maximum Likelihood Estimation, MLE) 建立起了直观的认识。
在本篇阅读材料中，你将学习这两个估计量背后的完整数学推导过程。

## 数学形式化表述 (Mathematical formulation) 

假设你拥有来自均值为 $\mu$、方差为 $\sigma^2$ 的高斯分布 (Gaussian distribution) 的 $n$ 个样本 (samples) $X = (X_1, X_2, \dots, X_n)$。这意味着各个随机变量满足独立同分布 (i.i.d.) 条件，即 $X_i \stackrel{\text{i.i.d.}}{\sim} N(\mu, \sigma^2)$。
    你知道这些样本是出自正态分布, 但是你不知道具体参数是多少

所以你想要求解参数 (parameters) $\mu$ 和 $\sigma$ 的 MLE，
第一步是写出似然函数 (Likelihood function) 的定义。
    当 $\mu$ 和 $\sigma$ 均为未知参数时，似然 (likelihood) 将是这两个参数的函数。
对于随机样本 $X$ 的一组观测实现值 (realization) $x = (x_1, x_2, \dots, x_n)$，似然函数可以表示为：
    连乘
$$
\begin{array}{rl}
L(\mu,\sigma; \boldsymbol{x}) &= \prod_{i=1}^n f_{X_i}(x_i) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma } e^{-\frac{1}{2}\frac{(x_i-\mu)^2}{\sigma^2}} \\
&\\
&= \frac{1}{\left(\sqrt{2\pi}\right)^n\sigma^n }e^{-\frac{1}{2}\frac{\sum_{i=1}^n (x_i-\mu)^2}{\sigma^2}}
\end{array}
$$

现在，你只需要找到能够使似然函数 $L(\mu, \sigma; x)$ 取得最大值的 $\mu$ 和 $\sigma$ 的取值即可。

你或许还记得在微积分 (calculus) 课程中学过的方法：在解析求解这类问题时，一种标准思路是对似然函数求导数 (derivative) 并令其等于 0。使导数等于零的 $\mu$ 和 $\sigma$ 的取值即为函数的极值点 (extreme points) 。具体到我们所讨论的这个问题，这些极值点恰好就是最大值点 (maximums) 。

不过，由于似然函数中包含大量的连乘项，直接对它求导会非常繁琐。好在我们有一个非常巧妙的数学技巧可以大幅简化计算：注意到**对数函数 (logarithm function)** 是严格单调递增的，因此能够最大化似然函数 $L(\mu, \sigma; x)$ 的参数值，也必然会同时最大化它的对数。我们将取对数后的函数称为对数似然 (log-likelihood) ，其定义如下：

$$
\ell(\mu, \sigma) = \log(L(\mu, \sigma; x))
$$

对数运算具有将乘积转化为加和的优良性质，即满足 $\log(a \cdot b) = \log(a) + \log(b)$。这一性质使得对对数似然求导变得格外简单直接。为了将高斯总体的对数似然化简为最简形式，你还需要用到对数的另外两条基本性质：

$$
\log(1/a) = -\log(a)
$$

以及

$$
\log(a^k) = k\log(a)
$$

将以上性质结合起来进行化简，可以得到：
    拆三项
$$
\begin{array}{rl}
\ell(\mu,\sigma) &= \log\left( \frac{1}{\left(\sqrt{2\pi}\right)^n\sigma^n }e^{-\frac{1}{2}\frac{\sum_{i=1}^n (x_i-\mu)^2}{\sigma^2}}\right)\\
& = -\frac{n}{2}\log(2\pi) - n\log(\sigma) - \frac{1}{2}\frac{\sum_{i=1}^n (x_i-\mu)^2}{\sigma^2}
\end{array}
$$

接下来，为了求出 $\mu$ 和 $\sigma$ 的 MLE，我们只需分别计算对数似然关于这两个参数的**偏导数 (partial derivatives)** ，并令它们等于零即可。

首先计算关于 $\mu$ 的偏导数。请注意，==上式中的前两项并不包含 $\mu$，因此求导后直接消去==，由此可得：

$$
\begin{array}{rl}
\frac{\partial }{\partial \mu}\ell(\mu, \sigma) &= -\frac{1}{2}\frac{\sum_{i=1}^n 2(x_i -\mu)}{\sigma^2}(-1)\\
&\\
&=\frac{1}{\sigma^2}\left(\sum_{i=1}^n x_i -\sum_{i=1}^n \mu\right) = \frac{1}{\sigma^2}\left(\sum_{i=1}^n x_i -n\mu\right)
\end{array}
$$

接着，计算对数似然关于 $\sigma$ 的偏导数，可得：

$$
\frac{\partial}{\partial \sigma}\ell(\mu, \sigma) = -\frac{n}{\sigma} - \frac{1}{2}\left(\sum_{i=1}^n (x_i - \mu)^2\right)(-2)\frac{1}{\sigma^3} = -\frac{n}{\sigma} + \left(\sum_{i=1}^n (x_i - \mu)^2\right)\frac{1}{\sigma^3}
$$

下一步是令这两个偏导数分别等于 0，从而求解出 $\mu$ 和 $\sigma$ 的估计值 (estimates) 。让我们先从关于 $\mu$ 的偏导数开始：

$$
\frac{\partial}{\partial \mu}\ell(\mu, \sigma) = \frac{1}{\sigma^2}\left(\sum_{i=1}^n x_i - n\mu\right) = 0
$$

首先不难观察到，由于标准差满足 $\sigma > 0$，要使上式成立，唯一的可能是括号内的部分等于零，即 $\sum_{i=1}^n x_i - n\mu = 0$。经过简单的代数变形即可得出，$\mu$ 的 MLE 必然为：

$$
\hat{\mu} = \frac{\sum_{i=1}^n x_i}{n} = \overline{x},
$$

这恰好就是**样本均值 (sample mean)** 。

接下来，我们继续寻找能够满足 $\frac{\partial}{\partial \sigma}\ell(\mu, \sigma) = 0$ 的 $\sigma$ 值：

$$
\frac{\partial}{\partial \sigma}\ell(\mu, \sigma) = -\frac{n}{\sigma} + \left(\sum_{i=1}^n (x_i - \mu)^2\right)\frac{1}{\sigma^3} = 0
$$

在求解该方程时，首先同样注意到 $\sigma > 0$，因此在方程两边同乘 $\sigma$ 后，可以将表达式简化为：

$$
\frac{\partial}{\partial \sigma}\ell(\mu, \sigma) = -n + \left(\sum_{i=1}^n (x_i - \mu)^2\right)\frac{1}{\sigma^2} = 0
$$

此外，由于我们需要让两个偏导数同时等于 0，因此==可以直接用刚才求出的估计值 $\hat{\mu} = \overline{x}$ 替换式中的 $\mu$==，从而得到：

$$
\frac{\partial}{\partial \sigma}\ell(\mu, \sigma) = -n + \left(\sum_{i=1}^n (x_i - \overline{x})^2\right)\frac{1}{\sigma^2} = 0
$$

移项整理后即可得出方差的估计结果：

$$
\sigma^2 = \frac{\sum (x_i - \overline{x})^2}{n},
$$

因此，标准差 (standard deviation) 的 MLE 为：

$$
\hat{\sigma} = \sqrt{\frac{\sum (x_i - \overline{x})^2}{n}}
$$

这个表达式告诉我们：高斯总体标准差的 MLE，等于每个样本与样本均值之差的平方和的平均值的算术平方根。你会发现，这个公式与你在第 2 周学过的样本标准差 (sample standard deviation) 计算公式非常相似。两者唯一的区别在于归一化常数 (normalizing constant) 不同：==**在 MLE 中乘以的是 $1/n$，而在计算样本标准差时使用的则是 $1/(n-1)$。**==

最后补充一点说明：从严格的数学形式上讲，你刚才完成的只是求解临界点 (critical point) 的推导过程。如果追求推导的绝对严密性，你还需要进一步证明这些坐标对应的是一个最大值点 (而非最小值点 (minimum) 或鞍点 (saddle point) ) 。不过，这一步的证明需要用到稍微复杂一些的高等数学工具，因此我们在这里就略过不表了。

## 一个简单的例子 (A simple example) 

现在，让我们结合一个具体的例子来看看如何在实际数据中运用上述公式。假设你对美国 18 岁青年的身高分布情况感兴趣，并收集到了以下 10 个身高测量数据 (单位：英寸) ：

$$
\begin{matrix}66.75 & 70.24 & 67.19 & 67.09 & 63.65\\ 64.64 & 69.81 & 69.79 & 73.52 & 71.74 \end{matrix}
$$

假设每一个测量值都来自同一个参数 $\mu$ 和 $\sigma$ 未知的高斯分布。那么基于这组样本数据，这两个参数的 MLE 估计结果分别为：

$$
\hat{\mu} = \frac{66.75+70.24+67.19+67.09+63.65+64.64+69.81+69.79+73.52+71.74}{10} = 68.442
$$

以及

$$
\hat{\sigma} = \sqrt{\frac{1}{10}\left((66.75-68.442)^2+(70.24-68.442)^2+(67.19-68.442)^2+(67.09-68.442)^2+(63.65-68.442)^2+(64.64-68.442)^2+(69.81-68.442)^2+(69.79-68.442)^2+(73.52-68.442)^2+(71.74-68.442)^2\right)} = 2.954
$$
