___
# 实验：中心极限定理 (Central Limit Theorem)



欢迎！在这个不计分实验中，你将看到中心极限定理 (Central Limit Theorem, CLT) 在不同数据分布中的实际应用。你将直观观察到该定理如何在实践中发挥作用，同时也会探索该定理失效的特殊场景。

让我们立即开始！

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

import utils

```

## 高斯分布总体 (Gaussian population)



我们从最直观的场景切入：当你的总体 (population) 服从高斯分布 (正态分布) 时。你可以使用 np.random.normal 函数来生成该总体的模拟数据。

```python
mu = 10
sigma = 5

gaussian_population = np.random.normal(mu, sigma, 100_000)

```

该总体的真实均值 (mean) 为 10，标准差 (standard deviation) 为 5 (因为这是我们生成数据时设定的真实参数) ，总共包含 100,000 个观测值。运行下方代码单元可以查看其直方图 (histogram)：

```python
sns.histplot(gaussian_population, stat="density")
plt.show()

```

## 从总体中抽样 (Sampling from the population)



由于本实验使用的是仿真数据，你完全可以直接利用整个总体来得出统计结论。例如，即使事先不知道 $\mu$ 和 $\sigma$ 的取值，你也可以通过直接计算整个总体的均值和标准差，获得与真实参数极度接近的估计值：

```python
gaussian_pop_mean = np.mean(gaussian_population)
gaussian_pop_std = np.std(gaussian_population)

print(f"Gaussian population has mean: {gaussian_pop_mean:.1f} and std: {gaussian_pop_std:.1f}")

```

然而在现实世界中，获取全量总体数据几乎是不可能的，你必须借助远小于总体的样本 (samples) 来推断整体数据的潜在规律。毕竟，这正是统计推断的核心所在。

根据抽样技术的不同，样本会表现出不同的统计特性，而这正是中心极限定理大显身手的地方。对于许多分布而言 (但**并非全部**) ，以下规律均成立：

无论各个随机变量本身的原始分布为何，大量独立同分布 (independent and identically distributed, i.i.d.) 随机变量的和或均值，其分布往往趋近于正态分布 (normal distribution)。这一结论至关重要，因为正态分布具有完备且成熟的理论体系，是进行统计推断 (statistical inference) 和假设检验 (hypothesis testing) 的基石。

明确了这一背景后，我们需要一种从总体中反复抽取样本并计算均值的方法。为此，我们定义了 `sample_means` 函数：

```python
def sample_means(data, sample_size):
    # 用列表保存所有的样本均值
    means = []

    # 抽取大量样本
    # 该数值并不影响定理本身的成立，但决定了直方图的平滑美观程度 (样本数越多，图表越细腻)
    for _ in range(10_000):
        # 从数据中有放回地抽取样本 (WITH replacement)
        sample = np.random.choice(data, size=sample_size)

        # 保存该样本的均值
        means.append(np.mean(sample))

    # 将均值列表转换为 numpy 数组返回
    return np.array(means)

```

让我们梳理一下上述函数的执行逻辑：

* 从总体中进行随机抽样 (采用有放回抽样，意味着某个元素被选出后会被重新放回抽样空间中，同一个元素可能被多次抽中)。这确保了抽样过程满足独立性条件。


* 计算每个样本的均值。


* 将所有样本的均值整理并保存在一个 NumPy 数组中。



该定理表明，只要样本量 `sample_size` 足够大 (通常认为大于 30) ，样本均值的抽样分布就应该近似服从高斯分布。运行下方代码直观验证这一结论：

```python
# 计算样本均值
gaussian_sample_means = sample_means(gaussian_population, sample_size=5)

# 绘制样本均值的直方图
sns.histplot(gaussian_sample_means, stat="density")
plt.show()

```

样本均值的分布看起来非常符合高斯分布的形态。但仅凭视觉印象还不足以严谨断定定理是否成立，毕竟在本次演示中设定的样本量 `sample_size` 非常小。检验样本均值是否严格服从高斯分布有多种科学方法。

第一种方法是计算样本均值的理论均值 $\mu_{\bar{X}}$ 与理论标准差 $\sigma_{\bar{X}}$。这两个理论值的计算公式如下：

* $\mu_{\bar{X}} = \mu$

* $\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$


**注：此处的 $n$ 代表单次抽样的样本量 (sample size)。**

接着，使用这两个理论参数绘制一条理论高斯概率密度曲线。若定理成立，则实验得到的样本均值经验分布应当与该理论曲线高度契合。运行下一个代码单元将其叠加入图中：

```python
# 计算估计均值
mu_sample_means = mu

# 计算估计标准差
# 此处分母取 5 是因为此前设定的 sample_size 为 5
sigma_sample_means = sigma / np.sqrt(5)

# 定义高斯曲线的 x 轴取值范围 (仅用于绘图渲染)
x_range = np.linspace(min(gaussian_sample_means), max(gaussian_sample_means), 100)

# 将直方图与理论曲线绘制在同一张图上
sns.histplot(gaussian_sample_means, stat="density")
plt.plot(
    x_range,
    norm.pdf(x_range, loc=mu_sample_means, scale=sigma_sample_means),
    color="black",
)
plt.show()

```

二者呈现出极高的相似度。更进一步，我们可以利用核密度估计 (kernel density estimation, KDE) 拟合出样本均值平滑的经验概率密度函数。如果该平滑曲线与理论高斯曲线高度重合，便能更确凿地说明样本均值的分布与高斯分布高度一致：

```python
# 样本均值的直方图 (蓝色)
sns.histplot(gaussian_sample_means, stat="density", label="hist")

# 样本均值的估计概率密度曲线 KDE (红色)
sns.kdeplot(
    data=gaussian_sample_means,
    color="crimson",
    label="kde",
    linestyle="dashed",
    fill=True,
)

# 基于估计 mu 和 sigma 绘制的理论高斯曲线 (黑色)
plt.plot(
    x_range,
    norm.pdf(x_range, loc=mu_sample_means, scale=sigma_sample_means),
    color="black",
    label="gaussian",
)

plt.legend()
plt.show()

```

两条曲线几乎完全重叠！

另一种检验正态性的经典工具是分位数-分位数图 (QQ plot, Quantile-Quantile plot)。如果样本数据服从高斯分布，QQ 图中的散点将紧密排布在一条对角直线上：

```python
# 创建 QQ 图
fig, ax = plt.subplots(figsize=(6, 6))
res = stats.probplot(gaussian_sample_means, plot=ax, fit=True)
plt.show()

```

绘制出的 QQ 图呈现出近乎完美的直线排列，这再次强有力地证实了样本均值确实服从高斯分布。

现在，让我们在交互式小部件中整合上述全套分析，以便自由调节 $\mu$、$\sigma$ 以及 `sample_size`。**请注意：修改参数后需要点击 `Run Interact` 按钮以刷新图表**：

```python
utils.gaussian_clt()

```

在高斯总体下，即便 `sample_size` 极小，样本均值依然服从高斯分布。这实际上源于高斯分布自身的优良数学性质。

接下来，让我们在非高斯分布中检验中心极限定理！

## 二项分布总体 (Binomial Population)



现在我们尝试一种非高斯的总体分布。课程中讲解过的二项分布 (Binomial distribution) 便是典型代表。为了生成服从二项分布的数据，需要预先指定试验次数 `n` 与成功概率 `p` 两个核心参数：

```python
n = 5
p = 0.8

binomial_population = np.random.binomial(n, p, 100_000)

```

该总体总共包含 100,000 个观测值。运行下方单元格可以查看其频数直方图：

```python
sns.histplot(binomial_population, stat="count")
plt.show()

```

二项分布的均值和标准差无法直接从生成函数的入参中一步读出 (不像高斯分布生成时直接传入了 $\mu$ 和 $\sigma$)。但我们同样可以直接对总体样本进行统计计算：

```python
binomial_pop_mean = np.mean(binomial_population)
binomial_pop_std = np.std(binomial_population)

print(f"Gaussian population has mean: {binomial_pop_mean:.1f} and std: {binomial_pop_std:.1f}")

```

同样地，在现实场景中你无法触达全量总体，因此需要理论公式来求解这些指标。二项分布的理论均值与标准差拥有清晰严格的解析式：

* $\mu = np$

* $\sigma = \sqrt{np(1-p)}$


```python
binomial_pop_mean = n * p
binomial_pop_std = np.sqrt(n * p * (1 - p))

print(f"Gaussian population has mean: {binomial_pop_mean:.1f} and std: {binomial_pop_std:.1f}")

```

借助理论公式，无需遍历全量总体即可精确算出相同的统计量，非常便捷！

在对二项分布验证定理之前，有一条广为人知的经验法则 (rule of thumb) 用于粗略判断中心极限定理是否能够良好成立：

若 $\min(Np, N(1-p)) \ge 5$ ，则中心极限定理良好成立

其中 $N = n \times \text{sample\_size}$

不过必须指出，该法则仅作为粗略的参考准则，在选择恰当的统计方法时，还必须结合离群值 (outliers) 的存在以及具体的分析目标进行综合考量。

现在让我们验证定理的表现。首先从较小的样本量 `sample_size` 开始：

```python
sample_size = 3
N = n * sample_size

condition_value = np.min([N * p, N * (1 - p)])
print(f"The condition value is: {int(condition_value*10)/10:.1f}. CLT should hold?: {True if condition_value >= 5 else False}")

```

执行抽样并计算样本均值的理论期望与标准误。回顾其计算公式：

* $\mu_{\bar{X}} = \mu$

* $\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{\text{sample\_size}}}$


```python
# 计算样本均值
binomial_sample_means = sample_means(binomial_population, sample_size=sample_size)

# 计算理论均值估计
mu_sample_means = n * p

# 计算理论标准差估计
sigma_sample_means = np.sqrt(n * p * (1 - p)) / np.sqrt(sample_size)

```

绘制 KDE 与理论高斯曲线对比图以及 QQ 图，观察此时定理的吻合程度：

```python
# 绘制图表
utils.plot_kde_and_qq(binomial_sample_means, mu_sample_means, sigma_sample_means)

```

拟合效果明显不如高斯总体的例子。当单次抽样的 `sample_size` 较小时，样本均值的分布并不能很好地契合正态分布。

现在加大每个样本的容量，重新试验：

```python
sample_size = 30
N = n * sample_size

condition_value = np.min([N * p, N * (1 - p)])
print(f"The condition value is: {int(condition_value*10)/10:.1f}. CLT should hold?: {True if condition_value >= 5 else False}")

```

根据经验法则，在此参数条件下中心极限定理应当能够稳定生效。运行下方单元格进行验证：

```python
binomial_sample_means = sample_means(binomial_population, sample_size=sample_size)

# 计算理论均值估计
mu_sample_means = n * p

# 计算理论标准差估计
sigma_sample_means = np.sqrt(n * p * (1 - p)) / np.sqrt(sample_size)

# 绘制图表
utils.plot_kde_and_qq(binomial_sample_means, mu_sample_means, sigma_sample_means)

```

各项图表均清晰显示，此时定理得到了极其出色的印证！

与前文类似，运行下方单元格将启动交互式组件，你可以自由调整 $n$、$p$ 和 $\text{sample\_size}$ 的取值。

尝试探索其中的统计趣味：例如当 $p$ 接近 0.5 时，定理在较小样本量下是否表现得更加稳健？

```python
utils.binomial_clt()

```

让我们继续探究另一种分布！

## 泊松分布总体 (Poisson Population)



另一种常见的概率分布是泊松分布 (Poisson distribution)。它用于刻画在固定时间或空间区间内独立随机事件发生的总次数，其唯一核心参数为事件的平均发生率 $\mu$。

鉴于你已经熟悉了检验定理的标准流程，我们跳过中间步骤，直接通过交互式小部件进行体验。

在此只需牢记泊松分布的理论均值与标准差公式：

* $\mu = \mu$

* $\sigma = \sqrt{\mu}$


```python
utils.poisson_clt()

```

不出所料，你可以观察到 `sample_size` 越大，样本均值的经验分布就越紧密地趋附于高斯分布。

## 柯西分布 (Cauchy Distributions)



柯西分布 (Cauchy distribution) 不如前几种分布常见。它具有厚尾 (heavy tails) 特征，意味着相比于离散程度相似的常规分布，它出现极端极端值的概率要高得多。更为特殊的是，柯西分布在数学上**不存在良好定义的有限均值与方差**，这使得它在许多常规统计分析场景中无法直接套用经典模型。

由于这种特殊的数学性质，**中心极限定理对柯西分布不再成立**。运行下方代码生成一个包含 1,000 个数据点的柯西分布总体：

```python
cauchy_population = np.random.standard_cauchy(1000)

```

查看该总体的直方图：

```python
sns.histplot(cauchy_population, stat="density", label="hist")
plt.show()

```

由于数据中充斥着极端的异常点，甚至很难完整看清直方图的常规轮廓。现在将单次抽样量 `sample_size` 设为 30 (在常规分布下该样本量通常足以保证 CLT 生效) 并计算样本均值：

```python
cauchy_sample_means = sample_means(cauchy_population, sample_size=30)

```

由于该分布不存在理论均值和标准差，且直方图极难解读，我们直接通过 QQ 图来评估样本均值的正态性：

```python
# 创建 QQ 图
fig, ax = plt.subplots(figsize=(6, 6))
res = stats.probplot(cauchy_sample_means, plot=ax, fit=True)
plt.show()

```

图中的散点形态与直线大相径庭，这明确显示样本均值根本不服从正态分布。如果我们大幅提高抽样容量呢？

```python
cauchy_sample_means = sample_means(cauchy_population, sample_size=100)

# 创建 QQ 图
fig, ax = plt.subplots(figsize=(6, 6))
res = stats.probplot(cauchy_sample_means, plot=ax, fit=True)
plt.show()

```

即便将 `sample_size` 提升到 100 这种在实际场景中往往偏高的数值，样本均值依然完全无法收敛到正态分布。这是一个非常关键的统计事实：中心极限定理绝非放之四海而皆准，必须满足有限方差等先决条件，在实际应用该定理时务必牢记这一理论边界。

**祝贺你顺利完成本次实验！**