___
# A/B 测试 (AB Testing) - 平均会话时长 (Average Session Duration)

欢迎！在本次作业中，你将面对两个需要通过 A/B 测试来选择行动方案以改进现有产品的案例。你将分别针对连续型指标 (continuous metric) 和比例型指标 (proportion metric) 执行 A/B 测试。为此，你需要定义函数来从样本中估计关键信息，根据具体情况计算相应的统计量 (statistic) ，并最终决定是否拒绝原假设 (null hypothesis) 。

让我们开始吧！

___
- [[#1 - 简介|1 - 简介]]
- [[#2 - 探索与处理数据|2 - 探索与处理数据]]
- [[#3 - 理论回顾|3 - 理论回顾]]
- [[#4 - 分步计算|4 - 分步计算]]
	- [[#4 - 分步计算#练习 1|练习 1]]
			- [[#**预期输出**|**预期输出**]]
	- [[#4 - 分步计算#练习 2|练习 2]]
	- [[#4 - 分步计算#**预期输出**|**预期输出**]]
	- [[#4 - 分步计算#练习 3|练习 3]]
			- [[#**预期输出**|**预期输出**]]
	- [[#4 - 分步计算#练习 4|练习 4]]
	- [[#4 - 分步计算#**预期输出**|**预期输出**]]
	- [[#4 - 分步计算#练习 5|练习 5]]

___

```python
import math
import numpy as np
import pandas as pd
from scipy import stats

```

```python
import w4_unittest

```

## 1 - 简介

假设你运营着一个以博客形式提供机器学习 (machine learning) 内容的网站。最近你读到一篇文章，声称类似的网站只需采用特定的背景配色方案，就能有效提升用户参与度。由于这项改动实现起来非常简单，你决定开展一次 A/B 测试，看看这一改动是否真的能让用户在你的网站上停留更长时间。

你决定评估的指标是 `average session duration` (平均会话时长) ，它用于衡量用户在你的网站上平均花费的时间。目前该指标的数值为 30.87 分钟。

无需过多复杂的准备，你决定将用户随机划分为两个群体，进行为期 20 天的测试：

* `control` (对照组) ：这部分用户将继续看到你原来的网站界面。
* `variation` (实验组) ：这部分用户将看到采用了新背景配色的网站界面。

## 2 - 探索与处理数据

运行下一个单元格来加载本次测试的数据：

```python
# 使用 pd.read_csv 加载测试数据
data = pd.read_csv("background_color_experiment.csv")

# 打印前 10 行数据
data.head(10)

```

```python
print(f"The dataset size is: {len(data)}")

```

这份数据展示了每位用户的平均会话时长以及他们所访问的网站版本。为了将两个群体分开以便于后续计算，你可以运行下面的单元格对 Pandas 数据框 (dataframe) 进行切片。如果你对 Pandas 还不够熟悉，可以回顾一下我们之前的非评分实验课。不过完全不必担心，完成本次作业并不需要你是 Pandas 专家。

```python
# 将两组数据分开 (sd 代表 session duration，即会话时长) 
control_sd_data = data[data["user_type"]=="control"]["session_duration"]
variation_sd_data = data[data["user_type"]=="variation"]["session_duration"]

print(f"{len(control_sd_data)} users saw the original website with an average duration of {control_sd_data.mean():.2f} minutes\n")
print(f"{len(variation_sd_data)} users saw the new website with an average duration of {variation_sd_data.mean():.2f} minutes")

```

请注意，两组的人数分配并不是完全均等的。这在 A/B 测试中非常常见，因为用户被分配到哪个组别的过程本身带有随机性。

乍一看，更换背景颜色似乎确实让用户在网站上停留了更长时间。但你很清楚，不能仅凭数据的表面差异就草率下结论，因此你决定进行一次假设检验 (hypothesis test) ，以判断这两个群体的均值 (means) 之间是否存在显著差异。

## 3 - 理论回顾

让我们回顾一下课程中学过的理论，并将其应用到这个问题中。如果你对理论已经胸有成竹，觉得不需要复习，可以直接跳过本节进入第 4 节！

请记住，你的任务是衡量更换网站背景颜色是否会增加访客的停留时间。将这个问题转化为假设检验，原假设就是这项改动并没有影响访客的停留时间。让我们先定义相关变量：

* $\mu_c$ 是对照组 (control group) 用户在网站上花费的平均时间。回想一下，对照组是指访问未更改背景颜色的原始网站的用户群体。
* $\mu_v$ 是实验组 (variation group) 用户在网站上花费的平均时间。回想一下，实验组是指访问更新了背景颜色的新版网站的用户群体。

同时请记住，你的目的是检验更换背景颜色是否带来了访客停留时间的增加。因此，将这个实验写成假设检验的形式，原假设就是 $H_0: \mu_c = \mu_v$，而备择假设 (alternative hypothesis) 则是 $H_1: \mu_v > \mu_c$，或者等价地写为 $H_1: \mu_v - \mu_c > 0$。

因此，你要检验的假设为：

$$H_0: \mu_v = \mu_c \quad \text{vs.} \quad H_1: \mu_v - \mu_c > 0$$

正如你在课程中所学到的，这是一个右尾检验 (right-tailed test) ，因为你关注的是平均时间是否有所增加。从上面的数据可以看出，每个组别都有超过 2000 名用户，数据量非常充足，因此我们可以合理地依据中心极限定理 (Central Limit Theorem, CLT) ，认为每个组别的平均时间服从正态分布 (normal distribution) 。请记住，这一结论针对的是整个群体的平均时间，而不是说单个用户在网站上停留的时间服从正态分布。你并不清楚单个用户在网站上停留时间的确切分布，但中心极限定理向我们保证：只要收集到足够多的数据，群体平均时间的分布就会非常接近正态分布，且该正态分布的均值正是用户在网站上花费的真实平均时间。接下来，让我们定义另外几个量：

* $\overline{X}_c$ - 对照组的样本均值 (sample mean) 。
* $\overline{X}_v$ - 实验组的样本均值。
* $n_c$ - 对照组的样本量 (size) 。
* $n_v$ - 实验组的样本量。

因此，根据中心极限定理，你可以假设：

* $$\overline{X}_c \sim N\left(\mu_c, \left(\frac{\sigma_c}{\sqrt{n_c}}\right)^2\right)$$


* $$\overline{X}_v \sim N\left(\mu_v, \left(\frac{\sigma_v}{\sqrt{n_v}}\right)^2\right)$$



需要注意的是，在正态分布的假设下，$\overline{X}_v - \overline{X}_c$ 同样服从正态分布。因此，如果 $H_0$ 为真，那么 $\mu_c = \mu_v$ 且 $\mu_v - \mu_c = 0$，由此可得：

$$\overline{X}_c - \overline{X}_v \sim N\left(\mu_v - \mu_c, \left(\dfrac{\sigma_v}{\sqrt{n_v}}\right)^2 + \left(\dfrac{\sigma_c}{\sqrt{n_c}}\right)^2\right) = N\left(0, \left(\dfrac{\sigma_v}{\sqrt{n_v}}\right)^2 + \left(\dfrac{\sigma_c}{\sqrt{n_c}}\right)^2\right)$$

或者等价地表示为：

$$\frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\left(\frac{\sigma_v}{\sqrt{n_v}}\right)^2 + \left(\frac{\sigma_c}{\sqrt{n_c}}\right)^2}} \sim N(0, 1)$$

不过请记住，你并不知道 $\sigma_v$ 和 $\sigma_c$ 的确切数值，因为它们是总体标准差 (population standard deviation) ，而你手头只有样本数据，所以最好的办法是计算样本标准差 (sample standard deviation) 。因此，你需要用样本标准差 $s_c$ 和 $s_v$ 分别替换 $\sigma_c$ 和 $\sigma_v$。你在课程中也学过，用样本标准差替换总体标准差后，随机变量 (random variable) 的分布就会从标准正态分布变为学生 t 分布 (t-student distribution) ：

$$t = \frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\left(\frac{s_v}{\sqrt{n_v}}\right)^2 + \left(\frac{s_c}{\sqrt{n_c}}\right)^2}} \sim t_d$$

其中 $d$ 是该场景下的自由度 (degrees of freedom) 。如果我们假设两个群体的标准差相同，那么 $d = n_c + n_v - 2$；但目前并没有任何依据支持这一假设，因此自由度的计算公式会稍微复杂一些：

$$d = \frac{\left[\frac{s_{v}^2}{n_v} + \frac{s_{c}^2}{n_c} \right]^2}{\frac{(s_{v}^2/n_v)^2}{n_v-1} + \frac{(s_{c}^2/n_c)^2}{n_c-1}}$$

一旦计算出 $t_d$ 的实际数值，在给定显著性水平 (significance level) $\alpha$ 的情况下，你就可以判断该数值是否落在学生 t 分布中“很可能出现”的数值范围内 (这里的“很可能”与你设定的显著性水平直接相关) 。为了完成这一步，你必须找到满足以下条件的 p 值 (p-value) $p$：

$$p = P(t_d > t \vert{} H_0)$$

如果这个值小于你的显著性水平 $\alpha$，那么你就拒绝原假设，因为这意味着在 $H_0$ 为真的前提下，你观测到了一个极不可能发生的数值 (这里的“极不可能”是指发生的概率低于你所设定的显著性水平) 。

此外请记住，$P(t_d \leq t)$ 是自由度为 $d$ 的学生 t 分布在 $x = t$ 处的 $\text{CDF}$ (累积分布函数, cumulative distribution function) ，因此要计算 $P(t_d > t)$，你可以通过以下公式计算：

$$P(t_d > t) = 1 - \text{CDF}_{t_d}(t)$$

因为 $P(t_d \leq t) + P(t_d > t) = 1$

## 4 - 分步计算

总结一下上面讨论的所有内容：

本次假设检验的表达式为：

$$H_0: \mu_v = \mu_c \quad \text{vs.} \quad H_1: \mu_v - \mu_c > 0$$

你将首先计算以下几个基础量：

* $n_c$ 和 $n_v$，分别代表对照组和实验组的样本量。
* $\overline{X}_c$ 和 $\overline{X}_v$，分别代表对照组和实验组用户在网站上花费的平均时间。
* $s_c$ 和 $s_v$，分别代表对照组和实验组用户停留时间的样本标准差。

拿到这些统计量后，接下来的步骤是计算：

* $d$，即学生 t 分布 $t_d$ 的自由度。
* t 值 (t-value) ，记为 $t$。
* 分布 $t_d$ 对应于该 t 值的 $p$ 值，即 $p = P(t_d > t \vert{} H_0)$。

最后，针对给定的显著性水平 $\alpha$，你可以根据 $p \leq \alpha$ 是否成立，来决定是否拒绝 $H_0$。

现在让我们动手实践吧！运行下面的单元格以提取对照组和实验组的会话时长数据。

```python
# X_c 存储对照组的会话时长，X_v 存储实验组的会话时长。
X_c = control_sd_data.to_numpy()
X_v = variation_sd_data.to_numpy()

```

```python
print(f"The first 10 entries for X_c are:\n{X_c[:20]}\n")
print(f"The first 10 entries for X_v are:\n{X_v[:20]}\n")

```

### 练习 1

在本练习中，你将编写一个函数来提取 `X_c` 和 `X_d` 的基础统计量。换句话说，对于给定的 numpy 数组，该函数将计算：

* 数组的大小 (在本例中即 $n_c$ 和 $n_v$) 。
* 数组的均值 (在本例中即 $\overline{X}_c$ 和 $\overline{X}_v$)
* 数组的样本标准差 (在本例中即 $s_c$ 和 $s_v$)

该函数接收一个 numpy 数组作为输入，并输出一个形式为 `(n, x, s)` 的元组 (tuple) ，其中 `n` 是 numpy 数组的大小，`x` 是其均值，`s` 是其样本标准差。

提示：

* 回想一下，样本标准差是通过将方差 (variance) 公式中的分母 $N$ 替换为 $N-1$ 来计算的。
* 你可以使用 `len` 函数来计算数组的大小。
* numpy 中的任何数组都有一个名为 `.mean()` 的方法来计算其均值。
* numpy 中的任何数组都有一个名为 `.std()` 的方法来计算标准差，并且包含一个名为 `ddof` 的参数；如果你传入 `ddof = 1`，它就会使用 $N-1$ 而不是 $N$。

```python
def get_stats(X):
    """
    Calculate basic statistics of a given data set.

    Parameters:
    X (numpy.array): Input data.

    Returns:
    tuple: A tuple containing:
        - n (int): Number of elements in the data set.
        - x (float): Mean of the data set.
        - s (float): Sample standard deviation of the data set.
    """

    ### START CODE HERE ###
    
    # Get the group size
    n = None
    # Get the group mean
    x = None
    # Get the group sample standard deviation (do not forget to pass the parameter ddof if using the method .std)
    s = None

    ### END CODE HERE ###

    return (n,x,s)

```

```python
w4_unittest.test_get_stats(get_stats)

```

```python
n_c, x_c, s_c = get_stats(X_c)
n_v, x_v, s_v = get_stats(X_v)

```

```python
print(f"For X_c:\n\tn_c = {n_c}, x_c = {x_c:.2f}, s_c = {s_c:.2f} ")
print(f"For X_v:\n\tn_v = {n_v}, x_v = {x_v:.2f}, s_v = {s_v:.2f} ")

```

##### **预期输出**

```Python
For X_c:
	n_c = 2069, x_c = 32.92, s_c = 17.54 
For X_v:
	n_v = 2117, x_v = 33.83, s_v = 18.24 

```

### 练习 2

在本练习中，你将实现一个函数来计算学生 t 分布的自由度 $d$。它的计算公式如下：

$$d = \frac{\left[\frac{s_{c}^2}{n_c} + \frac{s_{v}^2}{n_v} \right]^2}{\frac{(s_{c}^2/n_c)^2}{n_c-1} + \frac{(s_{v}^2/n_v)^2}{n_v-1}}$$

提示：在 Python 中，你可以使用语法 `x**2` 来计算一个数的平方，也可以使用函数 `np.square`。后者有助于让你的代码更加整洁。请特别注意括号的使用，因为括号决定了 Python 执行运算的先后顺序！

```python
def degrees_of_freedom(n_v, s_v, n_c, s_c):
    """Computes the degrees of freedom for two samples.

    Args:
        control_metrics (estimation_metrics_cont): The metrics for the control sample.
        variation_metrics (estimation_metrics_cont): The metrics for the variation sample.

    Returns:
        numpy.float: The degrees of freedom.
    """
    
    ### START CODE HERE ###
    
    # To make the code clean, let's divide the numerator and the denominator. 
    # Also, note that the value s_c^2/n_c and s_v^2/n_v appears both in the numerator and denominator, so let's also compute them separately

    # Compute s_v^2/n_v (remember to use Python syntax or np.square)
    s_v_n_v = None

    # Compute s_c^2/n_c (remember to use Python syntax or np.square)
    s_c_n_c = None


    # Compute the numerator in the formula given above
    numerator = None

    # Compute the denominator in the formula given above. Attention that s_c_n_c and s_v_n_v appears squared here!
    # Also, remember to use parenthesis to indicate the operation order. Note that a/b+1 is different from a/(b+1).
    denominator = None
    
    ### END CODE HERE ###

    dof = numerator/denominator
        
    return dof

```

```python
w4_unittest.test_degrees_of_freedom(degrees_of_freedom)

```

```python
d = degrees_of_freedom(n_v, s_v, n_c, s_c)
print(f"The degrees of freedom for the t-student in this scenario is: {d:.2f}")

```

### **预期输出**

`The degrees of freedom for the t-student in this scenario is: 4182.97 `

### 练习 3

在本练习中，你将计算 t 值，其计算公式为：

$$t = \frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\left(\frac{s_v}{\sqrt{n_v}}\right)^2 + \left(\frac{s_c}{\sqrt{n_c}}\right)^2}} = \frac{\left( \overline{X}_v - \overline{X}_c \right)}{\sqrt{\frac{s_v^2}{n_v} + \frac{s_c^2}{n_c}}}$$

请记住，你已经将 $\overline{X}_c$ 和 $\overline{X}_v$ 分别存储在变量 $x_c$ 和 $x_d$ 中。

```python
def t_value(n_v, x_v, s_v, n_c, x_c, s_c):

    ### START CODE HERE ###

    # As you did before, let's split the numerator and denominator to make the code cleaner.
    # Also, let's compute again separately s_c^2/n_c and s_v^2/n_v.

    # Compute s_v^2/n_v (remember to use Python syntax or np.square)
    s_v_n_v = None

    # Compute s_c^2/n_c (remember to use Python syntax or np.square)
    s_c_n_c = None

    # Compute the numerator for the t-value as given in the formula above
    numerator = None

    # Compute the denominator for the t-value as given in the formula above. You may use np.sqrt to compute the square root.
    denominator = None
    
    ### END CODE HERE ###

    t = numerator/denominator

    return t

```

```python
w4_unittest.test_t_value(t_value)

```

```python
t = t_value(n_v, x_v, s_v, n_c, x_c, s_c)
print(f"The t-value for this experiment is: {t:.2f}")

```

##### **预期输出**

```
The t-value for this experiment is: 1.64

```

### 练习 4

在本练习中，针对给定的显著性水平 $\alpha$，你将计算分布 $t_d$ 的 $p$ 值。回想一下，本次实验是一个右尾 t 检验，因为你正在探究更换背景颜色是否会增加用户在你网站上的停留时间。

在本实验中，给定显著性水平 $\alpha$ 下的 $p$ 值计算公式为：

$$p = P(t_d > t) = 1 - \text{CDF}_{t_d}(t)$$

提示：

* 你可以使用 scipy 函数 `stats.t(df = d)` 来获取自由度为 `d` 的学生 t 分布。
* 要计算其 CDF，可以使用它的 `.cdf` 方法。

示例：

假设你想计算自由度 $d = 10$ 的学生 t 分布在 t 值为 $1.21$ 处的 CDF。

```python
t_10 = stats.t(df = 10)
cdf = t_10.cdf(1.21)
print(f"The CDF for the t-student distribution with 10 degrees of freedom and t-value = 1.21, or equivalently P(t_10 < 1.21) is equal to: {cdf:.2f}")

```

这意味着，当从自由度为 10 的学生 t 分布中抽样时，你观测到小于 1.21 的数值的概率为 87%。

好了，现在你已经准备好编写一个函数，来计算具有 $d$ 个自由度和给定 t 值的学生 t 分布的 $p$ 值了。

```python
def p_value(d, t_value):

    ### START CODE HERE ###

    # Load the t-student distribution with $d$ degrees of freedom. Remember that the parameter in the stats.t is given by df.
    t_d = None

    # Compute the p-value, P(t_d > t). Remember to use the t_d.cdf with the proper adjustments as discussed above.
    p = None


    ### END CODE HERE ###

    return p

```

```python
w4_unittest.test_p_value(p_value)

```

```python
print(f"The p-value for t_15 with t-value = 1.10 is: {p_value(15, 1.10):.4f}")
print(f"The p-value for t_30 with t-value = 1.10 is: {p_value(30, 1.10):.4f}")

```

### **预期输出**

```
The p-value for t_15 with t-value = 1.10 is: 0.1443
The p-value for t_30 with t-value = 1.10 is: 0.1400

```

### 练习 5

在本练习中，你将整合目前构建的所有函数，在给定显著性水平 $\alpha$ 的情况下，决定是否接受 $H_0$。

该函数将接收对照组和实验组的数据作为输入，并根据检验结果相应地输出 `Reject H_0` 或 `Do not reject H_0`。

请记住，如果 p 值小于 $\alpha$，你就应当拒绝 $H_0$。

```python
def make_decision(X_v, X_c, alpha = 0.05):

    ### START CODE HERE ###

    # Compute n_v, x_v and s_v
    n_v, x_v, s_v = None

    # Compute n_c, x_c and s_c
    n_c, x_c, s_c = None

    # Compute the degrees of freedom for the t-student distribution for this experiment.
    # Pay attention to the arguments order. You may look the function definition above to make sure you don't swap values.
    # Also, remember that x_c and x_v are not used in this computation
    d = None
    
    # Compute the t-value
    t = None

    # Compute the p-value for the t-student distribution with d degrees of freedom
    p = None

    # This is the decision step. Compare p with alpha to decide about rejecting H_0 or not. 
    # Pay attention to the return value for each block to properly write the condition.

    if None:
        return 'Reject H_0'
    else:
        return 'Do not reject H_0'

    ### END CODE HERE ###

```

```python
w4_unittest.test_make_decision(make_decision)

```

```python
alphas = [0.06, 0.05, 0.04, 0.01]
for alpha in alphas:
    print(f"For an alpha of {alpha} the decision is to: {make_decision(X_v, X_c, alpha = alpha)}")

```

**恭喜你完成了本次作业！**

现在你已经亲手实现了在简单场景下执行 A/B 测试所需的全部步骤！

**这是本课程以及整个专项课程的最后一次作业，为自己取得如此出色的成就鼓个掌吧！干得漂亮！！！！**