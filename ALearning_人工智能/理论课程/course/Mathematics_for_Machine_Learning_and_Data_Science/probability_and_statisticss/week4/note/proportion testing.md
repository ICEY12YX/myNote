___
在之前的视频中，你已经学习了如何对高斯总体 (Gaussian Population) 的均值进行假设检验 (Hypothesis Testing) 。在实际应用中，另一个极其常见且实用的场景，是对**总体比例 (Population Proportion)** $p$ 进行检验。

### 一个实例

设想你手里有一枚硬币，但并不确定它质地是否均匀、是否公平。你关心的核心指标是它抛出正面的真实概率，即总体比例 $p = P(H)$ 。针对这个问题，我们可以设立这样一组假设：

$$H_0: p = 0.5 \quad \text{vs.} \quad H_1: p \neq 0.5$$

假设你将这枚硬币抛掷了 20 次，其中有 7 次正面朝上。你的随机样本可以用一个随机变量 (Random Variable) 来刻画：$X =$ “20 次抛掷中出现正面的总次数”，它服从参数为 20 和 $p$ 的**二项分布 (Binomial Distribution)** ，记作 $\text{Binomial}(20, p)$ 。要估计硬币出现正面的比例，最理想的估计量就是正面的相对频率 (Relative Frequency) ：

$$\hat{p} = \frac{X}{20}$$

回顾中心极限定理 (Central Limit Theorem) ：
    回忆 学 ## central limit theorem 这节的时候
    [[ALearning_人工智能/理论课程/course/Mathematics_for_Machine_Learning_and_Data_Science/probability_and_statisticss/week3/note/01|01]]

在满足特定条件时，样本比例 $\hat{p}$ 会近似服从正态分布，即 $\hat{p} \sim N\left(p, \frac{p(1-p)}{20}\right)$ ；经过标准化处理后，可等价地写为：

$$Z = \frac{\frac{X}{20} - p}{\sqrt{\frac{p(1-p)}{20}}} \sim N(0, 1)$$

这里的 $Z$ 即为**检验统计量 (Test Statistic)** 。如果原假设 $H_0$ 为真 (即 $p = 0.5$ ) ，代入参数后检验统计量便化为：

$$Z = \frac{\frac{X}{20} - 0.5}{\sqrt{\frac{0.5(1-0.5)}{20}}} = \frac{\frac{X}{20} - 0.5}{\frac{0.5}{\sqrt{20}}} \sim N(0, 1)$$

设定显著性水平 (Significance Level) 为 $\alpha = 0.05$ 。接下来，为了做出统计推断，我们需要基于实际观测数据计算对应的 $p$ 值 (p-value) 。由于观察到的样本中 $x = 7$ ，代入可得观测统计量 (Observed Statistic) 为：

$$z = \frac{\frac{7}{20} - 0.5}{\frac{0.5}{\sqrt{20}}} = -1.3416$$

此时的 $p$ 值即为在原假设成立的前提下，出现比当前观测值更极端情况的概率，也就是 $Z > \vert{}z\vert{}$ 或 $Z < -\vert{}z\vert{}$ 的概率：

$$\text{p-value} = P(\vert{}Z\vert{} > \vert{}z\vert{}) = P(\vert{}Z\vert{} > 1.3416) = 0.1797$$
![[proportion testing-1793011943089.webp]]

**结论**：由于算得的 $p$ 值为 0.1797 ，大于预设的 0.05 显著性水平，因此现有样本证据不足以推翻“硬币是均匀公平的 (即 $p=0.5$ ) ”这一原假设。

### 一般情况
    就是没有具体可以代入的数据了

* $p$：目标群体中属于特定类别的总体比例 (Population Proportion)  (例如：硬币正面朝上的真实概率)
* $p_0$：原假设下预设的基准总体比例 (例如：$p_0 = 0.5$ )
* $x$：样本中属于该特定类别的实际个体数 (例如：硬币正面朝上的次数)
* $n$：样本容量 (Sample Size)  (例如：抛掷硬币的总次数)
* $\hat{p} = \frac{x}{n}$：由实际观测样本算得的样本比例 (Sample Proportion)

此时，用于比较比例的检验统计量为：

$$Z = \frac{\frac{X}{n} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}} \sim N(0, 1)$$

将实际样本数据代入后算得的具体数值，即为观测统计量：

$$z = \frac{\frac{x}{n} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}$$

针对不同方向的假设设定，$p$ 值的计算方法分别如下：

* **右尾检验 (Right-tailed Test)**：$H_0: p = p_0 \quad \text{vs.} \quad H_1: p > p_0$ ，其对应 $\text{p-value} = P(Z > z)$
    ![[proportion testing-1793011984295.webp]]
* **左尾检验 (Left-tailed Test)**：$H_0: p = p_0 \quad \text{vs.} \quad H_1: p < p_0$ ，其对应 $\text{p-value} = P(Z < z)$
    ![[proportion testing-1793011990303.webp]]
* **双尾检验 (Two-tailed Test)**：$H_0: p = p_0 \quad \text{vs.} \quad H_1: p \neq p_0$ ，其对应 $\text{p-value} = P(\vert{}Z\vert{} > \vert{}z\vert{})$
    ![[proportion testing-1793011996794.webp]]


要确保上述推断结果科学可靠，必须满足以下适用条件：

* **总体容量至少是样本容量的 20 倍**：这一要求是为了确保各个样本抽取之间相互独立。而在类似抛硬币这类天然保证独立性的重复试验中，无需单独满足该限制。
* **总体中的个体必须具备明确的二元属性**：即个体要么属于特定类别，要么不属于。
* **满足大样本近似条件：$n p_0 > 10$ 且 $n(1 - p_0) > 10$**：必须验证这两个不等式成立，以保证在原假设 $H_0$ 为真的前提下，二项分布能够很好地近似为高斯正态分布。