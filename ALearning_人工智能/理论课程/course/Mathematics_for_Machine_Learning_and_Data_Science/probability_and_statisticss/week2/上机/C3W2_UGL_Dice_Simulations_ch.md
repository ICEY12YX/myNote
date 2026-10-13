___
# 实验：使用 NumPy 模拟掷骰子 🎲🤖



欢迎！本实验将带你探索如何使用 NumPy 模拟掷骰子实验，内容涵盖从单次投掷骰子到多次投掷求和的全过程。此外，你还将学习如何处理“灌铅骰子” (loaded dice) 的情况——即骰子的某一面出现的概率高于其他面。

让我们立即开始！

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

```

## 骰子的表示



首先，你需要确定骰子包含多少个面。更进一步，你可以使用 NumPy 数组 (NumPy array) 来表示一枚骰子，并为每个面赋予对应的点数标签：

```python
# 定义期望的面数 (你可以尝试修改这个数值！)
n_sides = 6

# 使用 numpy 数组表示一枚骰子
dice = np.array([i for i in range(1, n_sides+1)])

dice

```

## 投掷骰子



骰子构建完成后，就可以进行投掷了。在此，我们先假设骰子是质地均匀的均匀骰子 (fair dice) ，这意味着每一面落地的概率均等，服从均匀分布 (uniform distribution)。你可以使用 np.random.choice 函数来实现这一逻辑，该函数能够从给定的 NumPy 数组中随机抽取一个元素：

```python
# 多次运行该单元格 (每次运行都会随机获得不同的结果)
np.random.choice(dice)

```

单次抽取非常方便，但如果你想投掷 20 次，就必须反复运行 20 次并分别记录结果。因此，我们需要一种能够批量模拟多次投掷的方法。你可以先设定预期的投掷次数，随后借助列表推导式 (list comprehension) 自由模拟多次投掷，并将结果整合成一个 NumPy 数组：

```python
# 模拟掷骰子 20 次
n_rolls = 20

# 保存每次投掷的结果
rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

rolls

```

现在，你拥有了一种便捷记录每次投掷结果的方法，非常棒！

如果你想了解这一过程的均值 (mean) 和方差 (variance) ，可以直接调用 NumPy 的内置函数 np.mean 和 np.var：

```python
# 计算 20 次投掷的均值
m = np.mean(rolls)

# 计算 20 次投掷的方差
v = np.var(rolls)

print(f"mean of rolls: {m:.2f}\nvariance of rolls: {v:.2f}")

```

你还可以绘制直方图 (histogram) ，直观查看保存每次投掷结果的 NumPy 数组的分布特征。在此，我们将使用 Seaborn 可视化库中的 sns.histplot 函数：

```python
# 绘制 20 次投掷的直方图
n_rolls_hist = sns.histplot(rolls, discrete=True)
n_rolls_hist.set(title=f"Histogram of {n_rolls} rolls")
plt.show()

```

此时的直方图看起来可能并不像理想的均匀分布，这是因为随机实验本身存在波动，且当前仅模拟了 20 次投掷。现在，让我们将投掷次数提升至 20,000 次再试一次：

```python
n_rolls = 20_000

rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

print(f"mean of rolls: {np.mean(rolls):.2f}\nvariance of rolls: {np.var(rolls):.2f}")

n_rolls_hist = sns.histplot(rolls, discrete=True)
n_rolls_hist.set(title=f"Histogram of {n_rolls} rolls")
plt.show()

```

绘制出的直方图以及计算出的均值和方差，是否与你在课程中学习到的均匀分布理论相吻合？

蒙特卡洛模拟 (simulation) 是一种极佳的工具，能够将实验结果与理论解析解进行直观对比。例如，在本例中，六面均匀骰子的理论均值为 3.5，理论方差为 2.916 (具体推导公式可参考 这里)。需要牢记的核心原则是：模拟的样本量越大，实验结果就越逼近真实理论值，因此在开展仿真时务必选择充分的模拟次数！

NumPy 的底层运算十分高效，执行 20,000 次模拟依然能够在瞬间完成。

## 两次投掷结果求和



接下来，我们模拟连续投掷两次骰子并将两次结果相加的场景。与此前类似，我们可以将第一次投掷的结果保存在一个 NumPy 数组中，同时用第二个数组保存第二次投掷的结果。

要求和时，只需直接将两个数组相加。得益于 NumPy 的向量化运算 (vectorized operations) ，两个数组相加会自动对对应位置上的数值进行逐元素相加 (element-wise sum) ，并返回一个新的求和数组。

此时，你不仅可以分别计算第一次投掷、第二次投掷以及求和结果各自的均值与方差，还能计算出两次投掷之间的协方差 (covariance)：

```python
n_rolls = 20_000

# 第一次投掷 (逻辑同前)
first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

# 第二次投掷 (逻辑相同，保存在新的 numpy 数组中)
second_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

# 将两次投掷结果相加 (借助 numpy 向量化特性轻松实现)
sum_of_rolls = first_rolls + second_rolls

# 打印均值、方差与协方差
print(f"mean of first_rolls: {np.mean(first_rolls):.2f}\nvariance of first_rolls: {np.var(first_rolls):.2f}\n")
print(f"mean of second_rolls: {np.mean(second_rolls):.2f}\nvariance of second_rolls: {np.var(second_rolls):.2f}\n")
print(f"mean of sum_of_rolls: {np.mean(sum_of_rolls):.2f}\nvariance of sum_of_rolls: {np.var(sum_of_rolls):.2f}\n")
print(f"covariance between first and second roll:\n{np.cov(first_rolls, second_rolls)}")

# 绘制直方图
sum_2_rolls_hist = sns.histplot(sum_of_rolls, stat = "probability", discrete=True)
sum_2_rolls_hist.set(title=f"Histogram of {n_rolls} rolls (sum of rolling twice)")
plt.show()

```

正如你所预期的，最终绘制出的直方图呈现出非常典型的钟形高斯分布 (Gaussian distribution)。同时可以注意到，第一次投掷与第二次投掷之间的协方差非常接近于 0，这是因为两次投掷在统计上属于彼此独立的独立事件 (independent events)。

另外请注意，你可以通过调节 `sns.histplot` 函数的 `stat` 参数来改变直方图纵坐标的统计指标。在前面的练习中我们展示的是频数 (frequency) ，而在这里绘制的是概率 (probability) ，这在当前的概率论场景下更为直观贴切。如需了解更多可用的统计参数，请参阅 官方文档。

## 模拟灌铅骰子



到目前为止，我们模拟的都是完全公平的均匀骰子 (即所有点数朝上的概率完全相同) ，但如果需要模拟被动手脚的“灌铅骰子” (loaded dice)  (即某一个或几个点数朝上的概率明显更高) 呢？

实现这一逻辑其实非常直接。np.random.choice 提供了内置参数 `p`，该参数用于指定抽取数组中各个元素对应的概率分布权重。

为了进行演示，我们编写一个函数：假定骰子的某一个指定点数朝上的概率是其余所有点数的两倍，计算并返回各个面落地的概率分布：

```python
def load_dice(n_sides, loaded_number):
    
    # 初始状态下所有面的概率均等
    probs = np.array([1/(n_sides+1) for _ in range(n_sides)])
    
    # 将特定面的概率设为其她面的两倍
    probs[loaded_number-1] = 1 - sum(probs[:-1])
    
    # 验证所有概率之和是否为 1
    if not np.isclose(sum(probs), 1):
        print("All probabilities should add up to 1")
        return
    
    return probs 

```

在调用该函数之前，我们先观察一下正常均匀骰子的概率分布长什么样：

```python
# 计算每个面均等的概率
probs_fair_dice = np.array([1/n_sides]*n_sides)

# 绘制概率柱状图
fair_dice_sides = sns.barplot(x=dice, y=probs_fair_dice)
fair_dice_sides.set(title=f"Histogram for fair dice with {n_sides} sides")
fair_dice_sides.set_ylim(0,0.5)
plt.show()

```

现在调用 `load_dice` 函数获取灌铅骰子的概率分布。你可以尝试替换被加权的目标点数！

```python
# 获取偏向点数 2 的灌铅骰子概率
probs_loaded_dice = load_dice(n_sides, loaded_number=2)

# 绘制概率柱状图
loaded_dice_sides = sns.barplot(x=dice, y=probs_loaded_dice)
loaded_dice_sides.set(title=f"Histogram for loaded dice with {n_sides} sides")
loaded_dice_sides.set_ylim(0,0.5)
plt.show()

```

接着，将该概率数组 `probs_loaded_dice` 传入 `np.random.choice` 中，观察这会对统计指标与图表形态产生什么影响：

```python
n_rolls = 20_000

# 注意此处指定了概率参数 p
first_rolls = np.array([np.random.choice(dice, p=probs_loaded_dice) for _ in range(n_rolls)])

second_rolls = np.array([np.random.choice(dice, p=probs_loaded_dice) for _ in range(n_rolls)])

sum_of_rolls = first_rolls + second_rolls

print(f"mean of first_rolls: {np.mean(first_rolls):.2f}\nvariance of first_rolls: {np.var(first_rolls):.2f}\n")
print(f"mean of second_rolls: {np.mean(second_rolls):.2f}\nvariance of second_rolls: {np.var(second_rolls):.2f}\n")
print(f"mean of sum_of_rolls: {np.mean(sum_of_rolls):.2f}\nvariance of sum_of_rolls: {np.var(sum_of_rolls):.2f}\n")
print(f"covariance between first and second roll:\n{np.cov(first_rolls, second_rolls)}")

# 绘制直方图
loaded_rolls_hist = sns.histplot(sum_of_rolls, stat = "probability", discrete=True)
loaded_rolls_hist.set(title=f"Histogram of {n_rolls} rolls (sum of rolling twice a loaded dice)")
loaded_rolls_hist.set_xticks(range(min(sum_of_rolls),max(sum_of_rolls)+1))
plt.show()

```

此时的直方图发生了明显的偏斜，因为某些点数组合的和出现的可能性被显著拉高了。试着改变加权点数，观察直方图形态的实时演变！

需要注意的是，此时两次投掷之间的协方差依然十分趋近于 0，因为虽然单枚骰子的概率分布发生了改变，但两次投掷之间依然互相独立，不存在依赖关系。

## 存在依赖关系的投掷



在实验的最后环节，我们将模拟第二次投掷依赖于第一次投掷结果的条件关联场景。假设我们采用一种新的规则：只有当第一次投掷的点数大于或等于 4 时，才允许进行第二次投掷。

在动手模拟之前，不妨先从逻辑上推演一下这一新规则会带来哪些有趣的现象：

* 最终点数和可能出现 1 点：因为如果第一次掷出 1 点，游戏直接结束，无需投掷第二次。


* 1、2 和 3 点出现的综合概率显著上升。


* 最终点数和绝不可能为 4 点：因为只要第一次投掷得到 4 点就必须掷第二次，而第二次掷出的点数至少为 1 点，总和必然大于等于 5 点；若第一次投掷小于 4 点，则和直接等于 1、2 或 3 点。



为了实现这一条件判断逻辑，我们可以使用 np.where 函数。该函数能根据特定条件进行矢量化筛选，并将不满足条件的元素统一归零：

```python
n_rolls = 20_000

first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

second_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])

# 仅当第一次投掷点数大于等于 4 时才保留第二次投掷结果，否则置为 0
second_rolls = np.where(first_rolls>=4, second_rolls, 0)

sum_of_rolls = first_rolls + second_rolls

print(f"mean of first_rolls: {np.mean(first_rolls):.2f}\nvariance of first_rolls: {np.var(first_rolls):.2f}\n")
print(f"mean of second_rolls: {np.mean(second_rolls):.2f}\nvariance of second_rolls: {np.var(second_rolls):.2f}\n")
print(f"mean of sum_of_rolls: {np.mean(sum_of_rolls):.2f}\nvariance of sum_of_rolls: {np.var(sum_of_rolls):.2f}\n")
print(f"covariance between first and second roll:\n{np.cov(first_rolls, second_rolls)}")

# 绘制直方图
dependant_rolls_hist = sns.histplot(sum_of_rolls, stat = "probability", discrete=True)
dependant_rolls_hist.set(title=f"Histogram of {n_rolls} rolls (dependant sum of rolling twice)")
dependant_rolls_hist.set_xticks(range(min(sum_of_rolls),max(sum_of_rolls)+1))
plt.show()

```

模拟结果精准验证了我们此前的全部理论预测！值得特别关注的是，此时两次投掷之间的协方差已经显著偏离 0，这生动证明了两次投掷之间存在着真实的统计依赖性 (dependency)！

**恭喜你完成了本次不计分实验，表现出色！**