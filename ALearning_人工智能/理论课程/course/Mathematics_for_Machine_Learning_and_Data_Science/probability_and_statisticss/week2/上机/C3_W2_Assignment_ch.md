___
# 灌铅骰子 (Loaded dice)

欢迎来到《面向机器学习 (Machine Learning) 与数据科学 (Data Science) 的概率与统计 (Probability and Statistics) 》课程的第二次作业！在这份随堂测验式的作业中，我们将把经典的掷骰子案例全面升级，以此来检验你对课堂所学概念的直觉与理解。

**你完全可以只用纸和笔，甚至仅凭自己对概率 (Probability) 的直觉来完成这份作业；不过在很多问题上，动用你正在锤炼的编程技能或许会事半功倍**。

___
- [[#1 - 简介|1 - 简介]]
	- [[#1 - 简介#1.1 作业通关指南|1.1 作业通关指南]]
- [[#2 - 导入代码库|2 - 导入代码库]]
- [[#3 - 作业完成方式实例解析|3 - 作业完成方式实例解析]]
	- [[#3 - 作业完成方式实例解析#3.1 例题|3.1 例题]]
	- [[#3 - 作业完成方式实例解析#3.2 利用 Python 模拟求解|3.2 利用 Python 模拟求解]]
	- [[#3 - 作业完成方式实例解析#3.3 利用理论知识求解|3.3 利用理论知识求解]]
	- [[#3 - 作业完成方式实例解析#3.4 保存你的答案|3.4 保存你的答案]]
- [[#4 - 核心概念厘清 🎲🎲🎲|4 - 核心概念厘清 🎲🎲🎲]]
- [[#练习 1：|练习 1：]]
- [[#练习 2：|练习 2：]]
- [[#练习 3：|练习 3：]]
- [[#练习 4：|练习 4：]]
- [[#练习 5：|练习 5：]]
- [[#练习 6：|练习 6：]]
- [[#练习 7：|练习 7：]]
- [[#练习 8：|练习 8：]]
- [[#练习 9：|练习 9：]]
- [[#练习 10：|练习 10：]]
- [[#练习 11：|练习 11：]]
- [[#提交作业前的最后检查|提交作业前的最后检查]]


___


## 1 - 简介

接下来你将面对 11 道关于各类掷骰子游戏的问题。这些骰子有时是被动过手脚的灌铅骰子，有时则是完全均匀的普通骰子。每个练习都会配有清晰的操作说明。

### 1.1 作业通关指南

在每个练习中，你都会遇到一个关于投掷若干枚骰子 (可能均匀，也可能灌铅) 的场景。你需要针对不同场景的结果回答问题，例如计算投掷结果的期望值 (Expected Value) ，或者选出最能代表结果分布 (Distribution) 情况的图表。

无论面对哪道题，**你都可以通过以下两种方式之一来求解：**

* **手动推算：** 运用课堂上学到的理论知识，直接用纸笔进行推导计算。
* **使用 Python：** 利用题目下方提供的空白代码块编写计算和模拟 (Simulation) 程序，从而得出结果。

完成每道练习后，你需要运行一个特殊的代码单元格并填入答案来保存解题结果。这些单元格只包含一行形如 `utils.exercise_1()` 的代码，运行后会弹出一个用于保存答案的交互界面。**你可以在做题过程中随时保存每道题的答案，但直到最后正式提交整份作业前，系统都不会对你的答案进行评分。**

让我们先来看一个例子！在此之前，先把必要的库导入进来。

## 2 - 导入代码库


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils

```

## 3 - 作业完成方式实例解析

现在我们通过一道例题，带你熟悉整个作业的答题流程。

### 3.1 例题

问题：给定一枚均匀的 6 面骰子 (Fair Dice) ，将其连续投掷两次并记录结果。请问两次投掷的点数之和大于 5 的概率是多少？ (结果保留 1 位小数)。

在问题下方，你会看到如下代码块。

```python
# 定义期望的面数 (你可以尝试修改这个数值！)
n_sides = 6
# 使用 numpy 数组表示一枚骰子
dice = np.array([i for i in range(1, n_sides + 1)])

n_rolls = 20000

# 第一次投掷
first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
# 第二次投掷
second_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
sum_of_rolls = first_rolls + second_rolls
answer = sum_of_rolls[sum_of_rolls > 5].size / sum_of_rolls.size
# 小数点保留两位
print(f"{answer:.2f}")

# 小于等于5的情况 1,1 1,2 1,3 1,4 2,1 2,2 2,3 3,1 3,2 4,1 一共 10 种  10/36 ≈ 0.28
# 所以大于5的情况是 36 - 10 = 26 种  26/36 ≈ 0.72
```

```python
# 你可以使用此单元格进行计算 (不计分)
0.72

```

你可以随心所欲地利用这块区域来解题，也可以完全忽略它，直接拿起纸笔演算。全凭你的喜好！**你只需要在最后保存最终答案即可**。

### 3.2 利用 Python 模拟求解

我们用两种方法来解这道题。首先是使用 Python。你可以参考本次作业前紧挨着的不计分实验“骰子模拟 (Dice Simulations) ”，里面介绍了如何用代码模拟掷骰子。请记住，为了获得足够精确的近似值，你需要进行海量的模拟！在接下来的几周里你会学到背后的数学原理，不过这一点在直觉上也很容易理解。

```python
# 你可以使用此单元格进行计算 (不计分)
# 下面写的代码极其的简单,所以用最简单的代码也能实现功能

# 这个列表代表骰子的每一个面
dice = [1,2,3,4,5,6]

# 核心思路是从列表中随机抽取元素并求和。
# 每次抽取就相当于掷了一次骰子，抽到的数字就是朝上一面的点数。
# 这个列表用于存储每次重复实验的点数之和。我们的目标是把这项实验重复成百上千次。
sum_results = []

number_iterations = 1000

# 设置随机种子 (Random Seed) 以确保结果可复现
np.random.seed(42)
# 循环执行该游戏 number_iterations 次
for i in range(number_iterations):
    # 掷第一枚骰子
    throw_1 = np.random.choice(dice)
    # 掷第二枚骰子
    throw_2 = np.random.choice(dice)
    # 将两次结果相加
    sum_throw = throw_1 + throw_2
    # 将总和追加到结果列表中
    sum_results.append(sum_throw)

# 记录下所有点数之和后，真实概率就会非常接近 sum_results 列表中满足条件的元素所占的比例。
greater_5_count = 0

for x in sum_results:
    if x > 5:
        greater_5_count += 1

probability = greater_5_count/len(sum_results)    
print(f"The probability by this simulation is: {probability}")

```
![[C3_W2_Assignment_ch-1792449965502.webp]]

将模拟得到的结果四舍五入保留一位小数，答案就是 0.7！接下来我们试试“手动推算”。

### 3.3 利用理论知识求解



投掷两枚骰子时，总共有 $36$ 种可能的结果 (Outcomes)：

$$(1,1), (1,2), \ldots, (6,6)$$

我们需要数一数其中有多少种组合的点数之和大于 5。具体情况如下：

* 如果第一次掷出 $1$，第二次有 $2$ 种可能满足条件：5 或 6。
* 如果第一次掷出 $2$，第二次有 $3$ 种可能满足条件：4、5 或 6。
* 如果第一次掷出 $3$，第二次有 $4$ 种可能满足条件：3、4、5 或 6。
* 如果第一次掷出 $4$，第二次有 $5$ 种可能满足条件：2、3、4、5 或 6。
* 如果第一次掷出 $5$，第二次有 $6$ 种可能满足条件：1、2、3、4、5 或 6。
* 如果第一次掷出 $6$，第二次有 $6$ 种可能满足条件：1、2、3、4、5 或 6。

因此，点数之和大于 5 的可能性一共有 $2 + 3 + 4 + 5 + 6 + 6 = 26$ 种。

由此可得概率为 $\frac{26}{36} \approx 0.72$。四舍五入保留 1 位小数后，结果同样是 0.7！

### 3.4 保存你的答案

得出答案后，就该把它保存下来了。运行下方的代码来看看具体界面。你只需按要求填入答案，然后点击“Save your answer!”即可。

```python
utils.exercise_example()

```

就是这么简单！保存好一道题后，就可以继续攻克下一道了。如果你想修改某道题的答案，只需重新运行对应的单元格并输入新答案，系统就会自动覆盖之前的记录。在作业的最后，你还可以通过一行代码统一检查是否有遗漏未存的题目。

全部完成后，按平时的常规流程提交作业即可。系统会根据你每道题最后一次保存的答案进行评分。

## 4 - 核心概念厘清 🎲🎲🎲

在本次作业中，你会遇到形形色色的骰子场景。这些骰子不仅面数各异，而且有可能是均匀的，也有可能是灌铅的。

* 均匀骰子 (Fair Dice) 落在每一面上的概率完全相等。
* 灌铅骰子 (Loaded Dice) 落在每一面上的概率并不相等。通常其中一面 (或几面) 朝上的概率会明显高于其他面。

好了，以上就是通关本次作业所需的全部背景知识。是时候开始掷骰子了！

## 练习 1：

给定一枚 6 面均匀骰子 (每一面朝上的概率完全相等) ，请计算描述该骰子的概率分布 (Probability Distribution) 的均值 (Mean) 与方差 (Variance)。下图直观地展示了这一分布情况：
![[C3_W2_Assignment_ch-1792450138740.webp|298]]

**提交注意事项：**

* 请将答案以保留三位小数的浮点数 (Floating Point Numbers) 形式提交
* 示例：若要提交 1/4 的值，请输入 0.250

提示：

* 你可以使用 np.random.choice 来模拟投掷均匀骰子。
* 你可以使用 np.mean 和 np.var 来计算 numpy 数组 (Array) 的均值和方差。


```python
# You can use this cell for your calculations (not graded)
# 定义期望的面数 (你可以尝试修改这个数值！)
n_sides = 6
# 使用 numpy 数组表示一枚骰子
dice = np.array([i for i in range(1, n_sides + 1)])

n_rolls = 20000

# 第一次投掷
first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
mean = first_rolls.mean()
var = first_rolls.var()
# 小数点保留两位
print(f"Mean: {mean:.3f}")
print(f"Variance: {var:.3f}")

```
![[C3_W2_Assignment_ch-1792450298596.webp]]

```python
# 运行此单元格以提交你的答案
utils.exercise_1()

```

## 练习 2：

现在假设你将同一枚骰子 (与上一题相同) 连续投掷两次，并记录每次投掷的点数之和。在以下三个`概率质量函数 (Probability Mass Function)` 图表中，哪一个才是你最终应该得到的结果？
![[C3_W2_Assignment_ch-1792450329996.webp|509]]

提示：

* 你可以使用 numpy 数组来存储多次投掷的结果。
* 你可以使用 `+` 运算符将两个 numpy 数组直接相加，例如：`sum = first_throw + second_throw`
* 要模拟多次掷骰子，可以使用列表推导式 (List Comprehension) 或 for 循环

```python
# You can use this cell for your calculations (not graded)
# 定义期望的面数 (你可以尝试修改这个数值！)
n_sides = 6
# 使用 numpy 数组表示一枚骰子
dice = np.array([i for i in range(1, n_sides + 1)])

n_rolls = 20000

# 第一次投掷
first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
# 第二次投掷
second_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
sum_of_rolls = first_rolls + second_rolls

# 绘制直方图
sum_rolls_hist = sns.histplot(sum_of_rolls, stat = "probability", discrete=True)
sum_rolls_hist.set(title=f"Histogram of {n_rolls} rolls (sum of rolling twice)")
plt.show()
```
![[C3_W2_Assignment_ch-1792450427287.webp|367]]

```python
# 运行此单元格以提交你的答案
utils.exercise_2()

```

## 练习 3：

给定一枚均匀的 4 面骰子，将其连续投掷两次并记录点数之和。左图展示了单次投掷时骰子落在每一面上的概率，右图则展示了点数之和的直方图 (Histogram)。请算出每一个点数之和所对应的概率并填入空中 (注意，由于点数之和的分布是完全对称的，你总共只需要输入 4 个数值)：
![[C3_W2_Assignment_ch-1792450443409.webp]]

**提交注意事项：**

* 请将答案以保留三位小数的浮点数形式提交
* 示例：若要提交 1/4 的值，请输入 0.250

```python
# You can use this cell for your calculations (not graded)
# 定义期望的面数 (你可以尝试修改这个数值！)
n_sides = 4
# 使用 numpy 数组表示一枚骰子
dice = np.array([i for i in range(1, n_sides + 1)])

n_rolls = 20000

# 第一次投掷
first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
# 第二次投掷
second_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
sum_of_rolls = first_rolls + second_rolls

probabilities = np.bincount(sum_of_rolls)[2:] / n_rolls
# 保留三位小数
print(f"{probabilities:.3f}")

```
解释
    ![[C3_W2_Assignment_ch-1792450670066.webp]]
    但是返回的probabilities数组 不会有前面两个对0,1值的计数 
    (因为后面 `[2:]` 把前面2个元素切掉了)

```python
# 运行此单元格以提交你的答案
utils.exercise_3()

```

## 练习 4：

沿用上一题的场景，请计算两次投掷点数之和的均值与方差，以及第一次投掷和第二次投掷之间的协方差 (Covariance)：
![[C3_W2_Assignment_ch-1792450968371.webp|267]]

提示：
* 你可以使用 np.cov 来计算两个 numpy 数组的协方差 (不过对于这道特定题目，你未必需要动用代码)。

```python
# You can use this cell for your calculations (not graded)
mean = np.mean(sum_of_rolls)
var = np.var(sum_of_rolls)
cov = np.cov(first_rolls, second_rolls)
print(f"Mean: {mean:.3f}")
print(f"Variance: {var:.3f}")
print(f"Covariance: {cov[0, 1]:.3f}")
print(cov)

```

```python
# 运行此单元格以提交你的答案
utils.exercise_4()

```

## 练习 5：

现在假设你手里有一枚灌铅的 4 面骰子 (它被做了手脚，使得落在点数 2 这一面的概率是其他各面的两倍)：
![[C3_W2_Assignment_ch-1792451142426.webp|279]]

你将它连续投掷两次并记录点数之和。在以下三个`概率质量函数`图表中，哪一个才是你应该得到的结果？
![[C3_W2_Assignment_ch-1792451152544.webp|511]]

提示：

* 你可以利用 np.random.choice 中的 `p` 参数来模拟灌铅骰子。

```python
# You can use this cell for your calculations (not graded)
n_sides = 4
probs_loaded_dice = load_dice(n_sides, loaded_number=2)

n_rolls = 20000
first_rolls = np.array(
    [np.random.choice(dice, p=probs_loaded_dice) for _ in range(n_rolls)]
)

second_rolls = np.array(
    [np.random.choice(dice, p=probs_loaded_dice) for _ in range(n_rolls)]
)

sum_of_rolls = first_rolls + second_rolls

loaded_rolls_hist = sns.histplot(sum_of_rolls, stat="probability", discrete=True)
loaded_rolls_hist.set(
    title=f"Histogram of {n_rolls} rolls (sum of rolling twice a loaded dice)"
)
loaded_rolls_hist.set_xticks(range(min(sum_of_rolls), max(sum_of_rolls) + 1))
plt.show()
```
![[C3_W2_Assignment_ch-1792451280358.webp|442]]

```python
# 运行此单元格以提交你的答案
utils.exercise_5()

```

## 练习 6：

你手里有一枚 6 面灌铅骰子，它落在点数 3 这一面的概率是其他各面的两倍：
![[C3_W2_Assignment_ch-1792451338102.webp|260]]
将其连续投掷两次并记录点数之和。请问在满足累积概率 (Cumulative Probability) 小于或等于 0.5 的前提下，点数之和的最大可能值是多少？
![[C3_W2_Assignment_ch-1792451345037.webp|235]]

提示：

* 掷出点数 3 的概率等于 $\frac{2}{7}$

```python
# You can use this cell for your calculations (not graded)
n_sides = 6
dice = np.array([i for i in range(1, n_sides + 1)])
probs_loaded_dice = load_dice(n_sides, loaded_number=3)

n_rolls = 20000
first_rolls = np.array(
    [np.random.choice(dice, p=probs_loaded_dice) for _ in range(n_rolls)]
)
second_rolls = np.array(
    [np.random.choice(dice, p=probs_loaded_dice) for _ in range(n_rolls)]
)
sum_of_rolls = first_rolls + second_rolls

# 满足累积概率 (Cumulative Probability) 小于或等于 0.5 的前提下，点数之和的最大可能值是多少？
cumulative_probabilities = np.cumsum(np.bincount(sum_of_rolls)[2:] / n_rolls)
max_value_under_half_cumulative = np.max(np.where(cumulative_probabilities <= 0.5)) + 2
print(
    f"Maximum sum under cumulative probability <= 0.5: {max_value_under_half_cumulative}"
)

```
![[C3_W2_Assignment_ch-1792451572416.webp]]

```python
# 运行此单元格以提交你的答案
utils.exercise_6()

```

## 练习 7：

给定一枚 6 面均匀骰子，我们来尝试一种新玩法。只有当第一次投掷的结果**小于或等于** 3 时，你才能投掷第二次。在这一新限制条件下，你会得到以下哪一个`概率质量函数`？
![[C3_W2_Assignment_ch-1792451608514.webp|625]]

提示：

* 你可以先用 numpy 数组模拟出所有的第二次投掷结果，然后使用 np.where 将不符合条件的值直接置为 0



```python
# You can use this cell for your calculations (not graded)
n_rolls = 20000
first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
second_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
second_rolls = np.where(first_rolls <= 3, second_rolls, 0)
sum_of_rolls = first_rolls + second_rolls

dependant_rolls_hist = sns.histplot(sum_of_rolls, stat="probability", discrete=True)
dependant_rolls_hist.set(
    title=f"Histogram of {n_rolls} rolls (dependant sum of rolling twice)"
)
dependant_rolls_hist.set_xticks(range(min(sum_of_rolls), max(sum_of_rolls) + 1))
plt.show()
```
![[C3_W2_Assignment_ch-1792451974888.webp|399]]

```python
# 运行此单元格以提交你的答案
utils.exercise_7()

```

## 练习 8：

场景与上一题基本相同，但规则做了一点调整：只有当第一次投掷的结果**大于或等于** 3 时，你才能投掷第二次。在这个新的限制条件下，你会得到以下哪一个`概率质量函数`？

```python
# You can use this cell for your calculations (not graded)
n_rolls = 20000
first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
second_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
second_rolls = np.where(first_rolls >= 3, second_rolls, 0)
sum_of_rolls = first_rolls + second_rolls

dependant_rolls_hist = sns.histplot(sum_of_rolls, stat="probability", discrete=True)
dependant_rolls_hist.set(
    title=f"Histogram of {n_rolls} rolls (dependant sum of rolling twice)"
)
dependant_rolls_hist.set_xticks(range(min(sum_of_rolls), max(sum_of_rolls) + 1))
plt.show()
```
![[C3_W2_Assignment_ch-1792452017715.webp|375]]

```python
# 运行此单元格以提交你的答案
utils.exercise_8()

```

## 练习 9：

给定一枚 n 面均匀骰子，将其连续投掷两次并记录点数之和。请问随着骰子面数 `n` 的增加，点数之和的均值与方差，以及联合分布 (Joint Distribution) 的协方差会受到怎样的影响？

```python
# 你可以使用此单元格进行计算 (不计分)



```

```python
# 运行此单元格以提交你的答案
utils.exercise_9()

```
![[C3_W2_Assignment_ch-1792452132351.webp|375]]

## 练习 10：

给定一枚 6 面灌铅骰子，将其连续投掷两次并记录点数之和。以下哪项陈述是正确的？

```python
# 你可以使用此单元格进行计算 (不计分)



```

```python
# 运行此单元格以提交你的答案
utils.exercise_10()

```
![[C3_W2_Assignment_ch-1792452225847.webp]]

## 练习 11：

给定一枚 n 面骰子 (可能均匀，也可能灌铅)。将其连续投掷两次并记录点数之和 (两次投掷之间相互独立，不存在依赖关系)。如果只给你展示点数之和的直方图，你能否仅凭这张图反推出骰子落在每一面上的概率？

换句话说，如果只给你提供下面这张点数之和的直方图：
![[C3_W2_Assignment_ch-1792452249234.webp|286]]

你能否利用它推算出骰子每一面朝上的具体概率？这相当于还原出下面这张直方图：
![[C3_W2_Assignment_ch-1792452257202.webp|241]]


```python
# 你可以使用此单元格进行计算 (不计分)



```

```python
# 运行此单元格以提交你的答案
utils.exercise_11()

```

## 提交作业前的最后检查



运行下方单元格，检查你是否已经回答并保存了所有练习题

```python
utils.check_submissions()

```

**恭喜你完成了本次作业！**

在这次作业中，你全面检验了自己关于概率分布、描述性统计 (Descriptive Statistics) 以及这些概念的可视化解读能力。你可以自由选择是通过数学解析推导，还是编写模拟程序来辅助找到正确答案。在这个过程中你或许已经发现，有些题目其实根本不需要任何繁琐的计算，只要敏锐地捕捉到图表中暗藏的视觉线索，就能一眼看出答案。

**再接再厉，继续保持！**