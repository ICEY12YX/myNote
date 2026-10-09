___
# 优化单变量函数 (Functions of One Variable) ：成本最小化

在这份作业中，你将解决一个针对单变量函数的简单优化 (optimization) 问题。给定一份包含某产品在两家供应商处的历史价格数据集，你的任务是确定应该从每家供应商处采购多少比例的产品，以便为未来做出最佳的投资决策。通过将该问题转化为数学模型，你将构建一个需要最小化的目标函数 (target function) ，求解其最小值，并探索其导数 (derivative) 与最终结果之间的内在联系。

## 重要提示

请**不要删除**任何练习单元格，也不要在其他单元格中填写你的解答。**请务必在提供的原始单元格中完成作答**，因为更改单元格可能会导致自动评分系统 (autograder) 报错。

此外，**请避免导入任何新的代码库**，并且**不要在任何计分单元格内导入代码库**——这样做同样会干扰自动评分系统的正常运行。

留下任何未完成的练习 (即没有替换掉 'None' 值) 都会导致自动评分系统出现问题。如果你想在未完成所有练习的情况下提交作业进行测试，请参考[这里]的说明。

# 目录

* [ 1 - 优化问题的陈述](https://www.google.com/search?q=%231)
* [ 1.1 - 问题描述](https://www.google.com/search?q=%231.1)
* [ 1.2 - 问题的数学陈述](https://www.google.com/search?q=%231.2)
* [ 1.3 - 解决方案与思路](https://www.google.com/search?q=%231.3)


* [ 2 - 在 Python 中优化单变量函数](https://www.google.com/search?q=%232)
* [ 2.1 - 依赖库 (Packages) ](https://www.google.com/search?q=%232.1)
* [ 2.2 - 加载并分析数据集](https://www.google.com/search?q=%232.2)
* [ 练习 1](https://www.google.com/search?q=%23ex01)


* [ 2.3 - 构建要优化的目标函数 $L$ 并求解其极小值点](https://www.google.com/search?q=%232.3)
* [ 练习 2](https://www.google.com/search?q=%23ex02)
* [ 练习 3](https://www.google.com/search?q=%23ex03)
* [ 练习 4](https://www.google.com/search?q=%23ex04)





## 1 - 优化问题的陈述

### 1.1 - 问题描述

你的公司希望尽可能降低某批物资的生产成本。在生产过程中，不可避免地需要用到一种核心产品 P，该产品可以由两家合作伙伴——供应商 A 和供应商 B 供货。你的顾问团队收集了产品 P 在这两家供应商处的历史价格数据，数据形式为 2018 年 2 月至 2020 年 3 月期间的月度平均价格。

在编制未来 12 个月的公司预算时，你们计划每个月采购固定数量的产品 P。在挑选供应商时，你注意到在过去的一段时间里，有时选择供应商 A 更划算 (产品 P 的价格更低) ，而有时与供应商 B 合作更省钱。在构建预算模型时，你可以设定从供应商 A 采购商品的特定百分比 (例如 60%) ，剩下的部分则从供应商 B 采购 (例如 40%) ，但这个分配比例必须在接下来的 12 个月中保持不变。这份预算将作为接下来与两家供应商进行合同谈判的重要依据。

基于这些历史价格数据，是否存在一个最优的采购比例，使得从公司 A 和公司 B 组合采购的成本最低、利润最高？又或者，其实比例并不重要，你完全可以只和其中一家供应商合作？

### 1.2 - 问题的数学陈述

我们将公司 A 和公司 B 提供的产品 P 的价格分别记作 $p_A$ (美元) 和 $p_B$ (美元) ，每月的计划采购量记作 $n$ (件) ，那么每月的总成本 (美元) 可以表示为：

$$f\left(\omega\right) = p_A \omega \,n+ p_B \left(1 - \omega\right) n,$$

其中，$0\leq\omega\leq1$ 是一个参数。如果 $\omega = 1$，意味着所有商品都由公司 A 提供；如果 $\omega = 0$，则全由公司 B 提供。若 $0<\omega<1$，则表示按相应比例向两家公司分配采购量。

由于计划在未来十二个月内保持采购量 $n$ 恒定，在数学建模中，通常习惯直接设 $n = 1$。这样做是合理的，因为优化结果并不依赖于采购数量，最终得出的比例将是相同的。简化后的总成本公式如下：

$$f\left(\omega\right) = p_A \omega+ p_B \left(1 - \omega\right) \tag{1}$$

显然，你无法预知未来的价格 $p_A$ 和 $p_B$，你掌握的只有历史数据 (即 $k$ 个月内的价格集合 $\{p_A^1, \cdots, p_A^k\}$ 和 $\{p_B^1, \cdots, p_B^k\}$) 。回顾历史，确实在某些时期取 $\omega = 1$ ($p_A^i < p_B^i$) 更划算，而在另一些时期取 $\omega = 0$ ($p_A^i >p_B^i$) 更好。那么，现在是否能选出一个特定的 $\omega$ 值，以证明它能在未来带来最小的成本？

### 1.3 - 解决方案与思路

这在统计学 (statistics) 中是一个非常经典的**投资组合管理 (portfolio management)** 问题。简而言之，你需要基于历史价格做出投资决策，以实现利润最大化 (或者说成本最小化) 。由于本课程不涉及太多统计学知识，你不需要深入理解下一段即将引出的目标函数 $\mathcal{L}\left(\omega\right)$ (通常称为**损失函数 (loss function)** ) 的具体统计学含义。

我们的思路是：利用每个月的历史价格 $p_A^i$ 和 $p_B^i$，分别计算出对应的 $f\left(\omega\right)$ 值，即 $f^i\left(\omega\right)=p_A^i \omega+ p_B^i \left(1 - \omega\right)$。接着，计算这些历史成本的平均值 $\overline{f\left (\omega\right)}=\text{mean}\left(f^i\left(\omega\right)\right) = \frac{1}{k}\sum_{i=1}^{k}f^i\left(\omega\right)$。我们要寻找的，就是那个能让 $f^i\left(\omega\right)$ 波动最小、最“稳定”的 $\omega$ 值——也就是说，让它偏离平均值 $\overline{f\left (\omega\right)}$ 的程度尽可能小。这意味着你需要最小化差值 $\left(f^i \left(\omega\right) -  \overline{f\left (\omega\right)}\right)$ 的总和。由于差值有正有负，一种通用的做法是先将它们平方，再求这些平方值的平均数：

$$\mathcal{L}\left(\omega\right) = \frac{1}{k}\sum_{i=1}^{k}\left(f^i \left(\omega\right) -  \overline{f\left (\omega\right)}\right)^2\tag{2}$$

在统计学中，$\mathcal{L}\left(\omega\right)$ 实际上就是集合 $\{f^1 \left(\omega\right), \cdots , f^k \left(\omega\right)\}$ 的方差 (variance) 。我们的目标就是最小化这个方差 $\mathcal{L}\left(\omega\right)$，且满足 $\omega\in\left[0, 1\right]$。再说一次，如果你不明白为什么偏偏选中 $\mathcal{L}\left(\omega\right)$ 作为优化目标，不必担心。你可能会觉得直接最小化平均成本 $\overline{f\left (\omega\right)}$ 不是更符合直觉吗？但[风险管理]理论告诉我们，在这种特定的决策场景下，优化方差才是控制风险和成本的正确途径。

统计理论证明，必定存在一个 $\omega\in\left[0, 1\right]$ 能够使得函数 $\mathcal{L}\left(\omega\right)$ 达到极小值，而且这个值可以通过数据集 $\{p_A^1, \cdots, p_A^k\}$ 和 $\{p_B^1, \cdots, p_B^k\}$ 的某些特性直接求出。不过，因为这不是一门统计课，我们引入这个例子只是为了演示如何基于数据集进行单变量的函数优化。这正是你检验学习成果并将其应用于本周所学知识的绝佳练习。

现在，让我们导入数据集，看看是否能找出对应函数 $\mathcal{L}\left(\omega\right)$ 的极小值点吧。

## 2 - 在 Python 中优化单变量函数

### 2.1 - 依赖库 (Packages)

首先导入所有必需的代码库。除了本课程中你已经见过的库之外，这次你还需要导入 `pandas` 库。这是数据操作和数据分析领域最常用的工具包。

```python
# A function to perform automatic differentiation.
from jax import grad
# A wrapped version of NumPy to use JAX primitives.
import jax.numpy as np
# A library for programmatic plot generation.
import matplotlib.pyplot as plt
# A library for data manipulation and analysis.
import pandas as pd

# A magic command to make output of plotting commands displayed inline within the Jupyter notebook.
%matplotlib inline 

```

加载为此笔记本设计的单元测试用例。

```python
import w1_unittest

# Please ignore the warning message about GPU/TPU if it appears.

```

### 2.2 - 加载并分析数据集

供应商 A 和 B 的历史价格存储在 `data/prices.csv` 文件中。你可以使用 `pandas` 的 `read_csv` 函数来读取它。这个例子非常直白，不需要设置任何额外的参数。

```python
df = pd.read_csv('data/prices.csv')

```

数据现在已经被加载并保存到了变量 `df` 中，它的数据类型是 **DataFrame**，这是 `pandas` 中最核心的数据对象。你可以把它想象成一张表格或者电子表格，它是一个支持命名列的二维数据结构，且每一列可以包含不同的数据类型。完整的文档说明可以点击[这里]查看。

我们可以使用标准的 `print` 函数来预览数据：

```python
print(df)

```

如果只想打印列名列表，可以调用 DataFrame 的 `columns` 属性：

```python
print(df.columns)

```

观察输出的表格和列名，你可以看出提供的是每月的价格 (单位为美元) ，而你只需要提取 `price_supplier_a_dollars_per_item` 和 `price_supplier_b_dollars_per_item` 这两列的数据。在实际的工业场景中，数据集往往庞大得多，在输入模型之前需要进行彻底的审查和数据清洗。但这不属于本课程的讨论重点。

若要单独获取 DataFrame 中某一列的数据，可以直接将列名作为属性名调用。例如，下面的代码将输出 DataFrame `df` 的 `date` 列：

```python
df.date

```

### 练习 1

请将供应商 A 和供应商 B 的历史价格分别加载到变量 `prices_A` 和 `prices_B` 中。同时，请使用 `np.array` 函数将这些价格数据转换为数据类型为 `float32` 的 `NumPy` 数组。

```python
### START CODE HERE ### (~ 4 lines of code)
prices_A = None
prices_B = None
prices_A = None(None).astype('None')
prices_B = None(None).astype('None')
### END CODE HERE ###

```

```python
# Print some elements and mean values of the prices_A and prices_B arrays.
print("Some prices of supplier A:", prices_A[0:5])
print("Some prices of supplier B:", prices_B[0:5])
print("Average of the prices, supplier A:", np.mean(prices_A))
print("Average of the prices, supplier B:", np.mean(prices_B))

```

##### **预期输出**

```Python
Some prices of supplier A: [104. 108. 101. 104. 102.]
Some prices of supplier B: [76. 76. 84. 79. 81.]
Average of the prices, supplier A: 100.799995
Average of the prices, supplier B: 100.0

```

```python
w1_unittest.test_load_and_convert_data(prices_A, prices_B)

```

两家供应商的平均价格看起来差不多。但如果你把历史价格绘制成折线图，就会清楚地看到：在某些时期，供应商 A 的价格处于低位，而在另一些时期，供应商 B 的价格更具优势。

```python
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(prices_A, 'g', label="Supplier A")
plt.plot(prices_B, 'b', label="Supplier B")
plt.legend()

plt.show()

```

单看这些历史数据，你能直接判断出和哪家供应商合作利润更高吗？正如在第 [1.3](https://www.google.com/search?q=%231.3) 节中讨论过的，我们需要找出一个能让公式 $(2)$ 取得最小值的 $\omega \in \left[0, 1\right]$。

### 2.3 - 构建要优化的目标函数 $\mathcal{L}$ 并求解其极小值点

### 练习 2

计算 `f_of_omega`，它对应于公式 $f^i\left(\omega\right)=p_A^i \omega+ p_B^i \left(1 - \omega\right)$。价格 $\{p_A^1, \cdots, p_A^k\}$ 和 $\{p_B^1, \cdots, p_B^k\}$ 会通过数组 `pA` 和 `pB` 传入。因此，将它们分别乘以标量 `omega` 和 `1 - omega`，再将得到的数组相加，你就能得到一个包含 $\{f^1\left(\omega\right), \cdots, f^k\left(\omega\right)\}$ 的数组。

接着，根据表达式 $(2)$，可以利用 `f_of_omega` 数组来计算 `L_of_omega`：

$$\mathcal{L}\left(\omega\right) = \frac{1}{k}\sum_{i=1}^{k}\left(f^i \left(\omega\right) -  \overline{f\left (\omega\right)}\right)^2$$

```python
def f_of_omega(omega, pA, pB):
    ### START CODE HERE ### (~ 1 line of code)
    f = None
    ### END CODE HERE ###
    return f

def L_of_omega(omega, pA, pB):
    return 1/len(f_of_omega(omega, pA, pB)) * np.sum((f_of_omega(omega, pA, pB) - np.mean(f_of_omega(omega, pA, pB)))**2)

```

```python
print("L(omega = 0) =",L_of_omega(0, prices_A, prices_B))
print("L(omega = 0.2) =",L_of_omega(0.2, prices_A, prices_B))
print("L(omega = 0.8) =",L_of_omega(0.8, prices_A, prices_B))
print("L(omega = 1) =",L_of_omega(1, prices_A, prices_B))

```

##### **预期输出**

```Python
L(omega = 0) = 110.72
L(omega = 0.2) = 61.1568
L(omega = 0.8) = 11.212797
L(omega = 1) = 27.48

```

```python
w1_unittest.test_f_of_omega(f_of_omega)

```

分析上面的输出结果，你可以观察到：当 $\omega$ 从 $0$ 增加到 $0.2$，再增加到 $0.8$ 时，函数 $\mathcal{L}$ 的值是在递减的；但是当 $\omega = 1$ 时，函数 $\mathcal{L}$ 的值反而反弹变大了。那么，究竟哪个 $\omega$ 能使函数 $\mathcal{L}$ 达到真正的极小值呢？

在这个简单的模型中，$\mathcal{L}\left(\omega\right)$ 仅仅是一个单变量函数，要找到它在一定精度下的极小值点简直轻而易举。你只需要让 $\omega = 0, 0.001, 0.002, \cdots , 1$ 挨个取值并计算一遍，然后找出结果数组中的最小值即可。

注意：如果你向 `L_of_omega` 传递的是一个数组而不是单一的 `omega` 值，代码将会报错 (因为它最初的设计并没有考虑到数组运算) 。当然我们可以重写它使其支持向量化，但现在完全没必要——你可以直接写个循环来计算，毕竟计算量非常小。

### 练习 3

为 `omega_array` 数组中的每一个元素计算 `L_of_omega` 的值，并使用 `.at[<index>].set(<value>)` 语法，将计算结果写入到输出数组 `L_array` 的对应位置。

*注意*：这里我们使用的是 `jax.numpy` 而非原生 `NumPy`。虽然到目前为止还没真正用到 `jax` 的黑科技，但在接下来的步骤中就会大显身手。为了避免代码里同时混杂两个版本的 numpy 库，你必须使用 `.at[<index>].set(<value>)` 方法来原地更新数组。

```python
# Parameter endpoint=True will allow ending point 1 to be included in the array.
# This is why it is better to take N = 1001, not N = 1000
N = 1001
omega_array = np.linspace(0, 1, N, endpoint=True)

# This is organised as a function only for grading purposes.
def L_of_omega_array(omega_array, pA, pB):
    N = len(omega_array)
    L_array = np.zeros(N)

    for i in range(N):
        ### START CODE HERE ### (~ 2 lines of code)
        L = None(None[None], None, None)
        L_array = L_array.at[None].set(None)
        ### END CODE HERE ###
        
    return L_array

L_array = L_of_omega_array(omega_array, prices_A, prices_B)

```

```python
print("L(omega = 0) =",L_array[0])
print("L(omega = 1) =",L_array[N-1])

```

##### **预期输出**

```Python
L(omega = 0) = 110.72
L(omega = 1) = 27.48

```

```python
w1_unittest.test_L_of_omega_array(L_of_omega_array)

```

现在，可以直接调用 `NumPy` 的 `argmin()` 方法来找出函数 $\mathcal{L}\left(\omega\right)$ 的极小值点。因为我们在 $\left[0, 1\right]$ 这个区间内密集采样了 $N = 1001$ 个点，所以最终结果将精确到小数点后三位：

```python
i_opt = L_array.argmin()
omega_opt = omega_array[i_opt]
L_opt = L_array[i_opt]
print(f'omega_min = {omega_opt:.3f}\nL_of_omega_min = {L_opt:.7f}')

```

这个结果表明，基于历史数据，将 $\omega = 0.702$ 设为供应商 A 和供应商 B 之间的份额分配比例是最划算的。也就是说，计划将产品 P 总量的 $70.2\%$ 交给公司 A 供应，剩下的 $29.8\%$ 交给公司 B 供应，是一个非常合理的商业决策。

如果你想获得更高的精度，只需调大采样点数量 N 即可。这是单变量函数优化的一个非常入门的例子。由于计算成本极其低廉，我们可以通过“暴力”密集采样的方式找到极小值。但在机器学习 (Machine Learning) 中，模型动辄拥有成百上千个参数，如果你还想用这种“笨办法”，可能需要对目标函数进行数以百万计的重复求值。这在绝大多数情况下都是计算资源所不允许的。这正是微积分 (Calculus) 及其求导技术大显身手的地方。

在本课程接下来的几周里，你将学习如何利用微分技术来优化多变量函数。但作为目前的入门热身，让我们先计算一下函数 $\mathcal{L}\left(\omega\right)$ 在 `omega_array` 这些点上的导数，看看在刚才找到的极小值点处，它的导数是不是真的最接近零。

### 练习 4

对于 `omega_array` 中的每一个 $\omega$，使用 `JAX` 库提供的 `grad()` 函数计算导数 $\frac{d\mathcal{L}}{d\omega}$。请记住，你需要先将待求导的函数 (这里是 $\mathcal{L}\left(\omega\right)$) 作为参数传给 `grad()`，然后再对 `omega_array` 中的对应元素求取具体的导数值。最后，使用 `.at[<index>].set(<value>)` 方法将计算结果写入到输出数组 `dLdOmega_array` 的对应位置。

```python
# This is organised as a function only for grading purposes.
def dLdOmega_of_omega_array(omega_array, pA, pB):
    N = len(omega_array)
    dLdOmega_array = np.zeros(N)

    for i in range(N):
        ### START CODE HERE ### (~ 2 lines of code)
        dLdOmega = None(None)(None[None], None, None)
        dLdOmega_array = dLdOmega_array.at[None].set(None)
        ### END CODE HERE ###
        
    return dLdOmega_array

dLdOmega_array = dLdOmega_of_omega_array(omega_array, prices_A, prices_B)

```

```python
print("dLdOmega(omega = 0) =",dLdOmega_array[0])
print("dLdOmega(omega = 1) =",dLdOmega_array[N-1])

```

##### **预期输出**

```Python
dLdOmega(omega = 0) = -288.96
dLdOmega(omega = 1) = 122.47999

```

```python
w1_unittest.test_dLdOmega_of_omega_array(dLdOmega_of_omega_array)

```

现在，为了找出最接近 $0$ 的导数值，我们取每个导数的绝对值 $\left\vert{}\frac{d\mathcal{L}}{d\omega}\right\vert{}$，然后从中找出最小值所在的位置。

```python
i_opt_2 = np.np.abs(dLdOmega_array).argmin()
omega_opt_2 = omega_array[i_opt_2]
dLdOmega_opt_2 = dLdOmega_array[i_opt_2]
print(f'omega_min = {omega_opt_2:.3f}\ndLdOmega_min = {dLdOmega_opt_2:.7f}')

```

结果完全一致：$\omega = 0.702$。让我们将 $\mathcal{L}\left(\omega\right)$ 和 $\frac{d\mathcal{L}}{d\omega}$ 的图像绘制出来，直观地观察一下函数 $\mathcal{L}\left(\omega\right)$ 的极小值点，以及它的导数穿越 $0$ 轴的位置：

```python
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# Setting the axes at the origin.
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(omega_array,  L_array, "black", label = "$\mathcal{L}\\left(\omega\\right)$")
plt.plot(omega_array,  dLdOmega_array, "orange", label = "$\mathcal{L}\'\\left(\omega\\right)$")
plt.plot([omega_opt, omega_opt_2], [L_opt,dLdOmega_opt_2], 'ro', markersize=3)

plt.legend()

plt.show()

```

恭喜你完成了本周的作业！这个例子生动地展示了优化问题在现实商业世界中的真实应用场景，并让你有机会亲手探索并解决了一个单变量函数的最小化问题。接下来，准备好迎接多变量函数优化的挑战吧！
