___
# 特征值（Eigenvalues）与特征向量（Eigenvectors）

欢迎来到本课程的最后一次作业，祝贺你一路坚持学到了这里！在本次结课作业中，你将运用线性代数（Linear Algebra）知识以及 Python 和 NumPy 的编程技能，去攻克真实世界中的实际应用场景，亲身体会线性代数如何优雅地化繁为简、解决棘手问题。

**完成本次作业后，你将能够：**

* 在网页浏览模型中应用线性变换（Linear Transformations）、特征值与特征向量
* 在数据集上应用主成分分析（Principal Component Analysis, PCA）来实现降维（Dimensionality Reduction）

## 重要提示

请**切勿**删除任何练习代码块，或将答案写在其他代码块中。**请务必在原本提供的代码块中作答**，否则会导致自动评分系统（Autograder）崩溃。

此外，**请勿导入任何新的函数库**，并**避免在任何用于评分的代码块内部导入库**，这同样会干扰自动评分系统的正常运行。
___

___
## 依赖库

运行以下代码块以加载所需的依赖库。

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg

```

加载针对本作业专门编写的工具模块 utils 和单元测试模块。

```python
import utils
import w4_unittest

```

## 1 - 特征值与特征向量的应用：网页浏览模型

正如你在课程中所学到的，特征值与特征向量在所谓的（离散）动力系统（Discrete Dynamical Systems）中发挥着至关重要的作用。回顾一下，**离散动力系统**描述的是一种==状态会随着时间推移、依照某种特定规律发生演变的系统。==在刻画这种系统时，我们可以把所有可能出现的状态（例如晴天、雨天或阴天）打包记录在一个向量中，这个向量就被称为**状态向量（State Vector）**。

每个离散动力系统都可以用一个**转移矩阵（Transition Matrix）**$P$ 来表示，它刻画了当前处于某种状态时，下一步跳转到其他各个状态的概率大小。举个例子，矩阵中的元素 $(2,1)$ 就代表着系统从状态 $1$ 转移到状态 $2$ 的概率。

若初始状态记为 $X_0$，那么演变至下一个状态 $X_1$ 的过程本质上就是一个由转移矩阵 $P$ 所定义的线性变换：$X_1=PX_0$。顺藤摸瓜，系统随后的状态演变依次为 $X_2=PX_1=P^2X_0$、$X_3=P^3X_0$，以此类推。这就意味着对于 $t=0,1,2,3,\ldots$，均满足递推关系式 $X_t=PX_{t-1}$。换句话说，我们只要不断乘以转移矩阵 `P`，就能让系统一步步迈向未来的新状态。

离散动力系统的一个经典应用就是==模拟网民浏览网页的行为==。网页通常布满了跳转链接，因此动力系统可以生动地==模拟用户通过点击链接在网页之间来回穿梭的过程==。为简单起见，我们假设用户只通过点击网页内的链接跳转到新页面，而不会凭空输入新网址。

在这种设定下，==状态向量 $X_t$ 代表着在时间步 $t$ 时用户停留在各个特定网页上的概率分布==。用户每点击跳转一次，模型的状态向量就会从 $X_{t-1}$ 推进更新至 $X_t$。由转移矩阵 $P$ 构成的线性变换包含一系列元素 $p_{ij}$，它们代表用户==从网页 $i$ 浏览跳转到网页 $j$ 的概率==。对于任意固定的列 $j$，其内部元素构成了在当前位于状态 $j$ 的前提下、下一步所处位置的概率分布。正因如此，该矩阵每一列的所有元素之和必须严格等于 1。

### 练习 1

为了便于演算，我们先假设总共只有 $n=5$ 个网页。这意味着转移矩阵 $P$ 将是一个 $5 \times 5$ 的方阵。在这个特殊例子中，主对角线上的所有元素都应为 $0$，因为我们合理地==假设当前页面没有指向它自身的链接。==同时如前所述，每一列的所有数值相加必须等于 1。以下展示了一个当 $n=5$ 时的转移矩阵示例：

$$P= \begin{bmatrix} 0    & 0.75 & 0.35 & 0.25 & 0.85 \\ 0.15 & 0    & 0.35 & 0.25 & 0.05 \\ 0.15 & 0.15 & 0    & 0.25 & 0.05 \\ 0.15 & 0.05 & 0.05 & 0    & 0.05 \\ 0.55 & 0.05 & 0.25 & 0.25 & 0 \end{bmatrix}\tag{5}$$

请定义初始向量 $X_0$，设定用户是从第 $4$ 个网页开始浏览的（也就是说 $X_0$ 是一个仅在第 4 个位置为 1、其余元素均为 0 的列向量）。随后进行一次变换：$X_1=PX_0$，计算出用户在第一步跳转后分别身处各个网页的概率向量。

```python
P = np.array([ 
    [0, 0.75, 0.35, 0.25, 0.85], 
    [0.15, 0, 0.35, 0.25, 0.05], 
    [0.15, 0.15, 0, 0.25, 0.05], 
    [0.15, 0.05, 0.05, 0, 0.05], 
    [0.55, 0.05, 0.25, 0.25, 0]  
]) 

X0 = np.array([[0],[0],[0],[1],[0]])

### START CODE HERE ###

# Multiply matrix P and X_0 (matrix multiplication).
X1 = P @ X0

### END CODE HERE ###

print(f'Sum of columns of P: {sum(P)}')
print(f'X1:\n{X1}')

```

##### **预期输出**

```Python
Sum of columns of P: [1. 1. 1. 1. 1.]
X1:
[[0.25]
 [0.25]
 [0.25]
 [0.  ]
 [0.25]]

```

```python
# 运行测试用例检验你的代码
w4_unittest.test_matrix(P, X0, X1)

```

将上述变换重复执行 $m$ 次，你就能得到一个向量 $X_m$，它准确记录了用户连续点击浏览 $m$ 步之后，停留在各个网页上的最终概率。

```python
X = np.array([[0],[0],[0],[1],[0]])
m = 20

for t in range(m):
    X = P @ X
    
print(X)

```
![[Pasted image 20261004061315.png]]


当步数 $m$ 变得非常大时，==预测 $X_m$ 中的稳态概率==具有极大的实用价值——它能告诉我们，用户在网络世界里漫游许久之后，最有可能在哪些网页上流连。换句话说，我们能够洞察到底哪些网页最终汇聚了最多的互联网流量。
一种朴素的预测方法就是像上面那样不断重复矩阵乘法；面对这个轻量级的 $5 \times 5$ 小矩阵，计算机确实能够轻松应付。然而在现实世界中，网页网络构成的矩阵庞大得惊人，==暴力迭代相乘将带来无法承受的巨大计算开销。==而这正是==**特征值与特征向量**==施展魔力的地方——它们能够大幅削减计算负担。让我们一探究竟！

首先，计算前面定义的转移矩阵 $P$ 的特征值和特征向量：

```python
eigenvals, eigenvecs = np.linalg.eig(P)
print(f'Eigenvalues of P:\n{eigenvals}\n\nEigenvectors of P\n{eigenvecs}')

```
![[Pasted image 20261004061421.png]]

不难发现，==输出中恰好有一个特征值的大小为 $1$，而其余四个特征值的绝对值都小于 1==。事实证明，这正是所有转移矩阵所共有的神奇特质。实际上，这一类矩阵具备极为丰富的优良数学性质，在数学界被统称为**马尔可夫矩阵（Markov Matrix）**。

概括来说，==只要一个方阵内部的所有元素非负，且每一列的元素累加和均为 $1$，==它就被定义为**马尔可夫矩阵**。马尔可夫矩阵拥有一个极其便利的黄金定律：==**它必定存在一个等于 1 的特征值。**==正如课程中所强调的，==在转移矩阵中，与特征值 1 对应的特征向量，精准决定了系统在经历长时间演化后的终极极限状态。==

你可以轻松验证，前面定义的矩阵 $P$ 确实是一个标准的马尔可夫矩阵。
因此，当演步数 $m$ 足够大时，==**递推方程 $X_m=PX_{m-1}$ 便可近似化简写为 $X_m=PX_{m-1}=1\times X_m$。**==这意味着在预测极其遥远未来的概率时，我们根本不需要做漫长的幂次迭代，==**只需直接锁定对应于特征值 $1$ 的那个特征向量即可！**==

现在，让我们直接提取出与特征值 $1$ 相对应的特征向量。

```python
X_inf = eigenvecs[:,0]

print(f"Eigenvector corresponding to the eigenvalue 1:\n{X_inf[:,np.newaxis]}")

```
![[Pasted image 20261004061749.png]]
解释
    ![[Pasted image 20261004062058.png]]
    ![[Pasted image 20261004062109.png]]
    ![[Pasted image 20261004062225.png]]

### 练习 2

为了验证这一结论的可靠性，请执行矩阵与向量的乘法 $PX$（即将矩阵 `P` 与向量 `X_inf` 相乘），检验计算结果是否严格等于原向量 $X$（即 `X_inf`）。

```python
# 该函数仅用于自动评分检测
def check_eigenvector(P, X_inf):
    ### START CODE HERE ###
    X_check = P @ X_inf
    ### END CODE HERE ###
    return X_check

X_check = check_eigenvector(P, X_inf)
print("Original eigenvector corresponding to the eigenvalue 1:\n" + str(X_inf))
print("Result of multiplication:" + str(X_check))

# 函数 np.isclose 用于逐元素比对两个 NumPy 数组，支持通过 rtol 参数设置容差
print("Check that PX=X element by element:" + str(np.isclose(X_inf, X_check, rtol=1e-10)))

```
![[Pasted image 20261004062430.png]]

```python
# 运行测试用例检验你的代码
w4_unittest.test_check_eigenvector(check_eigenvector)

```

这个计算结果精确指明了特征向量的空间方向，但细心观察会发现，==里面的数值并不能直接当作概率来看待——**因为里面出现了负数，且各项之和并不等于 1**。==别担心，这完全不是问题。请记住，按照通用约定，`np.eig` 返回的是范数（模长）归一化为 1 的特征向量，但同一射线上任何伸缩后的向量本质上都是特征值 1 的合法特征向量。因此我们只需==**对它做一次等比例缩放，使得所有元素转为正数且总和恰好为 1 即可**==。如此一来，用户长期漫游后最终落在各个网页上的真实概率就浮出水面了。

```python
X_inf = X_inf/sum(X_inf)
print(f"Long-run probabilities of being at each webpage:\n{X_inf[:,np.newaxis]}")

```
![[Pasted image 20261004062555.png]]


这意味着，在网络中持续浏览很久很久之后，用户停留在网页 1 的概率约为 0.394，在网页 2 的概率为 0.134，在网页 3 的概率为 0.114，在网页 4 的概率为 0.085，而在网页 5 的概率为 0.273。

从这一结果可以清晰看出，网页 1 是最吸引人流的明星页面，而网页 4 则是访客最稀少的冷门页面。

如果将理论计算出的极限定态 `X_inf` 与前面反复模拟演化 20 次得到的实验数值相比，你会惊讶地发现，两者在小数点后三位以内已经完全吻合！

这里还有一个有趣的历史冷知识：正是这类基于马尔可夫矩阵的动力学模型，奠定了当年轰动全球的 PageRank 算法基石，进而造就了 Google 搜索引擎的非凡商业传奇。

## 2 - 特征值与特征向量的应用：主成分分析（PCA）

前情提要
![[Pasted image 20261008163735.png]]

在理论课上我们学过，特征值与特征向量最经典、最强大的舞台之一，就是降维算法——主成分分析（Principal Component Analysis，简称 PCA）。

在作业的第二部分，你将把 PCA 算法亲手应用到图像数据集上，体验一把纯粹依靠数学力量完成的图像压缩（Image Compression）技术。

我们采用的数据来源于 Kaggle 平台的 [猫狗面部数据集（Cat and dog face）](https://www.google.com/search?q=https://www.kaggle.com/datasets/alessiosanna/cat-dog-64x64-pixel/data)。这里我们专门抽取了其中的猫咪头像子集。

牢记 PCA 的标准流程：在任何数据集上应用 PCA，
    第一步都是构建样本的协方差矩阵（Covariance Matrix）。
    随后，求解该协方差矩阵的特征值与特征向量。
        这里的每一个特征向量，在几何上都代表着一个**主成分（Principal Component）**。
    降维的核心操作，就是挑选出对应于==**前 $k$ 个最大特征值的 $k$ 个主成分**==，
    然后把原始的高维数据正交投影（Project）到由这些主成分（特征向量）张成的低维子空间上。

### 2.1 加载数据

首先，调用 utils 模块中的 `load_images` 函数把猫咪图片读取进来，并自动转换为黑白灰度图。

```python
imgs = utils.load_images('./data/')

```

加载后的 `imgs` 应当是一个 Python 列表，列表里的==每个元素都是一个独立的图像数组==（即二维矩阵）。让我们来检查一下数据的基本维度：

```python
height, width = imgs[0].shape

print(f'\nYour dataset has {len(imgs)} images of size {height}x{width} pixels\n')

```
![[Pasted image 20261008164747.png]]
这次输出就是宽高了

接下来不妨随机画出一张图像瞧瞧。你可以调用灰度颜色映射 'gray' 来绘制黑白画面，也可以随心所欲多看几张萌猫头像。

```python
plt.imshow(imgs[0], cmap='gray')

```
![[Pasted image 20261004062822.png]]

在数字图像处理中，我们可以==将每一个像素点都看作一个独立的特征变量==。把图像保持为二维矩阵形式固然便于人类用肉眼直观欣赏，但在针对每个特征变量进行数学矩阵运算时却显得笨拙低效。

为了顺利运用 PCA 实施降维，我们==**必须把每张二维图像展平（Flatten）成一个单行向量**==。借助 NumPy 的 `reshape` 函数便能轻松完成这一操作。
(就是所有像素 展开成一行)

展平后得到的总数组将包含 55 行（每行代表一只猫咪样本）以及 64x64=4096 列（每列代表一个像素维度的变量）。

```python
imgs_flatten = np.array([im.reshape(-1) for im in imgs])
#

print(f'imgs_flatten shape: {imgs_flatten.shape}')

```
![[Pasted image 20261008164927.png]]
行数是图像的个数  列数是每个图像的像素数
就是现在每个像素算一个维度 (跟像素的位置没关系了?)

解释
    ![[Pasted image 20261008165906.png]]

**注意**
- imgs是图像列表 im是每个元素, 先把每个元素im展平 再存到新的array imgs_flatten
- 所以最后的矩阵变成 第一列 是每张图第一个像素(比如左上角第一个) 第二列是第二个像素

**预想接下来会发生什么:**
    ![[Pasted image 20261008170438.png]]
    ![[Pasted image 20261008170508.png]]
    这步等会解释
## 2.2 - 计算协方差矩阵

在将图像整合成标准的二维矩阵形态后，我们就可以在展平的数据集上放手施展 PCA 了。

若我们将每个像素（列）视作一个变量，而将每张图像（行）视作一次独立的样本观测，那么整个数据集就构成了拥有 55 个样本、4096 个变量 $X_1, X_2, \ldots, X_{4096}$ 的数据矩阵：

$$\mathrm{imgs\_flatten} = \begin{bmatrix} x_{1,1} & x_{1,2} & \ldots & x_{1,4096}\\                                            x_{2,1} & x_{2,2} & \ldots & x_{2,4096} \\                                            \vdots & \vdots & \ddots & \vdots \\                                            x_{55,1} & x_{55,2} & \ldots & x_{55,4096}\end{bmatrix}$$

回顾理论课所学，执行 PCA 计算的首要前提，是求解特征之间的协方差矩阵：

$$\Sigma = \begin{bmatrix}Var(X_1) & Cov(X_1, X_2) & \ldots & Cov(X_1, X_{4096}) \\                           Cov(X_1, X_2) & Var(X_2) & \ldots & Cov(X_2, X_{4096})\\                           \vdots & \vdots & \ddots & \vdots \\                           Cov(X_1,X_{4096}) & Cov(X_2, X_{4096}) &\ldots & Var(X_{4096})\end{bmatrix}$$

### 练习 3

为了正确求得协方差矩阵，必须先对原始数据实施“去中心化”（零均值化处理），即让每个变量（每一列）减去其对应的算术平均值。

正如讲座中所介绍的，中心化后的数据矩阵呈现如下形式：

$$X = \begin{bmatrix} (x_{1,1}- \mu_1) & (x_{1,2}- \mu_2) & \ldots & (x_{1,4096}- \mu_{4096})\\                                            (x_{2,1}- \mu_1) & (x_{2,2}- \mu_2) & \ldots & (x_{2,4096}- \mu_{4096}) \\                                            \vdots & \vdots & \ddots & \vdots \\                                            (x_{55,1}- \mu_1) & (x_{55,2}- \mu_2) & \ldots & (x_{55,4096}- \mu_{4096})\end{bmatrix}$$

课程中提到，以第一个变量（即第一个像素位置）为例，其均值可以通过对该位置所有 55 个样本求平均直接算出：$\mu_1 = \frac{1}{55} \sum_{i=1}^{55} x_{i,1}$。

在接下来的练习中，你将实现一个函数：它接收尺寸为 $\mathrm{样本数}\times\mathrm{特征变量数}$ 的二维数组，并返回中心化之后的新数据矩阵。
(这里的特征变量就是像素数)

在执行中心化操作时，你需要用到三个常用的 NumPy 内置函数：

* [`np.mean`](https://numpy.org/doc/2.5/reference/generated/numpy.mean.html)：用于计算各个变量的平均值，调用时务必记得传入正确的 `axis` 轴参数。
* [`np.repeat`](https://numpy.org/doc/2.5/reference/generated/numpy.repeat.html)：用于将各个均值 $\mu_i$ 在样本维度上进行平铺复制。
* [`np.reshape`](https://numpy.org/doc/2.5/reference/generated/numpy.reshape.html#numpy-reshape)：用于将平铺的均值重新重构为与原始数据尺寸完全一致的均值矩阵。为确保在内存重塑时按正确的列优先顺序排布，请务必设置参数 `order='F'`。

```python
# 用于评分的函数单元
def center_data(Y):
    """
    对原始数据矩阵实施中心化去均值处理
    参数:
         Y (ndarray): 输入的原始数据，形状为 (样本数 x 像素变量数)
    返回:
        X (ndarray): 中心化后的数据矩阵
    """
    ### START CODE HERE ###
    mean_vector = np.mean(Y, axis=0)
    mean_matrix = np.repeat(mean_vector, Y.shape[0]) #现在还是一维数组
    #将mean_vector里的每个元素 重复Y.shape[0]次(就是55次)
    
    # 使用 np.reshape 重构为与 Y 同尺寸的矩阵。切记添加参数 order='F'
    mean_matrix = np.reshape(mean_matrix, Y.shape, order='F')
    
    X = Y - mean_matrix
    ### END CODE HERE ###
    return X

```
[[reshape用法]]
理解
    ![[Pasted image 20261004063953.png]]
    ![[Pasted image 20261004064104.png]]
    ![[Pasted image 20261004065008.png]]

现在，将写好的 `center_data` 函数应用到先前展平的图像数据 `imgs_flatten` 上吧！

中心化处理之后，你可以把处理后的向量重新画成图像看一看，你会发现猫咪的面容轮廓依然清晰可辨。这是因为 Matplotlib 的==图像色阶并非绝对固定，而是根据当前像素数值区间自适应拉伸映射的。==

```python
X = center_data(imgs_flatten)
plt.imshow(X[0].reshape(64,64), cmap='gray')

```

```python
# 运行测试用例检验你的代码
w4_unittest.test_center_data(center_data)

```

### 练习 4

在拿到中心化后的数据矩阵 $X$ 后，接下来就可以顺利推导并求解其协方差矩阵了。

你可能还记得课堂上介绍的数学捷径：一旦完成了数据的零均值中心化，协方差矩阵就可以极其优雅地通过转置矩阵 $X^T$ 与原始矩阵 $X$ 进行点积（Dot Product），随后整体除以样本自由度（即样本数量减去 1）直接得到。

在 Python 中执行矩阵点积，直接调用 NumPy 的 [`np.dot`](https://www.google.com/search?q=https://numpy.org/doc/stable/reference/generated/numpy.dot.html%23numpy-dot) 函数即可。

```python
def get_cov_matrix(X):
    """ 基于中心化数据 X 计算特征协方差矩阵
    参数:
        X (np.ndarray): 中心化后的数据矩阵
    返回:
        cov_matrix (np.ndarray): 协方差矩阵
    """

    ### START CODE HERE ###
    cov_matrix = (X.T @ X) / (X.shape[0] - 1)
    ### END CODE HERE ###
    
    return cov_matrix

```

```python
cov_matrix = get_cov_matrix(X)

```

检查协方差矩阵的维度：它应该是一个行列对称的方阵，大小恰好为 4096 行乘以 4096 列。

```python
print(f'Covariance matrix shape: {cov_matrix.shape}')

```
![[Pasted image 20261004065238.png]]

```python
# 运行测试用例检验你的代码
w4_unittest.test_cov_matrix(get_cov_matrix)

```

### 2.3 - 计算特征值与特征向量

万事俱备，现在我们可以正式对协方差矩阵进行特征分解，提取其特征值和特征向量了。
考虑到大型矩阵的运算性能瓶颈，我们本次不直接调用 `np.linalg.eig`，❤️而是采用功能高度相似且速度更快的稀疏矩阵函数 [`scipy.sparse.linalg.eigsh`](https://www.google.com/search?q=https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)。这个函数巧妙利用了协方差矩阵必定对称（即 $\mathrm{cov\_matrix}^T=\mathrm{cov\_matrix}$）的优良性质，并且允许我们==只定向计算前几个关键的特征值-特征向量对==，从而极大地节省算力。

虽然严格的数学证明超出了本课程的范围，但在线性代数中可以证明：协方差矩阵 `cov_matrix` 中非零特征值的数量最多只有 16 个，这恰恰受限于原始数据矩阵 `X` 极度扁平的最小维度（16 个样本）。因此出于计算效率考虑，我们只需计算前 16 个最大的特征值 $\lambda_1, \ldots, \lambda_{16}$ 以及它们对应的特征向量 $v_1, \ldots, v_{16}$ 即可。你可以尝试将 `scipy.sparse.linalg.eigsh` 中的 `k` 参数稍微调大一点，亲自验证多出来的那些特征值是否全为 0；不过建议参数不要超过 80，否则会消耗较长的计算时间。

这个 SciPy 函数的输出格式与 `np.linalg.eig` 基本一致，唯一的不同在于它==默认将特征值按从小到大的升序排列==；因此若要检视最大的特征值，你==**需要从返回向量的最末尾去查找**==。

```python
scipy.sparse.random.seed(7) #这句暂时不用管
eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(cov_matrix, k=55)
print(f'Ten largest eigenvalues: \n{eigenvals[-10:]}')

```
![[Pasted image 20261008172827.png]]
解释
    ![[Pasted image 20261004070206.png]]
    ![[Pasted image 20261004070213.png]]
    但是好像有点问题
    ![[Pasted image 20261004070341.png]] 这样才不报错
    ![[Pasted image 20261008182327.png]]

在上面的代码中，我们==固定了随机数种子（Random Seed），以确保每次代码重新运行时都能得到完全一致的特征向量。==这是因为对于任意一个特征向量，都存在两个模长为 1 的有效数学表达：它们落在同一条直线上，但指向恰好相反。例如下面这两个向量：

$$\begin{bmatrix}0.25 \\0.25 \\ -0.25 \\ 0.25 \end{bmatrix} \text{与 } \begin{bmatrix}-0.25 \\ -0.25 \\ 0.25 \\ -0.25 \end{bmatrix}.$$

在数学上这两种表示都是完全正确的，==而固定种子能够确保代码在评分时具有稳定的确定性。==

为了与大家习惯的 `np.linalg.eig` 排序逻辑保持统一，我们将对 `eigenvals` 和 `eigenvecs` 执行一次反向切片，让它们按照特征值从大到小的降序重新排列。

```python
eigenvals = eigenvals[::-1]
eigenvecs = eigenvecs[:,::-1]

print(f'Ten largest eigenvalues: \n{eigenvals[:10]}')

```
![[Pasted image 20261008172818.png]]
解释
    ![[Pasted image 20261004070708.png]]
    [[关于数组冒号的问题]]

至此，求出的每一个特征向量都代表着一个独立的主成分。对应最大特征值的特征向量就是第一主成分（First Principal Component），对应第二大特征值的特征向量则是第二主成分，依此类推。

非常有意思的是，每个主成分往往会在视觉上提炼出图像的关键特征或某种抽象模式（Patterns）。在接下来的代码块中，我们将前 16 个主成分分别可视化呈现出来：

```python
fig, ax = plt.subplots(4,4, figsize=(20,20))
for n in range(4):
    for k in range(4):
        ax[n,k].imshow(eigenvecs[:,n*4+k].reshape(height,width), cmap='gray')
        ax[n,k].set_title(f'component number {n*4+k+1}')
#怎么显示出来的
```
![[Pasted image 20261004070849.png]]
![[Pasted image 20261004070900.png]]

仔细观察这些主成分图像，你能看出它们分别捕捉到了猫咪面部的哪些视觉特征吗？

理解
    首先上面 输入矩阵是 55x4096
    那算出的特征向量 每个就是 1x4096  那不就是一副图片的像素数, 那就当然可以把他当做图像显示了
    ![[Pasted image 20261008174339.png]]
    ![[Pasted image 20261008174449.png]]

### 2.4 利用 PCA 转换中心化后的数据

现在你已经掌握了前 16 对特征值与特征向量，接下来就可以对原始数据进行投影变换，实现真正的空间降维。请记住，最初的猫咪数据散落在 4096 维的超高维空间中。假设你想把数据极致压缩到仅剩 2 个维度，那么只需将中心化后的数据矩阵与投影矩阵 $\boldsymbol{V}=\begin{bmatrix} v_1 & v_2 \end{bmatrix}$ 进行点积即可；该投影矩阵的列向量正是对应前两个最大特征值的主成分特征向量。

### 练习 5

在下一个单元中，你将定义一个专门执行 PCA 降维的通用函数。该函数接收数据矩阵、已经按特征值降序排好序的特征向量矩阵、以及希望保留的目标主成分数量 $k$。

```python
# 用于评分的函数单元
def perform_PCA(X, eigenvecs, k):
    """
    利用 PCA 算法实施降维映射
    输入:
        X (ndarray): 原始数据矩阵，尺寸为 (样本数) x (变量维度)
        eigenvecs (ndarray): 特征向量矩阵，每一列代表一个特征向量，且第 k 列对应第 k 大的特征值
        k (int): 期望保留的主成分数量
    返回:
        Xred: 降维后的新数据矩阵
    """
    
    ### START CODE HERE ###
    V = eigenvecs[:, :k]
    Xred = X @ V
    ### END CODE HERE ###
    return Xred

```
解释
    ![[Pasted image 20261004071537.png]]

运行这个函数，尝试将图像数据一口气降维至仅有两个主成分的分离空间：

```python
Xred2 = perform_PCA(X, eigenvecs,2)
print(f'Xred2 shape: {Xred2.shape}')

```
![[Pasted image 20261008174537.png]]

```python
# 运行测试用例检验你的代码
w4_unittest.test_check_PCA(perform_PCA)

```

### 2.5 分析二维空间下的降维效果

将数据压缩至仅仅两个主成分，最令人兴奋的好处就在于：我们可以把每一只猫咪的特征毫无保留地直接绘制在一张普通的二维平面散点图上！请记住，这个全新二维平面上的两个坐标轴，实际上是由两个主成分特征向量的方向所决定的原始像素变量的线性组合（Linear Combination）。

调用 `utils` 模块中的 `plot_reduced_data` 函数来可视化降维后的数据点。图中的每个蓝色圆点代表一张猫咪图片，点旁的数字标注了该图像在数据集中的原始索引序号。这样方便我们随后按图索骥，调出原图建立几何直觉。

```python
utils.plot_reduced_data(Xred2)

```
![[outputkdjk.png]]


从几何直觉上推断：如果两个点在二维投影图上靠得非常近，那么它们所对应的原始猫咪照片在现实中大概率也长得非常相似。
让我们来验证这个猜想。以散点图上方正中央位置紧挨着的第 15,2,16 号图像为例，画出它们的原始照片，看看它们是否真是一组“孪生猫”。

```python
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(imgs[15], cmap="gray")
ax[0].set_title("Image 15")
ax[1].imshow(imgs[2], cmap="gray")
ax[1].set_title("Image 2")
ax[2].imshow(imgs[16], cmap="gray")
ax[2].set_title("Image 16")
plt.suptitle("Similar cats")

```
![[Pasted image 20261008174858.png]]
瞧，这三只猫咪都有着白色的口鼻区域，且双眼周围都环绕着一圈深色毛发，长相确实极为相像！

反过来，我们再挑选三张在二维平面上相距甚远的图像样本——例如右侧中部的 18 号、顶部中央的 41 号以及左下方的 51 号，同样将它们绘制出来：

```python
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(imgs[1], cmap="gray")
ax[0].set_title("Image 1")
ax[1].imshow(imgs[7], cmap="gray")
ax[1].set_title("Image 7")
ax[2].imshow(imgs[12], cmap="gray")
ax[2].set_title("Image 12")
plt.suptitle("Different cats")

```
![[Pasted image 20261004072018.png]]

果不其然，这三只猫咪的外貌风格迥异：一只浑身通黑，一只通体雪白，而第三只则是黑白杂糅的花猫。

你也可以随意挑选其他任意几对数据点，亲自验证一下降维坐标上的远近与视觉风格差异是否一致。

### 2.6 利用特征向量重构图像

当我们使用 PCA 压缩图像时，由于只使用了极少数的特征变量来代表原始样本，不可避免地会损失部分细节信息。

这时一个核心问题油然而生：我们到底需要保留多少个主成分，才能高保真地重构出一张“足够好”的图像？当然，如何界定重构效果是否“足够好”，完全取决于下游具体的业务需求。

妙不可言的是，仅需一次简单的矩阵点积，我们就能把降维后的低维特征重新逆向映射回原始图像的高维空间中！这意味着我们可以从极度压缩的编码中重新还原、重构出原图，并直观评估不同成分数量下的图像失真度。

假设我们此前仅保留了 2 个特征向量，得到了压缩矩阵 $X_{red}$，即满足数学关系：$X_{red} = \mathrm{X}\underbrace{\left[v_1\  v_2\right]}_{\boldsymbol{V_2}}$。
为了==将图像解码还原回原本的像素空间，我们只需计算 $X_{red}$ 与转置矩阵 $\boldsymbol{V_2}^T$ 的点积即可==。如果希望保留更多的特征成分（比如 $k$ 个），只需将 $\boldsymbol{V_2}$ 相应替换为 $\boldsymbol{V_k} = \left[v_1\ v_2\ \ldots\ v_k\right]$ 即可。请注意，两边的维度必须严格对齐：如果你在压缩降维时只截取了前 $k$ 个成分，那么在逆变换重构时也只能使用这对应的前 $k$ 个特征向量，否则矩阵乘法在维度法则上将无法执行。

在接下来的代码块中，你将定义一个图像重构函数，它接收降维后的数据 $X_{red}$ 与特征向量矩阵，并返回还原重构后的图像矩阵。

```python
def reconstruct_image(Xred, eigenvecs):
    X_reconstructed = Xred.dot(eigenvecs[:,:Xred.shape[1]].T)
 # Xred.shape是55 x n   n就是降到几维 
    return X_reconstructed

```
理解
    ![[Pasted image 20261008180538.png]]
    Xred 55 x n (这个配发表对于每个模版(中的每个像素)是不变的)
    eigenvecs  4096 x 55
    `eigenvecs[:,:Xred.shape[1]]`  4096 x n
    T以后 n x 4096  (每个数值代表每个像素的模版)
    55 x 4096
    想一下矩阵相乘的过程就能大概理解了

为什么有损失 
    ![[Pasted image 20261008181945.png]]
    ![[Pasted image 20261008182041.png]]
    (前面我们规定了求 55个特征向量 所以这边最多55个模版, 最多按维数应该是4096个模版)

重要
    ![[Pasted image 20261008182903.png]]
    ![[Pasted image 20261008182922.png]]
    而且这两个都是独立操作 彼此不是你操作  所以不能理解为正过来反过来算!!!!!  都应该当做独立的算法来记 

让我们来比对一下，在选用不同数量的主成分时，重构出的图像质量到底有何天壤之别：

```python
Xred1 = perform_PCA(X, eigenvecs,1) # reduce dimensions to 1 component
Xred5 = perform_PCA(X, eigenvecs, 5) # reduce dimensions to 5 components
Xred10 = perform_PCA(X, eigenvecs, 10) # reduce dimensions to 10 components
Xred20 = perform_PCA(X, eigenvecs, 20) # reduce dimensions to 20 components
Xred30 = perform_PCA(X, eigenvecs, 30) # reduce dimensions to 30 components
Xrec1 = reconstruct_image(Xred1, eigenvecs) # reconstruct image from 1 component
Xrec5 = reconstruct_image(Xred5, eigenvecs) # reconstruct image from 5 components
Xrec10 = reconstruct_image(Xred10, eigenvecs) # reconstruct image from 10 components
Xrec20 = reconstruct_image(Xred20, eigenvecs) # reconstruct image from 20 components
Xrec30 = reconstruct_image(Xred30, eigenvecs) # reconstruct image from 30 components

fig, ax = plt.subplots(2,3, figsize=(22,15))
ax[0,0].imshow(imgs[1], cmap='gray')
ax[0,0].set_title('original', size=20)
ax[0,1].imshow(Xrec1[1].reshape(height,width), cmap='gray')
ax[0,1].set_title('reconstructed from 1 components', size=20)
ax[0,2].imshow(Xrec5[1].reshape(height,width), cmap='gray')
ax[0,2].set_title('reconstructed from 5 components', size=20)
ax[1,0].imshow(Xrec10[1].reshape(height,width), cmap='gray')
ax[1,0].set_title('reconstructed from 10 components', size=20)
ax[1,1].imshow(Xrec20[1].reshape(height,width), cmap='gray')
ax[1,1].set_title('reconstructed from 20 components', size=20)
ax[1,2].imshow(Xrec30[1].reshape(height,width), cmap='gray')
ax[1,2].set_title('reconstructed from 30 components', size=20)

```
![[outputff 1.png]]

肉眼可见，==随着纳入的主成分数量逐步攀升，重构出的图像变得越来越清晰、细腻，并迅速向原图逼近！==令人惊叹的是，即便是仅仅使用 1 个成分的极度压缩状态下，图像依然勾勒出了猫咪眼睛和鼻子的核心轮廓。

那么，如果我们把所有 55 个非零特征向量全部用上，又会呈现怎样的效果呢？不妨多试几组不同数量的主成分参数，亲自观察图像细节的变化过程。

### 2.7 解释方差

在决定到底该保留多少个主成分来进行降维时，一个极具指导意义的核心统计指标就是**解释方差（Explained Variance）**。

解释方差衡量的是==数据集内部的整体波动==（信息量）中，有多少比例能够被各个特定主成分（特征向量）所囊括和解释。简而言之，它告诉我们==每个成分究竟“消化”了整个系统多少的总体方差。==

在 PCA 理论中，由最大特征值对应的第一主成分拥有最高的方差解释能力。正如理论课所述，PCA 的核心思想就是沿着方差最大、即数据离散度最显著的方向去正交投影，从而在降维的同时最大程度保全原始信息。

在工程实践中，==**某个主成分所拥有的解释方差比率，正好等于其对应的特征值除以所有特征值之和。**==以本次实验为例，若要计算第一主成分的解释方差占比，只需计算公式 $\frac{\lambda_1}{\sum_{i=1}^{55} \lambda_i}$ 即可。

下面，让我们把全部 16 个主成分（特征向量）各自的解释方差比重绘制成曲线图。请不要介意我们前面仅仅算出了 16 对特征值与特征向量，正如先前所述，协方差矩阵其余所有的特征值全部在数学上严格为零，对解释方差没有任何额外贡献。

```python
explained_variance = eigenvals/sum(eigenvals)
plt.plot(np.arange(1,56), explained_variance)

```
![[Pasted image 20261008175312.png]]

正如曲线上所呈现的，解释方差比重衰减得极快，在超过第 20 个主成分之后，后续成分所蕴含的方差信息已经变得微乎其微。

在工业界，确定主成分==**截断数量的经典准则，是挑选能够累计覆盖极高方差比例（例如 95%）的最小成分组合。**==

为了更加直观地评估信息保留进度，我们可以绘制==**累积解释方差**==（Cumulative Explained Variance）曲线。借助 NumPy 的💛 `np.cumsum` 函数可以一键完成累加计算：

```python
explained_cum_variance = np.cumsum(explained_variance)
plt.plot(np.arange(1,56), explained_cum_variance)
plt.axhline(y=0.95, color='r')

```
![[Pasted image 20261008175331.png]]

图中的红色横虚线标出了 95% 的信息保留基准线。这意味着，如果你希望让压缩后的模型保留原始图像 95% 的核心信息与方差特征，你总共仅需保留 35 个主成分即可。

接下来，让我们见证奇迹：看看仅仅凭借这 35 个主成分重构出的猫咪图像，究竟能达到怎样的保真度！

```python
Xred35 = perform_PCA(X, eigenvecs, 35) # 将数据降维至 35 个成分
Xrec35 = reconstruct_image(Xred35, eigenvecs) # 基于 35 个成分重构原图

fig, ax = plt.subplots(4,2, figsize=(15,28))
ax[0,0].imshow(imgs[0], cmap='gray')
ax[0,0].set_title('original', size=20)
ax[0,1].imshow(Xrec35[0].reshape(height, width), cmap='gray')
ax[0,1].set_title('Reconstructed', size=20)

ax[1,0].imshow(imgs[15], cmap='gray')
ax[1,0].set_title('original', size=20)
ax[1,1].imshow(Xrec35[15].reshape(height, width), cmap='gray')
ax[1,1].set_title('Reconstructed', size=20)

ax[2,0].imshow(imgs[32], cmap='gray')
ax[2,0].set_title('original', size=20)
ax[2,1].imshow(Xrec35[32].reshape(height, width), cmap='gray')
ax[2,1].set_title('Reconstructed', size=20)

ax[3,0].imshow(imgs[54], cmap='gray')
ax[3,0].set_title('original', size=20)
ax[3,1].imshow(Xrec35[54].reshape(height, width), cmap='gray')
ax[3,1].set_title('Reconstructed', size=20)


```
![[outputjhh.png]]

绝大多数重构出来的图像在视觉上都极为出色，而我们仅仅用 35 个数字就完整保留了原本多达 4096 个像素变量的视觉神韵，节省了海量的内存与存储开销！

现在你已经深刻理解了解释方差的工作机制，不妨在课后随心设定不同的方差阈值，观察它们对图像重构清晰度的直接影响。你也可以测试更多未曾展出的样本图像，检验模型的泛化能力。

显而易见，主成分分析（PCA）是一把兼具数学优雅与工业实用性的降维利器。在本实验中，你领略了它在数字图像压缩领域的强大威力，但在未来的机器学习工程中，同样的数学法则能够无缝迁移到任何复杂的表格数据与高维特征矩阵上。

祝贺你！你已经圆满完成了本周的全部编程实战任务。