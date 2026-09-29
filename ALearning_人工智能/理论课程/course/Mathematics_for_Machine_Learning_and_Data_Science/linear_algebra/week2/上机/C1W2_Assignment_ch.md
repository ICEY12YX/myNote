___
# 编程作业 - 高斯消元法 (Gaussian Elimination)

欢迎来到关于高斯消元法 (Gaussian Elimination) 的编程作业！在这项作业中，你将亲自动手实现高斯消元法这一用于求解线性方程组 (systems of linear equations) 的基石算法。

线性代数 (Linear Algebra) 是机器学习 (Machine Learning) 的基石，为数以百计的算法提供了底层数学支撑。高斯消元法虽然不是当下最前沿的高阶算法，却是一项经典且不可或缺的核心技术。它不仅能帮助你直观洞察线性代数的核心原理，更为深入理解更复杂的数值计算方法奠定了坚实基础。

## 为什么这项内容至关重要？

* **夯实核心基础**：加深对关键线性代数概念的理解。
* **强化编程实战**：通过从零实现经典数学算法，提升代码实操能力。
* **领略历史意义**：高斯消元法虽非当今最前沿的计算手段，但在数学与计算科学史上占据着里程碑式的地位，是理解现代线性代数技术演进脉络的绝佳起点。

## 重要注意事项

请 **切勿** 删除任何练习代码单元格 (cells)，也不要在其他单元格中编写你的答案。**请务必将解答写在指定的原始单元格中**，否则自动评分系统将无法正常判定。
___


___
## 1 - 简介

### 1.1 如何完成本作业

这是“机器学习与数据科学的数学基础”专项课程的第一份编程作业！让我们先快速熟悉一下作业流程。

本次作业包含 $3$ 个计分函数。每个计分函数中都会有部分代码被替换为 `None`。你需要将这些 `None` 替换为正确的计算逻辑或数值。例如，在第一个计分函数中包含这样一行代码：

```Python
pivot_candidate = M[None, None]

```

这意味着你需要填入正确的行索引 (第一个 None) 和列索引 (第二个 None)。别担心，函数中的每行代码都配有详尽的注释，引导你顺利完成！

在每个计分函数之后，都配有专门的测试代码。它会通过一些简单快速的测试用例验证你的实现思路是否正确。**请注意，这些测试仅覆盖基础逻辑，即便通过了本地单元测试，在最终提交代码时仍有可能无法得分。** 这是因为正式评测时会运行更严苛复杂的测试集。不过，系统在任何情况下都会给出反馈，帮助你高效排查和调试代码。

### 1.2 高斯消元算法

高斯消元法提供了一种系统化的解题路径：它通过**初等行变换**将增广矩阵( augmented matrix )化为行阶梯形 (row-echelon form)，进而精确解出各个未知数。该算法主要包含以下几个核心步骤：

**提示**：

* 为降低实现复杂度，本次构建的算法==仅适用于 **非奇异** (non-singular) 方程组==，即存在唯一解的线性方程组。
* 别忘了，==判断矩阵是否非奇异，最直接的方法就是计算它的行列式 (determinant) 是否为零==。

### 步骤 1：构造增广矩阵 (Augmented Matrix)

考虑如下线性方程组：

$$\begin{align*} 2x_1 + 3x_2 + 5x_3&= 12 \\ -3x_1 - 2x_2 + 4x_3 &= -2 \\ x_1 + x_2 - 2x_3  &= 8 \\ \end{align*}$$

我们构造形如 $[A \vert{} B]$ 的增广矩阵，其中 $A$ 为系数矩阵 (coefficient matrix)，$B$ 为常数项构成的列向量 (column vector)：

$$A = \begin{bmatrix} \phantom{-}2 & \phantom{-}3 & \phantom{-}5  \\ -3 & -2 & \phantom-4 \\ \phantom{-}1 & \phantom{-}1 & -2  \\ \end{bmatrix}$$

$$B = \begin{bmatrix} \phantom-12 \\ -2 \\ \phantom-8  \end{bmatrix}$$

于是，增广矩阵 $[A \vert{} B]$ 可以直观写为：

$$\begin{bmatrix} \phantom{-}2 & \phantom{-}3 & \phantom{-}5 & \vert{} & \phantom{-}12 \\ -3 & -2 & \phantom-4 & \vert{} & -2 \\ \phantom{-}1 & \phantom{-}1 & -2 & \vert{} & \phantom{-}8 \\ \end{bmatrix}$$

注：在本次作业中，矩阵 $A$ **始终为方阵** (square matrix)，即对应 $n$ 个方程包含 $n$ 个未知数的情形。

### 步骤 2：将矩阵化为行阶梯形 (Row Echelon Form)

利用初等行变换 (elementary row operations) 将增广矩阵化为行阶梯形。核心目标是通过消元，使主对角线下方的一切元素全部归零：

* **行对换 (Row Switching)：** 调换两行的相对位置，把当前最靠左的非零元素移至上方。
* **行倍乘 (Row Scaling)：** 用一个非零标量乘以某一行中的所有元素。
* **行倍加 (Row Replacement)：** 将某一行替换为“自身加上另一行的若干倍”。

### 步骤 3：回代求解 (Back Substitution)

当矩阵化为规范的阶梯形后，我们便能==从最后一行==最简单的单未知数方程入手，==由底向上逐层“回代”==，依次解出所有变量。

记住，这一步的精髓在于化繁为简，让最终的求值过程水到渠成！

### 步骤 4：组装高斯消元算法

将上述各步骤拆解实现的独立函数组合起来，封装为一个完整高效的主计算函数。

## 2 - 导入必要库

运行下面的代码块将导入本作业所需的全部依赖库。请不要在此处擅自添加或删除任何代码。

```python
import numpy as np

```

```python
import w2_unittest

```

## 3 - 辅助函数

本节将介绍三个为你扫清障碍的辅助函数。它们已经提前封装完毕，无需你重新编写底层实现。不过，认真研读其调用逻辑对于正确使用它们至关重要。

**提示：在 Python 中，索引下标从 $0$ 开始计数而非 $1$。因此，对于一个具有 $n$ 行的矩阵，行索引依次为 $0, 1, 2, \ldots, n-1$。**

### 3.1 - 行交换函数

该函数接收一个 NumPy 数组 (numpy array) 以及两个需要互换位置的行索引。它 **绝不会直接修改原始矩阵**，而是以安全纯函数的方式返回一个全新的矩阵副本。
(就是这里定义好了, 不在别的文件里,就在这篇笔记里, 运行一次这个cell,下面就能调用这个函数了)
```python
def swap_rows(M, row_index_1, row_index_2):
    """
    交换给定矩阵中的两行。

    参数:
    - matrix (numpy.array): 需要执行行对换的目标矩阵。
    - row_index_1 (int): 待交换的第一行索引。
    - row_index_2 (int): 待交换的第二行索引。
    """

    # 深度拷贝矩阵 M，确保行变换不会污染原始数据
    M = M.copy()
    # 交换行索引对应的内容
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M

```
解释
    ![[Pasted image 20260929224816.png]]
    ![[Pasted image 20260929224941.png]]
    ![[Pasted image 20260929225038.png]]

让我们通过简单的例子来感受其实际效果。观察以下矩阵 $M$：

```python
M = np.array([
[1, 3, 6],
[0, -5, 2],
[-4, 5, 8]
])
print(M)

```
![[Pasted image 20260929225127.png]]


对调第 $0$ 行与第 $2$ 行：

```python
M_swapped = swap_rows(M, 0, 2)
print(M_swapped)

```
![[Pasted image 20260929225118.png]]

### 3.2 - 从指定位置开始查找某列的首个非零元

在行消元过程中，如果我们不幸在==主对角线位置撞见 $0$==，这个函数就成了救场法宝。它==能向下探查是否存在非零元素，从而指导程序适时进行行对换==。我们来看一个方阵内部的典型消元切片：

假定在消元化为行阶梯形的过程中，前两行已妥善处理，但在第三行主对角线处出现了一个红色的“零主元”。此时，算法必须 **仅在主元下方的剩余行中** 寻找非零元来进行行对换：

$$\begin{bmatrix} 6 & 4 & 8 & 1 \\ 0 & 8 & 6 & 4 \\ \color{darkred}0 & \color{darkred}0 & \color{red}0 & \color{darkred}3 \\ 0 & 0 & 5 & 9  \end{bmatrix}$$

将索引为 2 和 3 的两行进行对换 (别忘了索引从 0 开始计！)，矩阵即刻变身规整形态：

$$\begin{bmatrix} 6 & 4 & 8 & 1 \\ 0 & 8 & 6 & 4 \\ 0 & 0 & 5 & 9  \\ 0 & 0 & 0 & 3  \end{bmatrix}$$

如此一来，矩阵便顺畅达成了行阶梯形的目标。

```python
def get_index_first_non_zero_value_from_column(M, column, starting_row):
    """
    检索指定矩阵特定列中，自起始行往下的首个非零元素的行索引。

    参数:
    - matrix (numpy.array): 待检索的目标矩阵。
    - column (int): 目标列的索引。
    - starting_row (int): 检索的起始行索引。

    返回:
    int: 从起始行算起，该列首个非零元素所在的行索引；
            若全为零则返回 -1。
    """
    # 截取自指定起始行往下的列切片
    # 因为column后面没有冒号所以就是切一列
    column_array = M[starting_row:,column]
    for i, val in enumerate(column_array):
        # 遍历该列切片中的每个数值。
        # 针对浮点数零值判定，请始终使用 np.isclose，切忌直接使用 "val == 0"。
        if not np.isclose(val, 0, atol = 1e-5):
            # 一旦寻得非零元，把这个非零元的行序号还原为全局矩阵行索引并返回
            index = i + starting_row
            return index
    # 若下方不存在任何非零元，则返回 -1
    return -1

```
解释
    ![[Pasted image 20260929225543.png]]
    ![[Pasted image 20260929225617.png]]

让我们通过实操来检验该函数的表现。定义如下矩阵：

```python
N = np.array([
[0, 5, -3 ,6 ,8],
[0, 6, 3, 8, 1],
[0, 0, 0, 0, 0],
[0, 0, 0 ,0 ,7],
[0, 2, 1, 0, 4]
]
)
print(N)

```
![[Pasted image 20260929225948.png]]


若从第 0 行开始向下检索第 0 列的非零元素，该函数将如期返回 -1：

```python
print(get_index_first_non_zero_value_from_column(N, column = 0, starting_row = 0))
# 返回-1
```

若从索引为 2 的行开始向下检索最后一列的首个非零元，它将精准返回 3 (数值 7 所在的行索引)：

```python
print(get_index_first_non_zero_value_from_column(N, column = -1, starting_row = 2))
# -1指倒数第一列
# 返回3 
# (所以说调用这个函数后,要2r和3r互换 然后还要调用一次这个函数,找到5r,再调换)
```

### 3.3 - 查找任意行的首个非零元

该函数旨在协助定位指定行中的主元位置。它会自左向右扫描并返回该行首个非零元素的列索引；若该行整行全为零，则返回 -1。

```python
def get_index_first_non_zero_value_from_row(M, row, augmented = False):
    """
    查找指定矩阵某一行中首个非零元素的列索引。

    参数:
    - matrix (numpy.array): 待扫描的目标矩阵。
    - row (int): 待扫描的目标行索引。
    - augmented (bool): 若处理的是增广矩阵，请将其置为 True，
                        算法将自动忽略末列常数项。
                        默认为false

    返回:
    int: 该行首个非零元素的列索引；若整行全为零则返回 -1。
    """

    # 深度拷贝以防破坏原始矩阵
    M = M.copy()


    # 若属于增广矩阵，则忽略最右侧的常数项列
    if augmented == True:
        # 分离纯系数矩阵 (剥离末尾常数列)
        # 不到最后一列,即去掉了最后一列
        M = M[:,:-1]
        
    # 获取目标行向量
    row_array = M[row]
    for i, val in enumerate(row_array):
        # 命中非零元即刻返回列索引；否则返回 -1
        if not np.isclose(val, 0, atol = 1e-5):
            return i
    return -1

```

让我们以刚才的矩阵 $N$ 为例继续演练：
(还是刚刚那个矩阵)
```python
print(N)

```
![[Pasted image 20260929233518.png]]

若不显式传递 `augmented` 参数，函数默认按照普通矩阵处理。

检索第 $2$ 行的首个非零元将得到 -1；而在第 $3$ 行中，返回的列索引则为 $4$ (即数值 $7$ 对应的列位置)：

```python
print(f'第 2 行输出: {get_index_first_non_zero_value_from_row(N, 2)}')
print(f'第 3 行输出: {get_index_first_non_zero_value_from_row(N, 3)}')

```
![[Pasted image 20260929233619.png]]

现在，我们传入 `augmented = True` 参数。这会指示算法将 $N$ 视作增广矩阵，因此最后一列常数项将从检索范围中被剥除。此时第 3 行的输出结果将截然不同：由于排除了最后一列，系数矩阵对应行已空无一物，故同样返回 `-1`：

```python
print(f'第 3 行输出: {get_index_first_non_zero_value_from_row(N, 3, augmented = True)}')

```
![[Pasted image 20260929233644.png]]

### 3.4 - 构造增广矩阵

该函数用于将描述 $n$ 元方程组的 $n \times n$ 阶系数方阵与表示常数项的 $n \times 1$ 阶列向量水平拼接，组合成标准的增广矩阵并输出。
(其实自己调用hstack也可以啊明明)
```python
def augmented_matrix(A, B):
    """
    通过将矩阵 A 与矩阵 B 水平拼接来构建增广矩阵。

    参数:
    - A (numpy.array): 系数矩阵。
    - B (numpy.array): 常数项列矩阵。

    返回:
    - numpy.array: A 与 B 水平拼接得到的增广矩阵。
    """
    augmented_M = np.hstack((A,B))
    return augmented_M

```

```python
A = np.array([[1,2,3], [3,4,5], [4,5,6]])
B = np.array([[1], [5], [7]])

print(augmented_matrix(A,B))

```
![[Pasted image 20260929233755.png]]

必须B定义成那样, 否则
![[Pasted image 20260929233846.png]]

要么这样
![[Pasted image 20260929233922.png]]

## 4 - 行阶梯形矩阵

### 4.1 - 行阶梯形矩阵概念

正如理论课程中所探讨的，一个标准的行阶梯形矩阵需严格满足以下几何与代数特征：

* 全零行必须整齐沉底，集中排列在矩阵最下方。
* 每一个非零行从左往右数第一个非零元素 (称为 **主元 (pivot)**)，其所在的列必须严格位于上一行主元所在列的右侧。由此推论，同一列中主元下方的一切元素必须全部归零。

**提示：**

* ==本作业构建的算法专注于求解非奇异方程组，这意味着系数矩阵的行列式绝不为 $0$。==这带来了一个极其优美的性质：==**在化简后的行阶梯形矩阵中，所有主元都将规整地落在主对角线上**==。这一特性大大简化了后续的编程与求解逻辑。

规范的阶梯结构如同优雅的台阶，为高斯消元的后半程搭好了稳固的跳板。

**处于行阶梯形** 的矩阵示例：

$$M = \begin{bmatrix} 7 & 2 & 3 \\ 0 & 9 & 4 \\ 0 & 0 & 3 \\ \end{bmatrix}$$

**不属于行阶梯形** 的反面矩阵示例：

$$A = \begin{bmatrix} 1 & 2 & 2 \\ 0 & 5 & 3 \\ 1 & 0 & 8 \\ \end{bmatrix}$$

$$B =  \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 4 \\ 0 & 0 & 7 \\ \end{bmatrix}$$

矩阵 $A$ 未能达标，是因为其首个主元 (位于第 0 行) 下方赫然存在非零元素；而矩阵 $B$ 同样破戒，因为其第二个主元 (第 1 行中的数值 4) 正下方依然潜藏着未被消去的非零元 7。

### 4.2 - 实例推导详解

在动手编码之前，我们先以讲义中的经典微型案例推演一遍完整的算法流转逻辑。若你对行消元过程早已了然于胸，大可跳过本节直接挑战练习题。

设想抽象矩阵 $M$ 呈现如下状态：  
$$
M = 
\begin{bmatrix} 
* & * & * & \\
0 & \text{pivot} & * \\
0 & \text{value} & * 
\end{bmatrix}
$$
此处星号 ( * ) 代表任意数值。为将最后一行 (第 $2$ 行) 主元下方的数值彻底归零，我们需要实施两步精细操作：  - 首先，让第 1 行乘以主元的倒数，将该行主元归一化为 1：  $$
\text{行 1} \rightarrow \frac{1}{\text{主元 (pivot)}} \cdot \text{行 1} $$

由此得到主元归一化后的新矩阵：
$$
M = 
\begin{bmatrix} 
* & * & * & \\
0 & 1 & * \\
0 & \text{value} & * 
\end{bmatrix}
$$

接着，执行行倍减消元，将第 1 行主元正下方的数值彻底消去：

$$\text{行 2} \rightarrow \text{行 2} - \text{待消值 (value)}\cdot \text{行 1}$$

伴随这一步初等变换，矩阵顺利转化为：
$$
M = 
\begin{bmatrix} 
* & * & * & \\
0 & 1 & * \\
0 & 0 & * 
\end{bmatrix}
$$


**切记：虽然我们的目标是将系数方阵 \(A\) 转化为行阶梯形，但施加在矩阵行上的每一步初等变换，都必须不偏不倚地同步作用在右侧的常数增广部分！唯有如此，整个方程组的解空间才能得以原汁原味地精准保留！** 

我们以具体的三元一次方程组为例进行实战推导：


$$\begin{align*} 2x_2 + x_3 &= 3 \\ x_1 + x_2 +x_3 &= 6 \\ x_1 + 2x_2 + 1x_3 &= 12 \end{align*}$$

其对应的系数方阵 $A$ 提取如下：

$$A =  \begin{bmatrix}  0 & 2 & 1 & \\ 1 & 1 & 1 & \\ 1 & 2 & 1 &  \end{bmatrix}$$

常数列向量 (即 $n \times 1$ 矩阵) 为：

$$B =  \begin{bmatrix}  3\\ 6\\ 12 \end{bmatrix}$$

将 $A$ 与 $B$ 并置，构成完整的增广矩阵 $M$：

$$M =  \begin{bmatrix}  0 & 2 & 1 & \vert{} & 3 \\ 1 & 1 & 1 & \vert{} & 6 \\ 1 & 2 & 1 & \vert{} & 12  \end{bmatrix}$$

**步骤 1：**

从第 $0$ 行启程：主元的首选候选人永远是主对角线上的元素。记第 $0$ 行为 $R_0$：

$$R_0= \begin{bmatrix} 0 & 2 & 1 & \vert{} & 3 \end{bmatrix}$$

对角线上的元素对应 $M[0,0]$ (即第一行第一列的交点)。在代码中，整行可以通过 $M[0]$ 直接索引，即 $M[0] = R_0$。

第一步通常是 **乘以主元的倒数进行归一化**。==然而此时对角线元素偏偏为 $0$！==由于零无法求倒数，我们必须==果断实施行对换==。注意到第 1 行 ($R_1$) 在同列拥有非零元 1，我们立即将第 $0$ 行与第 $1$ 行对调：

$$R_0 \rightarrow R_1$$

$$R_1 \rightarrow R_0$$

互换之后，增广矩阵刷新为：

$$M =  \begin{bmatrix}  1 & 1 & 1 & \vert{} & 6 \\ 0 & 2 & 1 & \vert{} & 3 \\ 1 & 2 & 1 & \vert{} & 12  \end{bmatrix}$$

此时新主元天然为 $1$，无需再做缩放。依照==消元==公式：

$$R_1 \rightarrow  R_1 - 0 \cdot R_0 = R_1$$

第二行本就为 0，安然无恙。转向第三行 ($R_2$)，其位于 $R_0$ 主元下方的元素 $M[2,0]$ 为 $1$。我们执行行倍减：

$$R_2 = R_2 - 1 \cdot R_0 = \begin{bmatrix} 0 & 1 & 0 & \vert{} & 6  \end{bmatrix}$$

增广矩阵演变为：

$$M =  \begin{bmatrix}  1 & 1 & 1 & \vert{} & 6 \\ 0 & 2 & 1 & \vert{} & 3 \\ 0 & 1 & 0 & \vert{} & 6 \end{bmatrix}$$

==移步第二行== ($R_1$)，其对角线主元为非零的 $2$。全行==缩放== $\frac{1}{2}$ 倍：

$$R_1 = \frac{1}{2}R_1$$

矩阵即更新为：

$$M =  \begin{bmatrix}  1 & 1 & 1 & \vert{} & 6 \\ 0 & 1 & \frac{1}{2} & \vert{} & \frac{3}{2} \\ 0 & 1 & 0 & \vert{} & 6 \end{bmatrix}$$

此时下方仅剩一行待==消元==。$R_1$ 主元正下方的元素为 $M[2,1] = 1$。行倍减再次登场：

$$R_2 = R_2 - 1 \cdot R_1 = \begin{bmatrix} \phantom{-}0 & \phantom{-}0 & -\frac{1}{2} & \vert{} & \phantom{-}\frac{9}{2} \end{bmatrix}$$

增广矩阵进一步简化为：

$$M =  \begin{bmatrix}  \phantom{-}1 & \phantom{-}1 & \phantom{-}1 & \vert{} & \phantom{-}6 \\ \phantom{-}0 & \phantom{-}1 & \phantom{-}\frac{1}{2} & \vert{} & \phantom{-}\frac{3}{2} \\ \phantom{-}0 & \phantom{-}0 & -\frac{1}{2} & \vert{} & \phantom{-}\frac{9}{2}  \end{bmatrix}$$

最后，对尾行施加归一化==缩放==：

$$R_2 = -2 \cdot R_2$$

由此斩获终极行阶梯形态：

$$M =  \begin{bmatrix}  \phantom{-}1 & \phantom{-}1 & \phantom{-}1 & \vert{} & \phantom{-}6 \\ \phantom{-}0 & \phantom{-}1 & \phantom{-}\frac{1}{2} & \vert{} & \phantom{-}\frac{3}{2} \\ \phantom{-}0 & \phantom{-}0 & \phantom{-}1 & \vert{} & -9  \end{bmatrix}$$

至此，矩阵已完美化为各主元皆为 1 的规范行阶梯形。

万事俱备！接下来就请你在练习题中大显身手，亲手编码实现这一消元算法。

### 练习 1

本练习要求你实现高斯消元算法的前半程——将任意线性系统化为行阶梯形。核心思路如前所述：逐行沿主对角线向下探查，若对角线元素为 $0$，则果断在下方列中寻觅非零元并执行行对换。

```python
# GRADED FUNCTION: row_echelon_form

def row_echelon_form(A, B):
    """
    运用初等行变换将由系数矩阵和常数项构成的线性方程组转化为行阶梯形矩阵。

    参数:
    - A (numpy.array): 系数构成的输入方阵。
    - B (numpy.array): 常数项构成的输入列矩阵。

    返回:
    numpy.array: 主元已全部归一化为 1 的行阶梯形增广矩阵。
    """
    
    # 在开始计算前，先检验系数矩阵 A 的行列式是否非零。
    # 这里使用 numpy 的 np.linalg 子模块进行行列式计算。

    det_A = np.linalg.det(A)

    # 若行列式为零，直接判定为奇异方程组
    if np.isclose(det_A, 0) == True:
        return 'Singular system'

    # 对输入矩阵进行深拷贝，避免对原始数据造成就地修改
    A = A.copy()
    B = B.copy()


    # 统一转换为 float64 浮点类型，彻底规避整除截断误差
    A = A.astype('float64')
    B = B.astype('float64')

    # 获取系数矩阵的行数
    num_rows = len(A) 

    ### START CODE HERE ###

    # 将矩阵 A 和 B 拼接为增广矩阵 M
    #M = augmented_matrix(None,None)
    M = augmented_matrix(A,B)
    
    # 逐行遍历迭代
    for row in range(num_rows):

        # 主元的初始候选值永远位于主对角线上。
        # 矩阵对角线元素的行列索引完全一致。
        # 在 NumPy 中可通过 M[row, column] 访问元素。此时 column 应设为 None
        #pivot_candidate = M[None, None]
        pivot_candidate = M[row, 0]

        # 若候选主元为 0，则无法直接充当本行主元。
        # 第一步应顺着该列向下探查，寻找是否存在可用的非零元。
        # 在比较浮点数时，调用 np.isclose 是严谨的编程实践。
        if np.isclose(pivot_candidate, 0) == True: 
            # 查找候选主元下方首个非零元素的行索引
            first_non_zero_value_below_pivot_candidate = get_index_first_non_zero_value_from_column(M, row, row) 

            # 执行行对换
            M = swap_rows(M, row, first_non_zero_value_below_pivot_candidate) 

            # 获取对换后顺利入驻主对角线的新主元
            pivot = M[row,row] 
        
        # 若候选主元本身非零，则它就是合规主元
        else:
            pivot = pivot_candidate 
        
        # 接下来对当前行下方的所有行展开逐层行消元
            
        # 全行除以主元，使其主元规整归一为 1。应用公式：当前行 -> (1/主元) * 当前行
        # 当前行可以通过切片 M[row] 直接提取
        # M[row] = None * M[row]
        M[row] = 1/pivot * M[row]

        # 对当前行以下的所有行依次执行行消元
        for j in range(row + 1, num_rows): 
            # 提取待消行中位于主元正下方的元素值。
            # 由于我们限定处理非奇异矩阵，主元始终驻留在主对角线上。
            # 因此，第 j 行中位于主元下方的元素，其列索引必定与主元的列索引一致。
            # value_below_pivot = M[None, None]
            value_below_pivot = M[j,row]
            
            # 依照行消元公式实施行倍减：
            # 待消行 -> 待消行 - 待消值 * 主元所在行
            # M[j] = M[None] - value_below_pivot * M[None]
            M[j] = M[j] - value_below_pivot*M[row]
            
    ### END CODE HERE ###

    return M
            

```
解释
    ![[Pasted image 20260929235219.png]]

```python
A = np.array([[1,2,3],[0,1,0], [0,0,5]])
B = np.array([[1], [2], [4]])
row_echelon_form(A,B)

```
![[Pasted image 20260930001245.png]]

```python
w2_unittest.test_row_echelon_form(row_echelon_form)

```

## 5 - 回代求解

算法的收官之作是回代求解 (back substitution)，这是从简化结构中高效提取未知数真解的关键一跃。与消元过程相反，回代是由矩阵最底层向上回溯的逆向流动。我们继续借助初等行变换，把主元上方的元素也悉数消减为零，最终将矩阵彻底转化为 **简化行阶梯形** (reduced row echelon form)。这期间反复调用的核心变换法则为：

$$\text{上方目标行} \rightarrow \text{上方目标行} - \text{待消值} \cdot \text{主元行}$$

在该式中，$\text{待消值}$ 即为主元上方待清零的系数数值。我们以下面的行阶梯形矩阵为例，见证逆向回代的神奇过程：

$$M =  \begin{bmatrix}  \phantom{-}1 & -1 & \phantom{-}\frac{1}{2} & \vert{} & \phantom{-}\frac{1}{2} \\ \phantom{-}0 & \phantom{-}1 & \phantom{-}1 & \vert{} & -1 \\ \phantom{-}0 & \phantom{-}0 & \phantom{-}1 & \vert{} & -1  \end{bmatrix}$$

自底向上回溯消元：

* 以 $R_2$ 为基准主元行：
* $R_1 = R_1 - 1 \cdot R_2 = \begin{bmatrix} 0 & 1 & 0 & \vert{} & 0 \end{bmatrix}$
* $R_0 = R_0 - \frac{1}{2} \cdot R_2 = \begin{bmatrix} 1 & -1& 0 & \vert{} & 1 \end{bmatrix}$


由此更新得到中间形态：

$$M =  \begin{bmatrix}  \phantom{-}1 & -1 & \phantom{-}0 & \vert{} & \phantom{-}1  \\ \phantom{-}0 & \phantom{-}1 & \phantom{-}0 & \vert{} & \phantom{-}0 \\ \phantom{-}0 & \phantom{-}0 & \phantom{-}1 & \vert{} & -1  \end{bmatrix}$$

继续向上移步至 $R_1$：

* 以 $R_1$ 为基准主元行：
* * $R_0 = R_0 - \left(-1 \cdot R_1 \right) = \begin{bmatrix} 1 & 0 & 0 & \vert{} & 1 \end{bmatrix}$



最终斩获的简化行阶梯形矩阵为：

$$M =  \begin{bmatrix}  \phantom{-}1 & \phantom{-}0 & \phantom{-}0 & \vert{} & \phantom{-}1  \\ \phantom{-}0 & \phantom{-}1 & \phantom{-}0 & \vert{} & \phantom{-}0 \\ \phantom{-}0 & \phantom{-}0 & \phantom{-}1 & \vert{} & -1 \end{bmatrix}$$

发现了吗？经过这一通自底向上的利落回代，右侧最后一列的数值就变成了我们梦寐以求的最终解！

$$x_0 = 1 \\ x_1 =0\\ x_2 = -1$$

### 练习 2

在本练习中，你将亲手编写一个函数，对一个 **拥有唯一解且主元已全归一化的行阶梯形增广矩阵** 执行回代求解。

```python
# GRADED FUNCTION: back_substitution

def back_substitution(M):
    """
    对已处于行阶梯形且存在唯一解的增广矩阵执行回代求解，提取线性方程组的解向量。

    参数:
    - M (numpy.array): 主元为 1 的行阶梯形增广矩阵 (维度为 n x n+1)。

    返回:
    numpy.array: 方程组的解向量。
    """
    
    # 建立矩阵深拷贝，避免对传入对象造成就地篡改
    M = M.copy()

    # 读取系数方阵的行数 (同时也是变量的维度)
    num_rows = M.shape[0]

    ### START CODE HERE ####
    
    # 由底向上逆序遍历每一行
    for row in reversed(range(num_rows)): 
        substitution_row = M[row]

        # 检索当前回代基准行的首个非零元索引。记得给 augmented 参数传入恰当的值
        index = M[row][row]

        # 遍历当前基准行之上的所有目标行
        for j in range(row): 

            # 提取待消行的引用。此处的切片索引与上方逻辑类似，只需将 row 替换为变量 j
            row_to_reduce = M[j]

            # 抓取待消行中对应主元列上的系数数值
            value = row_to_reduce[row]
            
            # 执行回代核心算式：待消行 -> 待消行 - 系数值 * 基准回代行
            row_to_reduce = row_to_reduce - value * substitution_row

            # 将更新后的数据行写回矩阵，请务必核对索引边界！
            M[j,:] = row_to_reduce

    ### END CODE HERE ####

     # 从末尾常数列中利落地提取最终求解结果
    solution = M[:,-1]
    
    return solution

```

```python
w2_unittest.test_back_substitution(back_substitution)

```

## 6 - 完整高斯消元法

### 6.1 - 融会贯通

此时此刻，是时候将此前打造的全部积木拼装合体了！你将从给定的 $n \times n$ 阶系数方阵 $A$ 以及 $n \times 1$ 阶常数列矩阵 $B$ 起步，先将其融合成增广矩阵 $[A \vert{} B]$ 并化为简化行阶梯形；随后判定其解的存在性；若存在唯一解，则启动回代函数算出具体数值。当遭遇无解或无穷多解的情形时，程序也应具备稳健的捕获与响应机制。

### 练习 3

在本练习中，你将把刚才独立封装的各部件整合成完整的高斯消元算法。

```python
# GRADED FUNCTION: gaussian_elimination

def gaussian_elimination(A, B):
    """
    运用高斯消元法求解由增广矩阵表示的线性方程组。

    参数:
    - A (numpy.array): 维度为 n x n 的线性方程组系数方阵
    - B (numpy.array): 维度为 n x 1 的常数项列矩阵

    返回:
    numpy.array: 方程组的解向量。
    """

    ### START CODE HERE ###

    # 将增广矩阵转化为行阶梯形
    row_echelon_M = row_echelon_form(A,B)

    # 基于方程组非奇异的前提假设，直接调用回代函数提取最终解
    solution = back_substitution(row_echelon_M)

    ### END SOLUTION HERE ###

    return solution
        

```

```python
w2_unittest.test_gaussian_elimination(gaussian_elimination)

```

## 7 - 测试任意方程组！

下方为你准备了一套便捷的小工具：你可以直接按照日常直观的纯文本数学格式书写任意线性方程组 (支持任意英文字母命名的未知数变量，顺序任意)，代码会自动将其转化为对应的增广矩阵，并调度你刚刚亲手打造的高斯消元求解器进行破解！

你只需灵活修改 `equations` 字符串中的方程内容即可；请务必保留 `*` 作为未知数与系数之间的乘号，且保持每个方程各占一行！

```python
from utils import string_to_augmented_matrix

```

```python
equations = """
3*x + 6*y + 6*w + 8*z = 1
5*x + 3*y + 6*w = -10
4*y - 5*w + 8*z = 8
4*w + 8*z = 9
"""

variables, A, B = string_to_augmented_matrix(equations)

sols = gaussian_elimination(A, B)

if not isinstance(sols, str):
    for variable, solution in zip(variables.split(' '),sols):
        print(f"{variable} = {solution:.4f}")
else:
    print(sols)

```

热烈祝贺！你已顺利通关本课程的第一项大型编程实战！你凭借自己的智慧与代码，从零打造了一款能自主破解线性方程组的强大数学求解器！