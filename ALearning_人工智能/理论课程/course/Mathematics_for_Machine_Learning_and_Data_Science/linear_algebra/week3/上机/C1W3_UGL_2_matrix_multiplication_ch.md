___
# 矩阵乘法（Matrix Multiplication）

在本实验中，你将使用 `NumPy` 内置函数来执行矩阵乘法（Matrix Multiplication），并一探它在机器学习（Machine Learning）应用中的具体发挥空间。

___

___

## 常用工具包

导入 `NumPy` 工具包以调用其相关函数。

```python
import numpy as np

```

## 1 - 矩阵乘法的定义

若矩阵 $A$ 的维度为 $m \times n$，矩阵 $B$ 的维度为 $n \times p$，则它们的矩阵乘积 $C = AB$  (通常省略乘号或点号) 结果为一个 $m \times p$ 的新矩阵，满足以下条件：
$c_{ij}=a_{i1}b_{1j}+a_{i2}b_{2j}+\ldots+a_{in}b_{nj}=\sum_{k=1}^{n} a_{ik}b_{kj}, \tag{4}$

其中 $a_{ik}$ 代表矩阵 $A$ 中的元素，$b_{kj}$ 代表矩阵 $B$ 中的元素，且 $i = 1, \ldots, m$，$k=1, \ldots, n$，$j = 1, \ldots, p$。换言之，$c_{ij}$ 本质上就是矩阵 $A$ 的第 $i$ 行与矩阵 $B$ 的第 $j$ 列之间的点积（Dot Product）。

## 2 - 使用 Python 执行矩阵乘法

如同计算点积一样，在 Python 中实现矩阵乘法也有多种方式。正如上一个实验所讨论的，采用向量化形式（Vectorized Form）计算不仅简洁，效率也显著更高。接下来我们看看向量化中最常用的几个实现函数。首先，定义两个矩阵：

```python
A = np.array([[4, 9, 9], [9, 1, 6], [9, 2, 3]])
print("Matrix A (3 by 3):\n", A)

B = np.array([[2, 2], [5, 7], [4, 4]])
print("Matrix B (3 by 2):\n", B)

```

你可以直接调用 `NumPy` 的内置函数 `np.matmul()` 来计算矩阵 $A$ 和 $B$ 的乘积：

```python
np.matmul(A, B)

```

该操作会返回一个形状为 $3 \times 2$ 的 `np.array`。此外，Python 专门提供的矩阵乘法运算符 `@` 在这里同样适用，并且会输出完全一致的结果：

```python
A @ B

```

## 3 - 矩阵运算约定与广播机制（Broadcasting）



在数学规则中，矩阵乘法成立的前提是：前一个矩阵 $A$ 的列数必须严格等于后一个矩阵 $B$ 的行数 (你可以回头复习第 1 节中的定义，不满足此条件则行列之间根本无法进行对应维度的点积运算)。

正因如此，在上述第 2 节的例子中，如果调换乘法顺序执行 $BA$，程序就无法正常运行，因为原有的维度匹配规则被破坏了。运行下方代码单元格便可验证这一现象——两者均会报错：

```python
try:
    np.matmul(B, A)
except ValueError as err:
    print(err)

```

```python
try:
    B @ A
except ValueError as err:
    print(err)

```

因此，在进行矩阵乘法时，必须对维度保持高度敏感——第一个矩阵的列数务必与第二个矩阵的行数保持一致。掌握这一规律，对于后续深入理解神经网络（Neural Networks）及其内部运转逻辑至关重要。

不过，针对向量（Vector）之间的相乘，`NumPy` 巧妙地提供了一套便捷规则。我们先定义两个相同维度的向量 $x$ 和 $y$  (在概念上可将其理解为两个 $3 \times 1$ 的矩阵)。观察向量 $x$ 的结构：

```python
x = np.array([1, -2, -5])
y = np.array([4, 3, -1])

print("Shape of vector x:", x.shape)
print("Number of dimensions of vector x:", x.ndim)
print("Shape of vector x, reshaped to a matrix:", x.reshape((3, 1)).shape)
print("Number of dimensions of vector x, reshaped to a matrix:", x.reshape((3, 1)).ndim)

```

按照标准矩阵乘法约定，两个 $3 \times 1$ 的矩阵相乘是未定义的。照常理推断，执行下面的单元格应该抛出错误，但让我们看看实际的运行输出：

```python
np.matmul(x,y)

```

代码不仅没有报错，返回的结果恰好是点积 $x \cdot y\,$！ 原来，底层自动将一维向量 $x$ 转置为了 $1 \times 3$ 的行向量，从而顺利完成了等价于 $x^Ty$ 的矩阵乘法。这项特性虽然极度方便，但在 Python 编程中一定要多加留心，避免因随意依赖隐式转换而写出逻辑错误的边缘代码。下面的单元格就会明确抛出错误：

```python
try:
    np.matmul(x.reshape((3, 1)), y.reshape((3, 1)))
except ValueError as err:
    print(err)

```

此时你可能会好奇：原本用于求点积的 `np.dot()` 函数能否直接用于矩阵乘法？ 让我们来测试一下：

```python
np.dot(A, B)

```

完全可行！ 这背后依赖的是 Python 科学计算中著名的 **广播机制（Broadcasting）**：`NumPy` 自动将点积计算广播（Broadcasting）扩展到了全部行与全部列之间，从而计算出完整的乘积矩阵。广播机制在很多日常运算中同样大显身手，例如：

```python
A - 2

```

从严谨的数学定义来看，一个 $3 \times 3$ 的矩阵 $A$ 减去一个标量是未定义的；但 Python 借助广播机制，将该标量自动扩展为一个对应的 $3 \times 3$ `np.array`，并按元素逐一相减。矩阵乘法最典型的工程实战场景之一便是线性回归（Linear Regression）模型，在本周后续的编程作业中你将亲手实现它！

祝贺你，顺利完成了本节实验！

```python


```