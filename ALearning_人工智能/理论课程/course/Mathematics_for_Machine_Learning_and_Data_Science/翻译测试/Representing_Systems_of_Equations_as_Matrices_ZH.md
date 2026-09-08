# 将线性方程组表示为矩阵

通过完成本实验，你将能够运用 Python 及 [`NumPy`](https://numpy.org/doc/stable/index.html) 库的基础编程技能，将线性方程组表示为矩阵。在本 Notebook 中，你将：

- 使用 [`NumPy`](https://numpy.org/doc/stable/index.html) 线性代数模块将线性方程组建模为矩阵。
- 计算矩阵的行列式，并探究矩阵奇异性与线性方程组解的数量之间的关系。

# 目录

- [ 1 - 使用矩阵表示和求解线性方程组](#1)
  - [ 1.1 - 线性方程组](#1.1)
  - [ 1.2 - 使用矩阵表示线性方程组](#1.2)
  - [ 1.3 - 计算矩阵的行列式](#1.3)
- [ 2 - 将 2x2 方程组可视化为直线图像](#2)
  - [ 2.1 - 消元法](#2.1)
  - [ 2.2 - 解的图形化表示](#2.2)
- [ 3 - 无解的线性方程组](#3)
- [ 4 - 有无穷多解的线性方程组](#4)

## 依赖库

加载 `NumPy` 库以调用其相关函数。此外，加载用于绘制图形的 `matplotlib.pyplot` 库。


```python
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_lines
```

<a name='1'></a>
## 1 - 使用矩阵表示线性方程组

<a name='1.1'></a>
### 1.1 - 线性方程组

**线性方程组**（或称**线性系统**）是由一个或多个包含相同变量的线性方程组成的集合。例如：


$$
\begin{cases} 
-x_1+3x_2=7, \\ 3x_1+2x_2=1, \end{cases}\tag{1}
$$

这是一个包含两个未知变量 $x_1$ 和 $x_2$ 的二元一次方程组。**求解**线性方程组意味着找到一组变量 $x_1$ 和 $x_2$ 的值，使得该方程组中的所有方程同时成立。

如果一个线性方程组没有唯一解，则称其为**奇异的**（singular）；反之，如果存在唯一解，则称为**非奇异的**（non-singular）。

<a name='1.2'></a>
### 1.2 - 使用矩阵表示线性方程组
在课程讲授中，你已经了解到可以用矩阵来表示线性方程组。方程组 $(1)$ 用增广矩阵表示如下：

$$
\begin{bmatrix}
-1 & 3 & 7 \\
3 & 2 & 1
\end{bmatrix}
$$ 

每一行代表方程组中的一个方程。第一列代表方程组中 $x_1$ 的系数，第二列代表 $x_2$ 的系数，第三列代表方程等号右侧的常数项。

我们还可以进一步选择将方程组 $(1)$ 的系数单独表示为一个系数矩阵 $A$：

$$
\begin{bmatrix}
-1 & 3\\
3 & 2
\end{bmatrix}
$$

并将方程组的常数项表示为一个向量 $b$：

$$
\begin{bmatrix}
7 \\
1
\end{bmatrix}
$$

下面我们在 `NumPy` 中定义并展示矩阵 $A$ 和向量 $b$：


```python
A = np.array([
        [-1, 3],
        [3, 2]
    ], dtype=np.dtype(float))

b = np.array([7, 1], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)
```

矩阵 $A$ 和向量 $b$ 的维度（形状）是多少？

你可以使用 `shape` 属性来查看 $A$ 和 $b$ 的维度（也可以使用 `np.shape()` 函数）：


```python
print(f"Shape of A: {A.shape}")
print(f"Shape of b: {b.shape}")

# print(f"Shape of A: {np.shape(A)}")
# print(f"Shape of A: {np.shape(b)}")
```

在课程讲授中，你手动求解过一些简单的二元线性方程组。不过，我们还没有正式系统化线性方程组的求解方法。在本实验中，我们将使用一个非常便捷的函数来求解方程组。

`NumPy` 线性代数模块提供了一种快速且可靠的方法来求解线性方程组，即使用 `np.linalg.solve(A, b)` 函数。在这里，正如你前面看到的，$A$ 是一个矩阵，每一行对应系统中的一个方程，每一列对应变量 $x_1$ 和 $x_2$；$b$ 是一维数组，包含等号右侧的自由常数项。关于 `np.linalg.solve()` 函数的更多信息可以参阅[官方文档](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)。

为了求出方程组 $(1)$ 的解，我们直接调用 `np.linalg.solve(A, b)` 函数。计算结果将保存在一维数组 $x$ 中，其中的元素分别对应变量 $x_1$ 和 $x_2$ 的值：


```python
x = np.linalg.solve(A, b)

print(f"Solution: {x}")
```

该输出中的第一个值是变量 $x_1$ 的解，第二个值是变量 $x_2$ 的解。你可以将 $x_1$ 和 $x_2$ 的值代入原方程组中，验证该解是否正确。

<a name='1.3'></a>
### 1.3 - 计算矩阵的行列式

与线性方程组 $(1)$ 对应的矩阵 $A$ 是一个**方阵**——它的行数和列数相同。对于方阵，我们可以计算其行列式（determinant）——这是一个表征矩阵某些特性的实数。一个包含相同数量未知变量和方程的线性方程组拥有唯一解，**当且仅当**矩阵 $A$ 的行列式不为零。

在本课程中，通过手算行列式等性质有助于建立直观理解，但借助计算机可以极其轻松地完成这些计算。

让我们使用 `NumPy` 线性代数模块来计算行列式。你可以使用 `np.linalg.det(A)` 函数完成此操作。更多相关信息可以查阅[官方文档](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html)。


```python
d = np.linalg.det(A)

print(f"Determinant of matrix A: {d:.2f}")
```

注意它的值不为零，这与方程组恰好有唯一解的结论完全吻合。

<a name='2'></a>
## 2 - 将 2x2 方程组可视化为直线图像

可以看到，使用现代计算库来求解线性方程组以及计算矩阵性质（如行列式）是多么容易。在本节中，我们将把 2x2 方程组可视化为直线，正如你在非评分插件中看到的那样。

<a name='2.1'></a>
### 2.1 - 方程组的矩阵表示

在可视化方程组 $(1)$ 之前，你需要将方程组表示为如下形式的增广矩阵：

$$
\begin{bmatrix}
-1 & 3 & 7 \\
3 & 2 & 1
\end{bmatrix}
$$

为此，你既可以创建一个包含这些值的新矩阵，也可以将之前创建的矩阵 $A$ 和 $b$ 进行水平拼接（horizontal stack）。需要注意的是，`np.hstack()` 函数要求在拼接前对数组 $b$ 进行变形（reshape），因为其当前形状是 $(2,)$。下面的代码使用了 `.reshape((2, 1))` 命令以顺利完成水平拼接。


```python
A_system = np.hstack((A, b.reshape((2, 1))))

print(A_system)
```

让我们复习一下如何提取矩阵的某一行，这有助于后续对矩阵行执行必要的操作。请记住，Python 中数组的索引是从零开始的，因此提取矩阵的第二行需要使用如下代码：


```python
print(A_system[1])
```

<a name='2.2'></a>
### 2.2 - 解的图形化表示

包含两个变量（此处为 $x_1$ 和 $x_2$）的线性方程在几何上可以表示为平面内的一条直线。这被称为**线性方程的图像**。对于包含两个方程的方程组，会有两条直线分别对应每一个方程，而方程组的解就是这两条直线的交点。

在以下代码中，你将使用已定义的 `plot_lines()` 函数来绘制直线，并用它来展示先前求得的解。如果暂时不理解该单元格内的代码细节也不必担心——在现阶段，理解底层绘图实现并不是重点。


```python
plot_lines(A_system)
```

观察这两条直线是如何相交于 $(x_1,x_2) = (-1, 2)$ 的，该交点正是方程组的解。

<a name='3'></a>
## 3 - 无解的线性方程组

给定另一个线性方程组：

$$
\begin{cases} 
-x_1+3x_2=7, \\ 3x_1-9x_2=1, \end{cases}\tag{2}
$$

让我们计算其对应系数矩阵的行列式。


```python
A_2 = np.array([
        [-1, 3],
        [3, -9]
    ], dtype=np.dtype(float))

b_2 = np.array([7, 1], dtype=np.dtype(float))

d_2 = np.linalg.det(A_2)

print(f"Determinant of matrix A_2: {d_2:.2f}")
```

该行列式等于 0，因此该方程组不可能存在唯一解。它要么有无穷多解，要么无解。具体属于哪种相容性情况取决于常数项（等号右侧系数）。你可以运行下方单元格中的代码，观察 `np.linalg.solve()` 函数因矩阵奇异而报错：


```python
try:
    x_2 = np.linalg.solve(A_2, b_2)
except np.linalg.LinAlgError as err:
    print(err)
```

构建该线性方程组对应的增广矩阵：


```python
A_2_system = np.hstack((A_2, b_2.reshape((2, 1))))
print(A_2_system)
```


```python
plot_lines(A_2_system)
```

正如预期的那样，这两个方程对应的直线相互平行。

<a name='4'></a>
## 4 - 有无穷多解的线性方程组

通过改变方程组 $(2)$ 的常数项，你可以使其变为相容方程组（有解）：

$$
\begin{cases} 
-x_1+3x_2=7, \\ 3x_1-9x_2=-21, \end{cases}\tag{3}
$$


```python
b_3 = np.array([7, -21], dtype=np.dtype(float))
```

准备对应于方程组 $(3)$ 的新矩阵：


```python
A_3_system = np.hstack((A_2, b_3.reshape((2, 1))))
print(A_3_system)
```

因此，从对应的简化线性方程组

$$
\begin{cases} 
-x_1+3x_2=7, \\ 0=0, \end{cases}\tag{4}
$$

可以得出线性方程组 $(3)$ 的解为：

$$
x_1=3x_2-7, \tag{5}
$$

其中 $x_2$ 可取任意实数。

如果你绘制该方程组的直线，你预期在图表中会看到几条线呢？使用下方代码来检验一下吧：


```python
plot_lines(A_3_system)
```
