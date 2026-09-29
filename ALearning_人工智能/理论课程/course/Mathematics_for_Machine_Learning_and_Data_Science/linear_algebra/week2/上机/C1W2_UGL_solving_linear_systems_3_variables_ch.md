___
# Numpy.linalg 子库简介



在本实验中，你将通过使用 numpy.linalg 子库，继续进阶利用 Python 求解线性方程组的技能。在这份实验指南中，你将：

* 使用 NumPy 线性代数 (Linear Algebra) 软件包求解线性方程组
* 计算矩阵的行列式 (Determinant)，再次直观感受矩阵奇异性 (Singularity) 与线性方程组解的数量之间的内在联系
* 探索 numpy.linalg 子库，熟悉其基本特性
___
- [[#软件库|软件库]]
- [[#1 - 使用矩阵 表示 并 求 解线性方程组(Representing and Solving)|1 - 使用矩阵 表示 并 求 解线性方程组(Representing and Solving)]]
	- [[#1 - 使用矩阵 表示 并 求 解线性方程组(Representing and Solving)#1.1 - 线性方程组|1.1 - 线性方程组]]
	- [[#1 - 使用矩阵 表示 并 求 解线性方程组(Representing and Solving)#1.2 - 使用矩阵求解线性方程组|1.2 - 使用矩阵求解线性方程组]]
	- [[#1 - 使用矩阵 表示 并 求 解线性方程组(Representing and Solving)#1.3 - 计算矩阵的行列式|1.3 - 计算矩阵的行列式]]
	- [[#1 - 使用矩阵 表示 并 求 解线性方程组(Representing and Solving)#1.4 - 如果方程组没有唯一解会发生什么？|1.4 - 如果方程组没有唯一解会发生什么？]]

___
## 软件库

加载 NumPy 库以调用相关数学功能。

```python
import numpy as np

```

## 1 - 使用矩阵 表示 并 求 解线性方程组(Representing and Solving)

### 1.1 - 线性方程组

这是一个包含三个方程和三个未知数的**线性方程组 (System of Linear Equations**，或简称 **Linear System)**：

$$\begin{cases}  4x_1-3x_2+x_3=-10, \\ 2x_1+x_2+3x_3=0, \\ -x_1+2x_2-5x_3=17, \end{cases}\tag{1}$$

所谓**求解**该线性方程组，就是找到一组特定的变量值 $x_1$、$x_2$ 和 $x_3$，使得其中的每一个等式都能同时成立。

### 1.2 - 使用矩阵求解线性方程组

接下来让我们准备用 NumPy 来求解线性方程组 $(1)$。我们用==矩阵 $A$ 来表示系数==，
(其中每一行代表方程组中的一个方程，每一列分别对应未知数 $x_1$、$x_2$ 和 $x_3$ 的系数)
同时用一维==数组 $b$== 来存储等号右侧的==常数项==（自由项）：

```python
A = np.array([
        [4, -3, 1],
        [2, 1, 3],
        [-1, 2, -5]
    ], dtype=np.dtype(float))

b = np.array([-10, 0, 17], dtype=np.dtype(float))

print("Matrix A:")
print(A)
print("\nArray b:")
print(b)

```
(这个就是结果, 只要下载md之前, jypter那边已经运行了,下载出来的就自带结果)
```
Matrix A:
[[ 4. -3.  1.]
 [ 2.  1.  3.]
 [-1.  2. -5.]]

Array b:
[-10.   0.  17.]

```

使用 `shape()` 函数检查 $A$ 和 $b$ 的形状维度：

```python
print(f"Shape of A: {np.shape(A)}")
print(f"Shape of b: {np.shape(b)}")

```

```
Shape of A: (3, 3)
Shape of b: (3,)

```

现在调用💛 `np.linalg.solve(A, b)` 函数来计算方程组 $(1)$ 的解。==结果==将存储在一维==数组 $x$== 中，其元素依次对应未知数 $x_1$、$x_2$ 和 $x_3$ 的数值：

```python
x = np.linalg.solve(A, b)

print(f"Solution: {x}")

```

```
Solution: [ 1.  4. -2.]

```

你可以尝试将算出的 $x_1$、$x_2$ 和 $x_3$ 带入原始方程组中，验证等式是否完全吻合相容。

### 1.3 - 计算矩阵的行列式

对应线性方程组 $(1)$ 的系数矩阵 $A$ 是一个**方阵 (Square Matrix)** —— 其行数与列数完全相同。对于方阵，我们可以计算它的行列式 —— 这是一个刻画矩阵内在特性的实数标量。
对于包含三个未知数和三个方程的线性方程组，它==**拥有唯一解的充要条件是矩阵 $A$ 的行列式不为零。**==

让我们使用💛 `np.linalg.det(A)` 函数来计算行列式：

```python
A = np.array([
        [4, -3, 1],
        [2, 1, 3],
        [-1, 2, -5]
    ], dtype=np.dtype(float))
d = np.linalg.det(A)

print(f"Determinant of matrix A: {d:.2f}")

```

```
Determinant of matrix A: -60.00

```

可以看到，计算结果正如预期那样是一个非零值。

### 1.4 - 如果方程组没有唯一解会发生什么？

接下来我们探索一下：当把 `np.linalg.solve` 应用于一个没有唯一解的方程组 (完全无解或存在无穷多解) 时，会发生什么情况。

考察另一个线性方程组：

$$\begin{cases}  x_1+x_2+x_3=2, \\ x_2-3x_3=1, \\ 2x_1+x_2+5x_3=0, \end{cases}\tag{2}$$

我们尝试用矩阵求解并观察输出。

```python
A_2= np.array([
        [1, 1, 1],
        [0, 1, -3],
        [2, 1, 5]
    ], dtype=np.dtype(float))
b_2 = np.array([2, 1, 0], dtype=np.dtype(float))

print(np.linalg.solve(A_2, b_2))

```

![[Pasted image 20260929200840.png]]

正如你所见，程序抛出了一个 `LinAlgError` 异常，提示==这是一个奇异矩阵==。你可以通过计算该矩阵的行列式来亲自验证这一原因：

```python
d_2 = np.linalg.det(A_2)

print(f"Determinant of matrix A_2: {d_2:.2f}")

```

```
Determinant of matrix A_2: 0.00

```

子库 np.linalg 提供了多个用于辅助线性代数计算的函数，到目前为止，我们已经将课堂上学到的相关函数悉数演练了一遍。

随着你掌握更多理论知识，该库中各类函数的运作机制会变得更加清晰直观。在第 3 周和第 4 周的课后作业中你还会继续使用它们，不过不必担心，届时都会有详细的引导带你逐步完成。

干得漂亮！你已经成功掌握了利用 NumPy 内置函数求解方程组的方法。正如预期那样，直接调用封装好的函数确实非常轻松快捷，但它对底层算法运行细节的呈现较少。**正因如此，下一节作业我们将重点学习高斯消元法 (Gaussian Elimination) —— 一种深入算法底层求解线性系统的经典方法**。请时刻牢记，当方程组无解或有无穷多解时，np.linalg.solve 会抛出错误，因此==在实际编程中需要注意捕获并处理这种边界情况，以避免程序异常崩溃==。