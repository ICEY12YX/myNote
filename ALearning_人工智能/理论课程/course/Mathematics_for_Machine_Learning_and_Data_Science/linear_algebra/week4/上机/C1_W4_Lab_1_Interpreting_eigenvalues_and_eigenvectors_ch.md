___
# 理解特征值 (eigenvalues) 与特征向量 (eigenvectors)

欢迎来到第 4 周的实验环节。在这里，你将动手练习求解并直观理解各种线性变换 (linear transformations) 下的特征值与特征向量。

**完成本次实验后，你将能够：**

* 使用 Python 计算特征值与特征向量
* 可视化展示并直观理解特征值与特征向量
___
- [[#依赖库|依赖库]]
- [[#1 - 特征值与特征向量：定义与直观解读|1 - 特征值与特征向量：定义与直观解读]]
	- [[#1 - 特征值与特征向量：定义与直观解读#1.1 - 特征值与特征向量的定义|1.1 - 特征值与特征向量的定义]]
	- [[#1 - 特征值与特征向量：定义与直观解读#1.2 - 使用 Python 求解特征值与特征向量|1.2 - 使用 Python 求解特征值与特征向量]]
- [[#2 - 平面上常见标准变换的特征值与特征向量|2 - 平面上常见标准变换的特征值与特征向量]]
	- [[#2 - 平面上常见标准变换的特征值与特征向量#2.1 - 示例 1：关于 y 轴（垂直轴）的反射变换|2.1 - 示例 1：关于 y 轴（垂直轴）的反射变换]]
	- [[#2 - 平面上常见标准变换的特征值与特征向量#2.2 - 示例 2：沿 x 轴方向的错切变换|2.2 - 示例 2：沿 x 轴方向的错切变换]]
	- [[#2 - 平面上常见标准变换的特征值与特征向量#2.3 - 示例 3：单位矩阵与各向同性缩放|2.3 - 示例 3：单位矩阵与各向同性缩放]]
	- [[#2 - 平面上常见标准变换的特征值与特征向量#2.4 - 示例 4：向 x 轴的正交投影|2.4 - 示例 4：向 x 轴的正交投影]]

___

## 依赖库

运行以下代码单元来加载所需的代码包。`utils.py` 文件中包含了一个稍后用于绘制空间变换图像的辅助函数。

```python
import numpy as np
import matplotlib.pyplot as plt
import utils

```

## 1 - 特征值与特征向量：定义与直观解读

### 1.1 - 特征值与特征向量的定义

我们先来看一个由矩阵 $A=\begin{bmatrix}2 & 3 \\ 2 & 1 \end{bmatrix}$ 所定义的线性变换。我们将该变换应用到标准基向量 (standard basis vectors) $e_1=\begin{bmatrix}1 \\ 0\end{bmatrix}$ 和 $e_2=\begin{bmatrix}0 \\ 1\end{bmatrix}$ 上，并观察变换后的效果。回顾本课程前面的讲座内容，相信你对利用矩阵变换基向量的操作已经非常熟悉了。

```python
A = np.array([[2, 3],[2, 1]])
e1 = np.array([[1],[0]])
e2 = np.array([[0],[1]])

```

你可以使用 `utils.py` 中定义的 `plot_transformation` 函数，将矩阵 $A$ 产生的空间变换可视化地呈现出来。

```python
utils.plot_transformation(A, e1, e2, vector_name='e');

```
![[Pasted image 20261004052602.png]]

可以看到，在经过矩阵 $A$ 的变换之后，原本的两个基向量 $e_1$ 与 $e_2$ 无论在长度还是方向上都发生了改变。
那么，我们能否找到这样一组特殊的基向量：
    在变换之后，它们的方向保持不变，仅仅是长度被拉伸或压缩了呢？也就是说，对于某个向量 $v$，经过变换后满足 $Av=\lambda v$。

正如你在课程中所学到的，具备这种性质的向量 $v$ 就被称为**特征向量（eigenvector）**，而对应的缩放倍数 $\lambda$ 则被称为**特征值（eigenvalue）**。

需要说明的是，如果 $v$ 是一个满足 $Av = \lambda v$ 的特征向量，那么==对 $v$ 乘以任意非零常数缩放后的向量，也同样是具有相同特征值的特征向量。== 如果我们用 $k$ 来表示这个缩放因子，用数学公式表达就是：

$$A(kv)=k(Av)=k \lambda v = \lambda (kv),$$


其中 $k$ 为任意非零实数常数。

换句话说，==对于每一个特征值，其实都存在着**无数个**有效的特征向量==。你可以把它们想象成全都落在同一条直线上、只是长度（即范数）各不相同的箭头。在实际应用中，我们通常只需要从中挑选一个代表即可，业界最常见的做法是选取范数（模长）为 1 的单位向量。

### 1.2 - 使用 Python 求解特征值与特征向量

在 Python 中，我们可以通过 `NumPy` 库提供的💛 `np.linalg.eig()` 函数来快速求解特征值与特征向量。
该函数会返回一个包含向量与数组的元组 (tuple)。其中的一维向量存储了特征值；二维数组则按列存储了对应的特征向量，即每一列对应一个特征向量。请注意，==该函数默认会对特征向量进行归一化处理，使其范数均为 1。==

通过运行以下代码，你可以求出前面定义的矩阵 $A$ 的特征值与特征向量：

```python
A_eig = np.linalg.eig(A)

print(f"Matrix A:\n{A} \n\nEigenvalues of matrix A:\n{A_eig[0]}\n\nEigenvectors of matrix A:\n{A_eig[1]}")

```
![[Pasted image 20261004053015.png]]
理解
    ![[Pasted image 20261004053138.png]]
    ![[Pasted image 20261004053230.png]]
    ![[Pasted image 20261004053320.png]]
    ![[Pasted image 20261004053343.png]]
    

切记，返回元组的第一个元素包含全部特征值，第二个元素则是**一列列**排布的特征向量。
    ![[Pasted image 20261004053831.png|399]]

这意味着==第一个特征向量==可以通过切片代码 `A_eig[1][:,0]` (所有行,0列) 提取，而==第二个特征向量==则通过 `A_eig[1][:,1]` 提取。

现在，让我们直观地观察一下特征向量在矩阵变换下的表现：

```python
utils.plot_transformation(A, A_eig[1][:,0], A_eig[1][:,1]);

```
![[Pasted image 20261004053649.png]]

正如图像所呈现的那样，$v_1$ 的长度被拉伸至原先的 4 倍；而 $v_2$ 的指向完全反转，这相当于乘以了缩放倍数 -1。然而不可思议的是，这两个向量始终与它们原本所指的轴线方向保持平行，因此完全符合特征向量的定义。

## 2 - 平面上常见标准变换的特征值与特征向量

### 2.1 - 示例 1：关于 y 轴（垂直轴）的反射变换

如果想要让图形==关于 y 轴产生镜面反射==，我们需要==保持 y 轴方向上的分量固定不变，同时将 x 轴方向的分量取反==。
这一操作可以通过如下矩阵来实现：

$$A_{\text{reflection\_yaxis}}= \left[\begin{matrix}-1& 0\\0 &1\end{matrix}\right].$$

在下面的代码中，你将定义该变换矩阵，计算其特征值与特征向量，并可视化其几何效果。如果从特征值与特征向量的视角来看，你会如何理解这种线性变换的物理本质呢？

```python
# 将变换矩阵 A_reflection_yaxis 定义为 numpy 数组。
A_reflection_yaxis = np.array([[-1,0],[0,1]])
# 求解矩阵 A_reflection_yaxis 的特征值和特征向量。
A_reflection_yaxis_eig = np.linalg.eig(A_reflection_yaxis)

print(f"Matrix A:\n {A_reflection_yaxis} \n\nEigenvalues of matrix A:\n {A_reflection_yaxis_eig[0]}",
        f"\n\nEigenvectors of matrix A:\n {A_reflection_yaxis_eig[1]}")

utils.plot_transformation(A_reflection_yaxis, A_reflection_yaxis_eig[1][:,0],A_reflection_yaxis_eig[1][:,1]);

```
![[Pasted image 20261004054010.png]]
(发现此时 特征向量就是1,0  0,1 基向量)
![[Pasted image 20261004054019.png]]

在目前接触到的所有示例中，我们处理的都是 2 $\times$ 2 矩阵，而且它们都拥有 2 个互不相同的特征值和 2 个不同的特征向量。这时一个自然的问题便浮出水面：==对于平面上的任何线性变换，是否总能找到两个不同的特征向量？==正如你在课程中所学到的那样，很遗憾，答案是==否定的==。在接下来的示例中，你就会看到这样一种反例。

### 2.2 - 示例 2：沿 x 轴方向的错切变换

**错切变换**（shear transformation）的几何形变如下图所示。这种变换==会把平面上的每个点沿着某一固定方向平移一段距离，而平移量与该点到平行参考线的带符号距离成正比。==你可以将其形象地理解为把整幅平面切成一副扑克牌，然后将这叠纸牌横向推斜滑动的过程。接下来让我们探索一下，这类变换到底拥有几个特征向量。

为了构造一个沿 x 轴方向错切的矩阵变换，我们需要根据 y 轴分量的大小在 x 方向施加一定的偏移量，假设这个比例系数为 0.5。这可以通过以下矩阵来实现：

$$A_{\text{shear\_x}}= \left[\begin{matrix}1& 0.5\\0 &1\end{matrix}\right].$$

需要注意的是，原本的向量 $e_1=\begin{bmatrix}1 \\ 0\end{bmatrix}$ 保持原位不动，而向量 $e_2=\begin{bmatrix}0 \\ 1\end{bmatrix}$ 则变换为了 $\begin{bmatrix}0.5 \\ 1\end{bmatrix}$。

在下一个代码单元中，你将定义这个错切矩阵，求解其特征值与特征向量，并观察作用在所求特征向量上的变换效果。

```python
# 将变换矩阵 A_shear_x 定义为 numpy 数组。
A_shear_x = np.array([[1, 0.5],[0, 1]])
# 求解矩阵 A_shear_x 的特征值和特征向量。
A_shear_x_eig = np.linalg.eig(A_shear_x)

print(f"Matrix A_shear_x:\n {A_shear_x}\n\nEigenvalues of matrix A_shear_x:\n {A_shear_x_eig[0]}",
      f"\n\nEigenvectors of matrix A_shear_x \n {A_shear_x_eig[1]}")

utils.plot_transformation(A_shear_x, A_shear_x_eig[1][:,0], A_shear_x_eig[1][:,1]);

```
![[Pasted image 20261004054208.png]]
![[Pasted image 20261004054218.png]]
解释
    ![[Pasted image 20261004054513.png]]
    ![[Pasted image 20261004054657.png]]
    反正可能底层库会在张成空间中 填满输出的特征值,特征向量

```python
A_rotation = np.array([[0, 1],[-1, 0]])
A_rotation_eig = np.linalg.eig(A_rotation)

print(f"Matrix A_rotation:\n {A_rotation}\n\nEigenvalues of matrix A_rotation:\n {A_rotation_eig[0]}",
      f"\n\nEigenvectors of matrix A_rotation \n {A_rotation_eig[1]}")


```
![[Pasted image 20261004054323.png]]


可以看到，控制台虽然输出了两个特征值，但它们实际上是**复数**。请注意，==在 Python 中，复数的虚数单位是用字母 `j` 来表示的，而非数学中更常见的 $i$。==

该矩阵包含两个复数特征值以及对应的两个复数特征向量。然而，由于不存在任何实数特征向量，我们完全可以从几何上直观理解这个结果：==在二维实数平面上，经历 90 度刚性旋转之后，没有任何一个向量能够保持原有的指向不变==。稍微思考一下就会发现这非常合乎常理——当整个平面被整体旋转后，每一个向量的朝向都会发生偏移。

如果你对实数与复数的概念不太熟悉也完全不必担心。这里的核心思想在于：==**某些 2 $\times$ 2 矩阵可能仅有 1 个甚至 0 个实数特征向量**==，希望你已经体会到这背后的几何直觉。如果在矩阵变换之后没有任何向量能够维持原方向，那么我们自然无法在实数域中找到特征向量。带着这个认识，我们再来看另一个有趣的案例。

### 2.3 - 示例 3：单位矩阵与各向同性缩放

如果使用==单位矩阵 (identity matrix) 来变换平面==，会发生什么呢？这意味着==平面上的所有向量都不会发生任何改变。==既然每个点和向量都原地不动，那么所有的向量显然都维持了原有的指向，因而==平面上的**每一个**向量都严格符合特征向量的定义！==

在下一个代码单元中，你将探索在尝试计算单位矩阵的特征值与特征向量时，`NumPy` 究竟会返回怎样的结果。

```python
A_identity = np.array([[1, 0],[0, 1]])
A_identity_eig = np.linalg.eig(A_identity)

utils.plot_transformation(A_identity, A_identity_eig[1][:,0], A_identity_eig[1][:,1]);

print(f"Matrix A_identity:\n {A_identity}\n\nEigenvalues of matrix A_identity:\n {A_identity_eig[0]}",
      f"\n\nEigenvectors of matrix A_identity\n {A_identity_eig[1]}")

```
![[Pasted image 20261004054934.png]]
![[Pasted image 20261004054942.png]]


正如你所看到的，`np.linalg.eig()` 函数输出了两个数值相等的特征值 $\lambda = 1$，这完全正确。然而，==输出所给出的特征向量列表并没有（也不可能）涵盖平面上的全部向量。==在代数上可以严格证明，平面上的任意非零向量都是单位矩阵的特征向量。==软件往往只会返回一组正交基，无法直接展示这“无限可能”==，所以千万要留心！这也正是为什么深入理解代码与算法模型背后的数学机理如此至关重要。

同样地，你可以验证在 x 和 y 方向上同时放大 $2$ 倍的缩放变换（均匀膨胀）。在这种情况下，每个向量都保持原有朝向，只是长度变成了原来的 2 倍。此时每一个向量同样都满足特征向量的定义，但 `NumPy` 依然只会返回其中的两个基底向量。

```python
A_scaling = np.array([[2, 0],[0, 2]])
A_scaling_eig = np.linalg.eig(A_scaling)

utils.plot_transformation(A_scaling, A_scaling_eig[1][:,0], A_scaling_eig[1][:,1]);

print(f"Matrix A_scaling:\n {A_scaling}\n\nEigenvalues of matrix A_scaling:\n {A_scaling_eig[0]}",
      f"\n\nEigenvectors of matrix A_scaling\n {A_scaling_eig[1]}")

```
![[Pasted image 20261004055029.png]]
![[Pasted image 20261004055036.png]]

### 2.4 - 示例 4：向 x 轴的正交投影

最后，让我们探讨一个非常具有启发性的案例：==向 x 轴的正交投影变换==。这种==投影变换只保留向量在 x 轴上的分量，同时将所有的 y 分量直接清零。==

实现向 x 轴投影的变换矩阵可表示为：

$$A_{\text{projection}}=\begin{bmatrix}1 & 0 \\ 0 & 0 \end{bmatrix}.$$

```python
A_projection = np.array([[1, 0],[0, 0]])
A_projection_eig = np.linalg.eig(A_projection)

utils.plot_transformation(A_projection, A_projection_eig[1][:,0], A_projection_eig[1][:,1]);


print(f"Matrix A_projection:\n {A_projection}\n\nEigenvalues of matrix A_projection:\n {A_projection_eig[0]}",
      f"\n\nEigenvectors of matrix A_projection\n {A_projection_eig[1]}")

```
![[Pasted image 20261004055114.png]]
![[Pasted image 20261004055128.png]]

该矩阵拥有两个实数特征值，==其中一个特征值恰好为 $0$。==这完全合情合理，❤️❤️❤️因为==**特征值 $\lambda$ 是允许等于 $0$ 的！**==
(应该是特征向量 0,0 没意义  特征值还是可以是0的)
在此场景下，特征值为 0 意味着任何落在 y 轴上的向量都会被直接压缩坍缩为零向量，因为它们在 x 方向上没有任何投影分量。由于该矩阵拥有两个互不相等的特征值，该变换仍然拥有两个特征向量。

祝贺你！你已经顺利完成了本次实验。希望通过这些可视化的实例，你已经对特征值与特征向量的几何直观有了更加清晰深刻的认识，也彻底明白了为什么不同的 2 $\times$ 2 矩阵会拥有不同数量的实特征向量。