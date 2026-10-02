___
# 向量运算：数乘、向量加法与点积

在本实验中，你将使用 Python 和 `NumPy` 函数来完成几种核心的向量运算：数乘（Scalar Multiplication）、向量加法（Sum of Vectors）以及点积（Dot Product）。此外，你还将探究使用传统循环与向量化形式（Vectorized Form）执行这些线性代数（Linear Algebra）基本运算时的计算速度差异。

___
- [[#工具包|工具包]]
- [[#1 - 向量的数乘与加法|1 - 向量的数乘与加法]]
	- [[#1 - 向量的数乘与加法#1.1 - 向量 $v\in\mathbb{R}^2$ 的可视化|1.1 - 向量 $v\in\mathbb{R}^2$ 的可视化]]
	- [[#1 - 向量的数乘与加法#1.2 - 数乘运算|1.2 - 数乘运算]]
	- [[#1 - 向量的数乘与加法#1.3 - 向量加法|1.3 - 向量加法]]
	- [[#1 - 向量的数乘与加法#1.4 - 向量的范数|1.4 - 向量的范数]]
- [[#2 - 点积|2 - 点积]]
	- [[#2 - 点积#2.1 - 点积的代数定义|2.1 - 点积的代数定义]]
	- [[#2 - 点积#2.2 - 使用 Python 计算点积|2.2 - 使用 Python 计算点积]]
	- [[#2 - 点积#2.3 - 向量化形式下的计算速度|2.3 - 向量化形式下的计算速度]]
	- [[#2 - 点积#2.4 - 点积的几何定义|2.4 - 点积的几何定义]]
	- [[#2 - 点积#2.5 - 点积的应用：向量相似度|2.5 - 点积的应用：向量相似度]]

___
## 工具包

导入 `NumPy` 库以便调用其内置函数。

```python
import numpy as np

```

## 1 - 向量的数乘与加法

### 1.1 - 向量 $v\in\mathbb{R}^2$ 的可视化

在视频和之前的实验中你已经了解到，向量可以用带箭头的线段形象地表示出来；在二维空间 $v\in\mathbb{R}^2$ 中，这种可视化非常直观，例如：
$$
v=\begin{bmatrix}
          1 & 3
\end{bmatrix}^T 
$$

通过以下代码，我们可以把它画出来。

```python
import matplotlib.pyplot as plt

def plot_vectors(list_v, list_label, list_color):
    _, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))
    
    plt.axis([-10, 10, -10, 10])
    for i, v in enumerate(list_v):
        sgn = 0.4 * np.array([[1] if i==0 else [i] for i in np.sign(v)])
        plt.quiver(v[0], v[1], color=list_color[i], angles='xy', scale_units='xy', scale=1)
        ax.text(v[0]-0.2+sgn[0], v[1]-0.2+sgn[1], list_label[i], fontsize=14, color=list_color[i])

    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()

v = np.array([[1],[3]])
# Arguments: list of vectors as NumPy arrays, labels, colors.
plot_vectors([v], [f"\(v\)"], ["black"])

```
[[解释函数1]]
![[output.png]]


向量的核心特征在于其 **范数 (Norm，即长度或模长)** 与 **方向 (Direction)**，而==**与它在空间中的具体位置无关**==。不过，为了直观和作图方便，我们==通常将坐标原点 (在 $\mathbb{R}^2$ 中即点 $(0,0)$) 作为向量的起点。==

### 1.2 - 数乘运算

标量 $k$ 与向量 $v=\begin{bmatrix}           v_1 & v_2 & \ldots & v_n  \end{bmatrix}^T\in\mathbb{R}^n$ 的 **数乘（Scalar multiplication）** 运算，实质上是将标量与向量的==每个分量逐一相乘==，得到新向量 $kv=\begin{bmatrix}           kv_1 & kv_2 & \ldots & kv_n  \end{bmatrix}^T$ (按元素逐项相乘)。
当 $k>0$ 时，新向量 $kv$ 的方向与 $v$ 保持一致，长度扩展为原长度的 $k$ 倍；
当 $k=0$ 时，$kv$ 变成一个零向量；
而当 $k<0$ 时，新向量 $kv$ 将指向完全相反的方向。
在 Python 中，你可以直接使用 `*` 运算符来实现数乘操作。来看下面的具体示例：

```python
plot_vectors([v, 2*v, -2*v], [f"\(v\)", f"$2v$", f"\(-2v\)"], ["black", "green", "blue"])

```
(这个函数不止可以画一个向量, 所以每个输入参数是列表的形式)
![[output1.png]]
### 1.3 - 向量加法

**向量加法 (Sum of vectors)** 通过对应位置的分量相加来实现：若有两个向量 $v=\begin{bmatrix}           v_1 & v_2 & \ldots & v_n  \end{bmatrix}^T\in\mathbb{R}^n$ 和
$w=\begin{bmatrix}           w_1 & w_2 & \ldots & w_n  \end{bmatrix}^T\in\mathbb{R}^n$，两者的和即为 $v + w=\begin{bmatrix}           v_1 + w_1 & v_2 + w_2 & \ldots & v_n + w_n  \end{bmatrix}^T\in\mathbb{R}^n$。几何学中的 **平行四边形定则 (Parallelogram law)** 直观展示了向量加法的物理意义：若将从同一点出发的两个向量 $u$ 和 $v$ 视作平行四边形的两条邻边 (大小与方向均保持对应)，那么两向量相加所得的向量和 $u+v$，恰好就是从该公共顶点引出的对角线：
![[Pasted image 20261001213236.png]]
在 Python 中，你可以直接使用常规的 `+` 运算符，也可以调用 `NumPy` 的内置函数 `np.add()`。取消下面代码中的注释行，你可以亲自验证两种方式的结果完全一致：

```python
v = np.array([[1],[3]])
w = np.array([[4],[-1]])

plot_vectors([v, w, v + w], [f"\(v\)", f"\(w\)", f"\(v + w\)"], ["black", "black", "red"])
# plot_vectors([v, w, np.add(v, w)], [f"\(v\)", f"\(w\)", f"\(v + w\)"], ["black", "black", "red"])

```
![[output2.png]]


向量写成`([[1],[3]])`  就是竖着的1,3 
python里面习惯这样写
记住横着的向量才是不正常的 要转置的
![[Pasted image 20261001195155.png|258]]

### 1.4 - 向量的范数

向量 $v$ 的范数通常记作 $\lvert v\rvert$。它是一个非负实数，用来度量向量在空间中的跨度或几何长度。你可以直接借助 `NumPy` 的内置函数 `np.linalg.norm()` 快速计算出向量的范数：

```python
print("Norm of a vector v is", np.linalg.norm(v))

```
![[Pasted image 20261001213103.png]]
(根号10)

## 2 - 点积

### 2.1 - 点积的代数定义

**点积 (Dot product)** (亦称 **数量积 (Scalar product)**) 是一种代数运算：它接收两个相同维度的向量 $x=\begin{bmatrix}           x_1 & x_2 & \ldots & x_n  \end{bmatrix}^T\in\mathbb{R}^n$ 与
$y=\begin{bmatrix}           y_1 & y_2 & \ldots & y_n  \end{bmatrix}^T\in\mathbb{R}^n$，最终输出一个单一的标量数值。点积通常用点运算符 $x\cdot y$ 表示，其代数计算公式如下：

$$x\cdot y = \sum_{i=1}^{n} x_iy_i = x_1y_1+x_2y_2+\ldots+x_ny_n \tag{1}$$

### 2.2 - 使用 Python 计算点积

在 Python 中，计算点积最纯粹、直观的方式就是将==两个向量对应位置的元素分别相乘==，==再将乘积全部累加==。我们首先通过列出坐标分量来定义两个普通列表形式的向量 $x$ 和 $y$：

```python
x = [1, -2, -5]
y = [4, 3, -1]

```

接着，定义一个自定义函数 `dot(x,y)` 来实现这一累加过程：

```python
def dot(x, y):
    s=0
    for xi, yi in zip(x, y):
        s += xi * yi
    return s

```
💖 zip还能这么用哦!!!

为了保持代码整洁清晰，我们默认传入该函数的向量长度始终一致(这里指不会出现两个向量维数对不上的情况)，省去了额外的长度校验逻辑。
现在万事俱备，我们可以调用 `dot(x,y)` 来执行点积计算了：

```python
print("The dot product of x and y is", dot(x, y))

```
![[Pasted image 20261001213435.png]]


由于点积在科学计算中极其常用，`NumPy` 线性代数库提供了更为高效的封装函数💛 `np.dot()`：

```python
# x = [[1], [-2], [-5]]
# y = [[4], [3], [-1]]
# x = np.array([[1], [-2], [-5]])
# y = np.array([[4], [3], [-1]])

x = [1, -2, -5]
y = [4, 3, -1]

print("np.dot(x,y) function returns dot product of x and y:", np.dot(x, y)) 

```
![[Pasted image 20261001213454.png]]
解释
    ![[Pasted image 20261001215024.png]]
    ![[Pasted image 20261001215051.png]]
    所以dot 应该理解为 矩阵乘法 的函数


注意，即使我们==传入的是原生 Python 列表而非 `NumPy` 数组，`np.dot()` 依然能够顺畅执行==。不过在 Python 中，💛还有专门用于矩阵与向量乘法的专用操作符 `@`，该操作符则==严格要求操作数必须是 `NumPy` 数组对象==。运行以下单元格即可看到两者的区别：

```python
print("This line output is a dot product of x and y: ", np.array(x) @ np.array(y))
print("\nThis line output is an error:")
try:
    print(x @ y)
except TypeError as err:
    print(err)

```
![[Pasted image 20261001215224.png]]

鉴于 `np.dot()` 与 `@` 都是工程实践中的高频用法，最佳实践是==将向量统一声明为 `NumPy` 数组，以规避潜在的类型错误。==我们不妨将 $x$ 和 $y$ 正式转换为 `NumPy` 数组：

```python
x = np.array(x)
y = np.array(y)

```

### 2.3 - 向量化形式下的计算速度

在 机器学习（Machine Learning） 的实际应用中，点积操作往往需要处理包含数百甚至数千维分量的超大向量 (常被称为 **高维向量 (High dimensional vectors)**)。面对海量数据集，即使在算力强大的计算集群上，模型训练也常常耗费数小时乃至数天之久。因此，底层计算的速度对于模型的训练迭代与线上推理部署至关重要。

理解基于传统循环的实现方式与基于向量化（Vectorized）底层实现的性能差距，是非常核心的一课。
在显式循环模式下，计算指令只能按部就班、串行地逐项处理；
而在向量化计算模式下，底层硬件能够利用现代处理器指令集实现高度并行化计算。
前面我们自己编写的 `dot()` 函数就是典型的循环实现，==而 `np.dot()` 和 `@` 则是向量化的高效代表。==

让我们做一个小实验，直观感受一下两者的性能悬殊。首先生成两个长度均为 $1,000,000$ 的大型随机向量 $a$ 和 $b$：

```python
a = np.random.rand(1000000)
b = np.random.rand(1000000)

```

借助 `time.time()` 记录时间戳，测试我们自己编写的循环版本 `dot(x,y)` 需要耗费多少毫秒：

```python
import time

tic = time.time()
c = dot(a,b)
toc = time.time()
print("Dot product: ", c)
print ("Time for the loop version:" + str(1000*(toc-tic)) + " ms")

```
![[Pasted image 20261001215427.png]]

随后，测试高度优化的向量化版本性能：

```python
tic = time.time()
c = np.dot(a,b)
toc = time.time()
print("Dot product: ", c)
print ("Time for the vectorized version, np.dot() function: " + str(1000*(toc-tic)) + " ms")

```
![[Pasted image 20261001215452.png]]
减少好多!!!!

```python
tic = time.time()
c = a @ b
toc = time.time()
print("Dot product: ", c)
print ("Time for the vectorized version, @ function: " + str(1000*(toc-tic)) + " ms")

```
![[Pasted image 20261001215516.png]]

测试结果显而易见：向量化带来的吞吐提升和算力加速是极其惊人的！

### 2.4 - 点积的几何定义

在欧几里得空间（Euclidean Space）中，欧氏向量同时具有**大小**与**方向**属性。从几何角度来看，两个向量 $x$ 与 $y$ 的点积可以通过模长与夹角定义：

$$x\cdot y = \lvert x\rvert \lvert y\rvert \cos(\theta),\tag{2}$$

其中 $\theta$ 表示两向量之间的空间夹角：
![[Pasted image 20261001215612.png]]

这个几何性质为我们==判断两个向量是否垂直（正交，Orthogonal）==提供了非常优雅的途径。若向量 $x$ 与 $y$ 相互正交 (即夹角恰好为 $90^{\circ}$)，由于 $\cos(90^{\circ})=0$，这就必然推导出：**任意两个相互正交的向量，其点积必然恒等于 $0$**。我们不妨选取两个已知相互正交的标准基向量 $i$ 与 $j$ 进行实测验证：

```python
i = np.array([1, 0, 0])
j = np.array([0, 1, 0])
print("The dot product of i and j is", dot(i, j))

```
![[Pasted image 20261001215621.png]]

### 2.5 - 点积的应用：向量相似度

点积的几何定义在现代工程中有着极为广泛的应用，其中最经典的一个场景就是评估 **向量相似度 (Vector similarity)**。
在 自然语言处理（Natural Language Processing, NLP） 领域，词汇表中的每个词语或短语都会==被映射为一个高维实数向量== (即词嵌入)。
此时，两个向量之间夹角的余弦值，便可以用来量化它们在语义上的相似度：当两个向量指向完全相同时，余弦相似度达到最大值 1；随着夹角逐渐张开，相似度逐步衰减。

通过将公式 $(2)$ 简单移项，我们就能得出通过点积求解向量间夹角余弦的公式：
$\cos(\theta)=\frac{x \cdot y}{\lvert x\rvert \lvert y\rvert}.\tag{3}$

==若计算结果为 0，说明两向量正交，语义完全无关（相似度为 0）==；
结果最大时 代表两向量完全同向（语义高度一致）；  就是==夹角为0还同方向==
而结果取得极小负值时，则代表两向量在几何空间中背道而驰。   就是==夹角为0但反方向==

这里之所以引入向量相似度的案例，是为了帮助大家建立线性代数与 机器学习（Machine Learning） 前沿应用之间的直观联系。在本门课程中我们暂不深入探讨其完整的工程落地，感兴趣的同学可以在后续的自然语言处理（NLP）专项课程中看到更加详尽的实战项目。

祝贺你，本节实验到此顺利完成！
