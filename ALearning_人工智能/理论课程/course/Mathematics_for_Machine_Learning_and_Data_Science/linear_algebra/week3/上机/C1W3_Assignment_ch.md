___
# 线性变换（Linear Transformations）与神经网络（Neural Networks）

___
- [[#1 - 导论|1 - 导论]]
	- [[#1 - 导论#1.1 - 变换|1.1 - 变换]]
	- [[#1 - 导论#1.2 - 线性变换|1.2 - 线性变换]]
	- [[#1 - 导论#1.3 - 用矩阵乘法定义的变换|1.3 - 用矩阵乘法定义的变换]]
- [[#2 - 平面上的标准变换|2 - 平面上的标准变换]]
	- [[#2 - 平面上的标准变换#2.1 - 水平缩放（拉伸）|2.1 - 水平缩放（拉伸）]]
	- [[#2 - 平面上的标准变换#2.2 - 示例 2：沿 y 轴（垂直轴）镜像翻转|2.2 - 示例 2：沿 y 轴（垂直轴）镜像翻转]]
	- [[#2 - 平面上的标准变换#2.3 标量缩放|2.3 标量缩放]]
	- [[#2 - 平面上的标准变换#练习 1|练习 1]]
	- [[#2 - 平面上的标准变换#2.4 水平错切变换|2.4 水平错切变换]]
	- [[#2 - 平面上的标准变换#练习 2|练习 2]]
	- [[#2 - 平面上的标准变换#2.5 旋转变换|2.5 旋转变换]]
	- [[#2 - 平面上的标准变换#练习 3|练习 3]]
	- [[#2 - 平面上的标准变换#练习 4|练习 4]]
- [[#3 - 神经网络（Neural Networks）|3 - 神经网络（Neural Networks）]]
	- [[#3 - 神经网络（Neural Networks）#3.1 - 线性回归|3.1 - 线性回归]]
	- [[#3 - 神经网络（Neural Networks）#3.2 - 单感知机双输入节点的神经网络模型|3.2 - 单感知机双输入节点的神经网络模型]]
	- [[#3 - 神经网络（Neural Networks）#3.3 神经网络的参数|3.3 神经网络的参数]]
	- [[#3 - 神经网络（Neural Networks）#3.4 前向传播|3.4 前向传播]]
	- [[#3 - 神经网络（Neural Networks）#练习 5|练习 5]]
	- [[#3 - 神经网络（Neural Networks）#3.5 定义损失函数|3.5 定义损失函数]]
	- [[#3 - 神经网络（Neural Networks）#练习 6|练习 6]]
	- [[#3 - 神经网络（Neural Networks）#3.6 - 训练神经网络|3.6 - 训练神经网络]]
- [[#4 - 开启你的模型预测！|4 - 开启你的模型预测！]]
	- [[#4 - 开启你的模型预测！#练习 7|练习 7]]

___

欢迎来到线性代数（Linear Algebra）课程的第三周作业！

本作业分为两个相对独立的部分，深入剖析线性变换与神经网络的核心基石。在第一部分中，我们将通过编写函数构造对应的变换矩阵，亲自动手实现图形的拉伸、错切以及旋转等线性变换操作。第二部分则转向神经网络，带你一步步搭建一个由双输入节点和单个感知机构成的极简架构，并实现完整的前向传播（Forward Propagation）流程。通过将这些基础组件层层拆解，本作业旨在生动展现线性代数在向量几何变换与现代神经网络底层运算中的核心纽带作用。

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils

```

```python
import w3_unittest

```

## 1 - 导论

### 1.1 - 变换

(之前lab有这部分内容,这里复习一下)

**变换（Transformation）** 本质上是两个向量空间之间的映射函数，并且它完好保留了空间底层的（线性）代数结构。我们通常用英文字母 $T$ 来表示某个具体变换。如果要指明变换前后输入向量与输出向量所在的几何空间（例如二维空间 $\mathbb{R}^2$ 与三维空间 $\mathbb{R}^3$），可以记作 $T: \mathbb{R}^2 \rightarrow \mathbb{R}^3$。当二维向量 $v \in \mathbb{R}^2$ 经由变换 $T$ 映射为三维向量 $w\in\mathbb{R}^3$ 时，数学上记作 $T(v)=w$，口语上常读作“ *T 作用于 v 等于 w* ”，或者称“ *向量 w 是向量 v 在变换 T 作用下的**像（Image）*** ”。

下面的 Python 函数实现了一个将二维向量映射为三维向量的具体变换 $T: \mathbb{R}^2 \rightarrow \mathbb{R}^3$，其对应的代数表达式如下：

$$T\begin{pmatrix}           \begin{bmatrix}            v_1 \\                       v_2           \end{bmatrix}\end{pmatrix}=           \begin{bmatrix}            3v_1 \\            0 \\            -2v_2           \end{bmatrix}           \tag{1}$$

```python
def T(v):
    w = np.zeros((3,1))
    w[0,0] = 3*v[0,0]
    w[2,0] = -2*v[1,0]
    
    return w

v = np.array([[3], [5]])
w = T(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

```
![[Pasted image 20261002220143.png]]

### 1.2 - 线性变换

对于任意标量（Scalar）$k$ 以及任意输入向量 $u$ 和 $v$，如果一个变换 $T$ 同时严格满足以下两条性质，那么它就被称为 **线性的（Linear）** ：

1. $T(kv)=kT(v)$，
2. $T(u+v)=T(u)+T(v)$。

在上一小节的例子中，变换 $T$ 便是一个严格意义上的线性变换：

$$T (kv) =           T \begin{pmatrix}\begin{bmatrix}           kv_1 \\           kv_2           \end{bmatrix}\end{pmatrix} =            \begin{bmatrix}            3kv_1 \\            0 \\            -2kv_2           \end{bmatrix} =           k\begin{bmatrix}            3v_1 \\            0 \\            -2v_2           \end{bmatrix} =            kT(v),\tag{2}$$

$$T (u+v) =           T \begin{pmatrix}\begin{bmatrix}           u_1 + v_1 \\           u_2 + v_2           \end{bmatrix}\end{pmatrix} =            \begin{bmatrix}            3(u_1+v_1) \\            0 \\            -2(u_2+v_2)           \end{bmatrix} =            \begin{bmatrix}            3u_1 \\            0 \\            -2u_2           \end{bmatrix} +           \begin{bmatrix}            3v_1 \\            0 \\            -2v_2           \end{bmatrix} =            T(u)+T(v).\tag{3}$$

你可以在下方单元格中自行修改标量 $k$ 或向量 $u$ 与 $v$ 的数值，直观验证这两条性质在具体数值代入下是否完全成立。

```python
u = np.array([[1], [-2]])
v = np.array([[2], [4]])

k = 7

print("T(k*v):\n", T(k*v), "\n k*T(v):\n", k*T(v), "\n\n")
print("T(u+v):\n", T(u+v), "\n\n T(u)+T(v):\n", T(u)+T(v))

```
![[Pasted image 20261002220133.png]]

### 1.3 - 用矩阵乘法定义的变换

假设变换 $L: \mathbb{R}^m \rightarrow \mathbb{R}^n$ 可以由矩阵 $A$ 来表达，满足 $L(v)=Av$，即由一个 $n\times m$ 维度的矩阵 $A$ 乘以一个 $m\times 1$ 维度的列向量 $v$，运算后生成一个 $n\times 1$ 维度的全新向量 $w$。

现在不妨思考一下：针对前文给出的具体映射变换 $L: \mathbb{R}^2 \rightarrow \mathbb{R}^3$，矩阵 $A$ 中的每个元素应该填入什么数字呢？

$$L\begin{pmatrix}           \begin{bmatrix}            v_1 \\                       v_2           \end{bmatrix}\end{pmatrix}=           \begin{bmatrix}            3v_1 \\            0 \\            -2v_2           \end{bmatrix}=           \begin{bmatrix}            ? & ? \\            ? & ? \\            ? & ?           \end{bmatrix}           \begin{bmatrix}            v_1 \\            v_2           \end{bmatrix}           \tag{4}$$

为了找到对应数值，我们把变换 $L$ 展开为常规的矩阵相乘形式：

$$L\begin{pmatrix}           \begin{bmatrix}            v_1 \\                       v_2           \end{bmatrix}\end{pmatrix}=           A\begin{bmatrix}            v_1 \\                       v_2           \end{bmatrix}=           \begin{bmatrix}            a_{1,1} & a_{1,2} \\            a_{2,1} & a_{2,2} \\            a_{3,1} & a_{3,2}           \end{bmatrix}           \begin{bmatrix}            v_1 \\                       v_2           \end{bmatrix}=           \begin{bmatrix}            a_{1,1}v_1+a_{1,2}v_2 \\            a_{2,1}v_1+a_{2,2}v_2 \\            a_{3,1}v_1+a_{3,2}v_2 \\           \end{bmatrix}=           \begin{bmatrix}            3v_1 \\            0 \\            -2v_2           \end{bmatrix}\tag{5}$$

对比等式两端，想必你已经看清各个矩阵元素 $a_{i,j}$ 该如何取值才能让公式 (5) 成立了。运行下方代码单元格，核对你的答案吧：

```python
def L(v):
    A = np.array([[3,0], [0,0], [0,-2]])
    print("Transformation matrix:\n", A, "\n")
    w = A @ v
    
    return w

v = np.array([[3], [5]])
w = L(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

```
![[Pasted image 20261002220200.png]]

## 2 - 平面上的标准变换

正如第 1 节所讲，任意二维平面线性变换 $L: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ 都可以等价表示为一个 $2 \times 2$ 阶矩阵与平面坐标向量 $v\in\mathbb{R}^2$ 的乘积。注意到目前为止，我们都只是随便代入了一个测试向量 $v\in\mathbb{R}^2$ (例如 $v=\begin{bmatrix}3 \\ 5\end{bmatrix}$)。但为了能在几何空间中更加直观地“看穿”一个变换的本质，挑选基准向量时大有讲究。

绝佳的选择莫过于标准基（Standard Basis）向量 $e_1=\begin{bmatrix}1 \\ 0\end{bmatrix}$ 与 $e_2=\begin{bmatrix}0 \\ 1\end{bmatrix}$。我们把线性变换 $L$ 分别施加在这两个标准基向量上：$L(e_1)=Ae_1$ 与 $L(e_2)=Ae_2$。如果把基向量组 $\{e_1, e_2\}$ 按列拼接成一个矩阵，并执行矩阵乘法：

$$A\begin{bmatrix}e_1 & e_2\end{bmatrix}=\begin{bmatrix}Ae_1 & Ae_2\end{bmatrix}=\begin{bmatrix}L(e_1) & L(e_2)\end{bmatrix},\tag{3}$$

你会立刻发现，$\begin{bmatrix}e_1 & e_2\end{bmatrix}=\begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}$ 恰好就是二维单位矩阵（Identity Matrix）。由此可得 $A\begin{bmatrix}e_1 & e_2\end{bmatrix} = AI=A$，即：

$$A=\begin{bmatrix}L(e_1) & L(e_2)\end{bmatrix}.\tag{4}$$

这一结论极为重要：**变换矩阵的每一列，其实正是各个标准基向量在变换后所落到的新位置（像）**！
选取 {$e_1, e_2$} 作为基准，为我们以图形化方式直观呈现线性变换 $L$ 提供了极大便利（下文将给出具体演示）。

接下来我们将加载一张具体的图像，把理论付诸实践。==图像==在本质上无非就是==平面坐标系内密集排列的点集==，因此完全==可以将其当成一组二维向量集合==来处理——我们可以随心所欲地对它进行拉伸、旋转等各种空间几何变换！

```python
img = np.loadtxt('data/image.txt')
print('Shape: ',img.shape)
print(img)

```
![[Pasted image 20261002220539.png]]
本来以为shape,会呈现出图像宽高的信息
为什么只有2行
可能图像变成txt 就是这样的吧
![[Pasted image 20261002221133.png]]

这幅图像在数据结构上表现为一个 $2 \times 329076$ 的矩阵，其中==**每一列都代表二维平面上的一个点向量**==。因此，调用 `img[0]` 便能取出所有点的 $x$ 坐标，调用 `img[1]` 则能取出对应的 $y$ 坐标。现在，我们把这幅原始图像绘制出来：

```python
plt.scatter(img[0], img[1], s = 0.001, color = 'black')

# 这段代码的意思是将图像的像素点在二维平面上绘制出来。
# img[0] 表示所有像素点的 x 坐标，img[1] 表示所有像素点的 y 坐标。
# s=0.001 表示每个点的大小非常小，color="black" 表示点的颜色为黑色。
```
![[Pasted image 20261002221252.png]]
就像是拼豆那样 把图片画出来了

### 2.1 - 水平缩放（拉伸）

水平方向上的线性缩放（本例中设定拉伸系数为 $2$）可以这样定义：它将基向量 $e_1=\begin{bmatrix}1 \\ 0\end{bmatrix}$ 沿水平轴延伸变换为 $\begin{bmatrix}2 \\ 0\end{bmatrix}$，而垂直基向量 $e_2=\begin{bmatrix}0 \\ 1\end{bmatrix}$ 保持完全静止不动。下面的函数 `T_hscaling()` 实现了对单向量的水平拉伸（系数为 2），而辅助函数 `transform_vectors()` 则负责将该变换批量应用到一组向量上（此处为两个基向量）。

```python
def T_hscaling(v):
    A = np.array([[2,0], [0,1]])
    w = A @ v
    
    return w
    
    
def transform_vectors(T, v1, v2):
    V = np.hstack((v1, v2))
    W = T(V)
    
    return W
    
e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

transformation_result_hscaling = transform_vectors(T_hscaling, e1, e2)

print("Original vectors:\n e1= \n", e1, "\n e2=\n", e2, 
      "\n\n Result of the transformation (matrix form):\n", transformation_result_hscaling)

```
![[Pasted image 20261002221331.png]]


为了帮助你直观理解变换带来的几何形变，我们编写了可视化工具函数，将输入向量与变换后的对应向量绘制在一起。底层绘图细节无需过多纠结，直接运行下方单元格即可：

```python
utils.plot_transformation(T_hscaling,e1,e2)

```
![[Pasted image 20261002221359.png]]


现在，让我们把这种变换真正应用到那张树叶点云图像上，看看整体画面的变化：

```python
plt.scatter(img[0], img[1], s=0.001, color="red")
plt.scatter(T_hscaling(img)[0], T_hscaling(img)[1], s=0.001, color="grey")
#T_hscaling(img)[0] 意思是传入整个img,得到的输出取[0] 就是0行
```
![[Pasted image 20261002221516.png]]
(红色的是未做变换的 之所以看上去比上面那张图小 因为图的x,y坐标轴范围不一样)
显而易见，变换后的灰色叶片在水平方向上被明显拉宽了！

### 2.2 - 示例 2：沿 y 轴（垂直轴）镜像翻转

下面定义的函数 `T_reflection_yaxis()` 对应了沿 y 轴进行镜像翻转（Reflection）的线性变换：

```python
def T_reflection_yaxis(v):
    A = np.array([[-1,0], [0,1]])
    w = A @ v
    
    return w
    
e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

transformation_result_reflection_yaxis = transform_vectors(T_reflection_yaxis, e1, e2)

print("Original vectors:\n e1= \n", e1,"\n e2=\n", e2, 
      "\n\n Result of the transformation (matrix form):\n", transformation_result_reflection_yaxis)

```
![[Pasted image 20261002221843.png]]

我们可以将这个镜像反射过程进行图形化渲染：

```python
utils.plot_transformation(T_reflection_yaxis, e1, e2)

```
![[Pasted image 20261002221856.png]]


```python
plt.scatter(img[0], img[1], s=0.001, color="red")
plt.scatter(
    T_reflection_yaxis(img)[0], T_reflection_yaxis(img)[1], s=0.001, color="grey"
)
```
![[Pasted image 20261002221926.png]]

### 2.3 标量缩放

接下来要体验的线性变换是通过一个非零标量进行全向等比例缩放（Stretching）。换句话说，给定一个非零常数 $a \neq 0$，二维平面内的线性变换满足：

$$T(v) = a \cdot v$$

若记平面坐标为 $v = (x,y)$，则该变换将点映射为 $T(v) = T((x,y)) = (ax, ay)$。

### 练习 1

在本练习中，你将编写一个 Python 函数，接收一个非零标量 $a$ 以及待变换的向量 $v$，并在平面上将向量 $v$ **等比例**拉伸 $a$ 倍。

**提示**：在构造对应的变换矩阵时，可以参考前面学过的方法：观察 *特殊* 基向量 $e_1 = (1,0)$ 和 $e_2 = (0,1)$ 在变换后会变成什么样！

```python
# GRADED FUNCTION: T_stretch

def T_stretch(a, v):
    """
    Performs a 2D stretching transformation on a vector v using a stretching factor a.

    Args:
        a (float): The stretching factor.
        v (numpy.array): The vector (or vectors) to be stretched.

    Returns:
        numpy.array: The stretched vector.
    """

    ### START CODE HERE ###
    # Define the transformation matrix
    T = np.array([[a,0], [0,a]])
    
    # Compute the transformation
    w = T @ v
    ### END CODE HERE ###

    return w

```

```python
w3_unittest.test_T_stretch(T_stretch)

```

```python
plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_stretch(2,img)[0], T_stretch(2,img)[1], s = 0.001, color = 'grey')

```
![[Pasted image 20261002222301.png]]

```python
utils.plot_transformation(lambda v: T_stretch(2, v), e1,e2)

```
![[Pasted image 20261002222315.png]]

### 2.4 水平错切变换

带有剪切参数 $m$ 的**水平错切变换**（Horizontal Shear Transformation），会将平面上的点 $(x,y)$ 映射到新位置 $(x + my, y)$。其代数形式定义为：

$$T((x,y)) = (x+my, y)$$

### 练习 2

你需要实现函数 `T_hshear`，它接收错切系数标量 $m$ 和输入向量 $v$，并计算执行上述错切变换后的结果。

**提示**：若想确定对应的变换矩阵，不妨推导一下基向量 $e_1 = (1,0)$ 和 $e_2 = (0,1)$ 在错切后分别变成了什么。

```python
# GRADED FUNCTION: T_hshear

def T_hshear(m, v):
    """
    Performs a 2D horizontal shearing transformation on an array v using a shearing factor m.

    Args:
        m (float): The shearing factor.
        v (np.array): The array to be sheared.

    Returns:
        np.array: The sheared array.
    """

    ### START CODE HERE ###
    # Define the transformation matrix
    T = np.array([[1, m], [0, 1]])  # 好聪明!!!
    
    # Compute the transformation
    w = T @ v
    
    ### END CODE HERE ###
    
    return w

```

```python
w3_unittest.test_T_hshear(T_hshear)

```

```python
plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_hshear(2,img)[0], T_hshear(2,img)[1], s = 0.001, color = 'grey')

```
![[Pasted image 20261002222708.png]]

```python
utils.plot_transformation(lambda v: T_hshear(2, v), e1,e2)

```
![[Pasted image 20261002222719.png]]


### 2.5 旋转变换



若要将平面上的向量**逆时针**旋转 $\theta$ 弧度（Radians），该变换对应的标准旋转矩阵为：

$$M = \begin{bmatrix} \cos \theta & - \sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}$$

### 练习 3

请实现函数 `T_rotation`，它接收以弧度表示的旋转角度 $\theta$ 以及输入向量 $v$，并对向量执行**逆时针方向**的旋转操作。

```python
# GRADED FUNCTION: T_rotation
def T_rotation(theta, v):
    """
    Performs a 2D rotation transformation on an array v using a rotation angle theta.

    Args:
        theta (float): The rotation angle in radians.
        v (np.array): The array to be rotated.

    Returns:
        np.array: The rotated array.
    """
    
    ### START CODE HERE ###
    # Define the transformation matrix
    T = np.array([[None,None], [None,None]])
    
    # Compute the transformation
    w = None @ None
    
    ### END CODE HERE ###
    
    return w

```

```python
w3_unittest.test_T_rotation(T_rotation)

```

```python
plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_rotation(np.pi,img)[0], T_rotation(np.pi,img)[1], s = 0.001, color = 'grey')

```
![[Pasted image 20261002222838.png]]

```python
utils.plot_transformation(lambda v: T_rotation(np.pi, v), e1,e2)

```
![[Pasted image 20261002222848.png]]


### 练习 4

在本节的最后一个练习中，你需要实现一个复合变换函数：
    首先将向量旋转 $\theta$ 弧度，
    紧接着再按比例因子 $a$ 进行拉伸。
牢记若分别用 $T_{\text{stretch}}$ 表示拉伸变换、$T_{\text{rotation}}$ 表示旋转变换，复合函数运算定义如下：

$$T_{\text{rotation and stretch}} (v) = \left(T_{\text{rotation}} \circ T_{\text{stretch}}\right) (v) = T_{\text{rotation}} \left(T_{\text{stretch}} \left(v \right) \right).$$

因此，要一并执行这两项变换，只需将对应的变换矩阵依次相乘即可！

```python
def T_rotation_and_stretch(theta, a, v):
    """
    Performs a combined 2D rotation and stretching transformation on an array v using a rotation angle theta and a stretching factor a.

    Args:
        theta (float): The rotation angle in radians.
        a (float): The stretching factor.
        v (np.array): The array to be transformed.

    Returns:
        np.array: The transformed array.
    """
    ### START CODE HERE ###

    rotation_T = np.array([[None,None], [None,None]])
    stretch_T = np.array([[None,None], [None,None]])

    w = None @ (None @ None)

    ### END CODE HERE ###

    return w


```

```python
w3_unittest.test_T_rotation_and_stretch(T_rotation_and_stretch)

```

```python
plt.scatter(img[0], img[1], s = 0.001, color = 'black') 
plt.scatter(T_rotation_and_stretch(np.pi,2,img)[0], T_rotation_and_stretch(np.pi,2,img)[1], s = 0.001, color = 'grey')

```
![[Pasted image 20261002223014.png]]

```python
utils.plot_transformation(lambda v: T_rotation_and_stretch(np.pi, 2, v), e1,e2)

```
![[Pasted image 20261002223024.png]]

## 3 - 神经网络（Neural Networks）

在作业的这一部分，你将完成：

* 针对线性回归（Linear Regression）任务，从零搭建一个包含两个输入节点和单个感知机（Perceptron）的简易神经网络
* 运用矩阵乘法高效实现模型的前向传播（Forward Propagation）流程

*注*：带有参数更新的反向传播（Backward Propagation）需要掌握微积分（Calculus）的相关求导知识。该内容将在本专项课程的第二门课《微积分》中全面展开。在本次作业中，反向传播以及底层参数自动更新的函数已经提前封装隐藏好了。

### 3.1 - 线性回归

**线性回归（Linear Regression）** 是一种通过线性关系建模标量输出响应（**因变量，Dependent Variable**）与一个或多个解释变量（**自变量，Independent Variables**）之间关联的统计方法。在这里，我们将处理包含 $2$ 个自变量的经典多元线性回归问题。

拥有两个自变量 $x_1, x_2$ 的线性回归方程可以写成：

$$\hat{y} = w_1x_1 + w_2x_2 + b = Wx + b,\tag{6}$$

其中 $Wx$ 代表输入向量 $x = \begin{bmatrix} x_1 & x_2\end{bmatrix}$ 与权重参数向量 $W = \begin{bmatrix} w_1 & w_2\end{bmatrix}$ 之间的点积（Dot Product），标量参数 $b$ 则代表线性方程的截距（偏置，Bias）。

我们建模的核心目标始终不变：==寻找一组“最理想”的参数 $w_1$、$w_2$ 和 $b$，使得模型输出的预测值 $\hat{y}_i$ 与真实观测标签 $y_i$ 之间的误差达到最小。==

我们可以使用神经网络模型优雅地完成这一拟合过程——而矩阵乘法，正是这套网络计算的核心心脏！

### 3.2 - 单感知机双输入节点的神经网络模型

本例中我们仅使用一个感知机神经元，但它具有两个并行的输入节点，结构图如下所示：
![[Pasted image 20261003010047.png]]

对于单个样本数据 $x = \begin{bmatrix} x_1& x_2\end{bmatrix}$，感知机的输出计算可以借由点积写成：

$$z = w_1x_1 + w_2x_2+ b = Wx + b$$

其中权重数值整齐排列在行向量 $W = \begin{bmatrix} w_1 & w_2\end{bmatrix}$ 中，偏置项 $b$ 是一个单独的标量。输出层由单一节点组成，满足恒等映射 $\hat{y}= z$。

在实际工程中，我们将所有训练样本按列排布，==规整为一个尺寸为 ($2 \times m$) 的输入矩阵 $X$。==此时，权重行向量 $W$ ($1 \times 2$) 与输入数据矩阵 $X$ ($2 \times m$) 直接进行矩阵乘法，便能瞬间生成一个包含全部预测结果的 ($1 \times m$) 行向量：

$$WX =  \begin{bmatrix} w_1 & w_2\end{bmatrix}  \begin{bmatrix}  x_1^{(1)} & x_1^{(2)} & \dots & x_1^{(m)} \\  x_2^{(1)} & x_2^{(2)} & \dots & x_2^{(m)} \\ \end{bmatrix} =\begin{bmatrix}  w_1x_1^{(1)} + w_2x_2^{(1)} &  w_1x_1^{(2)} + w_2x_2^{(2)} & \dots &  w_1x_1^{(m)} + w_2x_2^{(m)}\end{bmatrix}.$$

整套模型的矩阵化数学表达可以精简为：

![[Pasted image 20261003010104.png]]

这里标量偏置 $b$ 借助**广播机制（Broadcasting）** 自动扩展为一个 ($1 \times m$) 的行向量进行加和。以上公式正是我们在前向传播计算环节中需要执行的核心运算。

计算完毕后，我们便可将模型输出的整批==预测值向量 $\hat{Y}$ ($1 \times m$) 与真实的真实标签向量 $Y$ 进行系统性比对==。❤️这一评估环节由 **损失函数（Cost Function）** 来量化衡量，它精确衡量了当前模型的预测输出与真实数据之间的契合差距，从而客观评估参数 $w$ 与 $b$ 对该任务的拟合优劣。根据业务场景的不同，可供挑选的损失函数非常丰富；在当前这个基础神经网络中，我们采用经典的均方误差作为损失函数：

$$\mathcal{L}\left(w, b\right)  = \frac{1}{2m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.\tag{5}$$

在模型训练过程中，我们的**终极目标**就是==通过参数优化把损失函数降到最低，进而将所有样本上预测值 $\hat{y}_i$ 与真实值 $y_i$ 的偏差压缩至极限== (公式中除以 $2m$ 纯粹是为了后续梯度计算时数学推导的缩放便利)。

由于最初权重矩阵完全由随机数初始化赋值，在模型未经训练打磨之前，预测表现自然谈不上精准。

接下来的关键步骤就是==设法调整权重与偏置参数==，一步步==压缩损失函数的值==。这一关键过程被称为 **反向传播（Backward Propagation）** ，并且是以迭代形式循环开展的：我们根据误差方向给予参数细微的更新调整，并反复迭代这一过程。

*注*：反向传播的具体求导原理超出了本门课程的范畴——它将在本专项课程的后续篇章中深入拆解。

搭建并训练一个神经网络的 **标准工作流（Methodology）** 通常包含以下四步：

1. 确定神经网络架构 (包括输入层神经元数、隐藏层神经元数等)。
2. 初始化模型各项参数。
3. 循环迭代训练：
    * 执行前向传播 (计算感知机输出)，
    * 执行反向传播 (计算各参数所需调整的误差梯度方向)，
    * 正式更新网络参数。

4. 利用训练好的参数进行预测。

### 3.3 神经网络的参数

我们接下来要操控的简易神经网络总共包含 $3$ 个核心参数：两个权重参数和一个偏置参数。首先通过随机数来初始化赋值这些参数(就是随机搞出w和b的数值)，为优化算法提供一个探索起点。这组参数随后将被整齐存放在一个 Python 字典结构中。

```python
parameters = utils.initialize_parameters(2) #2 表示w向量有两个数,b写死了初始化为0
print(parameters)

```

### 3.4 前向传播

### 练习 5

实现函数 `forward_propagation()`。

**操作指引**：

* 回顾我们模型的前向传播数学表达式：
![[Pasted image 20261003010630.png]]


* 你的代码逻辑应包含：
1. 使用字典取值语句 `parameters[".."]` 从包含参数的字典 "parameters" 中提取出各个参数。
2. 实现前向传播：通过数组 `W` 与 `X` 的矩阵相乘并累加向量 `b` 计算出中间结果 `Z`。最后将预测输出数组 $A$ 赋值为 `Z`。


```python
# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m), where n_x is the dimension input (in our example is 2) and m is the number of training samples
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    Y_hat -- The output of size (1, m)
    """
    # Retrieve each parameter from the dictionary "parameters".
    W = parameters["W"]
    b = parameters["b"]
    
    # Implement Forward Propagation to calculate Z.
    ### START CODE HERE ### (~ 2 lines of code)
    Z = W @ X + b
    Y_hat = Z
    ### END CODE HERE ###
    
    return Y_hat

```

```python
w3_unittest.test_forward_propagation(forward_propagation)

```

### 3.5 定义损失函数

指导该模型优化训练的损失函数为：

$$\mathcal{L}\left(w, b\right)  = \frac{1}{2m}\sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

下面的损失计算实现已提前提供，无需作为作业评分项：

```python
def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares
    
    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)
    
    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y)**2)/(2*m)
    
    return cost

```

### 练习 6

现在万事俱备，我们可以把神经网络装配起来了。下面的函数封装了完整的模型训练全流程，运行结束后它将返回更新收敛后的参数字典，供我们后续进行推理预测。

```python
# GRADED FUNCTION: nn_model

def nn_model(X, Y, num_iterations=1000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (1, number of examples)
    num_iterations -- number of iterations in the loop
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """
    
    n_x = X.shape[0]
    
    # Initialize parameters
    parameters = utils.initialize_parameters(n_x) 
    
    # Loop (指迭代训练次数)
    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (~ 2 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "Y_hat".
        Y_hat = forward_propagation(X,parameters)
        
        # Cost function. Inputs: "Y_hat, Y". Outputs: "cost".
        cost = compute_cost(Y_hat,Y)
        ### END CODE HERE ###
        
        
        # Parameters update.
        parameters = utils.train_nn(parameters, Y_hat, X, Y, learning_rate = 0.001) 
        
        # Print the cost every iteration.
        if print_cost:
            if i%100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

```

```python
w3_unittest.test_nn_model(nn_model)

```

### 3.6 - 训练神经网络



现在我们正式加载示例数据集，让神经网络展开实战训练。

```python
df = pd.read_csv("data/toy_dataset.csv")

```

```python
df.head()
# 意思是显示数据框的前几行内容
```
![[Pasted image 20261003011559.png]]

首先把表格中的数据转换为 NumPy 数组格式，方便传递给我们编写的函数：

```python
X = np.array(df[["x1", "x2"]]).T  # 变成横着的
Y = np.array(df["y"]).reshape(1, -1)  # 把数组重塑成 1 行，列数让 NumPy 自己算出来。

```

运行下方代码块，启动 5000 次训练循环，用拟合出的最优权重更新我们的参数字典：

```python
parameters = nn_model(X,Y, num_iterations = 5000, print_cost= True)

```
![[Pasted image 20261003011905.png]]
## 4 - 开启你的模型预测！

拥有了训练拟合完毕的最优参数后，你就可以借助这套神经网络去预测任意未知数据的值了！ 你只需执行如下一步简单的矩阵计算：

$$Z = W X + b$$

其中所需的矩阵 $W$ 和标量偏置 $b$ 都可以从刚才的参数字典中直接读取。

### 练习 7

最后一步，编写我们的预测器函数 `predict`。它接收训练好的参数字典以及一组新数据点 X，并输出计算好的批量预测值。

```python
# GRADED FUNCTION: predict

def predict(X, parameters):

    W = parameters['W']
    b = parameters['b']

    Z = np.dot(W, X) + b

    return Z

```

```python
y_hat = predict(X,parameters)

```

```python
df['y_hat'] = y_hat[0]

```

现在让我们挑选前 10 条数据样本，将神经网络给出的预测值与真实数据进行直接对照：

```python
for i in range(10):
    print(f"(x1,x2) = ({df.loc[i,'x1']:.2f}, {df.loc[i,'x2']:.2f}): Actual value: {df.loc[i,'y']:.2f}. Predicted value: {df.loc[i,'y_hat']:.2f}")

```
![[Pasted image 20261003011948.png]]


拟合效果相当不错对吧？祝贺你！你已经圆满完成了第三周的所有作业内容！