___
# 线性变换（Linear Transformations）

在本实验中，你将探索线性变换（Linear Transformations）的奥秘，直观观察它们的几何变换效果，并熟练运用矩阵乘法（Matrix Multiplication）来施加各种各样的线性变换。

___
- [[#常用工具包|常用工具包]]
- [[#1 - 变换|1 - 变换]]
- [[#2 - 线性变换|2 - 线性变换]]
- [[#3 - 用矩阵乘法定义的变换|3 - 用矩阵乘法定义的变换]]
- [[#4 - 平面上的标准变换|4 - 平面上的标准变换]]
	- [[#4 - 平面上的标准变换#4.1 - 示例 1：水平缩放（拉伸）|4.1 - 示例 1：水平缩放（拉伸）]]
	- [[#4 - 平面上的标准变换#4.2 - 示例 2：沿 y 轴（垂直轴）对称翻转|4.2 - 示例 2：沿 y 轴（垂直轴）对称翻转]]
- [[#5 - 线性变换的应用：计算机图形学|5 - 线性变换的应用：计算机图形学]]

___

## 常用工具包

运行以下代码单元格以加载所需的工具库。

```python
import numpy as np
# 用于图像变换的 OpenCV 库
import cv2

```

## 1 - 变换

**变换（Transformation）** 本质上是一种从一个向量空间到另一个向量空间的映射函数，它能够很好地保持各个向量空间底层的（线性）代数结构。我们通常用一个特定符号来==指代某项变换==，比如 $T$。为了明确输入向量与输出向量所在的几何空间（例如 $\mathbb{R}^2$ 与 $\mathbb{R}^3$），可以记作 $T: \mathbb{R}^2 \rightarrow \mathbb{R}^3$。如果通过变换 $T$ 将二维向量 $v \in \mathbb{R}^2$ 变换为三维向量 $w\in\mathbb{R}^3$，通常记作 $T(v)=w$，读作==“ *T 作用于 v 等于 w* ”==或“ *向量 w 是向量 v 在变换 T 下的**像（Image）*** ”。

下面的 Python 函数对应了一个将二维空间映射到三维空间的变换 $T: \mathbb{R}^2 \rightarrow \mathbb{R}^3$，其数学公式定义如下：

$$T\begin{pmatrix}           \begin{bmatrix}            v_1 \\                       v_2           \end{bmatrix}\end{pmatrix}=           \begin{bmatrix}            3v_1 \\            0 \\            -2v_2           \end{bmatrix}           \tag{1}$$

```python
def T(v):
    w = np.zeros((3, 1))
    w[0, 0] = 3 * v[0, 0]
    w[2, 0] = -2 * v[1, 0]

    return w
#非常朴实无华的写法

v = np.array([[3], [5]])
w = T(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

```

```
Original vector:
 [[3]
 [5]] 

 Result of the transformation:
 [[  9.]
 [  0.]
 [-10.]]

```

## 2 - 线性变换

❤️如果一个变换 $T$ 对于任意标量（Scalar）$k$ 以及任意输入向量 $u$ 和 $v$，都严格满足以下两条性质，那么该变换就被称为 **线性的（Linear）** ：

1. $T(kv)=kT(v)$，
2. $T(u+v)=T(u)+T(v)$。
    联想 就是维度没有变化
        这边这个T改变了维度 仍然是线性的 因为多出来那个维度永远是0,相当于向量还在二维平面上动
在前面的例子中，$T$ 就是一个典型的线性变换：

$$T (kv) =           T \begin{pmatrix}\begin{bmatrix}           kv_1 \\           kv_2           \end{bmatrix}\end{pmatrix} =            \begin{bmatrix}            3kv_1 \\            0 \\            -2kv_2           \end{bmatrix} =           k\begin{bmatrix}            3v_1 \\            0 \\            -2v_2           \end{bmatrix} =            kT(v),\tag{2}$$

$$T (u+v) =           T \begin{pmatrix}\begin{bmatrix}           u_1 + v_1 \\           u_2 + v_2           \end{bmatrix}\end{pmatrix} =            \begin{bmatrix}            3(u_1+v_1) \\            0 \\            -2(u_2+v_2)           \end{bmatrix} =            \begin{bmatrix}            3u_1 \\            0 \\            -2u_2           \end{bmatrix} +           \begin{bmatrix}            3v_1 \\            0 \\            -2v_2           \end{bmatrix} =            T(u)+T(v).\tag{3}$$

你可以在下方单元格中随意修改标量 $k$ 或向量 $u$ 与 $v$ 的数值，亲自动手验证这些性质在具体数值下是否完全成立。

```python
u = np.array([[1], [-2]])
v = np.array([[2], [4]])

k = 7

print("T(k*v):\n", T(k*v), "\n k*T(v):\n", k*T(v), "\n\n")
print("T(u+v):\n", T(u+v), "\n T(u)+T(v):\n", T(u)+T(v))

```

```
T(k*v):
 [[ 42.]
 [  0.]
 [-56.]] 
 k*T(v):
 [[ 42.]
 [  0.]
 [-56.]] 


T(u+v):
 [[ 9.]
 [ 0.]
 [-4.]] 
 T(u)+T(v):
 [[ 9.]
 [ 0.]
 [-4.]]

```

日常生活中常见的==旋转（Rotation）、反射翻转（Reflection）、缩放（Scaling/Dilation）等几何操作，本质上都是线性变换。==本实验将带你逐一探索其中的代表性变换。

## 3 - 用矩阵乘法定义的变换

假设==变换 $L: \mathbb{R}^m \rightarrow \mathbb{R}^n$ 是通过矩阵 $A$ 来定义的==，即满足 $L(v)=Av$。这里的运算是一个 $n\times m$ 的矩阵 $A$ 乘以一个 $m\times 1$ 的向量 $v$，最终生成一个 $n\times 1$ 的全新向量 $w$。

现在不妨猜猜看：对于前面提到的映射变换 $L: \mathbb{R}^2 \rightarrow \mathbb{R}^3$，矩阵 $A$ 中的元素应该分别取什么值呢？

$$L\begin{pmatrix}           \begin{bmatrix}            v_1 \\                       v_2           \end{bmatrix}\end{pmatrix}=           \begin{bmatrix}            3v_1 \\            0 \\            -2v_2           \end{bmatrix}=           \begin{bmatrix}            ? & ? \\            ? & ? \\            ? & ?           \end{bmatrix}           \begin{bmatrix}            v_1 \\            v_2           \end{bmatrix}           \tag{4}$$

为了找出答案，我们把变换 $L$ 写成 $Av$ 的形式，并按照规则展开矩阵乘法：

$$L\begin{pmatrix}           \begin{bmatrix}            v_1 \\                       v_2           \end{bmatrix}\end{pmatrix}=           A\begin{bmatrix}            v_1 \\                       v_2           \end{bmatrix}=           \begin{bmatrix}            a_{1,1} & a_{1,2} \\            a_{2,1} & a_{2,2} \\            a_{3,1} & a_{3,2}           \end{bmatrix}           \begin{bmatrix}            v_1 \\                       v_2           \end{bmatrix}=           \begin{bmatrix}            a_{1,1}v_1+a_{1,2}v_2 \\            a_{2,1}v_1+a_{2,2}v_2 \\            a_{3,1}v_1+a_{3,2}v_2 \\           \end{bmatrix}=           \begin{bmatrix}            3v_1 \\            0 \\            -2v_2           \end{bmatrix}\tag{5}$$

对比两边，你是否一眼就能看出矩阵 $A$ 的各个元素 $a_{i,j}$ 该取何值才能让等式 (5) 完美成立？ 下面的代码单元格揭晓了答案：

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

```
Transformation matrix:
 [[ 3  0]
 [ 0  0]
 [ 0 -2]] 

Original vector:
 [[3]
 [5]] 

 Result of the transformation:
 [[  9]
 [  0]
 [-10]]

```

==任何线性变换都可以用矩阵乘法来执行==；反过来，==当我们面对一个矩阵乘法运算时，也可以很自然地将其看作一种具体的几何线性变换。==这意味着矩阵与线性变换之间存在着完美的对应关系——这正是线性代数（Linear Algebra）与几何空间变换之间最为美妙的核心纽带。

## 4 - 平面上的标准变换

如第 3 节所述，二维平面上的线性变换 $L: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ 可以表示为一个 $2 \times 2$ 的矩阵乘以平面坐标向量 $v\in\mathbb{R}^2$。注意到在前面的测试中，我们随意选取了一个普通的向量 $v\in\mathbb{R}^2$ (比如 $v=\begin{bmatrix}3 \\ 5\end{bmatrix}$)。但若想直观把握一个变换在整个 $\mathbb{R}^2$ 空间中究竟扮演了什么几何角色，==有目的地挑选基准测试向量==往往更加一目了然。

最经典且强大的选择莫过于==**标准基（Standard Basis）**==向量 $e_1=\begin{bmatrix}1 \\ 0\end{bmatrix}$ 和 $e_2=\begin{bmatrix}0 \\ 1\end{bmatrix}$。我们把线性变换 $L$ 分别施加在 $e_1$ 与 $e_2$ 上：$L(e_1)=Ae_1$ 以及 $L(e_2)=Ae_2$。如果把这两个基向量 $\{e_1, e_2\}$ 拼成一个矩阵的列，再进行矩阵乘法：

$$A\begin{bmatrix}e_1 & e_2\end{bmatrix}=\begin{bmatrix}Ae_1 & Ae_2\end{bmatrix}=\begin{bmatrix}L(e_1) & L(e_2)\end{bmatrix},\tag{3}$$

你会发现 $\begin{bmatrix}e_1 & e_2\end{bmatrix}=\begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}$ 恰好就是**单位矩阵（Identity Matrix）**。因此，$A\begin{bmatrix}e_1 & e_2\end{bmatrix} = AI=A$，即：

$$A=\begin{bmatrix}L(e_1) & L(e_2)\end{bmatrix}.\tag{4}$$

这个公式告诉我们：==线性变换对应的矩阵 的各列，其实就是各个标准基向量在经历变换后所映射成的“像”！==
    (看3b1b的视频应该很好理解)

选用基向量组 {$e_1, e_2$}，让我们可以将抽象的线性变换 $L$ 极其直观地在二维平面上绘制出来（见下文的示例）。

### 4.1 - 示例 1：水平缩放（拉伸）

水平方向缩放（本例中缩放系数为 $2$）可以定义为：将基向量 $e_1=\begin{bmatrix}1 \\ 0\end{bmatrix}$ 拉伸为新向量 $\begin{bmatrix}2 \\ 0\end{bmatrix}$，同时保持基向量 $e_2=\begin{bmatrix}0 \\ 1\end{bmatrix}$ 垂直方向不变。下面的 `T_hscaling()` 函数实现了水平缩放（倍数为 2），而辅助函数 `transform_vectors()` 则负责将定义好的变换批量应用到一组向量上（这里为两个基向量）。

```python
def T_hscaling(v):
    A = np.array([[2,0], [0,1]])
    w = A @ v
    
    return w
    
#从这里也可以看出 python里面可以自由把函数传进函数并调用   
def transform_vectors(T, v1, v2):
    V = np.hstack((v1, v2)) #以此实现向量批量变换
    W = T(V)
    
    return W
    
e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

transformation_result_hscaling = transform_vectors(T_hscaling, e1, e2)

print("Original vectors:\n e1= \n", e1, "\n e2=\n", e2, 
      "\n\n Result of the transformation (matrix form):\n", transformation_result_hscaling)

```

```
Original matrix V:
 [[1 0]
 [0 1]]
Original vectors:
 e1= 
 [[1]
 [0]] 
 e2=
 [[0]
 [1]] 

 Result of the transformation (matrix form):
 [[2 0]
 [0 1]]
```

我们可以通过绘制对比图，把输入向量及其变换后的效果直观呈现出来。如果你一时看不懂下面单元格里的绘图代码也不必担心，现阶段这并不是必须掌握的核心知识点。

```python
import matplotlib.pyplot as plt


def plot_transformation(T, e1, e2):
    color_original = "#129cab"
    color_transformed = "#cc8933"

    # 把列向量展平成一维，方便绘图
    e1 = np.asarray(e1).flatten()
    e2 = np.asarray(e2).flatten()

    _, ax = plt.subplots(figsize=(7, 7))
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xticks(np.arange(-5, 5))
    ax.set_yticks(np.arange(-5, 5))

    plt.axis([-5, 5, -5, 5])
    plt.quiver(
        [0, 0],
        [0, 0],
        [e1[0], e2[0]],
        [e1[1], e2[1]],
        color=color_original,
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    plt.plot([0, e2[0], e1[0], e1[0]], [0, e2[1], e2[1], e1[1]], color=color_original)

    e1_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(e1)])
    ax.text(
        e1[0] - 0.2 + e1_sgn[0],
        e1[1] - 0.2 + e1_sgn[1],
        "$e_1$",
        fontsize=14,
        color=color_original,
    )
    e2_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(e2)])
    ax.text(
        e2[0] - 0.2 + e2_sgn[0],
        e2[1] - 0.2 + e2_sgn[1],
        "$e_2$",
        fontsize=14,
        color=color_original,
    )

    e1_transformed = np.asarray(T(e1)).flatten()
    e2_transformed = np.asarray(T(e2)).flatten()

    plt.quiver(
        [0, 0],
        [0, 0],
        [e1_transformed[0], e2_transformed[0]],
        [e1_transformed[1], e2_transformed[1]],
        color=color_transformed,
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    plt.plot(
        [
            0,
            e2_transformed[0],
            e1_transformed[0] + e2_transformed[0],
            e1_transformed[0],
        ],
        [
            0,
            e2_transformed[1],
            e1_transformed[1] + e2_transformed[1],
            e1_transformed[1],
        ],
        color=color_transformed,
    )

    e1_transformed_sgn = 0.4 * np.array(
        [[1] if i == 0 else [i] for i in np.sign(e1_transformed)]
    )
    ax.text(
        e1_transformed[0] - 0.2 + e1_transformed_sgn[0],
        e1_transformed[1] - 0.2 + e1_transformed_sgn[1],
        "$T(e_1)$",
        fontsize=14,
        color=color_transformed,
    )
    e2_transformed_sgn = 0.4 * np.array(
        [[1] if i == 0 else [i] for i in np.sign(e2_transformed)]
    )
    ax.text(
        e2_transformed[0] - 0.2 + e2_transformed_sgn[0],
        e2_transformed[1] - 0.2 + e2_transformed_sgn[1],
        "$T(e_2)$",
        fontsize=14,
        color=color_transformed,
    )

    plt.gca().set_aspect("equal")
    plt.show()


e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])
plot_transformation(T_hscaling, e1, e2)

```
![[Pasted image 20261002210959.png]]
从图中可以清晰地看出，经历线性变换后，原本的多边形在水平方向上被明显拉伸展平了。

### 4.2 - 示例 2：沿 y 轴（垂直轴）对称翻转

下方定义的函数 `T_reflection_yaxis()` 对应了沿 y 轴进行镜像反射翻转的变换：

```python
def T_reflection_yaxis(v):
    A = np.array([[-1,0], [0,1]])
    w = A @ v
    
    return w
    
e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

# transform_vectors函数 是总的线性变换函数(有拼接向量操作的那个)
# transform_vectors函数需要传入一个函数 定义不同的线性变换,并施加
transformation_result_reflection_yaxis = transform_vectors(T_reflection_yaxis, e1, e2)

print("Original vectors:\n e1= \n", e1,"\n e2=\n", e2, 
      "\n\n Result of the transformation (matrix form):\n", transformation_result_reflection_yaxis)

```
![[Pasted image 20261002211042.png]]

你可以将其几何效果可视化呈现出来：

```python
plot_transformation(T_reflection_yaxis, e1, e2)

```
![[Pasted image 20261002211340.png]]


二维空间中还有许多经典线性变换等待你去探索。而掌握了上述核心思路后，你已经具备了亲自构建并可视化任意变换的基础能力。

## 5 - 线性变换的应用：计算机图形学

计算机图形学（Computer Graphics）在绘制场景时大量采用基础几何图形。这些图元（如三角形、四边形）完全由其关键顶点（Vertices/Corners）的坐标确定。==通过缩放、镜像、旋转、错切（Shearing）等线性变换手段，我们可以轻松地从简单基础形状衍生出极度复杂的图形构造==，从而高效地操纵和驱动几何模型。

渲染引擎在绘制庞大画面时，往往需要并行处理数以百万计的顶点数据。==将各种变换统一表示为矩阵乘法==，最大的优势在于能够将==**多步连续变换“打包合并”**==——只需将各个变换矩阵依次相乘，即可融合成单一矩阵一次性生效。更令人振奋的是，现代计算机专用的硬件加速芯片，如图形处理器（Graphics Processing Units, GPUs），正是专门为海量矩阵运算的高并发、超高速吞吐而量身定制的。

可以说，==矩阵乘法与线性变换正是现代三维渲染与图形学在规模化运算中立于不败之地的“超级力量”！==

下面就是一个==利用线性变换大幅简化图像生成工作量==的经典案例：

这株巴恩斯利蕨类植物（Barnsley Fern）中的所有子叶片形态高度自相似，每一片都可以简单地通过母叶片经由一次线性变换直接生成。
![[Pasted image 20261002211518.png]]
接下来我们看一个对叶片图像连续施加两次几何变换的简易例子。在图像处理领域，我们可以直接借助强大的 `OpenCV` 库。首先加载并展示原始图像：

```python
img = cv2.imread('images/leaf_original.png', 0)
plt.imshow(img)

```
![[Pasted image 20261002211813.png]]


当然，这只是一张极其简单的叶片素材（并非专业美术设计中的复杂案例），但足以帮助你清晰理解如何将多个连续变换串联起来。我们先将图像顺时针旋转 90 度，接着对其施加错切变换（Shear Transformation，其变形效果示意如下）：

第一步，旋转图像：

```python
image_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

plt.imshow(image_rotated)

```
![[Pasted image 20261002211843.png]]

第二步，施加错切变换，得到如下变形结果：

```python
rows, cols = image_rotated.shape
print(rows, cols)

# 3 by 3 matrix as it is required for the OpenCV library, don't worry about the details of it for now.
M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
image_rotated_sheared = cv2.warpPerspective(image_rotated, M, (int(cols), int(rows)))

# 这段代码的意思是对已经旋转过的图像进行剪切变换。
# M 是一个 3x3 的仿射变换矩阵，其中 [1, 0.5, 0] 表示在 x 方向上进行剪切，0.5 是剪切系数。
# cv2.warpPerspective 函数根据矩阵 M 对图像进行透视变换，得到剪切后的图像。
# row 和 col 分别表示图像的行数和列数，也就是图像的高度和宽度。

plt.imshow(image_rotated_sheared)

```
![[Pasted image 20261002212040.png]]

假如我们将这两次变换的顺序调换一下呢？你觉得最后生成的画面会一样吗？ 运行下面的代码验证你的直觉：

```python
image_sheared = cv2.warpPerspective(img, M, (int(cols), int(rows)))
image_sheared_rotated = cv2.rotate(image_sheared, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(image_sheared_rotated)

```
![[Pasted image 20261002212058.png]]

对比最后两张图片，可以非常直观地看出它们的呈现截然不同。这是因为==线性变换在底层严格等价于矩阵乘法==；若连续施加变换矩阵 $A$ 和 $B$，计算本质上是在对向量 $v$ 执行复合运算 $B(Av)=(BA)v$。千万别忘了：==在普遍情况下，矩阵乘法是 **不满足交换律的** （即通常 $BA\neq AB$）！==让我们在代码里亲手验证这一事实：定义分别代表 90 度顺时针旋转与沿 x 轴错切的两个变换矩阵：

```python
M_rotation_90_clockwise = np.array([[0, 1], [-1, 0]])
M_shear_x = np.array([[1, 0.5], [0, 1]])

print("90 degrees clockwise rotation matrix:\n", M_rotation_90_clockwise)
print("Matrix for the shear along x-axis:\n", M_shear_x)

```
![[Pasted image 20261002215208.png]]

现在来观察它们的乘积 `M_rotation_90_clockwise @ M_shear_x` 与 `M_shear_x @ M_rotation_90_clockwise` 是否各不相同：

```python
print("M_rotation_90_clockwise by M_shear_x:\n", M_rotation_90_clockwise @ M_shear_x)
print("M_shear_x by M_rotation_90_clockwise:\n", M_shear_x @ M_rotation_90_clockwise)

```
![[Pasted image 20261002215234.png]]

这个简单生动的例子清晰地提醒我们：在实际工程落地时，深入理解背后的数学对象与其内在性质是多么重要。

祝贺你，圆满完成了本实验的学习！

