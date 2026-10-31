# 选修实验：Python、NumPy 与向量化 (Vectorization) 
本实验将简要介绍本课程中用到的一些科学计算 (scientific computing) 工具，重点讲解 NumPy 科学计算工具包及其在 Python 中的使用方法。

___



___


```python
import numpy as np    # it is an unofficial standard to use np for numpy
import time
```

&lt;a name=&quot;toc_40015_1.1&quot;&gt;&lt;/a&gt;
## 1.1 实验目标
在本次实验中，你将：
- 系统回顾第 1 门课程 (Course 1) 中会用到的 NumPy 与 Python 核心特性

&lt;a name=&quot;toc_40015_1.2&quot;&gt;&lt;/a&gt;
## 1.2 实用参考资料
- 包含基础入门指南的 NumPy 官方文档：NumPy.org
- 一个颇具挑战性的进阶特性专题：NumPy 广播机制 (NumPy Broadcasting) 


&lt;a name=&quot;toc_40015_2&quot;&gt;&lt;/a&gt;
# 2 Python 与 NumPy &lt;a name=&#39;Python and NumPy&#39;&gt;&lt;/a&gt;
Python 是我们在本课程中全程使用的编程语言，它内置了一系列基础的数值数据类型和算术运算。而 NumPy 则是一个强大的扩展库，它在 Python 基础功能之上提供了更丰富的数据结构，包括更多样的数值类型、向量、矩阵以及大量的矩阵运算函数。NumPy 与 Python 能够无缝配合：Python 原生的算术运算符可以直接作用于 NumPy 数据类型，同时许多 NumPy 函数也能直接接收 Python 原生的数据类型。


&lt;a name=&quot;toc_40015_3&quot;&gt;&lt;/a&gt;
# 3 向量
&lt;a name=&quot;toc_40015_3.1&quot;&gt;&lt;/a&gt;
## 3.1 概念简介
&lt;img align=&quot;right&quot; src=&quot;./images/C1_W2_Lab04_Vectors.PNG&quot; style=&quot;width:340px;&quot; &gt;在本课程中，我们所说的向量其实就是按顺序排列的一组数字。在数学符号中，向量通常用小写的粗体字母来表示，比如 $\mathbf{x}$。同一个向量中的所有元素都必须属于同一种数据类型——举例来说，一个向量里不能既装字符又装数字。向量中包含的元素个数通常被称为该向量的*维度 (dimension) *，不过数学家们有时更喜欢称之为*秩 (rank) *。右图展示的向量维度为 $n$。我们可以通过索引 (index) 来访问向量中的特定元素。在纯数学教材中，索引编号通常从 1 到 n；而在计算机科学以及我们的实验课里，索引编号通常是从 0 到 n-1。在数学符号中，当我们单独引用向量里的某个元素时，会在字母右下角标出它的索引下标：例如，向量 $\mathbf{x}$ 的第 $0$ 个元素记作 $x_0$。请注意，此时的字母 x 不再加粗。  


&lt;a name=&quot;toc_40015_3.2&quot;&gt;&lt;/a&gt;
## 3.2 NumPy 数组

NumPy 最基础的数据结构是一种支持索引访问、内部元素类型一致 (`dtype`) 的多维*数组 (array) *。你可能马上就会发现，“维度”这个词在这里被赋予了另一层含义：在上一节里，它指的是向量中元素的个数；而在这里，维度指的是访问数组所需的索引个数。一维数组 (1-D array) 只需要一个索引。在第 1 门课程中，我们将统一使用 NumPy 的一维数组来表示向量。

 - 一维数组，形状 (shape) 为 (n,) ：包含 n 个元素，索引范围从 [0] 到 [n-1]
 

&lt;a name=&quot;toc_40015_3.3&quot;&gt;&lt;/a&gt;
## 3.3 创建向量


在 NumPy 中，用来创建数据的函数通常都将目标对象的形状作为第一个参数。对于一维数组来说，这个参数可以是一个单独的整数；而对于更高维的数组，则可以传入一个指定各维大小的元组 (tuple) `(n,m,...)`。下面展示了几个利用这类函数创建向量的具体示例。


```python
# NumPy routines which allocate memory and fill arrays with value
a = np.zeros(4);                print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,));             print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4); print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
```

也有一部分创建数据的函数不需要传入表示形状的元组：


```python
# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.);              print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4);          print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
```

当然，你也可以手动指定数组中的每一个具体数值：


```python
# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2]);  print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2]); print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
```

以上所有方法都创建了一个包含 4 个元素的一维向量 `a`。调用 `a.shape` 会返回它的维度信息。在这里我们看到 `a.shape` 的结果是 `(4,)`，代表这是一个拥有 4 个元素的一维数组。  

&lt;a name=&quot;toc_40015_3.4&quot;&gt;&lt;/a&gt;
## 3.4 向量运算
接下来，让我们一起探索几种常见的向量运算。
&lt;a name=&quot;toc_40015_3.4.1&quot;&gt;&lt;/a&gt;
### 3.4.1 索引 (Indexing) 
我们可以通过索引和切片 (slicing) 来访问向量中的元素。NumPy 提供了一套非常完备的索引与切片功能，这里我们只介绍本课程所需的基礎操作。如需了解更多细节，可以参考官方文档中的切片与索引 (Slicing and Indexing) 章节。  
**索引**是指根据元素在数组中的具体位置，直接引用数组中的*单个元素*。  
**切片**则是根据指定的索引范围，从数组中截取出一组元素构成的*子集 (subset) *。  
由于 NumPy 的索引从 0 开始计数，因此向量 $\mathbf{a}$ 中的第 3 个元素在代码中写作 `a[2]`。


```python
#vector indexing operations on 1-D vectors
a = np.arange(10)
print(a)

#access an element
print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

# access the last element, negative indexes count from the end
print(f"a[-1] = {a[-1]}")

#indexs must be within the range of the vector or they will produce and error
try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)
```

&lt;a name=&quot;toc_40015_3.4.2&quot;&gt;&lt;/a&gt;
### 3.4.2 切片 (Slicing) 
切片操作通过由三个数值组成的语法 (`start:stop:step`，即 `起始位置:结束位置:步长`) 来生成一组索引。在实际使用时，也可以只填写其中的一部分数值。通过下面的代码实例，你能最直观地理解它的用法：


```python
#vector slicing operations
a = np.arange(10)
print(f"a         = {a}")

#access 5 consecutive elements (start:stop:step)
c = a[2:7:1];     print("a[2:7:1] = ", c)

# access 3 elements separated by two 
c = a[2:7:2];     print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)

# access all elements
c = a[:];         print("a[:]     = ", c)
```

&lt;a name=&quot;toc_40015_3.4.3&quot;&gt;&lt;/a&gt;
### 3.4.3 单向量运算
有许多非常实用的运算只需要针对单个向量即可完成：


```python
a = np.array([1,2,3,4])
print(f"a             : {a}")
# negate elements of a
b = -a 
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a) 
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")
```

&lt;a name=&quot;toc_40015_3.4.4&quot;&gt;&lt;/a&gt;
### 3.4.4 向量与向量的逐元素运算 (element-wise operations) 
NumPy 中绝大部分的算术、逻辑和比较运算符都同样适用于向量。这些运算符会按照“逐个元素一一对应”的方式进行计算。例如：
$$ c_i = a_i + b_i $$


```python
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")
```

当然，要想让这类逐元素运算顺利执行，参与运算的两个向量必须具有完全相同的长度：


```python
#try a mismatched vector operation
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)
```

&lt;a name=&quot;toc_40015_3.4.5&quot;&gt;&lt;/a&gt;
### 3.4.5 标量与向量的运算
向量可以通过乘以一个标量 (scalar) 来实现整体“缩放”。所谓的标量，其实就是单个普通的数字。在运算时，这个标量会与向量中的每一个元素分别相乘。


```python
a = np.array([1, 2, 3, 4])

# multiply a by a scalar
b = 5 * a 
print(f"b = 5 * a : {b}")
```

&lt;a name=&quot;toc_40015_3.4.6&quot;&gt;&lt;/a&gt;
### 3.4.6 向量与向量的点积 (dot product) 
点积是线性代数 (Linear Algebra) 与 NumPy 中的核心支柱运算。在本课程中，我们会极其频繁地使用点积，因此请务必透彻理解它的原理。点积的计算过程如下图所示：

&lt;img src=&quot;./images/C1_W2_Lab04_dot_notrans.gif&quot; width=800&gt; 

简单来说，点积就是先把两个向量中对应位置的元素逐个相乘，然后再把所有乘积加总求和。
进行向量点积运算的前提是：两个向量的维度必须完全一致。

下面，我们不妨先动手编写一个属于自己的点积函数：

**使用 `for` 循环**，编写一个返回两个向量点积的函数。给定输入向量 $a$ 和 $b$，该函数应当返回：
$$ x = \sum_{i=0}^{n-1} a_i b_i $$
这里可以假设 `a` 和 `b` 的形状完全相同。


```python
def my_dot(a, b): 
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    x=0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x
```


```python
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")
```

请注意，两个向量的点积最终返回的应当是一个单独的标量值。

接下来，让我们直接调用 NumPy 内置的 `np.dot` 来完成同样的运算：


```python
# test 1-D
a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")

```

从上面的输出可以看出，在一维向量上，`np.dot` 的计算结果与我们自己编写的函数完全一致。

&lt;a name=&quot;toc_40015_3.4.7&quot;&gt;&lt;/a&gt;
### 3.4.7 对速度的极致追求：向量化 vs for 循环
我们之所以青睐 NumPy 库，很大程度上是因为它能大幅提升计算速度与内存效率。下面让我们用实验数据说话：


```python
np.random.seed(1)
a = np.random.rand(10000000)  # very large arrays
b = np.random.rand(10000000)

tic = time.time()  # capture start time
c = np.dot(a, b)
toc = time.time()  # capture end time

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory
```

可以看到，在这个包含一千万个元素的例子里，向量化 (vectorization) 带来了惊人的速度提升！这是因为 NumPy 能够充分挖掘底层硬件的数据并行计算能力 (data parallelism) 。无论是 GPU 还是现代 CPU，内部都配备了“单指令多数据流” (Single Instruction, Multiple Data, 简称 SIMD) 流水线，允许同时并行处理多组数据运算。在数据集动辄十分庞大的机器学习 (Machine Learning) 领域，这种加速能力至关重要。

&lt;a name=&quot;toc_12345_3.4.8&quot;&gt;&lt;/a&gt;
### 3.4.8 第 1 门课程中的向量-向量运算
在第 1 门课程中，你会频繁遇到向量与向量之间的运算，原因如下：
- 在接下来的课程里，我们的所有训练样本都会被统一存放在一个维度为 (m,n) 的数组 `X_train` 中。后续我们结合具体场景还会做更详细的解释，但这里你需要先记住，它是一个二维数组 (2 Dimensional array) ，也就是矩阵 (参见下一节关于矩阵的介绍) 。
- 参数 `w` 则是一个形状为 (n,) 的一维向量。
- 当我们通过循环遍历所有训练样本时，每次都会利用索引从矩阵 `X` 中单独提取出一个样本来进行处理，比如写作 `X[i]`。
- `X[i]` 返回的结果形状为 (n,) ，正好是一个一维向量。因此，凡是涉及 `X[i]` 的运算，本质上往往都是向量与向量之间的运算。  

这段解释虽然看起来稍微有点长，但在执行向量运算时，时刻厘清并对齐各个操作数 (operands) 的形状是避免出错的关键。


```python
# show common Course 1 example
X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")
```

&lt;a name=&quot;toc_40015_4&quot;&gt;&lt;/a&gt;
# 4 矩阵


&lt;a name=&quot;toc_40015_4.1&quot;&gt;&lt;/a&gt;
## 4.1 概念简介
简单来说，矩阵就是二维数组。同一个矩阵中的所有元素都具有相同的数据类型。在数学符号中，矩阵通常用大写的粗体字母来表示，例如 $\mathbf{X}$。在本实验及后续实验中，我们通常用 `m` 表示矩阵的行数，用 `n` 表示矩阵的列数。矩阵中的元素需要通过二维索引来访问。在数学公式里，索引编号通常从 1 到 n；而在计算机科学及本课程实验中，索引编号则是从 0 到 n-1。  
&lt;figure&gt;
    &lt;center&gt; &lt;img src=&quot;./images/C1_W2_Lab04_Matrices.PNG&quot;  alt=&#39;missing&#39;  width=900&gt;&lt;center/&gt;
    &lt;figcaption&gt; 通用矩阵符号表示：第 1 个索引代表行，第 2 个索引代表列 &lt;/figcaption&gt;
&lt;figure/&gt;

&lt;a name=&quot;toc_40015_4.2&quot;&gt;&lt;/a&gt;
## 4.2 NumPy 数组

正如前面所介绍的，NumPy 的基础数据结构是支持索引、元素类型一致 (`dtype`) 的多维*数组*。而矩阵则拥有一个二维 (2-D) 索引 `[m,n]`。

在第 1 门课程中，我们会用二维矩阵来存放训练数据。假设训练数据包含 $m$ 个样本，每个样本有 $n$ 个特征 (features) ，那么它们就会组成一个形状为 (m,n) 的二维数组。第 1 门课程一般不会直接对整个矩阵进行矩阵级运算，而是通常先从矩阵中提取出一个样本作为向量，再对该向量进行运算。接下来你将重点回顾：
- 数据的创建方式
- 矩阵的切片与索引

&lt;a name=&quot;toc_40015_4.3&quot;&gt;&lt;/a&gt;
## 4.3 创建矩阵
之前用来创建一维向量的那些函数，同样可以用来创建二维甚至更高维的数组。下面来看几个例子：


在下方代码中，我们通过传入表示形状的元组来创建二维数组。请留意 NumPy 是如何利用方括号的嵌套层级来表示各个维度的，同时在打印输出时，NumPy 会将矩阵的每一行单独打印在一行上。



```python
a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     

a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}") 
```

我们同样可以手动填入具体的数据。此时需要通过多层嵌套的方括号来指明维度，其格式与上面打印输出的结果完全对应：


```python
# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")
```

&lt;a name=&quot;toc_40015_4.4&quot;&gt;&lt;/a&gt;
## 4.4 矩阵运算
下面让我们一起来探索几种常见的矩阵操作。

&lt;a name=&quot;toc_40015_4.4.1&quot;&gt;&lt;/a&gt;
### 4.4.1 索引


矩阵引入了第二个索引维度，这两个索引分别对应 `[行, 列]` (`[row, column]`) 。通过索引访问矩阵时，既可以返回单个元素，也可以返回整行或整列的数据。请看下方示例：


```python
#vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \na= {a}")

#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")
```

最后一个例子非常值得注意：当我们访问矩阵却只指定行索引时，返回的结果将是一个*一维向量*。

**重塑形状 (Reshape) **  
上面的例子使用了 reshape 函数来调整数组的形状：  
`a = np.arange(6).reshape(-1, 2) `   
这行代码首先创建了一个包含 6 个元素的*一维向量*，随后利用 `reshape` 命令将该向量重新排列成了一个*二维*数组。它其实完全等价于下面这行写法：  
`a = np.arange(6).reshape(3, 2) `  
两者都会得到同一个 3 行 2 列的二维数组。
在这里，第一个参数传入 `-1` 就相当于告诉程序：“请根据数组元素的总个数以及我已经指定的列数，自动帮我推算出应该有多少行。”


&lt;a name=&quot;toc_40015_4.4.2&quot;&gt;&lt;/a&gt;
### 4.4.2 切片
切片操作利用一组包含三个数值的语法 (`start:stop:step`) 来生成索引数组，当然也可以只写出其中的部分数值。通过下面的实例可以最清楚地看到它的具体用法：


```python
#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")

```

&lt;a name=&quot;toc_40015_5.0&quot;&gt;&lt;/a&gt;
## 恭喜！
在本次实验中，你已经熟练掌握了第 1 门课程所需的全部 Python 与 NumPy 核心特性。


```python

```
