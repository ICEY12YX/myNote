___
- [[#Python 矩阵与 NumPy 导论|Python 矩阵与 NumPy 导论]]
	- [[#Python 矩阵与 NumPy 导论#关于 Jupyter Notebooks|关于 Jupyter Notebooks]]
- [[#1 - NumPy 基础|1 - NumPy 基础]]
	- [[#1 - NumPy 基础#1.1 - Packages 软件包|1.1 - Packages 软件包]]
	- [[#1 - NumPy 基础#1.2 - 使用 NumPy 数组的优势|1.2 - 使用 NumPy 数组的优势]]
	- [[#1 - NumPy 基础#1.3 - 如何创建 NumPy 数组|1.3 - 如何创建 NumPy 数组]]
	- [[#1 - NumPy 基础#1.4 - NumPy 数组进阶|1.4 - NumPy 数组进阶]]
- [[#2 - Multidimensional Arrays 多维数组|2 - Multidimensional Arrays 多维数组]]
	- [[#2 - Multidimensional Arrays 多维数组#2.1 - 查看数组的 size, shape and dimension 大小、形状与维度|2.1 - 查看数组的 size, shape and dimension 大小、形状与维度]]
- [[#3 - 数组的数学运算|3 - 数组的数学运算]]
	- [[#3 - 数组的数学运算#3.1 - 向量与标量相乘（广播机制）|3.1 - 向量与标量相乘（广播机制）]]
- [[#4 - Indexing and slicing 索引与切片|4 - Indexing and slicing 索引与切片]]
	- [[#4 - Indexing and slicing 索引与切片#4.1 - Indexing 索引|4.1 - Indexing 索引]]
	- [[#4 - Indexing and slicing 索引与切片#4.2 - Slicing 切片|4.2 - Slicing 切片]]
- [[#5 - 堆叠|5 - 堆叠]]
	- [[#5 - 堆叠#练习题|练习题]]
		- [[#练习题#问题 1|问题 1]]

___
# Python 矩阵与 NumPy 导论 #


欢迎来到本专项课程的第一个实验手册！在本文中，你将使用 NumPy 来创建二维数组，并轻松完成各种数学运算。NumPy  (Numerical Python)  是一个广泛应用于科学与工程领域的开源软件包。如果你已经能够熟练使用 NumPy，可以放心跳过本节内容。

**完成本节任务后，你将能够：**

* 熟练使用 Jupyter Notebook 及其各项功能。
* 使用 NumPy 函数创建数组并执行数组运算。
* 对二维数组进行索引  (indexing)  与切片  (slicing)  操作。
* 获取数组的形状，对其进行变形  (reshape)  以及水平和垂直堆叠  (stack)。

**使用说明**
* 你将使用 Python 3。
* 使用快捷键 `Shift`+`Enter` 依次运行代码单元格，也可以点击顶部菜单中的 `Run` 按钮执行。

___
## 关于 Jupyter Notebooks ##

Jupyter Notebook 是一种交互式编程笔记，它把可执行代码、解析文本、数学公式、可视化图表以及多媒体资源巧妙地融为一体。作为入门热身，不妨运行下方的小代码段，打印出经典的“Hello World”。

```python
# 运行下方单元格中的 "Hello World"，将其打印输出。
test = "Hello World"
```

```python
print(test)
```

# 1 - NumPy 基础 #

NumPy 是 Python 进行科学计算的核心基础包，能以极高效率执行各种高级数学运算。在本次实践练习中，你将掌握几个关键的 NumPy 函数，这些技能在后续任务中至关重要，涵盖数组创建、切片、索引、变形以及堆叠等操作。

## 1.1 - Packages 软件包 ##

在正式动手前，我们需要先导入 NumPy 模块以载入相关功能。你会发现，运行该单元格时虽然没有任何可见输出，但 Jupyter Notebook 已经悄悄完成了软件包（通常称为库）及其内置函数的加载。快来亲自尝试运行下方单元格吧。

```python
import numpy as np
```

## 1.2 - 使用 NumPy 数组的优势 ##

数组  (arrays)  是 NumPy 库的核心数据结构，也是组织和处理数据的基石。你可以把它想象成一个==网格状的数据结构==，其中容纳着同一种类型的值。如果你以前用过 Python 列表  (list) ，可能会觉得列表很灵活，因为它可以混合存储各种不同类型的数据。然而，列表的功能较为单一，在==内存占用和计算速度==上也远远逊色于 NumPy 数组。

相比 Python 原生列表，NumPy 提供的数组对象array不仅处理速度快得多，内存开销也更加轻量。得益于丰富的 API 集成，该库提供了大量内置函数，仅需几行代码便能轻松完成复杂的运算。这一优势在处理大规模数据集的数学计算时尤为突出。

NumPy 中的数组对象被称为💛 `ndarray`，全称为“n 维数组  (n-dimensional array) ”。
首先，我们将从最常见的数组类型之一 —— 一维数组  ('1-D')  入手。
一维数组在结构上类似于完全处于单一维度的一组常规数值。==
务必牢记：在 NumPy 中，==同一个数组内的所有元素数据类型必须严格保持一致。

```python
one_dimensional_arr = np.array([10, 12])
print(one_dimensional_arr)
```
![[Pasted image 20260924144222.png]]

## 1.3 - 如何创建 NumPy 数组 ##

在 NumPy 中创建数组有多种途径。要生成一维数组，只需调用 `array()` 函数，传入一个常规列表作为参数，它就会为你构建并返回对应的一维数组。

```python
# 创建并打印一个包含元素 1、2、3 的 NumPy 数组 'a'
a = np.array([1, 2, 3])
print(a)

```
![[Pasted image 20260924150450.png]]

另一种构建数组的方法是使用 `np.arange()`。该函数可以在给定区间内，生成由等间距数值组成的数组。如果你想了解该函数支持的具体参数，Jupyter Notebook 提供了非常便捷的查询技巧：只需用鼠标点击该函数名称，再按下键盘上的💙 `Shift+Tab` 组合键，就能直接调出其说明文档。不妨立即尝试查看 `np.arange()` 的内置文档说明。

```python
# 创建一个包含 3 个整数的数组，默认从 0 开始计数
b = np.arange(3)
print(b)

```
![[Pasted image 20260924144819.png]]
![[Pasted image 20260924150346.png]]

```python
# 创建一个从 1 开始、到 20 结束、步长为 3 的数组
c = np.arange(1, 20, 3)
print(c)

```
![[Pasted image 20260924144831.png]]
![[Pasted image 20260924150429.png]]


如果你希望在 0 到 100 这一区间内，均匀采集生成 5 个等间距的数值，该怎么做？显然，这里需要向函数传入 3 个关键参数：起点数值（此处为 0）、终点数值（此处为 100）以及生成的元素总数（此处为 5）。针对这种需求，NumPy 特意提供了专用的 `np.linspace()` 函数。

```python
lin_spaced_arr = np.linspace(0, 100, 5)
print(lin_spaced_arr)

```

你是否注意到上面输出的结果呈现为浮点数格式（例如 “... 25. 50. ...”）？原因在于，NumPy 的 `np.linspace` 函数默认生成的数值类型就是浮点型  (`np.float64`) 。
其实，你可以通过 `dtype` 参数轻松指定所需的数据类型。查阅内置文档你会发现，绝大多数 NumPy 函数都支持这个💛**可选参数** `dtype`。除了浮点数，NumPy 还支持诸如整型  (`int`)  和字符型  (`char`)  等多种数据类型。

如果想转换为整数类型，只需把 `dtype` 设置为 `int` 即可。前面的函数同样支持这一配置。欢迎随心修改单元格参数，打印出你所期待的数据类型。

```python
lin_spaced_arr_int = np.linspace(0, 100, 5, dtype=int)
print(lin_spaced_arr_int)

```
![[Pasted image 20260924150037.png]]

```python
c_int = np.arange(1, 20, 3, dtype=int)
print(c_int)

```
![[Pasted image 20260924150052.png]]

```python
b_float = np.arange(3, dtype=float)
print(b_float)

```
![[Pasted image 20260924150302.png]]

```python
char_arr = np.array(['Welcome to Math for ML!'])
print(char_arr)
print(char_arr.dtype) # 打印该数组的数据类型

```
![[Pasted image 20260924150653.png]]

你注意到 `char_arr` 数组的数据类型输出是 `<U23` 了吗？  
这意味着该字符串（`'Welcome to Math for ML!'`）在小端序架构（`<`）上是一个 23 个字符（23）的 Unicode 字符串（`U`）。你可以[在这里](https://numpy.org/doc/stable/user/basics.types.html)了解更多关于数据类型的信息。`

## 1.4 - NumPy 数组进阶 ##

使用 NumPy 的另一大便利在于，它提供了多种高效内置函数来快速生成特定数组，例如：
* `np.ones()` - 创建全 1 数组。
* `np.zeros()` - 创建全 0 数组。
* `np.empty()` - 创建未经初始化的新数组。
* `np.random.rand()` - 创建由随机数填充的数组。


```python
# 创建一个形状为 3、全部由 1 组成的新数组
ones_arr = np.ones(3)
print(ones_arr)

```
![[Pasted image 20260924150838.png]]

```python
# 创建一个形状为 3、全部由 0 组成的新数组
zeros_arr = np.zeros(3)
print(zeros_arr)

```
![[Pasted image 20260924150902.png]]

```python
# 创建一个形状为 3 的新数组，不对内存数据进行初始化
empt_arr = np.empty(3)
print(empt_arr)

```

```python
# 创建一个shape为 3、数值在 0 到 1 之间均匀分布的随机数数组
rand_arr = np.random.rand(3)
print(rand_arr)

```
![[Pasted image 20260924151234.png]]
![[Pasted image 20260924151226.png]]


# 2 - Multidimensional Arrays 多维数组 #

利用 NumPy，你还能轻松构建多维数组。在上文示例中，我们接触的都是一维数组，只需单个下标索引即可定位其中的元素。而多维数组则拥有多列结构。你可以把多维数组直观地类比为 Excel 表格，其中的每一行、每一列都映射着不同的维度。
![[Pasted image 20260924151352.png]]

```python
# 创建一个二维数组（2-D）
two_dim_arr = np.array([[1,2,3], [4,5,6]])
print(two_dim_arr)

```
![[Pasted image 20260924151440.png]]

构建多维数组的另一种常见思路，是对现有一维数组进行形状重塑  (reshaping) 。使用 `np.reshape()` 函数，可以把原数组中的元素按照指定的新结构重新排列。

```python
# 一维数组
one_dim_arr = np.array([1, 2, 3, 4, 5, 6])

# 使用 reshape() 构建多维数组
multi_dim_arr = np.reshape(
                one_dim_arr, # 待重塑形状的原数组
               (2,3) # 新数组的目标维度
              )
# 打印重塑后的 2 行 3 列二维数组
print(multi_dim_arr)

```

```python
#这样也可以
one_dim_arr = np.array([1, 2, 3, 4, 5, 6])
multi_dim_arr = one_dim_arr.reshape(2, 3)
print(multi_dim_arr)
```
![[Pasted image 20260924151717.png]]

## 2.1 - 查看数组的 size, shape and dimension 大小、形状与维度 ##

在接下来的编程任务中，快速获取数组的大小、维度与形状是一项基本功。它们都是 `ndarray` 对象的原生属性，可按如下方式直接访问：

* `ndarray.ndim` - 存储数组的维度数量（轴的个数）。
* `ndarray.shape` - 存储数组的形状元组，元组中的每个数值分别对应各维度上的长度。
* `ndarray.size` - 存储数组内包含的元素总数。

![[Pasted image 20260924151717.png]]
```python
# 二维数组 multi_dim_arr 的维度
multi_dim_arr.ndim #2

```

```python
# 二维数组 multi_dim_arr 的形状
# 返回结果代表 2 行 3 列
multi_dim_arr.shape #(2,3)

```

```python
# 数组 multi_dim_arr 的大小
# 返回元素总数
multi_dim_arr.size #6

```

# 3 - 数组的数学运算 #

在这一部分你将体会到，NumPy 能够对一维和多维数组疾速执行逐元素  (elementwise)  的加、减、乘、除运算( addition, substraction, multiplication and division )。
这些操作直接沿用标准的数学算符：`+`、`-` 和 `*`。
需要回想的是，❤️Python ==**原生列表的加法行为截然不同**==——它仅是将两个列表==首尾拼接==成更长的列表；更关键的是，原生列表根本无法直接进行减法和乘法操作。

```python
arr_1 = np.array([2, 4, 6])
arr_2 = np.array([1, 3, 5])

# 两个一维数组相加
addition = arr_1 + arr_2
print(addition)

# 两个一维数组相减
subtraction = arr_1 - arr_2
print(subtraction)

# 两个一维数组逐元素相乘
multiplication = arr_1 * arr_2
print(multiplication)

```
![[Pasted image 20260924152031.png]]

## 3.1 - 向量与标量相乘（广播机制） ##

假设你现在需要将英里数值换算为公里。运用刚才学到的 NumPy 数组功能即可轻松解决：
让一个表示英里的数组与单一常数（即换算系数，数学上称为**标量**  (scalar) ）直接相乘。因为 1 英里 ≈ 1.6 公里，NumPy 会自动在内部完成对每个元素的独立相乘计算。

这种让不同形状数组之间能够协调完成运算的精妙设计，被称为**广播**  (broadcasting) 机制。

```python
vector = np.array([1, 2])
print(vector * 1.6)

```
![[Pasted image 20260924152203.png]]
![[Pasted image 20260924152220.png]]

# 4 - Indexing and slicing 索引与切片 #

索引  (Indexing)  是一个非常实用的功能，它允许你按需提取数组中的特定元素。正如你在后续任务中所见，针对多维数组，索引还可以用来整行、整列甚至是整个切面地提取数据。

## 4.1 - Indexing 索引 ##

我们先来练习从指定数组中抓取特定位置的元素。

```python
# 提取数组中的第 3 个元素。注意：计数从 0 开始。
a = ([1, 2, 3, 4, 5])
print(a[2])
#print(a.dtype) #这里a是一个列表，不是numpy数组，所以没有dtype属性
print(type(a))

# 提取数组中的第 1 个元素。
print(a[0])

```
![[Pasted image 20260924152523.png]]

对于形状为 `n` 维的多维数组，如果想要精确锁定某个元素，必须依次传入 `n` 个坐标索引，每个维度对应一个索引值。

```python
# 二维数组上的索引操作
two_dim = np.array(([1, 2, 3],
          [4, 5, 6], 
          [7, 8, 9]))

# 使用坐标索引 i, j 选取二维数组中的数值 8
print(two_dim[2][1]) #8

```

## 4.2 - Slicing 切片 ##

切片  (Slicing)  能从数组中圈定并抽取出一部分连续元素形成子序列。
切片语法需要指定起始和结束位置，其截取区间遵循“左闭右开”规则（即包含起点，但不包含终点边界）。

标准语法格式如下：
`array[start:end:step]`

如果不填起始位置 `start`，程序将==默认从头开始==  (`start = 0`) ；
如果省略结束位置 `end`，则==默认截取至末尾==  (`end = 数组总长度`) ；
如果不指定步长 `step`，则**默认为每次递进 1 个步长**  (`step = 1`)  。

```python
# 对数组 a 进行切片，提取子数组 [2,3,4]
a = ([1, 2, 3, 4, 5])
sliced_arr = a[1:4]
print(sliced_arr)

```
`[2, 3, 4]`

```python
# 对数组 a 进行切片，提取子数组 [1,2,3]
sliced_arr = a[:3]
print(sliced_arr)

```
`[1, 2, 3]`

```python
# 对数组 a 进行切片，提取子数组 [3,4,5]
sliced_arr = a[2:]
print(sliced_arr)

```
`[3, 4, 5]`

```python
# 对数组 a 进行切片，隔位提取子数组 [1,3,5]
sliced_arr = a[::2]
print(sliced_arr)

```
`[1, 3, 5]`

```python
# Note that a == a[:] == a[::]
print(a[:], a[::])
print(a == a[:] == a[::])
```
`[1, 2, 3, 4, 5] [1, 2, 3, 4, 5]`
`True`

```python
# 对二维数组进行切片，获取前两行数据
sliced_arr_1 = two_dim[0:2]
sliced_arr_1

```
![[Pasted image 20260924153609.png]]

```python
# 类似地，对二维数组进行切片，获取最后两行数据
sliced_two_dim_rows = two_dim[1:3]
print(sliced_two_dim_rows)

```
![[Pasted image 20260924153619.png]]

```python
sliced_two_dim_cols = two_dim[:, 1]  # 意思是取所有行的第1列
print(sliced_two_dim_cols)

sliced_two_dim_subarray = two_dim[0:2, 0:2]  # 意思是取前两行前两列的子数组
print(sliced_two_dim_subarray)
```
![[Pasted image 20260924153642.png]]

# 5 - 堆叠 #

最后是堆叠  (Stacking)  功能，它让 NumPy 在重塑和组合数组时变得更为灵活强大。所谓堆叠，就是沿着一个新方向（新维度轴）将两个或更多数组在水平或垂直维度上拼接整合。

* `np.vstack()` - 沿垂直方向（上下）进行纵向堆叠
* `np.hstack()` - 沿水平方向（左右）进行横向堆叠
* `np.hsplit()` - 将单个数组沿水平方向横向拆分为若干更小的子数组


```python
a1 = np.array([[1,1], 
               [2,2]])
a2 = np.array([[3,3],
              [4,4]])
print(f'a1:\n{a1}')
print(f'a2:\n{a2}')

```
![[Pasted image 20260924153900.png]]

```python
# 将数组进行垂直堆叠
vert_stack = np.vstack((a1, a2))
print(vert_stack)

```
![[Pasted image 20260924154405.png]]

```python
# 将数组进行水平堆叠
horz_stack = np.hstack((a1, a2))
print(horz_stack)

```
![[Pasted image 20260924154414.png]]

## 练习题 ##

太棒了！回顾今天学到的知识点，试着回答下面这道小题检验一下掌握程度吧。

### 问题 1 ###

`np.zeros()` 和 `np.empty()` 之间存在区别吗？请从下列选项中选出正确的一项：

* A. 没有区别，它们生成的都是全 0 数组。
* B. `np.zeros()` 虽未经过显式初始化，但最终输出的内容是 0。
* C. `np.zeros()` 的执行效率要快于 `np.empty()`。
* D. `np.empty()` 输出的是 uninitialized array，而 `np.zeros()` 输出的是an initialized array of value zero。

```python
# 运行该单元格以选择你的答案
import quiz
import ipywidgets as widgets
q1 = quiz.mcq(quiz.question1, quiz.solution1)

```
![[Pasted image 20260924154637.png]]

恭喜你圆满完成本专项课程的首个实验手册！