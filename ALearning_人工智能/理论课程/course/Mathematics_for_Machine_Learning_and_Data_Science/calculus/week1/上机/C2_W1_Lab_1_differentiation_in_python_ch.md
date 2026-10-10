___
# Python 中的微分 (Differentiation) ：符号微分、数值微分与自动微分

在本实验中，你将探索 Python 中可用于计算导数的各种工具与代码库。你将使用 `SymPy` 库进行符号微分 (symbolic differentiation) ，使用 `NumPy` 进行数值微分 (numerical differentiation) ，并使用基于 `Autograd` 的 `JAX` 进行自动微分 (automatic differentiation) 。通过对比计算速度，你将深入探究这三种方法的计算效率。

___
- [[#1 - Python 中的函数|1 - Python 中的函数]]
- [[#2 - 符号微分|2 - 符号微分]]
	- [[#2 - 符号微分#2.1 - `SymPy` 符号计算入门|2.1 - `SymPy` 符号计算入门]]
	- [[#2 - 符号微分#2.2 - 使用 `SymPy` 进行符号微分|2.2 - 使用 `SymPy` 进行符号微分]]
	- [[#2 - 符号微分#2.3 - 符号微分的局限性|2.3 - 符号微分的局限性]]
- [[#3 - 数值微分|3 - 数值微分]]
	- [[#3 - 数值微分#3.1 - 使用 `NumPy` 进行数值微分|3.1 - 使用 `NumPy` 进行数值微分]]
	- [[#3 - 数值微分#3.2 - 数值微分的局限性|3.2 - 数值微分的局限性]]
- [[#4 - 自动微分|4 - 自动微分]]
	- [[#4 - 自动微分#4.1 - `JAX` 简介|4.1 - `JAX` 简介]]
	- [[#4 - 自动微分#4.2 - 使用 `JAX` 进行自动微分|4.2 - 使用 `JAX` 进行自动微分]]
- [[#5 - 符号微分、数值微分与自动微分的计算效率对比|5 - 符号微分、数值微分与自动微分的计算效率对比]]

___

## 1 - Python 中的函数

这里先简单复习一下如何在 Python 中定义函数。对于一个简单函数 $f\left(x\right) = x^2$，可以这样构建：

```python
def f(x):
    return x**2 #就是^

print(f(3)) #返回9

```

你可以轻松通过解析法求出该函数的导数，并将其写成一个独立的函数：

```python
def dfdx(x):
    return 2*x #写死了导数

print(dfdx(3))

```

既然你已经熟悉了 `NumPy` 数组，就可以直接将该函数作用于数组中的每一个元素：

```python
import numpy as np

x_array = np.array([1, 2, 3])

print("x: \n", x_array)
print("f(x) = x**2: \n", f(x_array))
print("f'(x) = 2x: \n", dfdx(x_array))
#直接对array进行数值运算 就是广播到每个元素

```
![[Pasted image 20261009225128.png]]


现在你可以把 `f` 和 `dfdx` 这两个函数应用到更大规模的数组上。接下来的代码将绘制出函数及其导数的图像 ：
*(现阶段你无需深究 `plot_f1_and_f2` 函数的具体实现细节)* 

```python
import matplotlib.pyplot as plt

# Output of plotting commands is displayed inline within the Jupyter notebook.
%matplotlib inline

def plot_f1_and_f2(f1, f2=None, x_min=-5, x_max=5, label1="f(x)", label2="f'(x)"):
    # 在 [-5, 5] 之间均匀切出 100 个点，作为输入下面f1,f2的 x。
    x = np.linspace(x_min, x_max,100)

    # Setting the axes at the centre.把坐标轴移到正中心
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.plot(x, f1(x), 'r', label=label1) #画原函数
    # 把函数整个传进去当y就好了 (实际是传入f1函数的输出值) 
    # 这样一步到位的做法是可以的 或者外面定义y=f1(x),再传入y
    
    if not f2 is None: #画导函数
        # If f2 is an array, it is passed as it is to be plotted as unlinked points.
        # If f2 is a function, f2(x) needs to be passed to plot it. 
        # isinstance 判断“前面的对象”是不是“后面指定的类型”  
        # 前面定义了     
        if isinstance(f2, np.ndarray):
            plt.plot(x, f2, 'bo', markersize=3, label=label2,)
        else:
            plt.plot(x, f2(x), 'b', label=label2)
    plt.legend()

    plt.show()
    
plot_f1_and_f2(f, dfdx)

```
函数功能
    ![[Pasted image 20261009232825.png]]但是其实 f2是函数 f2(x)还是离散的数值
    这里应该就是为了兼容, 传入f2函数,还是 传入y=f2(x)的y都可以

解释
    ![[Pasted image 20261009232609.png]]
    ![[Pasted image 20261009233603.png]]

![[Pasted image 20261009233644.png|443]]

在实际应用中，函数往往复杂得多，不可能每次都依赖人工推导来解析求导。接下来，让我们看看 Python 中有哪些无需手动推导就能计算导数的工具与库。

## 2 - 符号微分

符号计算 (Symbolic computation) 专门用于==处理精确表达而非近似表达==的数学对象 ==(例如 $\sqrt{2}$ 会直接保留原样，而不是写成 $1.41421356237$) ==。应用在微分上，意味着它的输出就像你根据运算法则亲手推导出的解析解一样。因此，符号微分能够得出完全精确的导数表达式。

### 2.1 - `SymPy` 符号计算入门

让我们借助常用的 `SymPy` 库来体验 Python 中的符号微分。
这个库呢, 让你感觉在手算一样, 就是保留了符号原来的数学表达

如果想计算 $\sqrt{18}$ 的十进制近似值，通常可以这样做：

```python
import math

math.sqrt(18) # 4.242640687119285

```

输出的 $4.242640687119285$ 只是一个近似值。或许你还记得 $\sqrt{18} = \sqrt{9 \cdot 2} = 3\sqrt{2}$，
但仅凭这个浮点数结果，几乎无法反推出它的精确形式。
而在符号计算系统中，根号不会被粗暴地算成小数，而是进行==代数化简==，从而给出绝对精确的输出：

```python
# This format of module import allows to use the sympy functions without sympy. prefix.
from sympy import *

# This is actually sympy.sqrt function, but sympy. prefix is omitted.
sqrt(18)

```
![[Pasted image 20261009234307.png]]

你也可以随时对其进行数值求值，甚至自由指定近似结果中展示的有效数字位数：

```python
N(sqrt(18),8)
# 4.2426407
```
解释
    ![[Pasted image 20261009234518.png]]

在 `SymPy` 中，==变量是用符号 (symbols) 来定义的==。在这个库里，符号必须提前声明 (提供一个符号列表) 。请看下方单元格，观察对应于数学表达式 $2x^2 - xy$ 的符号表达式是如何定义的：

```python
# List of symbols.
x, y = symbols('x y')
# Definition of the expression.
expr = 2 * x**2 - x * y
expr

```
![[Pasted image 20261009235132.png]]
解释
    SymPy 的核心设计理念是：**保持数学表达式的精确形式，不进行自动数值计算。**
    ![[Pasted image 20261009234936.png]]
    ![[Pasted image 20261009235028.png]]
    ![[Pasted image 20261009235115.png]]


现在你可以对这个表达式进行各种代数运算：增减项、乘以其它表达式等，就像在草稿纸上手算一样自如：

```python
expr_manip = x * (expr + x * y + x**3)
expr_manip

```
![[Pasted image 20261009235151.png]]


你还可以**展开**表达式：

```python
expand(expr_manip)

```
![[Pasted image 20261009235215.png]]

或者对其进行**因式分解**：

```python
factor(expr_manip)

```
![[Pasted image 20261009235232.png]]

若要将**特定数值代入**表达式中的变量进行计算，可以使用以下代码：

```python
expr = 2 * x**2 - x * y
expr.evalf(subs={x:-1, y:2})
# 4.0
```

解释
    ![[Pasted image 20261009235437.png]]

这一方法同样适用于计算函数 $f\left(x\right) = x^2$ 的具体取值：

```python
f_symb = x ** 2
f_symb.evalf(subs={x:3})
# 9.0
```
后面通过符号定义的式子 `expr = 2 * x**2 - x * y`
我们都称作 `符号函数`


现在你可能会好奇：==能否直接把一个数组传给符号函数==，计算出每个元素对应的值？在本实验开头，你曾定义过一个 `NumPy` 数组 `x_array`：

```python
print(x_array)

```
![[Pasted image 20261009235742.png]]

现在尝试对该数组中的每个元素计算函数 `f_symb` 的值，你会收到一个报错：

```python
try:
    f_symb(x_array)
except TypeError as err:
    print(err)

```
![[Pasted image 20261009235756.png]]

想要==对数组中的每个元素逐一计算符号函数==是完全可行的，但前提是==必须先将 符号函数和符号 转换为“适配 `NumPy`”的形式==：

```python
from sympy.utilities.lambdify import lambdify

f_symb_numpy = lambdify(x, f_symb, 'numpy')

```
解释
    ![[Pasted image 20261010000031.png]]
    ![[Pasted image 20261010000128.png]]

现在运行下方的代码便能顺利执行了：

```python
print("x: \n", x_array)
print("f(x) = x**2: \n", f_symb_numpy(x_array))

```
![[Pasted image 20261010000151.png]]

`SymPy` 拥有极其丰富的函数库，可用于公式变换以及微积分中的各类运算。更多细节可以查阅官方文档。

### 2.2 - 使用 `SymPy` 进行符号微分

让我们尝试使用 `SymPy` 来求一个简单幂函数的导数：

```python
diff(x**3,x)

```
解释
    ![[Pasted image 20261010000254.png]]

表达式中也可以包含各种标准初等函数，`SymPy` 会自动应用求导法则 (和差法则、乘积法则、链式法则) 来计算导数：

```python
dfdx_composed = diff(exp(-2*x) + 3*sin(3*x), x)
dfdx_composed

```
![[Pasted image 20261010000335.png]]

现在对在 [2.1](https://www.google.com/search?q=%232.1) 中定义的函数 `f_symb` 求导，并将其转换为支持 `NumPy` 的形式：

```python
dfdx_symb = diff(f_symb, x)
dfdx_symb_numpy = lambdify(x, dfdx_symb, 'numpy')
dfdx_symb
```
![[Pasted image 20261010000412.png]]

为 `x_array` 中的每一个元素计算 `dfdx_symb_numpy` 的导数值：

```python
print("x: \n", x_array)
print("f'(x) = 2x: \n", dfdx_symb_numpy(x_array))

```
![[Pasted image 20261010000349.png]]

你也可以将符号定义的函数作用于更大规模的数组。下方的代码绘制出了该函数及其导数图像，可以看到运行表现非常完美：

```python
plot_f1_and_f2(f_symb_numpy, dfdx_symb_numpy)

```
![[Pasted image 20261010000435.png|327]]


### 2.3 - 符号微分的局限性

==符号微分==看似是一门强大而完美的工具，但也==存在明显的短板==。有时它输出的==表达式会极其冗长复杂==，甚至根本无法求值。举个例子，求绝对值函数的一阶导数：$$\left\vert{}x\right\vert{} = \begin{cases} x, \ \text{if}\ x > 0\  -x, \ \text{if}\ x < 0 \ 0, \ \text{if}\ x = 0\end{cases}$$从解析数学的角度看，它的导数为：


$$\frac{d}{dx}\left(\left\vert{}x\right\vert{}\right) = \begin{cases} 1, \ \text{if}\ x > 0\\  -1, \ \text{if}\ x < 0\\\ \text{does not exist}, \ \text{if}\ x = 0\end{cases}$$

来看一下符号微分给出的输出结果：

```python
dfdx_abs = diff(abs(x),x)
dfdx_abs

```
![[Pasted image 20261010000527.png]]
解释
    ![[Pasted image 20261010000614.png|409]]
    **不用管**

形式看起来相当晦涩，不过只要能正常带入数值计算倒也无妨。但检查后会发现，当 $x=-2$ 时，它并没有给出导数值 $-1$，而是吐出了一个未化简的表达式：

```python
dfdx_abs.evalf(subs={x:-2})

```
![[Pasted image 20261010001006.png]]

而如果在转换为支持 `NumPy` 的版本后尝试调用，==同样会直接抛出错误==：

```python
dfdx_abs_numpy = lambdify(x, dfdx_abs,'numpy')

try:
    dfdx_abs_numpy(np.array([1, -2, 0]))
except NameError as err:
    print(err)

```
实际运行下来,甚至报错不支持
    ![[Pasted image 20261010001127.png]]

实际上，每当==导数中存在“跳变 (jump) ”== (例如函数在 $x$ 的不同区间内具有不同的表达式) 时，==符号表达式的数值求值往往就会遭遇困难==，正如在 $\frac{d}{dx}\left(\left\vert{}x\right\vert{}\right)$ 中发生的那样。

此外，从这个例子中你还可以发现，符号计算输出的函数往往可能变得异常繁琐膨胀。这种现象被称为**表达式膨胀** (expression swell) ，会==导致计算效率断崖式下跌==。在学习完 Python 中的其它微分库后，你会在后文看到具体的对比实例。

## 3 - 数值微分

数值微分法==并不关心函数的具体解析式是什么==。唯一的要求是==能够在相邻的两个点 $x$ 与 $x+\Delta x$ 处完成求值==，其中 $\Delta x$ 是一个充分小的步长。此时导数可近似表示为 $\frac{df}{dx}\approx\frac{f\left(x + \Delta x\right) - f\left(x\right)}{\Delta x}$，这被称为导数的数值近似 (numerical approximation) 。

基于这一思想衍生出了多种数值近似方法，它们在计算速度与精度上各有千秋。然而，所有这些方法==得出的结果都不是绝对精确的==——不可避免地存在舍入误差 (round off error) 。现阶段无需深入探讨各种算法的细枝末节，我们只需了解 `NumPy` 中提供的一个数值微分函数即可。

### 3.1 - 使用 `NumPy` 进行数值微分

你可以调用💛 `np.gradient` 函数来求上述函数 $f\left(x\right) = x^2$ 的导数。
它的第一个参数是函数值构成的数组，
第二个参数定义了用于计算的采样间距 $\Delta x$。
这里我们直接传入 $x$ 坐标数组，函数会自动计算出步长差值。相关文档可以查阅官方说明。

```python
x_array_2 = np.linspace(-5, 5, 100) #输入的x
dfdx_numerical = np.gradient(f(x_array_2), x_array_2)

plot_f1_and_f2(dfdx_symb_numpy, dfdx_numerical, label1="f'(x) exact", label2="f'(x) approximate")

```
![[Pasted image 20261010001531.png|401]]

解释
    ![[Pasted image 20261010001729.png]]

再来尝试对更复杂的复合函数进行数值微分：

```python
def f_composed(x):
    return np.exp(-2*x) + 3*np.sin(3*x)

plot_f1_and_f2(lambdify(x, dfdx_composed, 'numpy'), np.gradient(f_composed(x_array_2), x_array_2),
              label1="f'(x) exact", label2="f'(x) approximate")

```
![[Pasted image 20261010001817.png|355]]

效果令人赞叹——尤其考虑到它根本不需要理解函数的底层逻辑，==仅靠函数值就逼近了导数==！

### 3.2 - 数值微分的局限性

很显然，数值微分的首要==缺陷就是不够精确==。但在机器学习 (Machine Learning) 领域，==这种精度通常已经足够日常使用==。目前我们不必深究数值微分误差的理论界限。

另一个痛点与符号微分遇到的问题如出一辙：==在导数存在“突变跳跃”的点上，数值微分同样极不准确。==让我们对比一下绝对值函数的理论解析导数与数值近似导数：

```python
def dfdx_abs(x):
    if x > 0:
        return 1
    else:
        if x < 0:
            return -1
        else:
            return None

plot_f1_and_f2(np.vectorize(dfdx_abs), np.gradient(abs(x_array_2), x_array_2))

```
![[Pasted image 20261010001926.png|393]]

可以看到，在导数“突变”的附近，数值微分的结果竟给出了 $0.5$ 和 $-0.5$，而理论上本该是 $1$ 和 $-1$。在实际运算中，这类偏差会带来不可忽视的误差。

不过，数值微分最致命的==硬伤在于计算极其缓慢==。它==每次求导都必须重新计算一遍函数值==。在机器学习模型中，动辄包含数以亿计的参数，需要计算成千上万个导数；如果每次求导都要重新对整个函数求值一遍，会使整体计算变得奇慢无比。稍后你将看到直观的对比实验。

## 4 - 自动微分

**自动微分** (Automatic differentiation，简称 autodiff) 的方法是将复杂函数==逐层拆解为基本初等函数== ($sin$、$cos$、$log$ 以及幂函数等) ，并==构建出一张由基础计算节点构成的计算图 ==(computational graph) 。随后，==借助链式法则 (chain rule) 沿着计算图的节点求出各处的导数==。这是当前机器学习应用和神经网络 (Neural Networks) 中最普及的求导方案，因为在构建神经网络结构时就可以顺带生成函数及其导数的计算图，从而大幅节省后续反向传播中的计算开销。
*(原来大概了解就好)*

自动微分最主要的==缺点在于底层工程实现难度极高。好在如今已有许多成熟易用的开源库==，例如 MyGrad、Autograd 以及 JAX。其中，`Autograd` 和 `JAX` 是构建神经网络计算框架中最常用的利器。`JAX` 不仅融合了 `Autograd` 面向优化问题的高效求导机制，还引入了专门用于并行计算的 `XLA` (Accelerated Linear Algebra，加速线性代数) 编译器。

`Autograd` 与 `JAX` 在接口语法上略有区别，在同一篇教程中兼顾两者容易让人眼花缭乱。因此在本实验中，我们将聚焦于其中最先进的代表：`JAX`。

### 4.1 - `JAX` 简介

首先，导入所需的库文件。从 `jax` 核心库中，我们暂时只需要引入两个常用函数 (`grad` 和 `vmap`) 。
而 `jax.numpy` 则可以看作是==原生 `NumPy` 的高阶封装版==，在引入 `JAX` 生态后==几乎可以完全替代原生 `NumPy`==。在大多数项目代码中，它==**常被直接别名导入为 `np`**==。不过在本文档中，为了直观区分两者，我们将其导入为 `jnp`。

```python
from jax import grad, vmap
import jax.numpy as jnp

```

创建一个全新的 `jnp` 数组并查看其类型：

```python
x_array_jnp = jnp.array([1., 2., 3.])

print("Type of NumPy array:", type(x_array))
print("Type of JAX NumPy array:", type(x_array_jnp))
# Please ignore the warning message if it appears.

```
![[Pasted image 20261010002250.png]]

同样，也可以直接转换之前定义的 `x_array = np.array([1, 2, 3])`，不过在某些情况下 `JAX` 并不支持整型求导，因此需要提前转为浮点数类型。后文将对此进行专门演示。

```python
x_array_jnp = jnp.array(x_array.astype('float32'))
print("JAX NumPy array:", x_array_jnp)
print("Type of JAX NumPy array:", type(x_array_jnp))

```
![[Pasted image 20261010002308.png]]

请注意，`jnp` 数组的具体类型为 `jaxlib.xla_extension.DeviceArray`。在绝大多数情况下，原生 `NumPy` 支持的操作符和数学函数都可以无缝作用于它，例如：

```python
print(x_array_jnp * 2)
print(x_array_jnp[2])

```
![[Pasted image 20261010002325.png]]

但在某些操作习惯上，使用 `jnp` 数组时必须做出改变。例如在下面的代码中，如果你==尝试对数组的某个元素进行就地原位赋值修改，就会直接报错==：

```python
try:
    x_array_jnp[2] = 4.0
except TypeError as err:
    print(err)

```

要在 `jnp` 数组中更新某个元素的值，必须使用💛 `.at[i]` 指定要更新的索引位置，再调用 💛`.set(value)` 赋予新数值。
需要格外注意的是，这类方法采用的是非原地操作 (out-of-place) 策略，即==**更新后的结果会作为一个崭新的数组返回，而原始数组本身并不会发生任何改变。**==

```python
y_array_jnp = x_array_jnp.at[2].set(4.0)
print(y_array_jnp)

```
![[Pasted image 20261010002430.png]]

尽管如此，==部分 `JAX` 函数对于 `np` 与 `jnp` 数组是完全通用的==。在下方的代码中，两行调用能得到完全一致的结果：

```python
print(jnp.log(x_array))
print(jnp.log(x_array_jnp))

```
![[Pasted image 20261010002458.png]]

这或许会让人有些纠结：实际开发中究竟该用哪个 `NumPy` 呢？
通常在引入 `JAX` 的项目中，开发者会统一把 `jax.numpy` 别名导入为 `np`，从而彻底取代原生的 Python `NumPy` 模块。

### 4.2 - 使用 `JAX` 进行自动微分

接下来正式使用 `JAX` 体验自动微分。下方的代码将计算前面定义的函数 $f\left(x\right) = x^2$ 在点 $x = 3$ 处的导数值：

```python
print("Function value at x = 3:", f(3.0))
print("Derivative value at x = 3:",grad(f)(3.0))

```
![[Pasted image 20261010002642.png]]

解释
    ![[Pasted image 20261010002630.png|608]]

是不是非常简单利落？不过请务必牢记：自动微分==**不能直接作用于整型输入,必须传入浮点数**==。运行下方的代码将会产生类型错误：

```python
try:
    grad(f)(3)
except TypeError as err:
    print(err)

```

现在尝试直接对一个数组使用 `grad` 函数，希望能直接求出数组中每个元素对应的导数：

```python
try:
    grad(f)(x_array_jnp)
except TypeError as err:
    print(err)

```
![[Pasted image 20261010002726.png]]
解释
    ![[Pasted image 20261010002929.png]]

这里抛出了一个广播机制 (broadcasting) 相关的异常。现阶段无需深究其底层原因，
只需使用 `vmap` 函数即可优雅地解决这个==向量化映射==的问题。

*提示*：广播机制在专项课程的课程 1“线性代数 (Linear Algebra) ”中有专门讲解，你也可以直接阅读官方文档进行深入了解。

```python
dfdx_jax_vmap = vmap(grad(f))(x_array_jnp)
print(dfdx_jax_vmap)

```
![[Pasted image 20261010002842.png]]
解释
    ![[Pasted image 20261010002954.png]]

太棒了！现在可以使用 `vmap(grad(f))` 直接对更大尺寸的数组批量计算函数 `f` 的导数，并将结果完美绘制出来：

```python
plot_f1_and_f2(f, vmap(grad(f)))

```
![[Pasted image 20261010003026.png|402]]

在下面的代码中，你可以通过取消注释或注释相关行，来直观对比各种常见初等函数的导数曲线。它们全部由 `JAX` 自动微分自动计算得出，可视化效果非常理想！

```python
def g(x):
#     return x**3
#     return 2*x**3 - 3*x**2 + 5
#     return 1/x
#     return jnp.exp(x)
#     return jnp.log(x)
#     return jnp.sin(x)
#     return jnp.cos(x)
    return jnp.abs(x)
#     return jnp.abs(x)+jnp.sin(x)*jnp.cos(x)

plot_f1_and_f2(g, vmap(grad(g)))

```
![[Pasted image 20261010003052.png|314]]

## 5 - 符号微分、数值微分与自动微分的计算效率对比

在 [2.3](https://www.google.com/search?q=%232.3) 节与 [3.2](https://www.google.com/search?q=%233.2) 节中，我们曾分别指出了符号微分与数值微分在计算效率上的不足。现在，是时候通过实测数据正面较量这三种方法的计算速度了。我们对同一个简单的平方函数 $f\left(x\right) = x^2$ ==进行多次求导计算== (就是输入的x很多)，将其==作用于一个包含百万元素的大规模数组上==，并对比计算结果与耗时：

```python
import timeit, time

def f(x):
    return x**2 #就是^

x_array_large = np.linspace(-5, 5, 1000000)

tic_symb = time.time()
res_symb = lambdify(x, diff(f(x),x),'numpy')(x_array_large)
toc_symb = time.time()
time_symb = 1000 * (toc_symb - tic_symb)  # Time in ms.

tic_numerical = time.time()
res_numerical = np.gradient(f(x_array_large),x_array_large)
toc_numerical = time.time()
time_numerical = 1000 * (toc_numerical - tic_numerical)

tic_jax = time.time()
res_jax = vmap(grad(f))(jnp.array(x_array_large.astype('float32')))
toc_jax = time.time()
time_jax = 1000 * (toc_jax - tic_jax)

print(f"Results\nSymbolic Differentiation:\n{res_symb}\n" + 
      f"Numerical Differentiation:\n{res_numerical}\n" + 
      f"Automatic Differentiation:\n{res_jax}")

print(f"\n\nTime\nSymbolic Differentiation:\n{time_symb} ms\n" + 
      f"Numerical Differentiation:\n{time_numerical} ms\n" + 
      f"Automatic Differentiation:\n{time_jax} ms")

```
![[Pasted image 20261010003526.png]]

从数值上看三者的==计算结果高度一致==，但所需耗时却大相径庭。在需要高频重复求导的场景下，数值微分的低效暴露无遗——而这恰恰是深度学习 (Deep Learning) 模型训练中最频繁的操作。对于这个极其简单的二次函数，符号微分与自动微分的耗时表现看似旗鼓相当；但只要函数形式稍微复杂一些，符号计算就会遭遇严重的表达式膨胀，计算速度会随之急剧劣化。

*提示*：有时程序运行耗时会出现细微浮动，尤其是在 JAX 编译初期。你可以多运行几次上面的代码进行复现，但这并不影响“数值微分整体最为缓慢”的核心结论。虽然使用 `timeit` 模块可以对代码运行时间进行更严谨的基准评估，但那会使当前教学代码变得不必要的复杂。

现在让我们定义一个由简单多项式复合而成的函数——在数学推导上求导并不困难——借此==对比符号微分与自动微分==在更复杂场景下的计算速度表现：

```python
def f_polynomial_simple(x):
    return 2*x**3 - 3*x**2 + 5

def f_polynomial(x):
    for i in range(3):
        x = f_polynomial_simple(x) #算3次
    return x

tic_polynomial_symb = time.time()
res_polynomial_symb = lambdify(x, diff(f_polynomial(x),x),'numpy')(x_array_large)
toc_polynomial_symb = time.time()
time_polynomial_symb = 1000 * (toc_polynomial_symb - tic_polynomial_symb)

tic_polynomial_jax = time.time()
res_polynomial_jax = vmap(grad(f_polynomial))(jnp.array(x_array_large.astype('float32')))
toc_polynomial_jax = time.time()
time_polynomial_jax = 1000 * (toc_polynomial_jax - tic_polynomial_jax)

print(f"Results\nSymbolic Differentiation:\n{res_polynomial_symb}\n" + 
      f"Automatic Differentiation:\n{res_polynomial_jax}")

print(f"\n\nTime\nSymbolic Differentiation:\n{time_polynomial_symb} ms\n" +  
      f"Automatic Differentiation:\n{time_polynomial_jax} ms")

```
![[Pasted image 20261010003641.png]]


同样，数值计算结果依然一致，但==自动微分的速度已经是符号微分的数倍之快！==

随着计算图规模的不断扩展，自动微分凭借高效的链式法则计算模式，其相对于传统方法的性能优势将展现得更加淋漓尽致。

恭喜你！至此你已经熟练掌握了在 Python 中进行各类微分计算的核心工具与方法。

    反正就是自动微分 jax库好好好