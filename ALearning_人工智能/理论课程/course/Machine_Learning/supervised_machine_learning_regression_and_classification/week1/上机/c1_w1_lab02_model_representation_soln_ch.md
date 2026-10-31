
___
- [[#选修实验：模型表示 (Model Representation)|选修实验：模型表示 (Model Representation)]]
	- [[#选修实验：模型表示 (Model Representation)#实验目标|实验目标]]
	- [[#选修实验：模型表示 (Model Representation)#符号说明 (Notation)|符号说明 (Notation)]]
	- [[#选修实验：模型表示 (Model Representation)#实验工具|实验工具]]
- [[#问题背景|问题背景]]
		- [[#实验工具#训练样本数量 `m`|训练样本数量 `m`]]
		- [[#实验工具#训练样本 `x_i, y_i`|训练样本 `x_i, y_i`]]
		- [[#实验工具#绘制数据图|绘制数据图]]
	- [[#问题背景#模型函数 (Model function)|模型函数 (Model function)]]
		- [[#模型函数 (Model function)#小挑战|小挑战]]
			- [[#小挑战#提示：|提示：]]
		- [[#模型函数 (Model function)#房价预测 (Prediction)|房价预测 (Prediction)]]
- [[#恭喜！|恭喜！]]

___

# 选修实验：模型表示 (Model Representation) 
![[c1_w1_lab02_model_representation_soln_ch-1793361081200.webp]]

## 实验目标
在本次实验中，你将：
- 学习如何用代码实现单变量线性回归 (linear regression with one variable) 的模型 $f_{w,b}$

## 符号说明 (Notation) 
下面汇总了你在本实验中会遇到的一些常用数学符号。  

| **Notation**       | **Description**                                                                                            | **Python (if applicable)** |
| ------------------ | ---------------------------------------------------------------------------------------------------------- | -------------------------- |
| **General**        |                                                                                                            |                            |
| $a$                | scalar, non bold                                                                                           |                            |
| $\mathbf{a}$       | vector, bold                                                                                               |                            |
| **Regression**     |                                                                                                            |                            |
| $\mathbf{x}$       | Training Example feature values (in this lab - Size (1000 sqft))                                           | `x_train`                  |
| $\mathbf{y}$       | Training Example targets (in this lab Price (1000s of dollars))                                            | `y_train`                  |
| $x^{(i)}, y^{(i)}$ | $i_{th}$ Training Example                                                                                  | `x_i`, `y_i`               |
| $m$                | Number of training examples                                                                                | `m`                        |
| $w$                | parameter: weight                                                                                          | `w`                        |
| $b$                | parameter: bias                                                                                            | `b`                        |
| $f_{w,b}(x^{(i)})$ | The result of the model evaluation at $x^{(i)}$ parameterized by $w, b$: $f_{w,b}(x^{(i)}) = wx^{(i)} + b$ | `f_wb`                     |
|                    | 就是 y_hat                                                                                                   |                            |

中文翻译版本

| **符号 (Notation)**   | **说明 (Description)**                                                 | **Python（若适用）** |
| ------------------- | -------------------------------------------------------------------- | --------------- |
| **通用 (General)**    |                                                                      |                 |
| $a$                 | 标量，非粗体                                                               |                 |
| $\mathbf{a}$        | 向量，粗体                                                                |                 |
| **回归 (Regression)** |                                                                      |                 |
| $\mathbf{x}$        | 训练样本特征值（本实验中为面积：千平方英尺）                                               | `x_train`       |
| $\mathbf{y}$        | 训练样本目标值（本实验中为价格：千美元）                                                 | `y_train`       |
| $x^{(i)}, y^{(i)}$  | 第 $i$ 个训练样本                                                          | `x_i`, `y_i`    |
| $m$                 | 训练样本数量                                                               | `m`             |
| $w$                 | 参数：权重                                                                | `w`             |
| $b$                 | 参数：偏置（截距）                                                            | `b`             |
| $f_{w,b}(x^{(i)})$  | 由参数 $w, b$ 决定的模型在 $x^{(i)}$ 处的评估结果：$f_{w,b}(x^{(i)}) = wx^{(i)} + b$ | `f_wb`          |

## 实验工具
在本次实验中，你将使用以下工具：
- NumPy，一个广受欢迎的科学计算 (scientific computing) 库
- Matplotlib，一个广受欢迎的数据可视化绘图 (plotting data) 库


```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
```

# 问题背景

![[c1_w1_lab02_model_representation_soln_ch-1793361905444.webp|466]]
和理论课上一样，我们将以经典的“房价预测”作为切入点。  
本次实验使用的是一个非常精简的数据集 (data set) ，里面只包含两个数据点 (data points) —— 一套面积为 1000 平方英尺 (sqft) 的房子售价为 \$300,000，另一套面积为 2000 平方英尺的房子售价为 \$500,000。这两个样本点就构成了我们的数据或训练集 (training set) 。在本次实验中，房屋面积的单位是“1000 平方英尺”，而价格的单位则是“千美元”。

| 面积 (1000 平方英尺)     | 价格 (千美元) |
| -------------------| ------------------------ |
| 1.0               | 300                      |
| 2.0               | 500                      |

你的任务是拟合出一个穿过这两个数据点的线性回归模型 (linear regression model) (如上方图中的蓝色直线所示) ，这样一来，你就能借助它去预测其他房子的价格了——比如一套面积为 1200 平方英尺的房子。


请运行下方的代码单元格来创建 `x_train` 和 `y_train` 变量。这些数据都被存储在一维 (one-dimensional) NumPy 数组 (arrays) 中。
```python
# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
```
![[c1_w1_lab02_model_representation_soln_ch-1793362069211.webp]]

>**注意**：本课程在打印输出时会频繁使用这里介绍的 Python 格式化字符串 ('**f-string**') 语法。花括号内部的内容会在生成输出时自动完成求值并填入字符串中。


### 训练样本数量 `m`
我们将使用变量 `m` 来表示训练样本的总数。NumPy 数组自带一个 `.shape` 属性。调用 `x_train.shape` 会返回一个 ==Python 元组 (tuple) ==，其中的每个元素分别代表对应维度 (dimension) 的大小。如下所示，`x_train.shape[0]` 就代表该数组的长度，也就是样本的数量。

```python
# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
```
![[c1_w1_lab02_model_representation_soln_ch-1793362128687.webp]]


除此之外，你也可以使用如下所示的 Python 内置函数 `len()` 来获取样本数量。

```python
# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")
```
![[c1_w1_lab02_model_representation_soln_ch-1793362143014.webp]]


### 训练样本 `x_i, y_i`

我们将使用 (x$^{(i)}$, y$^{(i)}$) 来表示第 $i$ 个训练样本。由于 Python 采用从 0 开始的索引机制 (zero indexed) ，因此第 0 个样本 (x$^{(0)}$, y$^{(0)}$) 对应的值是 (1.0, 300.0) ，而第 1 个样本 (x$^{(1)}$, y$^{(1)}$) 对应的值则是 (2.0, 500.0) 。

想要读取 NumPy 数组中的某个具体数值，只需在方括号中填入对应的索引偏移量即可。例如，访问 `x_train` 第 0 号位置元素的语法是 `x_train[0]`。
请运行下方的代码块，获取第 $i$ 个训练样本的值。

```python
i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
```
![[c1_w1_lab02_model_representation_soln_ch-1793362181565.webp]]


### 绘制数据图

如下方单元格所示，你可以调用 `matplotlib` 库中的 `scatter()` 函数，把这两个样本点画在散点图 (scatter plot) 上。
- 函数参数 `marker` 和 `c` 用来将这些数据点显示为红色的叉号 (默认样式是蓝色圆点) 。

你还可以调用 `matplotlib` 库中的其他函数来设置图表的标题和坐标轴标签。

```python
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()
```
![[c1_w1_lab02_model_representation_soln_ch-1793362238603.webp|449]]


## 模型函数 (Model function) 

![[c1_w1_lab02_model_representation_soln_ch-1793362433587.webp|447]]
正如理论课上所讲，线性回归的模型函数 (即一个将输入 `x` 映射到输出 `y` 的数学函数) 可以表示为：

$$ f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$

上面的公式其实就是我们熟悉的直线方程——只要给 $w$ 和 $b$ 赋予不同的数值，就能在图表上画出不同斜率和截距的直线。

下面我们不妨通过几段代码来建立更直观的认识。让我们先从 $w = 100$ 和 $b = 100$ 开始试起。
**注意：你随时可以回到这个单元格，重新修改模型的 w 和 b 参数**


```python
# w = 100
# b = 100
w = 200                         
b = 100 
print(f"w: {w}")
print(f"b: {b}")
```

现在，让我们来计算这两个数据点对应的模型预测值 $f_{w,b}(x^{(i)})$。你可以像下面这样，针对每个数据点分别写出计算过程——

对于 $x^{(0)}$，`f_wb = w * x[0] + b`
对于 $x^{(1)}$，`f_wb = w * x[1] + b`

但如果数据点的数量非常庞大，这种逐行手写的方式就会变得既繁琐又重复。因此，更聪明的做法是像下方 `compute_model_output` 函数所展示的那样，用一个简单的 `for` 循环来批量计算函数的输出。
> **注意**：参数说明中的 `(ndarray (m,))` 表示一个形状为 (m,) 的 NumPy 多维数组 (n-dimensional array) 。而 `(scalar)` 则表示这是一个没有维度的标量参数，仅仅代表一个纯粹的数值大小。  
>     (m,) 就是一维数组
> 解释：`np.zero(n)` 会返回一个包含 $n$ 个元素的一维 NumPy 全零数组。


```python
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb
```

现在，让我们调用 `compute_model_output` 函数，并把模型的预测结果画出来。

```python
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
```
![[c1_w1_lab02_model_representation_soln_ch-1793362639802.webp|342]]
正如你所见，将参数设为 $w = 100$ 和 $b = 100$ 所画出的直线，并**不能**很好地拟合我们的数据。

𝑤=200 and 𝑏=100
![[c1_w1_lab02_model_representation_soln_ch-1793362797886.webp|429]]

### 小挑战
不妨亲手尝试调整一下 $w$ 和 $b$ 的不同取值。要想画出一条完美穿过这两个数据点的直线，$w$ 和 $b$ 应该分别取多少呢？

#### 提示：
你可以用鼠标点击下方绿色的“Hints”字样，展开查看关于如何选择 b 和 w 的提示。
- Try 𝑤=200 and 𝑏=100

### 房价预测 (Prediction) 
既然我们已经找到了合适的模型，现在就可以用它来完成最初设定的房价预测任务了。让我们来预测一套面积为 1200 平方英尺的房子的价格。由于输入特征 $x$ 的单位是“1000 平方英尺”，因此这里的 $x$ 应当取值为 1.2。

```python
w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")
```
![[c1_w1_lab02_model_representation_soln_ch-1793362743840.webp]]

# 恭喜！
通过本次实验，你已经掌握了以下核心内容：
 - 线性回归能够构建出一个数学模型，用来刻画输入特征 (features) 与目标值之间的内在关联
     - 在上面的例子中，输入特征是房屋面积，而预测目标则是房屋价格
     - 对于简单的单变量线性回归而言，模型一共包含两个参数 $w$ 和 $b$，它们的具体数值需要利用训练数据 (training data) 通过“拟合” (fit) 来确定。
     - 一旦确定了模型的参数，我们就可以用这个模型去对从未见过的新数据做出预测 (predictions) 。


这个实验用最原始的方式改m,b 看模型拟不拟合数据

