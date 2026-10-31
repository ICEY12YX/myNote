# 编程练习实验：线性回归 (Linear Regression) 

欢迎来到你的第一个实战编程练习实验！在本次实验中，你将亲手实现单变量线性回归 (linear regression with one variable) ，用来预测一家连锁餐饮品牌的门店利润。


# 目录大纲
-  1 - 依赖工具包 
-  2 - 单变量线性回归 
  -  2.1 问题背景
  -  2.2 数据集 (Dataset) 
  -  2.3 线性回归核心知识回顾
  -  2.4 计算代价 (Compute Cost) 
    -  练习 1
  -  2.5 梯度下降 (Gradient descent) 
    -  练习 2
  -  2.6 使用批量梯度下降学习模型参数 


_**注意：** 为了避免自动评分系统 (autograder) 报错，请不要修改或删除本笔记本中的任何非计分单元格，也请不要额外新建任何单元格。
**当你顺利通过本次作业的评测之后**，如果想要修改非计分代码来做一些自由探索，可以参考本笔记本文末附带的操作说明。_

&lt;a name=&quot;1&quot;&gt;&lt;/a&gt;
## 1 - 依赖工具包 

首先，让我们运行下方的代码单元格，导入本次作业所需的全部工具包：
- numpy 是 Python 中用于处理矩阵与科学计算 (scientific computing) 的基础核心库。
- matplotlib 是 Python 中广受欢迎的图表绘制库。
- ``utils.py`` 中包含了本次作业所需的辅助函数，你不需要修改该文件中的任何代码。



```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
%matplotlib inline
```

## 2 - 问题背景

假设你是一家餐饮连锁品牌的 CEO，目前正在考察多个不同的城市，准备开设一家新的分店。
- 你希望把业务拓展到那些能够为餐厅带来更高利润的城市。
- 你的连锁品牌已经在许多城市开设了门店，因此你手里掌握着这些现有所在城市的人口规模与门店利润数据。
- 同时，你也收集了一批候选新店所在城市的数据：
    - 对于这些候选城市，你已经掌握了当地的城市人口数量。
    
你能否利用手头的这些数据，帮自己分析并挑出哪些候选城市最有可能为餐厅带来丰厚的利润呢？

## 3 - 数据集

第一步，我们先来加载本次任务所需的数据集。
- 下方代码中的 `load_data()` 函数会将数据分别加载到变量 `x_train` 和 `y_train` 中：
  - `x_train` 表示某座城市的人口数量
  - `y_train` 表示该城市门店的利润。如果利润为负数，则代表该门店处于亏损状态。   
  - `X_train` 和 `y_train` 都是 NumPy 数组 (numpy arrays) 。


```python
# load the dataset
x_train, y_train = load_data()
```

#### 查看变量内容
在正式着手解决任何问题之前，先花点时间熟悉一下手头的数据集总是一个好习惯。  
- 一个最简单直观的切入点，就是把各个变量打印出来，看看里面究竟装了些什么数据。

下方的代码会打印出变量 `x_train` 的数据类型以及它的前几个元素：


```python
# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5]) 
```

可以看到，`x_train` 是一个全部由大于零的小数构成的 NumPy 数组。
- 这些数值代表以“万人” (即乘以 10,000) 为单位的城市人口数量
- 举例来说，数值 `6.1101` 就代表那座城市的人口为 61,101 人
  
接下来，让我们再把 `y_train` 打印出来看看：


```python
# print y_train
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])  
```

同样地，`y_train` 也是一个包含小数的 NumPy 数组，其中既有正数也有负数。
- 这些数值代表你的餐厅在各个城市的月均利润，单位为“万美元” (\$10,000) 。
  - 例如，`17.592` 代表该城市门店的月均利润为 \$175,920。
  - 而 `-2.6807` 则代表该城市门店平均每月亏损 -\$26,807。

#### 检查变量的维度 (dimensions) 

熟悉数据的另一个实用方法，是查看各个数组变量的维度大小。

请运行下方代码，打印出 `x_train` 和 `y_train` 的形状 (shape) ，看看你的数据集中一共包含多少个训练样本 (training examples) 。


```python
print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))
```

可以看到，城市人口数组一共包含 97 个数据点 (data points) ，而月均利润数组也同样包含 97 个数据点。它们都是一维 (1D) NumPy 数组。

#### 将数据可视化

通过绘制图表将数据直观地展现出来，往往能帮我们更好地理解数据背后的规律。
- 对于当前这个数据集，由于一共只有两个属性需要展示 (利润与人口) ，我们可以直接使用散点图 (scatter plot) 来进行可视化。
- 在现实生活中，你遇到的许多其他问题往往会包含两个以上的属性 (比如同时包含城市人口、家庭平均收入、月均利润、月销售额等) 。即便属性超过两个，你依然可以借助散点图来两两对比，观察每一对属性之间的相互关系。



```python
# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()
```

你的目标是构建一个能够拟合这批数据的线性回归模型 (linear regression model) 。
- 有了这个模型之后，你只需输入任意一座新城市的人口数量，模型就能为你估算出在该城市开店预计能获得的月均利润。

&lt;a name=&quot;4&quot;&gt;&lt;/a&gt;
## 4 - 线性回归核心知识回顾

在本次编程练习中，你将使用手头的数据集来拟合线性回归的两个参数 (parameters) $(w,b)$。
- 线性回归的模型函数 (即一个将输入 `x` (城市人口) 映射到输出 `y` (该城市门店月均利润) 的数学函数) 可以表示为：
    $$f_{w,b}(x) = wx + b$$
    

- 训练线性回归模型的本质，就是找出能够最好地拟合数据集的最佳参数组合 $(w,b)$。  

    - 为了衡量某一组 $(w,b)$ 的取值究竟比另一组取值更好还是更差，我们可以借助代价函数 (cost function) $J(w,b)$ 来进行评估。
      - $J$ 是关于 $(w,b)$ 的函数——也就是说，代价 $J(w,b)$ 的高低直接取决于你为 $(w,b)$ 挑选的具体数值。
  
    - 能够让代价 $J(w,b)$ 降到最低的那一组 $(w,b)$，就是最契合你手头数据的最佳参数。


- 为了找出能让代价 $J(w,b)$ 达到最小值的参数 $(w,b)$，我们可以使用一种叫做**梯度下降**的方法。
  - 随着梯度下降每向前迈出一步，你的参数 $(w,b)$ 就会离那个能让代价 $J(w,b)$ 降到最低的最优值 (optimal values) 更近一步。
  

- 当模型训练完成后，这个线性回归模型就可以接收输入的特征 (feature) $x$ (城市人口) ，并直接输出对应的预测值 (prediction) $f_{w,b}(x)$ (在该城市开店的预计月均利润) 。

&lt;a name=&quot;5&quot;&gt;&lt;/a&gt;
## 5 - 计算代价

梯度下降通过一步步反复调整参数 $(w,b)$ 的数值，让代价 $J(w,b)$ 变得越来越小。
- 在梯度下降的每一步更新中，随着 $(w,b)$ 不断变化，同步计算出当前的代价 $J(w,b)$ 能帮我们随时监控模型训练的推进情况。
- 在本节中，你将亲手编写一个用于计算 $J(w,b)$ 的函数，以便后续用来检验梯度下降的运行进展。

#### 代价函数
回想一下理论课上所学的内容，对于单变量线性回归而言，代价函数 $J(w,b)$ 的定义如下：

$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$ 

- 你可以把 $f_{w,b}(x^{(i)})$ 理解为模型对某家餐厅利润给出的预测值；而与之相对的 $y^{(i)}$，则是数据集中记录的真实利润。
- $m$ 代表数据集中训练样本的总数量。

#### 模型预测

- 对于单变量线性回归，模型 $f_{w,b}$ 针对第 $i$ 个样本 $x^{(i)}$ 给出的预测值可以表示为：

$$ f_{w,b}(x^{(i)}) = wx^{(i)} + b$$

这就是我们熟悉的直线方程，其中 $b$ 是截距 (intercept) ，而 $w$ 则是直线的斜率 (slope) 。

#### 代码实现

请补全下方的 `compute_cost()` 函数，用来计算代价 $J(w,b)$。

&lt;a name=&quot;ex01&quot;&gt;&lt;/a&gt;
### 练习 1

请补全下方的 `compute_cost` 函数，完成以下逻辑：

* 循环遍历所有训练样本，并针对每一个样本分别计算：
    * 模型对该样本给出的预测值：
    $$
    f_{wb}(x^{(i)}) =  wx^{(i)} + b 
    $$
   
    * 该样本对应的误差代价：$$cost^{(i)} =  (f_{wb} - y^{(i)})^2$$
    

* 返回所有样本汇集而成的总代价：
$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} cost^{(i)}$$
  * 其中，$m$ 是训练样本的数量，而 $\sum$ 则是求和算子 (summation operator) 

如果在编写过程中遇到困难，可以点击下方单元格后面的提示按钮，查看具体的实现思路。


```python
# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    ### START CODE HERE ###
    
    ### END CODE HERE ### 

    return total_cost
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示 (Click for hints) &lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
    
   * 在代码中，你可以用如下方式来表示求和算子 (例如 $h = \sum\limits_{i = 0}^{m-1} 2i$) ：
    
    ```python 
    h = 0
    for i in range(m):
        h = h + 2*i
    ```
  
   * 在本题中，你可以使用 `for` 循环遍历 `x` 中的所有样本，并将每一轮循环算出的 `cost` 累加到一个在循环外部初始化好的变量 (`cost_sum`) 中。

   * 最后，将 `cost_sum` 除以 `2m` 作为 `total_cost` 返回即可。
   * 如果你刚开始接触 Python，请务必检查代码的缩进是否统一使用了相同数量的空格或制表符 (Tab) 。否则可能会导致输出结果异常，或者抛出 `IndentationError: unexpected indent` 缩进错误。你可以参考我们社区中的这篇帖子了解更多细节。

    <details>
          <summary><font size="2" color="darkblue"><b> 点击查看进一步提示</b></font></summary>
        
    * 你可以按照下面的整体框架来组织该函数的代码实现：
    
    ```python 
    def compute_cost(x, y, w, b):
        # number of training examples
        m = x.shape[0] 
    
        # You need to return this variable correctly
        total_cost = 0
    
        ### START CODE HERE ###  
        # Variable to keep track of sum of cost from each example
        cost_sum = 0
    
        # Loop over training examples
        for i in range(m):
            # Your code here to get the prediction f_wb for the ith example
            f_wb = 
            # Your code here to get the cost associated with the ith example
            cost = 
        
            # Add to sum of cost for each example
            cost_sum = cost_sum + cost 

        # Get the total cost as the sum divided by (2*m)
        total_cost = (1 / (2 * m)) * cost_sum
        ### END CODE HERE ### 

        return total_cost
    ```
    
    * 如果还是觉得毫无头绪，可以继续点开下方的提示，看看具体该如何计算 `f_wb` 和 `cost`。
    
    <details>
          <summary><font size="2" color="darkblue"><b>关于如何计算 f_wb 的提示</b></font></summary>
           &emsp; &emsp; 对于标量 (scalars) $a$、$b$ 和 $c$ (这里的 <code>x[i]</code>、<code>w</code> 和 <code>b</code> 都是标量) ，你可以在代码中将公式 $h = ab + c$ 写为 <code>h = a * b + c</code>
          <details>
              <summary><font size="2" color="blue"><b>&emsp; &emsp; 关于计算 f 的更多提示</b></font></summary>
               &emsp; &emsp; 你可以这样计算 f_wb：<code>f_wb = w * x[i] + b </code>
           </details>
    </details>

     <details>
          <summary><font size="2" color="darkblue"><b>关于如何计算 cost 的提示</b></font></summary>
          &emsp; &emsp; 你可以用 z**2 来计算变量 z 的平方
          <details>
              <summary><font size="2" color="blue"><b>&emsp; &emsp; 关于计算 cost 的更多提示</b></font></summary>
              &emsp; &emsp; 你可以这样计算 cost：<code>cost = (f_wb - y[i]) ** 2</code>
          </details>
    </details>
        
    </details>

&lt;/details&gt;

    


你可以运行下方的测试代码，检验自己编写的函数是否正确：


```python
# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1

cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')

# Public tests
from public_tests import *
compute_cost_test(compute_cost)
```

**预期输出结果**：
&lt;table&gt;
  &lt;tr&gt;
    &lt;td&gt; &lt;b&gt;Cost at initial w:&lt;b&gt; 75.203 &lt;/td&gt; 
  &lt;/tr&gt;
&lt;/table&gt;

&lt;a name=&quot;6&quot;&gt;&lt;/a&gt;
## 6 - 梯度下降 

在本节中，你将为线性回归的参数 $w, b$ 编写梯度 (gradient) 计算函数。

正如理论课视频中所讲，梯度下降算法的具体流程如下：

$$\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & \phantom {0000} b := b -  \alpha \frac{\partial J(w,b)}{\partial b} \newline       \; & \phantom {0000} w := w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{1}  \; & 
\newline & \rbrace\end{align*}$$

其中，参数 $w, b$ 需要同时 (simultaneously) 完成更新；而对应的偏导数 (partial derivatives) 计算公式分别为：
$$
\frac{\partial J(w,b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{2}
$$
$$
\frac{\partial J(w,b)}{\partial w}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) -y^{(i)})x^{(i)} \tag{3}
$$
* m 是数据集中训练样本的数量

    
* $f_{w,b}(x^{(i)})$ 是模型给出的预测值，而 $y^{(i)}$ 则是真实的目标值 (target value) 


接下来，你将编写一个名为 `compute_gradient` 的函数，用来计算偏导数 $\frac{\partial J(w)}{\partial w}$ 和 $\frac{\partial J(w)}{\partial b}$。

&lt;a name=&quot;ex02&quot;&gt;&lt;/a&gt;
### 练习 2

请补全下方的 `compute_gradient` 函数，完成以下逻辑：

* 循环遍历所有训练样本，并针对每一个样本分别计算：
    * 模型对该样本给出的预测值：
    $$
    f_{wb}(x^{(i)}) =  wx^{(i)} + b 
    $$
   
    * 该样本对参数 $w, b$ 贡献的梯度分量：
        $$
        \frac{\partial J(w,b)}{\partial b}^{(i)}  =  (f_{w,b}(x^{(i)}) - y^{(i)}) 
        $$
        $$
        \frac{\partial J(w,b)}{\partial w}^{(i)}  =  (f_{w,b}(x^{(i)}) -y^{(i)})x^{(i)} 
        $$
    

* 返回汇总全部样本之后得到的总梯度更新量：
    $$
    \frac{\partial J(w,b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} \frac{\partial J(w,b)}{\partial b}^{(i)}
    $$
    
    $$
    \frac{\partial J(w,b)}{\partial w}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} \frac{\partial J(w,b)}{\partial w}^{(i)} 
    $$
  * 其中，$m$ 是训练样本的数量，而 $\sum$ 则是求和算子

如果在编写过程中遇到困难，可以点击下方单元格后面的提示按钮，获取一步步的代码引导。


```python
# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0
    
    ### START CODE HERE ###
    
    ### END CODE HERE ### 
        
    return dj_dw, dj_db
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示 (Click for hints) &lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
   * 在代码中，你可以用如下方式来表示求和算子 (例如 $h = \sum\limits_{i = 0}^{m-1} 2i$) ：
    
   ```python 
    h = 0
    for i in range(m):
        h = h + 2*i
   ```
    
   * 在本题中，你可以使用 `for` 循环遍历 `x` 中的所有样本；针对每一个样本，将它算出的梯度分量不断累加到循环外部初始化好的变量 `dj_dw` 和 `dj_db` 上。

   * 最后，将 `dj_dw` 和 `dj_db` 分别除以 `m` 并返回即可。    
    &lt;details&gt;
          &lt;summary&gt;&lt;font size=&quot;2&quot; color=&quot;darkblue&quot;&gt;&lt;b&gt; 点击查看进一步提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
        
    * 你可以按照下面的整体框架来组织该函数的代码实现：
    
    ```python 
    def compute_gradient(x, y, w, b): 
        """
        Computes the gradient for linear regression 
        Args:
          x (ndarray): Shape (m,) Input to the model (Population of cities) 
          y (ndarray): Shape (m,) Label (Actual profits for the cities)
          w, b (scalar): Parameters of the model  
        Returns
          dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
          dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
        """
    
        # Number of training examples
        m = x.shape[0]
    
        # You need to return the following variables correctly
        dj_dw = 0
        dj_db = 0
    
        ### START CODE HERE ### 
        # Loop over examples
        for i in range(m):  
            # Your code here to get prediction f_wb for the ith example
            f_wb = 
            
            # Your code here to get the gradient for w from the ith example 
            dj_dw_i = 
        
            # Your code here to get the gradient for b from the ith example 
            dj_db_i = 
     
            # Update dj_db : In Python, a += 1  is the same as a = a + 1
            dj_db += dj_db_i
        
            # Update dj_dw
            dj_dw += dj_dw_i
    
        # Divide both dj_dw and dj_db by m
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        ### END CODE HERE ### 
        
        return dj_dw, dj_db
    ```
        
    * 如果还是觉得不太确定，可以继续点开下方的提示，看看该如何具体写出 `f_wb` 以及各个梯度分量的算式。
    
    <details>
          <summary><font size="2" color="darkblue"><b>关于如何计算 f_wb 的提示</b></font></summary>
           &emsp; &emsp; 这一步你在上一个练习中已经写过了！对于标量 $a$、$b$ 和 $c$ (这里的 <code>x[i]</code>、<code>w</code> 和 <code>b</code> 都是标量) ，你可以在代码中将公式 $h = ab + c$ 写为 <code>h = a * b + c</code>
          <details>
              <summary><font size="2" color="blue"><b>&emsp; &emsp; 关于计算 f 的更多提示</b></font></summary>
               &emsp; &emsp; 你可以这样计算 f_wb：<code>f_wb = w * x[i] + b </code>
           </details>
    </details>
        
    <details>
          <summary><font size="2" color="darkblue"><b>关于如何计算 dj_dw_i 的提示</b></font></summary>
           &emsp; &emsp; 对于标量 $a$、$b$ 和 $c$ (这里的 <code>f_wb</code>、<code>y[i]</code> 和 <code>x[i]</code> 都是标量) ，你可以在代码中将公式 $h = (a - b)c$ 写为 <code>h = (a-b)*c</code>
          <details>
              <summary><font size="2" color="blue"><b>&emsp; &emsp; 关于计算 dj_dw_i 的更多提示</b></font></summary>
               &emsp; &emsp; 你可以这样计算 dj_dw_i：<code>dj_dw_i = (f_wb - y[i]) * x[i] </code>
           </details>
    </details>
        
    <details>
          <summary><font size="2" color="darkblue"><b>关于如何计算 dj_db_i 的提示</b></font></summary>
             &emsp; &emsp; 你可以这样计算 dj_db_i：<code> dj_db_i = f_wb - y[i] </code>
    </details>
        
    </details>

&lt;/details&gt;

    


请运行下方的代码单元格，分别使用两组不同的参数初始值 $w, b$ 来检验你编写的 `compute_gradient` 函数是否正确：


```python
# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)

compute_gradient_test(compute_gradient)
```

接下来，让我们再测试一组非零的参数初始值：

**预期输出结果**：
&lt;table&gt;
  &lt;tr&gt;
    &lt;td&gt; &lt;b&gt;Gradient at initial , b (zeros)&lt;b&gt;&lt;/td&gt;
    &lt;td&gt; -65.32884975 -5.83913505154639&lt;/td&gt; 
  &lt;/tr&gt;
&lt;/table&gt;


```python
# Compute and display cost and gradient with non-zero w
test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)

print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)
```

**预期输出结果**：
&lt;table&gt;
  &lt;tr&gt;
    &lt;td&gt; &lt;b&gt;Gradient at test w&lt;b&gt;&lt;/td&gt;
    &lt;td&gt; -47.41610118 -4.007175051546391&lt;/td&gt; 
  &lt;/tr&gt;
&lt;/table&gt;

&lt;a name=&quot;2.6&quot;&gt;&lt;/a&gt;
### 2.6 使用批量梯度下降学习模型参数 

现在，你将利用批量梯度下降 (batch gradient descent) 算法，找出线性回归模型的最优参数。回想一下，“批量” (batch) 指的是在每一轮迭代 (iteration) 中都使用全部的训练样本参与计算。
- 在这一部分中，你不需要编写任何新代码，直接运行下方的单元格即可。

- 检验梯度下降是否正常工作的一个绝佳方法，就是观察每一步迭代后代价 $J(w,b)$ 的数值变化，确认它是否在一步步持续减小。

- 只要你正确实现了梯度计算函数与代价计算函数，并且设置了大小合适的学习率 (learning rate) alpha，那么代价 $J(w,b)$ 就绝不会上升，而是会在算法运行结束前平稳收敛 (converge) 到一个稳定的数值。


```python
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(x)
    
    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )  

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing
```

现在，让我们运行上方的梯度下降算法，为我们的数据集自动学习出最优的模型参数：


```python
# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.

# some gradient descent settings
iterations = 1500
alpha = 0.01

w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                     compute_cost, compute_gradient, alpha, iterations)
print("w,b found by gradient descent:", w, b)
```

**预期输出结果**：
&lt;table&gt;
  &lt;tr&gt;
    &lt;td&gt; &lt;b&gt; w, b found by gradient descent&lt;b&gt;&lt;/td&gt;
    &lt;td&gt; 1.16636235 -3.63029143940436&lt;/td&gt; 
  &lt;/tr&gt;
&lt;/table&gt;

接下来，我们将使用梯度下降最终学到的参数，把拟合出的回归直线画出来。

回想一下，我们可以用公式 $f(x^{(i)})= wx^{(i)}+b$ 算出单个样本的预测值。

为了算出整个数据集上所有样本的预测值，我们可以通过一个循环遍历全部训练样本，逐一算出每个样本对应的预测结果，如下方代码块所示：


```python
m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b
```

现在，让我们把算出的预测值连成直线画在散点图上，直观地看看线性拟合的效果：


```python
# Plot the linear fit
plt.plot(x_train, predicted, c = "b")

# Create a scatter plot of the data. 
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
```

最后，你学到的这套最优参数 $w,b$ 还可以直接用来预测新城市的潜在门店利润。不妨让我们来预测一下：在人口分别为 35,000 人和 70,000 人的两座城市开店，预计每月能获得多少利润？

- 请注意，该模型接收的城市人口输入是以“万人” (10,000s) 为单位的。

- 因此，35,000 人的城市人口在输入模型时，应当转换成 `np.array([3.5])`。

- 同理，70,000 人的城市人口在输入模型时，应当转换成 `np.array([7.])`。



```python
predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))
```

**预期输出结果**：
&lt;table&gt;
  &lt;tr&gt;
    &lt;td&gt; &lt;b&gt; For population = 35,000, we predict a profit of&lt;b&gt;&lt;/td&gt;
    &lt;td&gt; $4519.77 &lt;/td&gt; 
  &lt;/tr&gt;
  
  &lt;tr&gt;
    &lt;td&gt; &lt;b&gt; For population = 70,000, we predict a profit of&lt;b&gt;&lt;/td&gt;
    &lt;td&gt; $45342.45 &lt;/td&gt; 
  &lt;/tr&gt;
&lt;/table&gt;

**恭喜你完成了本次关于线性回归的实战编程练习！下一周，你将亲手构建模型去解决另一类全新的核心问题：分类 (classification) 。我们下周见！**

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;2&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;如果你想要修改任何非计分单元格的代码进行自由探索，请点击此处查看说明。&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    &lt;p&gt;&lt;i&gt;&lt;b&gt;重要提示：为了避免引发自动评分系统报错，请务必在已经顺利通过本次作业评测之后再进行以下操作。&lt;/b&gt;&lt;/i&gt;
    &lt;ol&gt;
        &lt;li&gt; 在笔记本顶部菜单栏中，依次点击 “View” &gt; “Cell Toolbar” &gt; “Edit Metadata”&lt;/li&gt;
        &lt;li&gt; 点击你想要解锁/锁定的代码单元格旁边的 “Edit Metadata” 按钮&lt;/li&gt;
        &lt;li&gt; 将其中的 “editable” 属性值修改为：
            &lt;ul&gt;
                &lt;li&gt; “true” (如果你想要解锁该单元格以便编辑) &lt;/li&gt;
                &lt;li&gt; “false” (如果你想要重新锁定该单元格) &lt;/li&gt;
            &lt;/ul&gt;
        &lt;/li&gt;
        &lt;li&gt; 在笔记本顶部菜单栏中，依次点击 “View” &gt; “Cell Toolbar” &gt; “None” 关闭元数据编辑栏 &lt;/li&gt;
    &lt;/ol&gt;
    &lt;p&gt; 下方是一段演示上述操作步骤的简短动图： 
        &lt;br&gt;
        &lt;img src=&quot;https://lh3.google.com/u/0/d/14Xy_Mb17CZVgzVAgq7NCjMVBvSae3xO1&quot; align=&quot;center&quot; alt=&quot;unlock_cells.gif&quot;&gt;
&lt;/details&gt;
