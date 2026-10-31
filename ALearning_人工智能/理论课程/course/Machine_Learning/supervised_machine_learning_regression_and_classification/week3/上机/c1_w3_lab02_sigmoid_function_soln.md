# 可选实验：逻辑回归 (Logistic Regression) 

在这个非计分实验中，你将：
- 探索 S型函数 (sigmoid function)  (也称为逻辑函数 (logistic function) ) 
- 探索使用 S型函数的逻辑回归


```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('./deeplearning.mplstyle')
```

## S型函数或逻辑函数
&lt;img align=&quot;left&quot; src=&quot;./images/C1_W3_LogisticRegression_left.png&quot;     style=&quot; width:300px; padding: 10px; &quot; &gt;正如课程视频中所讨论的，面对一项分类任务 (classification task) 时，我们可以先从熟悉的线性回归模型 (linear regression model)  $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot  \mathbf{x}^{(i)} + b$ 入手，根据给定的 $x$ 来预测 $y$。
- 然而，由于我们的输出变量 (output variable)  $y$ 只能取 0 或 1，因此我们希望分类模型 (classification model) 的预测值能够限制在 0 到 1 之间。
- 这一目标可以通过使用“S型函数”来实现，它能将所有输入值映射到 0 到 1 之间的数值。


下面让我们亲手实现 S型函数，直观感受一下它的效果。

## S型函数的公式

S型函数的数学公式如下 -  

$g(z) = \frac{1}{1+e^{-z}}\tag{1}$

在逻辑回归中，$z$ (即 S型函数的输入) 正是线性回归模型的输出结果。
- 对于单个样本 (single example) 而言，$z$ 是一个标量 (scalar) 。
- 对于多个样本而言，$z$ 可能是一个包含 $m$ 个数值的向量 (vector) ，每个样本对应其中一个值。
- S型函数的代码实现应当能够同时兼容这两种潜在的输入格式。
下面让我们用 Python 来实现它。

NumPy 提供了一个名为 `exp()` 的函数，它可以非常方便地计算输入数组 (`z`) 中所有元素的指数 (exponential)  ( $e^{z}$) 。
 
如下方代码所示，当输入是单个数字时，该函数同样适用。


```python
# 输入是一个数组 (array) 。
input_array = np.array([1,2,3])
exp_array = np.exp(input_array)

print("Input to exp:", input_array)
print("Output of exp:", exp_array)

# 输入是单个数字
input_val = 1  
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)
```

下方代码单元格展示了 `sigmoid` 函数在 Python 中的具体实现。


```python
def sigmoid(z):
    """
    计算 z 的 sigmoid 值

    参数:
        z (ndarray): 一个标量，或任意大小的 numpy 数组。

    返回:
        g (ndarray): sigmoid(z)，形状与 z 保持一致
         
    """

    g = 1/(1+np.exp(-z))
   
    return g
```

让我们来看看当 `z` 取不同数值时，这个函数会输出什么结果。


```python
# 生成一个在 -10 到 10 之间均匀分布的数组
z_tmp = np.arange(-10,11)

# 调用上面实现的函数来获取 sigmoid 值
y = sigmoid(z_tmp)

# 将两个数组并排美观打印的代码
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])
```

左列中的数值是输入 `z`，右列中的数值则是计算出的 `sigmoid(z)`。正如你所看到的，S型函数的输入值范围从 -10 到 10，而对应的输出值则被限制在 0 到 1 之间。

现在，让我们尝试使用 `matplotlib` 库将这个函数绘制成图表。


```python
# 绘制 z 与 sigmoid(z) 的关系曲线
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
```

从图中可以清晰地看到，当 `z` 趋向于较大的负值时，S型函数的输出逼近 `0`；而当 `z` 趋向于较大的正值时，S型函数的输出逼近 `1`。


## 逻辑回归
&lt;img align=&quot;left&quot; src=&quot;./images/C1_W3_LogisticRegression_right.png&quot;     style=&quot; width:300px; padding: 10px; &quot; &gt; 逻辑回归模型将 S型函数应用于我们所熟悉的线性回归模型之上，具体表达式如下：

$$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b ) \tag{2} $$ 

  其中

  $g(z) = \frac{1}{1+e^{-z}}\tag{3}$


  
让我们把逻辑回归应用到肿瘤分类这个分类数据 (categorical data) 示例中。  
首先，加载训练样本以及参数 (parameters) 的初始值。
  



```python
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0
```

请尝试按以下步骤操作：
- 点击“Run Logistic Regression” (运行逻辑回归) ，为给定的训练数据 (training data) 寻找最佳的逻辑回归模型
    - 请注意，生成的模型能够非常好地拟合数据。
    - 请注意，图中的橙色直线代表上式中的“$z$”或 $\mathbf{w} \cdot \mathbf{x}^{(i)} + b$。它与线性回归模型中的直线并不相同。
接下来，通过引入一个*阈值 (threshold) *来进一步提升结果的表现。
- 勾选“Toggle 0.5 threshold” (切换 0.5 阈值) 复选框，查看应用阈值之后的预测效果。
    - 这些预测看起来非常准确，预测结果与实际数据完全吻合
    - 现在，在较大的肿瘤尺寸范围内 (靠近 10 的位置) 继续添加几个数据点，然后重新运行逻辑回归。
    - 与线性回归模型不同，该模型依然能够持续做出正确的预测


```python
plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)
```

## 恭喜！
你已经探索了 S型函数在逻辑回归中的核心应用。


```python

```
