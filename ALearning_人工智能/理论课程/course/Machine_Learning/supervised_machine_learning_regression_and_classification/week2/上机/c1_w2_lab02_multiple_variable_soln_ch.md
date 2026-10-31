# 选修实验：多变量线性回归 (Multiple Variable Linear Regression) 

在本次实验中，你将对之前构建的数据结构和函数例程进行扩展，使它们能够支持包含多个特征 (multiple features) 的数据。由于需要对好几个函数进行升级，这份实验表面上看起来篇幅较长，但其实每个函数都只是在原有基础上做了微小的调整，因此阅读和回顾起来会非常轻松。
# 目录大纲
- &nbsp;&nbsp;1.1 实验目标
- &nbsp;&nbsp;1.2 实验工具
- &nbsp;&nbsp;1.3 符号说明
- 2 问题背景
- &nbsp;&nbsp;2.1 存放训练样本的矩阵 X
- &nbsp;&nbsp;2.2 参数向量 w 与偏置 b
- 3 多变量模型的预测
- &nbsp;&nbsp;3.1 逐元素循环的单样本预测
- &nbsp;&nbsp;3.2 基于向量的单样本预测
- 4 计算多变量代价函数
- 5 多变量梯度下降
- &nbsp;&nbsp;5.1 计算多变量梯度
- &nbsp;&nbsp;5.2 执行多变量梯度下降
- 6 恭喜


&lt;a name=&quot;toc_15456_1.1&quot;&gt;&lt;/a&gt;
## 1.1 实验目标
- 扩展我们的回归模型 (regression model) 代码，使其支持多特征输入
    - 升级数据结构以容纳多个特征
    - 重写预测、代价 (cost) 以及梯度 (gradient) 计算函数，使其支持多个特征
    - 利用 NumPy 的点积函数 `np.dot` 对代码实现进行向量化 (vectorize) ，从而兼顾更快的运行速度与更简洁的代码结构

&lt;a name=&quot;toc_15456_1.2&quot;&gt;&lt;/a&gt;
## 1.2 实验工具
在本次实验中，我们将用到以下工具：
- NumPy，一个广受欢迎的科学计算 (scientific computing) 库
- Matplotlib，一个广受欢迎的数据可视化绘图 (plotting data) 库


```python
import copy, math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
```

&lt;a name=&quot;toc_15456_1.3&quot;&gt;&lt;/a&gt;
## 1.3 符号说明 (Notation) 
下面汇总了你在本实验中会用到的数学符号，其中针对多特征场景进行了相应更新。  

|通用 &lt;img width=70/&gt; &lt;br /&gt;  符号  &lt;img width=70/&gt; | 含义说明&lt;img width=350/&gt;| Python 变量名 (如适用) |
|: ------------|: ------------------------------------------------------------||
| $a$ | 标量 (scalar) ，非粗体                                                      ||
| $\mathbf{a}$ | 向量 (vector) ，小写粗体                                                 ||
| $\mathbf{A}$ | 矩阵 (matrix) ，大写粗体                                         ||
| **回归 (Regression) ** |         |    |     |
|  $\mathbf{X}$ | 训练样本矩阵 (training example matrix)                  | `X_train` |   
|  $\mathbf{y}$  | 训练样本目标值 (targets)                | `y_train` 
|  $\mathbf{x}^{(i)}$, $y^{(i)}$ | 第 $i$ 个训练样本 | `X[i]`, `y[i]`|
| m | 训练样本的总数量 | `m`|
| n | 每个样本中包含的特征数量 | `n`|
|  $\mathbf{w}$  |  参数 (parameter) ：权重向量 (weight)                       | `w`    |
|  $b$           |  参数：偏置 (bias)                                           | `b`    |     
| $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ | 以 $\mathbf{w},b$ 为参数的模型在样本 $\mathbf{x^{(i)}}$ 处的预测结果：$f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)}+b$  | `f_wb` | 


&lt;a name=&quot;toc_15456_2&quot;&gt;&lt;/a&gt;
# 2 问题背景

我们依然以经典的“房价预测”为例。如下表所示，本次的训练数据集 (training dataset) 一共包含 3 个样本，每个样本都有 4 个特征 (房屋面积、卧室数量、楼层数以及房龄) 。请特别注意：与之前的实验不同，这里的房屋面积直接以“平方英尺” (sqft) 为单位，而不是“1000 平方英尺”。这种数值尺度上的差异会引发一个新的问题，我们将在下一个实验中专门解决它！

| 面积 (平方英尺) | 卧室数量  | 楼层数 | 房龄 | 价格 (千美元)  |   
| ----------------| ------------------- |----------------- |--------------|-------------- |  
| 2104            | 5                   | 1                | 45           | 460           |  
| 1416            | 3                   | 2                | 40           | 232           |  
| 852             | 2                   | 1                | 35           | 178           |  

你将利用这些数据构建一个线性回归模型，以便后续用来预测其他房屋的价格——比如一套面积为 1200 平方英尺、拥有 3 间卧室、1 层楼高、房龄 40 年的房子。  

请运行下方的代码单元格来创建 `X_train` 和 `y_train` 变量。


```python
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
```

&lt;a name=&quot;toc_15456_2.1&quot;&gt;&lt;/a&gt;
## 2.1 存放训练样本的矩阵 X
与上方表格的结构类似，所有的训练样本都被存放在一个 NumPy 矩阵 `X_train` 中，矩阵的每一行都代表一个独立的样本。当你拥有 $m$ 个训练样本 (在本例中 $m$ 为 3) ，且每个样本包含 $n$ 个特征 (在本例中为 4) 时，$\mathbf{X}$ 就是一个维度为 ($m$, $n$) (即 m 行、n 列) 的矩阵。


$$\mathbf{X} = 
\begin{pmatrix}
 x^{(0)}_0 & x^{(0)}_1 & \cdots & x^{(0)}_{n-1} \\ 
 x^{(1)}_0 & x^{(1)}_1 & \cdots & x^{(1)}_{n-1} \\
 \cdots \\
 x^{(m-1)}_0 & x^{(m-1)}_1 & \cdots & x^{(m-1)}_{n-1} 
\end{pmatrix}
$$
符号约定：
- $\mathbf{x}^{(i)}$ 是包含第 i 个样本全部特征的向量，即 $\mathbf{x}^{(i)}$ $ = (x^{(i)}_0, x^{(i)}_1, \cdots,x^{(i)}_{n-1})$
- $x^{(i)}_j$ 表示第 i 个样本中的第 j 个特征元素。带括号的上标用来标明样本序号，而下标则用来标明特征序号。  

下面让我们打印查看一下输入数据：


```python
# data is stored in numpy array/matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)
```

&lt;a name=&quot;toc_15456_2.2&quot;&gt;&lt;/a&gt;
## 2.2 参数向量 w 与偏置 b

* $\mathbf{w}$ 是一个包含 $n$ 个元素的向量。
  - 其中的每一个元素都对应着某一个特征的权重参数。
  - 在我们的数据集中，特征数 n 为 4。
  - 在数学概念上，我们通常把它画成一个列向量 (column vector) ：

$$\mathbf{w} = \begin{pmatrix}
w_0 \\ 
w_1 \\
\cdots\\
w_{n-1}
\end{pmatrix}
$$
* $b$ 则是一个单独的标量参数。  

为了方便演示，下面我们先为 $\mathbf{w}$ 和 $b$ 载入一组预先选好的、非常接近最优解 (optimal) 的初始值。其中，$\mathbf{w}$ 是一个一维 (1-D) NumPy 向量。


```python
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")
```

&lt;a name=&quot;toc_15456_3&quot;&gt;&lt;/a&gt;
# 3 多变量模型的预测
多变量线性模型的预测公式可以展开写为：

$$ f_{\mathbf{w},b}(\mathbf{x}) =  w_0x_0 + w_1x_1 +... + w_{n-1}x_{n-1} + b \tag{1}$$
或者用更简洁的向量符号表示为：
$$ f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b  \tag{2} $$ 
其中 $\cdot$ 代表向量的`点积` (dot product) 运算。

为了直观展示点积的威力，下面我们将分别按照公式 (1) 和公式 (2) 来实现模型预测。

&lt;a name=&quot;toc_15456_3.1&quot;&gt;&lt;/a&gt;
## 3.1 逐元素循环的单样本预测
在之前的单变量预测中，我们只需把唯一的特征值乘以对应的权重参数，再加上偏置参数即可。如果要把这种思路直接推广到多特征场景，最直观的做法就是按照上方的公式 (1) ，写一个循环逐一遍历每个特征元素，将它与对应的权重参数相乘并累加起来，最后在循环结束时加上偏置参数。



```python
def predict_single_loop(x, w, b): 
    """
    single predict using linear regression
    
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters    
      b (scalar):  model parameter     
      
    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]  
        p = p + p_i         
    p = p + b                
    return p
```


```python
# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
```

请留意 `x_vec` 的形状：它是一个包含 4 个元素的一维 NumPy 向量，形状为 `(4,)`。而最终计算出的预测结果 `f_wb` 则是一个标量。

&lt;a name=&quot;toc_15456_3.2&quot;&gt;&lt;/a&gt;
## 3.2 基于向量的单样本预测

注意到上方的公式 (1) 其实完全可以用公式 (2) 中的向量点积来实现，我们就可以借助向量化运算来大幅提升预测的速度。

回想一下我们在 Python/NumPy 实验中了解过的内容，NumPy 的 `np.dot()` [链接] 函数可以直接用来计算两个向量的点积。


```python
def predict(x, w, b): 
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b     
    return p    
```


```python
# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")
```

可以看到，计算出的结果与数组形状跟前面使用 `for` 循环的版本完全一致。在后续的内容中，我们将统一使用 `np.dot` 来执行这类运算。如此一来，整个预测过程就浓缩成了短短一行代码，因此大多数函数都会直接内联写出这行算式，而不再单独调用一个外部的 `predict` 函数。

&lt;a name=&quot;toc_15456_4&quot;&gt;&lt;/a&gt;
# 4 计算多变量代价函数
多变量代价函数 (cost function) $J(\mathbf{w},b)$ 的计算公式为：
$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 \tag{3}$$ 
其中：
$$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b  \tag{4} $$ 


与之前的实验不同，这里的 $\mathbf{w}$ 和 $\mathbf{x}^{(i)}$ 不再是单个标量，而是支持多个特征的向量。

下方代码实现了公式 (3) 和公式 (4) 。请注意，这里采用了*本课程的标准编程模式*——即使用一个 `for` 循环来遍历全部 `m` 个训练样本。


```python
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost
```


```python
# Compute and display cost using our pre-chosen optimal parameters. 
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')
```

**预期输出结果**：Cost at optimal w : 1.5578904045996674e-12

&lt;a name=&quot;toc_15456_5&quot;&gt;&lt;/a&gt;
# 5 多变量梯度下降
多变量场景下的梯度下降 (Gradient descent) 算法表示如下：

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{5}  \; & \text{for j = 0..n-1}\newline
&b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
\end{align*}$$

其中，n 代表特征的数量，各个参数 $w_j$ 与 $b$ 需要同时完成更新；而对应的偏导数计算公式分别为：

$$
\begin{align}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{6}  \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{7}
\end{align}
$$
* m 是数据集中训练样本的总数

    
* $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ 是模型给出的预测值，而 $y^{(i)}$ 则是真实的目标值


&lt;a name=&quot;toc_15456_5.1&quot;&gt;&lt;/a&gt;
## 5.1 计算多变量梯度
下方给出了计算公式 (6) 和公式 (7) 的具体代码实现。实现这一逻辑的方法有很多，在当前这个版本中：
- 外层循环负责遍历全部 m 个样本。
    - 当前样本对 $\frac{\partial J(\mathbf{w},b)}{\partial b}$ 的贡献可以直接算出并累加起来
    - 内层嵌套的第二个循环则负责遍历全部 n 个特征：
        - 针对每一个权重参数 $w_j$，分别计算并累加它对应的偏导数分量 $\frac{\partial J(\mathbf{w},b)}{\partial w_j}$。
   


```python
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw
```


```python
#Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')
```

**预期输出结果**：   
dj_db at initial w,b: -1.6739251122999121e-06  
dj_dw at initial w,b:   
 [-2.73e-03 -6.27e-06 -2.22e-06 -6.92e-05]  

&lt;a name=&quot;toc_15456_5.2&quot;&gt;&lt;/a&gt;
## 5.2 执行多变量梯度下降
下方的函数实现了前面列出的公式 (5) 。


```python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing
```

在接下来的单元格中，我们将对这段梯度下降代码进行测试。


```python
# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
```

**预期输出结果**：    
b,w found by gradient descent: -0.00,[ 0.2   0.   -0.01 -0.07]   
prediction: 426.19, target value: 460  
prediction: 286.17, target value: 232  
prediction: 171.47, target value: 178  


```python
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()
```

*目前的运行结果看起来还不太理想*！代价虽然仍在缓慢下降，但模型给出的预测值还不够准确。在下一个实验中，我们将深入探索如何一步步改进这一表现。


&lt;a name=&quot;toc_15456_6&quot;&gt;&lt;/a&gt;
# 6 恭喜！
在本次实验中，你已经成功：
- 重新开发了支持多变量的线性回归核心函数。
- 利用 NumPy 的 `np.dot` 完成了算法实现的向量化升级
