# 可选实验：逻辑回归 (Logistic Regression) 的梯度下降 (Gradient Descent) 

## 实验目标
在本实验中，你将：
- 针对逻辑回归更新梯度下降算法。
- 在一个熟悉的数据集 (data set) 上探索梯度下降的运行过程


```python
import copy, math
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import  dlc, plot_data, plt_tumor_data, sigmoid, compute_cost_logistic
from plt_quad_logistic import plt_quad_logistic, plt_prob
plt.style.use('./deeplearning.mplstyle')
```

## 数据集
让我们继续从决策边界 (decision boundary) 实验中使用过的那个双特征数据集开始。


```python
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])
```

和之前一样，我们将使用一个辅助函数 (helper function) 把这些数据绘制出来。标签 (label) 为 $y=1$ 的数据点显示为红色叉号，而标签为 $y=0$ 的数据点则显示为蓝色圆圈。


```python
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()
```

## 逻辑回归梯度下降 (Logistic Gradient Descent) 
&lt;img align=&quot;right&quot; src=&quot;./images/C1_W3_Logistic_gradient_descent.png&quot;     style=&quot; width:400px; padding: 10px; &quot; &gt;

回想一下，梯度下降算法的核心是利用梯度计算 (gradient calculation) 不断迭代更新参数：
$$\begin{align*}
&\text{repeat until convergence:} \; \lbrace \\
&  \; \; \;w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{for j := 0..n-1} \\ 
&  \; \; \;  \; \;b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \\
&\rbrace
\end{align*}$$

其中，每次迭代 (iteration) 都会对所有的 $j$ 同步更新 (simultaneous updates) 参数 $w_j$，偏导数 (partial derivatives) 的计算公式如下：
$$\begin{align*}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{2} \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3} 
\end{align*}$$

* $m$ 是数据集中训练样本 (training examples) 的数量      
* $f_{\mathbf{w},b}(x^{(i)})$ 是模型的预测值 (prediction) ，而 $y^{(i)}$ 则是真实的目标值 (target) 
* 对于逻辑回归模型 (logistic regression model)   
    $z = \mathbf{w} \cdot \mathbf{x} + b$  
    $f_{\mathbf{w},b}(x) = g(z)$  
    其中 $g(z)$ 是 S型函数 (sigmoid function) ：  
    $g(z) = \frac{1}{1+e^{-z}}$   
    


### 梯度下降的代码实现
梯度下降算法的实现主要包含两个核心部分：
- 实现上方方程 (1) 的主循环。这对应下方的 `gradient_descent` 函数，在可选实验和编程练习实验中通常会直接提供给你。
- 计算当前的梯度，即实现上方方程 (2, 3) 。这对应下方的 `compute_gradient_logistic` 函数。在本周的编程练习实验中，你将需要亲手编写这一函数。

#### 计算梯度，代码逻辑说明
针对所有的参数 $w_j$ 和 $b$ 实现上方的方程 (2) 和 (3) 。
实现这一计算有许多种方式，下方代码所采用的整体思路如下：
- 初始化用于累加 `dj_dw` 和 `dj_db` 的变量
- 遍历每一个训练样本
    - 计算该样本的预测误差 (error)  $g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b) - \mathbf{y}^{(i)}$
    - 遍历该样本中的每一个输入特征值 $x_{j}^{(i)}$，  
        - 将误差与输入特征 $x_{j}^{(i)}$ 相乘，并累加到 `dj_dw` 的对应元素中。 (对应上方方程 2) 
    - 将误差直接累加到 `dj_db` 中 (对应上方方程 3) 

- 将累加完的 `dj_db` 和 `dj_dw` 除以样本总数 ($m$) 
- 请注意，在 numpy 中，$\mathbf{x}^{(i)}$ 可以表示为 `X[i,:]` 或 `X[i]`，而 $x_{j}^{(i)}$ 则表示为 `X[i,j]`


```python
def compute_gradient_logistic(X, y, w, b): 
    """
    计算逻辑回归的梯度
 
    参数:
      X (ndarray (m,n): 数据，包含 m 个样本，每个样本有 n 个特征
      y (ndarray (m,)): 目标值
      w (ndarray (n,)): 模型参数  
      b (scalar)      : 模型参数
    返回
      dj_dw (ndarray (n,)): 代价相对于参数 w 的梯度。
      dj_db (scalar)      : 代价相对于参数 b 的梯度。
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                           #(n,)
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=标量 (scalar) 
        err_i  = f_wb_i  - y[i]                       #标量
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #标量
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #标量
        
    return dj_db, dj_dw  
```

运行下方的代码单元格，检验一下梯度计算函数的实现是否正确。


```python
X_tmp = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.array([2.,3.])
b_tmp = 1.
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic(X_tmp, y_tmp, w_tmp, b_tmp)
print(f"dj_db: {dj_db_tmp}" )
print(f"dj_dw: {dj_dw_tmp.tolist()}" )
```

**预期输出**
``` 
dj_db: 0.49861806546328574
dj_dw: [0.498333393278696, 0.49883942983996693]
```

#### 梯度下降主函数代码
下方代码具体实现了前面的方程 (1) 。请花一点时间仔细阅读，将代码中的函数调用与上面的数学方程一一对应起来。


```python
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    """
    执行批量梯度下降 (batch gradient descent) 
    
    参数:
      X (ndarray (m,n)   : 数据，包含 m 个样本，每个样本有 n 个特征
      y (ndarray (m,))   : 目标值
      w_in (ndarray (n,)): 模型参数的初始值  
      b_in (scalar)      : 模型参数的初始值
      alpha (float)      : 学习率 (Learning rate) 
      num_iters (scalar) : 运行梯度下降的迭代次数
      
    返回:
      w (ndarray (n,))   : 更新后的参数值
      b (scalar)         : 更新后的参数值
    """
    # 用于存储每次迭代的代价 J 和参数 w 的数组，主要用于后续绘图
    J_history = []
    w = copy.deepcopy(w_in)  # 避免在函数内部修改全局变量 w
    b = b_in
    
    for i in range(num_iters):
        # 计算梯度并更新参数
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)   

        # 使用 w、b、alpha 和梯度来更新参数
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # 保存每次迭代的代价 J
        if i<100000:      # 防止耗尽内存资源
            J_history.append( compute_cost_logistic(X, y, w, b) )

        # 每隔总迭代次数的 1/10 打印一次代价 (如果迭代次数小于 10，则每次都打印) 
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history         # 返回最终的 w、b 以及用于绘图的 J 历史记录

```

现在，让我们在数据集上运行梯度下降。


```python
w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alph, iters) 
print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
```

#### 让我们将梯度下降的结果绘制成图表：


```python
fig,ax = plt.subplots(1,1,figsize=(5,4))
# 绘制预测概率分布
plt_prob(ax, w_out, b_out)

# 绘制原始数据点
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')   
ax.axis([0, 4, 0, 3.5])
plot_data(X_train,y_train,ax)

# 绘制决策边界
x0 = -b_out/w_out[0]
x1 = -b_out/w_out[1]
ax.plot([0,x0],[x1,0], c=dlc["dlblue"], lw=1)
plt.show()
```

在上方图表中：
 - 背景渐变阴影反映了模型预测 y=1 的概率 (probability)  (即通过决策边界进行二值化判定之前的连续输出结果) 
 - 决策边界正是预测概率恰好等于 0.5 的那条分界线
 

## 另一个数据集
接下来，让我们回到单变量数据集 (one-variable data set) 。当模型只有 $w$ 和 $b$ 两个参数时，我们可以用等高线图 (contour plot) 将代价函数绘制出来，从而更直观地洞察梯度下降的工作原理。


```python
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])
```

和之前一样，我们将使用一个辅助函数把这些数据绘制出来。标签为 $y=1$ 的数据点显示为红色叉号，而标签为 $y=0$ 的数据点则显示为蓝色圆圈。


```python
fig,ax = plt.subplots(1,1,figsize=(4,3))
plt_tumor_data(x_train, y_train, ax)
plt.show()
```

在下方的交互式图表中，请尝试以下操作：
- 在右上角的等高线图内部点击鼠标，改变参数 $w$ 和 $b$ 的取值。
    - 图像更新可能需要一两秒钟的时间
    - 观察左上角图表中代价值随之发生的变化。
    - 请注意，总代价是由每个样本上的损失累加而成的 (图中用垂直虚线表示每个样本的损失) 
- 点击橙色按钮运行梯度下降。
    - 观察代价是如何稳步下降的 (等高线图和代价曲线图均采用 log(cost) 对数刻度绘制) 
    - 在等高线图中重新点击鼠标，即可重置模型参数以开启新一轮运行
- 如需重置整个图表，请重新运行该代码单元格


```python
w_range = np.array([-1, 7])
b_range = np.array([1, -14])
quad = plt_quad_logistic( x_train, y_train, w_range, b_range )
```

## 恭喜！
你已经完成了以下学习内容：
- 深入研究了逻辑回归梯度计算的数学公式与代码实现
- 实际运用这些程序完成了以下探索：
    - 探索单变量数据集上的参数优化
    - 探索双变量数据集上的参数优化


```python

```
