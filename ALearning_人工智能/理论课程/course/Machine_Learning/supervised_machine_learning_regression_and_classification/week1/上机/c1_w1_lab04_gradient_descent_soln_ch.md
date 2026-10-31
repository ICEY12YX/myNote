
___
- [[#选修实验：线性回归中的梯度下降 (Gradient Descent for Linear Regression)|选修实验：线性回归中的梯度下降 (Gradient Descent for Linear Regression)]]
	- [[#选修实验：线性回归中的梯度下降 (Gradient Descent for Linear Regression)#实验目标|实验目标]]
	- [[#选修实验：线性回归中的梯度下降 (Gradient Descent for Linear Regression)#实验工具|实验工具]]
- [[#问题背景|问题背景]]
		- [[#实验工具#计算代价函数 (Compute_Cost)|计算代价函数 (Compute_Cost)]]
	- [[#问题背景#梯度下降算法核心回顾|梯度下降算法核心回顾]]
	- [[#问题背景#动手实现梯度下降|动手实现梯度下降]]
		- [[#动手实现梯度下降#实现 compute_gradient 函数|实现 compute_gradient 函数]]
		- [[#动手实现梯度下降#实现梯度下降主函数 (Gradient Descent)|实现梯度下降主函数 (Gradient Descent)]]
		- [[#动手实现梯度下降#代价随梯度下降迭代次数的变化曲线|代价随梯度下降迭代次数的变化曲线]]
		- [[#动手实现梯度下降#模型预测 (Predictions)|模型预测 (Predictions)]]
	- [[#问题背景#寻优轨迹可视化 (Plotting)|寻优轨迹可视化 (Plotting)]]
		- [[#寻优轨迹可视化 (Plotting)#调大学习率会发生什么|调大学习率会发生什么]]
	- [[#问题背景#恭喜！|恭喜！]]

___

# 选修实验：线性回归中的梯度下降 (Gradient Descent for Linear Regression) 

![[c1_w1_lab04_gradient_descent_soln_ch-1793365523305.webp]]


## 实验目标
在本次实验中，你将：
- 利用梯度下降 (gradient descent) 算法，让寻找最优参数 $w$ 和 $b$ 的过程实现自动化。

## 实验工具
在本次实验中，我们将用到以下工具：
- NumPy，一个广受欢迎的科学计算 (scientific computing) 库
- Matplotlib，一个广受欢迎的数据可视化绘图 (plotting data) 库
- 当前目录下 lab_utils.py 文件中预先封装好的绘图函数


```python
import math, copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients
```

# 问题背景

我们继续沿用之前的两个数据点 (data points) —— 一套面积为 1000 平方英尺的房子售价为 \$300,000，另一套面积为 2000 平方英尺的房子售价为 \$500,000。

| 面积 (1000 平方英尺)     | 价格 (千美元) |
| ----------------| ------------------------ |
| 1               | 300                      |
| 2               | 500                      |

```python
# Load our data set
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value
```


### 计算代价函数 (Compute_Cost) 
这个函数我们在上一个实验中已经编写过了，这次实验还会继续用到它。

```python
#Function to calculate the cost
def compute_cost(x, y, w, b):
   
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    total_cost = 1 / (2 * m) * cost

    return total_cost
```


## 梯度下降算法核心回顾
在前面的课程中，我们已经构建了一个能够给出预测值 $f_{w,b}(x^{(i)})$ 的线性模型 (linear model) ：
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{1}$$
在线性回归 (linear regression) 中，我们会利用输入的训练数据 (training data) ，通过不断缩小模型预测值 $f_{w,b}(x^{(i)})$ 与真实数据 $y^{(i)}$ 之间的误差指标，来拟合出最佳的参数 (parameters) $w$ 和 $b$。这个衡量误差的指标就叫做代价 ($cost$) ，记作 $J(w,b)$。在模型训练过程中，我们会在全部训练样本 (training samples) $x^{(i)},y^{(i)}$ 上计算总代价：
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2\tag{2}$$ 

在理论课上，梯度下降的执行过程被概括为：
$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3}  \; \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}$$

其中，参数 $w$ 和 $b$ 需要**同时 (simultaneously) 完成更新**。  
而梯度 (gradient) 的具体计算公式定义如下：
$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4}\\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{5}\\
\end{align}
$$

这里的同时更新意味着：你必须==先算==出所有参数在当前位置的==偏导数== (partial derivatives) ，然后==再统一去更新各个参数的数值==。

## 动手实现梯度下降
接下来，你将针对单特征 (one feature) 场景编写梯度下降算法。整个过程一共需要三个核心函数：
- `compute_gradient`：用来实现上方的公式 (4) 和公式 (5) 
- `compute_cost`：用来实现上方的公式 (2) (直接复用上一个实验的代码) 
- `gradient_descent`：综合调用 `compute_gradient` 和 `compute_cost` 来执行完整的梯度下降流程

命名约定：
- 在 Python 代码中，用来存储偏导数的变量名统一遵循以下命名规范：例如 $\frac{\partial J(w,b)}{\partial b}$ 在代码中会被命名为 `dj_db`。
- 注释中的缩写 `w.r.t` 代表“关于” (With Respect To) ，例如 $J(wb)$ 关于参数 $b$ 的偏导数。


### 实现 compute_gradient 函数

`compute_gradient` 函数负责实现上方的公式 (4) 和公式 (5) ，并最终返回偏导数 $\frac{\partial J(w,b)}{\partial w}$ 和 $\frac{\partial J(w,b)}{\partial b}$。代码内部的注释详细说明了每一步的具体操作。

```python
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw / m 
    dj_db = dj_db / m 
        
    return dj_dw, dj_db
```

![[c1_w1_lab04_gradient_descent_soln_ch-1793365907104.webp|380]]
理论课上曾讲过，梯度下降是如何利用代价函数在某一点处关于某个参数的偏导数来更新该参数的。   
下面，我们就调用刚刚写好的 `compute_gradient` 函数，求出代价函数关于其中一个参数 $w_0$ 的几处偏导数，并将它们画在图表上。

```python
plt_gradients(x_train,y_train, compute_cost, compute_gradient)
plt.show()
```
![[c1_w1_lab04_gradient_descent_soln_ch-1793365965636.webp|561]]

在上方左侧的图表中，展示了代价曲线在三个不同位置处关于参数 $w$ 的偏导数 $\frac{\partial J(w,b)}{\partial w}$ (也就是切线斜率 (slope) ) 。
可以看到，在曲线右半部分，导数 (derivative) 为正值；而在左半部分，导数则为负值。正是因为代价曲线呈现出这种“碗状结构” ('bowl shape') ，无论从哪一侧出发，导数都会始终引导着梯度下降一步步走向斜率为零的碗底。
 
左侧图表将参数 $b$ 固定在了 $b=100$。而在实际运行中，梯度下降会同时利用 $\frac{\partial J(w,b)}{\partial w}$ 和 $\frac{\partial J(w,b)}{\partial b}$ 来更新这两个参数。右侧的“箭头场图” ('quiver plot') 则为我们提供了一种同时观察两个参数梯度的直观视角。图中==箭头的大小反映了该点处梯度的幅值 (magnitude) ==，而==箭头的指向与倾斜角度则体现了该点处 $\frac{\partial J(w,b)}{\partial w}$ 与 $\frac{\partial J(w,b)}{\partial b}$ 之间的比例关系。==
    越长的箭头代表梯度向量的模越长  这个箭头颜色越浅
请特别留意：图中所有箭头的指向都是背离最低点 (minimum) 的。不妨回顾一下上方的公式 (3) ：我们在更新参数时，是用当前的 $w$ 或 $b$ 数值**减去**按比例缩放后的梯度。这样一减，参数就会顺着能够让代价不断降低的反方向迈进。
    因为梯度方向是函数变大最快的方向


### 实现梯度下降主函数 (Gradient Descent) 
既然现在已经能够顺利算出梯度，我们就可以在下方的 `gradient_descent` 函数中正式实现公式 (3) 所描述的梯度下降算法了。具体实现细节已经写在代码注释中。接下来，你将利用这个函数，在训练数据上自动找出参数 $w$ 和 $b$ 的最优值 (optimal values) 。

```python
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):  #num_iters 是你自己设定的迭代次数
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 防止资源耗尽 
            J_history.append(cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        # 在训练过程中打印 cost，总共大约打印 10 次；如果总迭代次数不到 10 次，那就每次迭代都打印（即打印 `num_iters` 次）
        if i % math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing
```
解释
    ![[c1_w1_lab04_gradient_descent_soln_ch-1793366543422.webp|541]]
    ![[c1_w1_lab04_gradient_descent_soln_ch-1793366568415.webp|540]]


```python
# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train,
    y_train,
    w_init,
    b_init,
    tmp_alpha,
    iterations,
    compute_cost,
    compute_gradient,
)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
```
![[c1_w1_lab04_gradient_descent_soln_ch-1793366651218.webp]]

![[c1_w1_lab04_gradient_descent_soln_ch-1793366669238.webp|445]]
不妨花点时间观察一下上方打印出的运行日志，看看梯度下降过程呈现出哪些**特点**：
- ==代价在起步阶段非常高==，随后便如理论课幻灯片所描述的那样==迅速下降==。
- 两个==偏导数== `dj_dw` 和 `dj_db` 的数值也在==不断变小==，起初下降得很快，随后==逐渐放缓==。正如理论课示意图所展示的，当寻优过程逐渐逼近“碗底”时，由于那里的导数绝对值越来越小，每次迈出的步伐也会自然变慢。
- 尽管学习率 (learning rate) alpha 始终保持固定不变，但==参数更新的步伐还是会随着接近谷底而自动放缓==。

### 代价随梯度下降迭代次数的变化曲线

绘制一张“代价随迭代次数 (iterations) 变化”的折线图，是检验梯度下降推进效果的绝佳方式。
==在一次成功的梯度下降运行中，代价应当自始至终保持下降趋势。==由于代价在最初几步下降得极为迅猛，==我们通常会把起步阶段的下降曲线和尾声阶段的下降曲线放在不同比例尺的坐标系中分别展示==。请留意下方两张子图中纵轴代价的刻度范围以及横轴的迭代步数。

```python
# plot cost versus iteration  
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])  #前100次迭代
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:]) #从1000次迭代往后
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel("Cost")
ax2.set_ylabel("Cost")
ax1.set_xlabel("iteration step")
ax2.set_xlabel("iteration step")
plt.show()
```
![[c1_w1_lab04_gradient_descent_soln_ch-1793366885130.webp]]

### 模型预测 (Predictions) 

既然你已经通过梯度下降找到了参数 $w$ 和 $b$ 的最优值，现在就可以利用学到的参数来预测房价了。不出所料，对于训练集里已有的房屋面积，模型给出的预测房价与真实售价几乎完全一致；而对于不在训练集中的房屋面积，模型给出的预测结果也完全符合预期。

```python
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")
```
![[c1_w1_lab04_gradient_descent_soln_ch-1793366911064.webp]]


## 寻优轨迹可视化 (Plotting) 

你还可以把每一轮迭代时的代价画在 cost(w,b) 的等高线图 (contour plot) 上，直观地展现出梯度下降在执行过程中的前进轨迹。

```python
fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)
```
![[c1_w1_lab04_gradient_descent_soln_ch-1793366950611.webp]]
就是在 代价函数-参数 图中  看迭代后,w,b变换趋势, 代价变化趋势

上方的等高线图展示了在不同 $w$ 和 $b$ 取值范围内的代价分布 $cost(w,b)$。图中的一圈圈椭圆环代表不同的代价水平，而叠加在上面的红色箭头则是梯度下降走过的路径。这里有几个值得留意的细节：
- 整条寻优路径始终==朝着目标稳步 (单调 (monotonic) ) 推进。==
- ==起步阶段的步幅==明显比接近目标时的步幅==大==得多。

**局部放大观察**，我们可以更清晰地看到梯度下降在==最后几个阶段迈出的微小步伐==。注意观察，随着梯度逐渐趋近于零，相邻两步之间的距离也在不断缩短。

```python
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
plt_contour_wgrad(
    x_train,
    y_train,
    p_hist,
    ax,
    w_range=[180, 220, 0.5],
    b_range=[80, 120, 0.5],
    contours=[1, 5, 10, 20],
    resolution=0.5,
)
```
![[c1_w1_lab04_gradient_descent_soln_ch-1793367767078.webp]]

### 调大学习率会发生什么

![[c1_w1_lab04_gradient_descent_soln_ch-1793367786506.webp|428]]
在理论课上，我们曾专门探讨过如何为公式 (3) 中的学习率 $\alpha$ 挑选一个合适的数值。$\alpha$ 设得越大，梯度下降收敛 (converge) 到最优解的速度就越快；但如果把 $\alpha$ 设得过大，梯度下降不仅无法收敛，反而会发生发散 (diverge) 。上面我们已经看到了一个平稳收敛的正面例子。

现在，让我们试着把 $\alpha$ 的数值调大一些，看看会发生什么情况：

```python
# initialize parameters
w_init = 0
b_init = 0
# set alpha to a large value
iterations = 10
tmp_alpha = 8.0e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(
    x_train,
    y_train,
    w_init,
    b_init,
    tmp_alpha,
    iterations,
    compute_cost,
    compute_gradient,
)
```
![[c1_w1_lab04_gradient_descent_soln_ch-1793367818277.webp]]

从上方的打印结果可以看到，==$w$ 和 $b$ 的数值在正负号之间来回剧烈震荡==，而且它们的绝对值随着每一次迭代越变越大。此外，每一轮迭代后偏导数 $\frac{\partial J(w,b)}{\partial w}$ 都会改变正负号，==总代价非但没有下降，反而在不断飙升==。这是一个非常典型的信号，说明**学习率设得太大了**，求解过程正在不断发散。
下面让我们用图表把这个发散的过程直观地画出来。

```python
plt_divergence(p_hist, J_hist,x_train, y_train)
plt.show()
```
![[c1_w1_lab04_gradient_descent_soln_ch-1793367868679.webp]]

在上方左侧的图表中，展示了梯度下降前几步里参数 $w$ 的变化轨迹。可以看到，$w$ 在正负两侧剧烈来回摆动，导致代价迅速暴涨。由于梯度下降是同时对 $w$ 和 $b$ 两个参数进行操作的，因此我们需要结合右侧的三维 (3-D) 图表才能看清完整的全貌。

## 恭喜！
在本次实验中，你已经完成了以下探索：
- 深入剖析了单变量梯度下降算法的内部细节。
- 编写了用于计算梯度的核心函数
- 通过图表直观认识了梯度的物理与几何意义
- 完成了梯度下降主函数的代码实现
- 利用梯度下降自动找出了最优模型参数
- 亲自验证了不同学习率大小对算法收敛与发散的影响

