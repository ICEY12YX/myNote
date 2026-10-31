# 可选实验：逻辑回归 (Logistic Regression) 与决策边界 (Decision Boundary) 


## 实验目标
在本实验中，你将：
- 绘制逻辑回归模型 (logistic regression model) 的决策边界。这将帮助你更直观地理解模型是如何做出预测的。



```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, draw_vthresh
plt.style.use('./deeplearning.mplstyle')
```

## 数据集 (Dataset) 

假设你拥有以下训练数据集 (training dataset) 
- 输入变量 (input variable)  `X` 是一个 numpy 数组 (numpy array) ，包含 6 个训练样本 (training examples) ，每个样本具有两个特征 (features) 
- 输出变量 (output variable)  `y` 也是一个包含 6 个样本的 numpy 数组，且 `y` 的取值只能是 `0` 或 `1`


```python
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 
```

### 绘制数据图表

让我们使用一个辅助函数 (helper function) 将这些数据绘制出来。标签 (label) 为 $y=1$ 的数据点显示为红色叉号，而标签为 $y=0$ 的数据点则显示为蓝色圆圈。


```python
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()
```



&lt;div style=&quot;display: inline-block;&quot;&gt;
    &lt;div class=&quot;jupyter-widgets widget-label&quot; style=&quot;text-align: center;&quot;&gt;
        图
    &lt;/div&gt;
    &lt;img src=&#39;data:image/png;base64,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/frXv9amTZu8aqzaZp31ZsU2+/Of/6wPP/xQv/nNbzqtDcbf0eDZj+tEMF+A2NXeBg8erKioqNZ1FixYoL179/Zob10RzBdtWrnNnnvuOW3ZskWbNm3SgAEDvJZbud06683qz9qiRYtUXl6us2fPeswPhs+av96s2Gbvv/++PvvsM6Wmpmr8+PGqra1Vdna2181lg2G7+WKbAAnmCxC72tupU6fkdDolSY2Njdq+fbtSU1N7tLeuCOaLNq3aZmvXrlVJSYk2b97scfy8Lau2W1d6C/R2a2ho0MmTJ1unt2zZopiYGA0ZMsSjzopt1tXerPisPfTQQzpy5Ig+/vhjffzxxxo5cqTefvttzZgxw6MuWH9HbXUdSDBfgNiV3l555RWtX79eoaGhunz5subNm6fHH3/c78DZ1fDoo49q27Ztqqur09ChQzVw4EDt378/KLZZV3qzYpvV1NTohhtuUFJSkgYNGiRJioiI0M6dOy3fbl3tLdDb7cSJE1q8eLGamprkcDg0bNgwPfnkk0pNTbV8m3W1Nys+a+21/CE6btw4y7dbV9gqQAAAwcM2h7AAAMGFAAEAGCFAAABGCBAAgBECBABghAABABghQAAARggQAIARAgTw45VXXtF1113X+lChxsZGTZ48WStWrLC4MyA4ECCAH7m5uQoPD1dxcbGcTqeWLVum+Ph4PfXUU1a3BgQFW93OHQikiIgIPfzww1q9erU+//xznThxQmVlZQoNDbW6NSAocC8soAPffvutxo0bp9DQUO3cudPjFtobN27UunXrJEm/+tWvdPvtt1vVJmAJ9kCADmzatEnnzp3TkCFDNHTo0Nb5586dU2Fhof74xz/qwoUL+od/+AdVVFSwd4I+hTEQwI+Kigr98z//s0pKStSvXz+9+uqrrcs++OAD3Xrrrerfv79iYmIUHx+vo0ePWtgtEHgECOBDZWWlFi9erFWrVun222/XQw89pGeffVbffPONJOmrr77yeJhTdHS0vvrqK4u6BaxBgADt1NfX64477lBubq4WLVokScrJyVF4eHjrXkhMTIzq6+tb1zl37pxiYmKsaBewDIPogIH6+nplZmZq165dunDhgmbPnq3y8nKFhTGsiL6DTztgIDo6Wg8++KDmzJkjh8OhlStXEh7oc9gDAQAYYQwEAGCEAAEAGCFAAABGCBAAgBECBABghAABABghQAAARggQAIARAgQAYIQAAQAYIUAAAEb+P2Wz/rU+Z9N0AAAAAElFTkSuQmCC&#39; width=400.0/&gt;
&lt;/div&gt;



## 逻辑回归模型


* 假设你想在这些数据上训练一个逻辑回归模型，其数学形式如下：

  $f(x) = g(w_0x_0+w_1x_1 + b)$
  
  其中 $g(z) = \frac{1}{1+e^{-z}}$，即 S型函数 (sigmoid function) 


* 假设你已经完成了模型训练，并求得参数 (parameters) 为 $b = -3, w_0 = 1, w_1 = 1$。也就是说，

  $f(x) = g(x_0+x_1-3)$

   (你将在后续课程中进一步学习如何根据数据拟合这些参数) 
  
  
下面让我们通过绘制决策边界，来深入理解这个训练好的模型是如何进行预测的。

### 温习逻辑回归与决策边界

* 回想一下，对于逻辑回归，模型的表达式为：

  $$f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b) \tag{1}$$

  其中 $g(z)$ 被称为 S型函数，它将所有输入值映射到 0 到 1 之间的数值：

  $g(z) = \frac{1}{1+e^{-z}}\tag{2}$
  而 $\mathbf{w} \cdot \mathbf{x}$ 表示向量点积 (vector dot product) ：
  
  $$\mathbf{w} \cdot \mathbf{x} = w_0 x_0 + w_1 x_1$$
  
  
 * 我们将模型的输出 ($f_{\mathbf{w},b}(x)$) 解释为：在给定输入 $\mathbf{x}$ 且以 $\mathbf{w}$ 和 $b$ 作为参数的条件下，$y=1$ 的概率 (probability) 。
* 因此，为了从逻辑回归模型中获得最终的分类预测结果 ($y=0$ 或 $y=1$) ，我们可以采用以下启发式规则 (heuristic) -

  如果 $f_{\mathbf{w},b}(x) >= 0.5$，则预测 $y=1$
  
  如果 $f_{\mathbf{w},b}(x) < 0.5$，则预测 $y=0$
  
  
* 让我们绘制出 S型函数的图像，看看在什么范围内满足 $g(z) >= 0.5$


```python
# 在 -10 到 10 的取值范围内绘制 sigmoid(z) 图像
z = np.arange(-10,11)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# 绘制 z 与 sigmoid(z) 的关系曲线
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
```

* 正如你所看到的，当 $z >=0$ 时，恒有 $g(z) >= 0.5$

* 对于逻辑回归模型而言，$z = \mathbf{w} \cdot \mathbf{x} + b$。因此，

  如果 $\mathbf{w} \cdot \mathbf{x} + b >= 0$，模型就会预测 $y=1$
  
  如果 $\mathbf{w} \cdot \mathbf{x} + b < 0$，模型就会预测 $y=0$
  
  
  
### 绘制决策边界

现在，让我们回到前面的示例，看看逻辑回归模型具体是如何做出预测的。

* 我们的逻辑回归模型具有以下形式：

  $f(\mathbf{x}) = g(-3 + x_0+x_1)$


* 根据你刚才学到的规律可以看出，当 $-3 + x_0+x_1 >= 0$ 时，该模型会预测 $y=1$

让我们来看看它在图形上是什么样子的。我们首先绘制直线 $-3 + x_0+x_1 = 0$，它等价于 $x_1 = 3 - x_0$。



```python
# 在 0 到 6 之间选择数值
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# 绘制决策边界
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# 填充直线下方的区域
ax.fill_between(x0,x1, alpha=0.2)

# 绘制原始数据点
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()
```

* 在上图中，蓝色直线代表方程 $x_0 + x_1 - 3 = 0$，它与 x1 轴相交于 3 (令 $x_1$ = 3，$x_0$ = 0) ，与 x0 轴也相交于 3 (令 $x_1$ = 0，$x_0$ = 3) 。


* 阴影区域代表 $-3 + x_0+x_1 < 0$。直线上方的区域则代表 $-3 + x_0+x_1 > 0$。


* 任何位于阴影区域内 (直线下方) 的点都会被分类为 $y=0$。任何位于直线上或直线上方的点都会被分类为 $y=1$。这条分界线就被称为“决策边界”。

正如我们在课程讲座中所了解到的，通过引入更高阶的多项式项 (polynomial terms)  (例如：$f(x) = g( x_0^2 + x_1 -1)$) ，我们可以构建出更加复杂的非线性边界 (non-linear boundaries) 。

## 恭喜！
你已经深入探索了逻辑回归中的决策边界概念。


```python

```


```python

```
