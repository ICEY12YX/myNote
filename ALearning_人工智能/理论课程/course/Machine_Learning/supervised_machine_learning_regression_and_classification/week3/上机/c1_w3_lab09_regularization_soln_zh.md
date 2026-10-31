# 选修实验 - 正则化代价函数与梯度 (Regularized Cost and Gradient)

## 实验目标
在本实验中，你将：
- 在之前的线性回归 (linear regression) 和逻辑回归 (logistic regression) 代价函数 (cost function) 基础上，引入正则化项 (regularization term) 进行扩展。
- 加入正则化项后，重新运行之前的过拟合 (over-fitting) 示例。



```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from plt_overfit import overfit_example, output
from lab_utils_common import sigmoid
np.set_printoptions(precision=8)
```

# 引入正则化
&lt;img align=&quot;Left&quot; src=&quot;./images/C1_W3_LinearGradientRegularized.png&quot;  style=&quot; width:400px; padding: 10px; &quot; &gt;
&lt;img align=&quot;Center&quot; src=&quot;./images/C1_W3_LogisticGradientRegularized.png&quot;  style=&quot; width:400px; padding: 10px; &quot; &gt;

上方的幻灯片展示了线性回归和逻辑回归的代价函数与梯度函数 (gradient function) 。请注意：
- 代价 (Cost)
    - 线性回归和逻辑回归的代价函数本身差异显著，但向方程中添加正则化项的方式是完全相同的。
- 梯度 (Gradient)
    - 线性回归和逻辑回归的梯度函数形式高度相似，二者的唯一区别在于模型函数 $f_{wb}$ 的具体实现。

## 带有正则化的代价函数
### 正则化线性回归的代价函数

正则化线性回归 (regularized linear regression) 的代价函数方程为：
$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2  + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2 \tag{1}$$ 
其中：
$$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b  \tag{2} $$ 


将其与不带正则化的代价函数 (你在之前的实验中已经实现过) 进行对比，后者的形式为：

$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 $$ 

二者的区别在于多了一个正则化项：&lt;span style=&quot;color:blue&quot;&gt;
    $\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$ &lt;/span&gt; 
    
引入这一项可以促使梯度下降 (gradient descent) 在优化过程中尽量缩小模型参数 (parameters) 的数值。请注意，在这个示例中，参数 $b$ 不参与正则化，这是业界通用的标准做法。

下方是方程 (1) 和 (2) 的代码实现。注意，这里采用了*本课程的标准实现模式*，即使用一个遍历全部 `m` 个样本的 `for loop` 。


```python
def compute_cost_linear_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m  = X.shape[0]
    n  = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b                                   #(n,)(n,)=scalar, see np.dot
        cost = cost + (f_wb_i - y[i])**2                               #scalar             
    cost = cost / (2 * m)                                              #scalar  
 
    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar
```

运行下方的代码单元格，看看它的实际运行效果。


```python
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)
```

**预期输出**：
&lt;table&gt;
  &lt;tr&gt;
    &lt;td&gt; &lt;b&gt;Regularized cost: &lt;/b&gt; 0.07917239320214275 &lt;/td&gt;
  &lt;/tr&gt;
&lt;/table&gt;

### 正则化逻辑回归的代价函数
对于正则化**逻辑**回归 (regularized logistic regression) ，其代价函数的形式为：
$$J(\mathbf{w},b) = \frac{1}{m}  \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \right] + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2 \tag{3}$$
其中：
$$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = sigmoid(\mathbf{w} \cdot \mathbf{x}^{(i)} + b)  \tag{4} $$ 

将其与不带正则化的代价函数 (你在之前的实验中已经实现过) 进行对比：

$$ J(\mathbf{w},b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\right] $$

与上文的线性回归一样，二者的区别同样在于正则化项，即 &lt;span style=&quot;color:blue&quot;&gt;
    $\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$ &lt;/span&gt; 

引入这一项可以促使梯度下降在优化过程中尽量缩小参数的数值。请注意，在这个示例中，参数 $b$ 同样不参与正则化，这是业界通用的标准做法。


```python
def compute_cost_logistic_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns:
      total_cost (scalar):  cost 
    """

    m,n  = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b                                      #(n,)(n,)=scalar, see np.dot
        f_wb_i = sigmoid(z_i)                                          #scalar
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)      #scalar
             
    cost = cost/m                                                      #scalar

    reg_cost = 0
    for j in range(n):
        reg_cost += (w[j]**2)                                          #scalar
    reg_cost = (lambda_/(2*m)) * reg_cost                              #scalar
    
    total_cost = cost + reg_cost                                       #scalar
    return total_cost                                                  #scalar
```

运行下方的代码单元格，看看它的实际运行效果。


```python
np.random.seed(1)
X_tmp = np.random.rand(5,6)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1]).reshape(-1,)-0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print("Regularized cost:", cost_tmp)
```

**预期输出**：
&lt;table&gt;
  &lt;tr&gt;
    &lt;td&gt; &lt;b&gt;Regularized cost: &lt;/b&gt; 0.6850849138741673 &lt;/td&gt;
  &lt;/tr&gt;
&lt;/table&gt;

## 带有正则化的梯度下降
引入正则化后，运行梯度下降的基本算法流程并没有改变，依然是：
$$\begin{align*}
&\text{repeat until convergence:} \; \lbrace \\
&  \; \; \;w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{for j := 0..n-1} \\ 
&  \; \; \;  \; \;b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \\
&\rbrace
\end{align*}$$
在每次迭代 (iteration) 中，算法会对所有的 $j$ 同步更新 $w_j$ 。

引入正则化后真正发生变化的是梯度的计算方式。

### 计算带有正则化的梯度 (适用于线性回归与逻辑回归)
线性回归和逻辑回归的梯度计算公式几乎完全一致，唯一的区别在于模型预测值 $f_{\mathbf{w}b}$ 的计算方式。
$$\begin{align*}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)}  +  \frac{\lambda}{m} w_j \tag{2} \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3} 
\end{align*}$$

* m 是数据集中的训练样本 (training examples) 数量      
* $f_{\mathbf{w},b}(x^{(i)})$ 是模型的预测值 (prediction) ，而 $y^{(i)}$ 是目标值 (target) 

      
* 对于&lt;span style=&quot;color:blue&quot;&gt; **线性** &lt;/span&gt;回归模型  
    $f_{\mathbf{w},b}(x) = \mathbf{w} \cdot \mathbf{x} + b$  
* 对于&lt;span style=&quot;color:blue&quot;&gt; **逻辑** &lt;/span&gt;回归模型  
    $z = \mathbf{w} \cdot \mathbf{x} + b$  
    $f_{\mathbf{w},b}(x) = g(z)$  
    其中 $g(z)$ 是 S 型函数 (sigmoid function) ：  
    $g(z) = \frac{1}{1+e^{-z}}$   
    
用于实现正则化的新增项是 &lt;span style=&quot;color:blue&quot;&gt;$\frac{\lambda}{m} w_j $&lt;/span&gt; 。

### 正则化线性回归的梯度函数


```python
def compute_gradient_linear_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
      
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
    
    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw
```

运行下方的代码单元格，看看它的实际运行效果。


```python
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )
```

**预期输出**
```
dj_db: 0.6648774569425726
Regularized dj_dw:
 [0.29653214748822276, 0.4911679625918033, 0.21645877535865857]
 ```

### 正则化逻辑回归的梯度函数


```python
def compute_gradient_logistic_reg(X, y, w, b, lambda_): 
    """
    Computes the gradient for linear regression 
 
    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters  
      b (scalar)      : model parameter
      lambda_ (scalar): Controls amount of regularization
    Returns
      dj_dw (ndarray Shape (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar)            : The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape
    dj_dw = np.zeros((n,))                            #(n,)
    dj_db = 0.0                                       #scalar

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
        err_i  = f_wb_i  - y[i]                       #scalar
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   #(n,)
    dj_db = dj_db/m                                   #scalar

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw  

```

运行下方的代码单元格，看看它的实际运行效果。


```python
np.random.seed(1)
X_tmp = np.random.rand(5,3)
y_tmp = np.array([0,1,0,1,0])
w_tmp = np.random.rand(X_tmp.shape[1])
b_tmp = 0.5
lambda_tmp = 0.7
dj_db_tmp, dj_dw_tmp =  compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)

print(f"dj_db: {dj_db_tmp}", )
print(f"Regularized dj_dw:\n {dj_dw_tmp.tolist()}", )
```

**预期输出**
```
dj_db: 0.341798994972791
Regularized dj_dw:
 [0.17380012933994293, 0.32007507881566943, 0.10776313396851499]
 ```

## 重新运行过拟合示例


```python
plt.close("all")
display(output)
ofit = overfit_example(True)
```

在上方的交互图表中，在之前的示例上亲自尝试一下正则化的效果。具体操作如下：
- 分类任务 (Categorical，即逻辑回归)
    - 将多项式阶数 (degree) 设为 6，将正则化参数 lambda 设为 0 (即不使用正则化) ，对数据进行拟合 (fit) 。
    - 接着将 lambda 设为 1 (即增强正则化) ，重新对数据进行拟合，观察前后的变化。
- 回归任务 (Regression，即线性回归)
    - 尝试相同的操作流程。

## 恭喜！
你已经完成了以下内容：
- 掌握了在线性回归和逻辑回归中加入正则化后的代价函数与梯度计算代码实现。
- 对正则化如何有效缓解过拟合建立起了直观的理解。


```python

```
