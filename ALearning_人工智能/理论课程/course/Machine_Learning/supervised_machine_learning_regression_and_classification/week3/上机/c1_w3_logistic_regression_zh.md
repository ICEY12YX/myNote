___
- [[#逻辑回归 (Logistic Regression)|逻辑回归 (Logistic Regression)]]
	- [[#逻辑回归 (Logistic Regression)#1 - 工具包|1 - 工具包]]
	- [[#逻辑回归 (Logistic Regression)#2 - 逻辑回归|2 - 逻辑回归]]
		- [[#2 - 逻辑回归#2.1 问题描述|2.1 问题描述]]
		- [[#2 - 逻辑回归#2.2 加载并可视化数据|2.2 加载并可视化数据]]
			- [[#2.2 加载并可视化数据#查看变量内容|查看变量内容]]
			- [[#2.2 加载并可视化数据#检查变量的维度|检查变量的维度]]
			- [[#2.2 加载并可视化数据#可视化你的数据|可视化你的数据]]
		- [[#2 - 逻辑回归#2.3 S 型函数|2.3 S 型函数]]
		- [[#2 - 逻辑回归#练习 1|练习 1]]
		- [[#2 - 逻辑回归#2.4 逻辑回归的代价函数|2.4 逻辑回归的代价函数]]
		- [[#2 - 逻辑回归#练习 2|练习 2]]
		- [[#2 - 逻辑回归#2.5 逻辑回归的梯度|2.5 逻辑回归的梯度]]
		- [[#2 - 逻辑回归#练习 3|练习 3]]
		- [[#2 - 逻辑回归#2.6 使用梯度下降学习参数|2.6 使用梯度下降学习参数]]
		- [[#2 - 逻辑回归#2.7 绘制决策边界|2.7 绘制决策边界]]
		- [[#2 - 逻辑回归#2.8 评估逻辑回归模型|2.8 评估逻辑回归模型]]
		- [[#2 - 逻辑回归#练习 4|练习 4]]
	- [[#逻辑回归 (Logistic Regression)#3 - 正则化逻辑回归|3 - 正则化逻辑回归]]
		- [[#3 - 正则化逻辑回归#3.1 问题描述|3.1 问题描述]]
		- [[#3 - 正则化逻辑回归#3.2 加载并可视化数据|3.2 加载并可视化数据]]
			- [[#3.2 加载并可视化数据#查看变量内容|查看变量内容]]
			- [[#3.2 加载并可视化数据#检查变量的维度|检查变量的维度]]
			- [[#3.2 加载并可视化数据#可视化你的数据|可视化你的数据]]
		- [[#3 - 正则化逻辑回归#3.3 特征映射|3.3 特征映射]]
		- [[#3 - 正则化逻辑回归#3.4 正则化逻辑回归的代价函数|3.4 正则化逻辑回归的代价函数]]
		- [[#3 - 正则化逻辑回归#练习 5|练习 5]]
		- [[#3 - 正则化逻辑回归#3.5 正则化逻辑回归的梯度|3.5 正则化逻辑回归的梯度]]
		- [[#3 - 正则化逻辑回归#练习 6|练习 6]]
		- [[#3 - 正则化逻辑回归#3.6 使用梯度下降学习参数|3.6 使用梯度下降学习参数]]
		- [[#3 - 正则化逻辑回归#3.7 绘制决策边界|3.7 绘制决策边界]]
		- [[#3 - 正则化逻辑回归#3.8 评估正则化逻辑回归模型|3.8 评估正则化逻辑回归模型]]

___
# 逻辑回归 (Logistic Regression)

在本练习中，你将亲手实现逻辑回归，并将其应用于两个不同的数据集 (datasets) 。

_**注意：** 为了避免自动评分系统 (autograder) 报错，请勿编辑或删除本实验中的非评分单元格 (non-graded cells) ，也请勿添加任何新的单元格。
**当你通过本次作业后**，如果想对非评分代码进行自由探索和实验，可以参考本笔记本底部的操作说明。_

## 1 - 工具包 

首先，我们运行下方的代码单元格，导入本次作业所需的全部工具包：
- numpy 是使用 Python 进行科学计算的基础工具包。
- matplotlib 是 Python 中著名的绘图库。
- ``utils.py`` 包含了本次作业的辅助函数 (helper functions) ，你无需修改该文件中的任何代码。

```python
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

%matplotlib inline
```

## 2 - 逻辑回归

在练习的这一部分，你将构建一个逻辑回归模型，用来预测一名学生是否能被大学录取。

### 2.1 问题描述

假设你是某大学院系的招生管理员，希望根据申请者在两门考试中的成绩，评估每位申请者被录取的概率。
* 你手头有一份往届申请者的历史数据，可以将其作为逻辑回归的训练集 (training set) 。
* 对于每一个训练样本 (training example) ，你都掌握了该申请者在==两门考试中的成绩以及最终的录取结果==。
* 你的任务是构建一个分类模型 (classification model) ，根据这两门考试的成绩来估算申请者被录取的概率。

### 2.2 加载并可视化数据

首先，我们来加载这项任务所需的数据集。
- 下方展示的 `load_dataset()` 函数会将数据加载到变量 `X_train` 和 `y_train` 中：
  - `X_train` 包含了学生在两门考试中的成绩。
  - `y_train` 是最终的录取决定：
      - 若 `y_train = 1` ，表示该学生已被录取。
      - 若 `y_train = 0` ，表示该学生未被录取。
  - `X_train` 和 `y_train` 均为 numpy 数组 (numpy arrays) 。

```python
# load dataset
X_train, y_train = load_data("data/ex2data1.txt")
```

#### 查看变量内容
让我们先熟悉一下你的数据集。
- 一个很好的起点是直接打印出每个变量，看看里面具体包含了什么数据。

下方的代码会打印出 `X_train` 的前五个元素以及该变量的数据类型。

```python
print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))
```

```
First five elements in X_train are:
 [[34.62365962 78.02469282]
 [30.28671077 43.89499752]
 [35.84740877 72.90219803]
 [60.18259939 86.3085521 ]
 [79.03273605 75.34437644]]
Type of X_train: <class 'numpy.ndarray'>
```

现在，打印出 `y_train` 的前五个元素：

```python
print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))
```

```
First five elements in y_train are:
 [0. 0. 0. 1. 1.]
Type of y_train: <class 'numpy.ndarray'>
```

#### 检查变量的维度

熟悉数据的另一个实用方法是查看它的维度 (dimensions) 。让我们打印出 `X_train` 和 `y_train` 的形状 (shape) ，看看数据集中一共有多少个训练样本。

```python
print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))
```

```
The shape of X_train is: (100, 2)
The shape of y_train is: (100,)
We have m = 100 training examples
```

#### 可视化你的数据

在动手实现任何机器学习算法之前，如果条件允许，先把数据可视化 (visualize) 出来总是一个好习惯。
- 下方的代码会将数据展示在一张二维散点图上 (如下图所示) ，其中横纵坐标轴分别代表两门考试的成绩，正样本 (positive examples) 和负样本 (negative examples) 则用不同的标记符号加以区分。
- 我们调用了 ``utils.py`` 文件中的一个辅助函数来生成这张图表。

```python
# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()
```
![[Pasted image 20261103215550.png|427]]

你的目标是构建一个逻辑回归模型来拟合这些数据。
- 有了这个模型，你就可以根据一名新学生在两门考试中的成绩，预测他是否会被录取。

### 2.3 S 型函数

回顾一下，在逻辑回归中，模型可以表示为：
$$ f_{\mathbf{w},b}(x) = g(\mathbf{w}\cdot \mathbf{x} + b)$$
其中的函数 $g$ 就是 S 型函数 (sigmoid function) 。S 型函数的定义如下：
$$g(z) = \frac{1}{1+e^{-z}}$$

让我们先来实现这个 S 型函数，以便在后续的作业中调用它。

### 练习 1

请补全 `sigmoid` 函数，用于计算：

$$g(z) = \frac{1}{1+e^{-z}}$$

请注意：
- `z` 不一定只是一个单独的数字 (标量，scalar) ，也可以是一个==包含多个数字的数组。==
    - 关于这点,用np.exp 就解决了,不用额外处理
- 如果输入是一个数组，我们希望对输入数组中的每一个数值都应用一次 S 型函数。

如果遇到困难，可以查看下方代码单元格后面的提示 (hints) 来协助你完成代码实现。

```python
# UNQ_C1
# GRADED FUNCTION: sigmoid

def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (scalar or ndarray): A scalar or numpy array of any size.
    Returns:
        g (scalar or ndarray): sigmoid(z), with the same shape as z if z is a numpy array.
         
    """
          
    ### START CODE HERE ### 
    g = 1/(1+np.exp(-z))
    ### END SOLUTION ###  
    
    return g
```

完成后，可以在下方的代码单元格中调用 `sigmoid(x)`，测试几个不同的数值：
- 当 x 为很大的正数时，S 型函数的输出应该接近 1；当 x 为很大的负数时，输出应该接近 0。
- 计算 `sigmoid(0)` 应该得到精确的 0.5。

```python
# Note: You can edit this value
value = 0

print (f"sigmoid({value}) = {sigmoid(value)}")
```

```
sigmoid(0) = 0.5
```

- 如前所述，你的代码还应当支持向量 (vectors) 和矩阵 (matrices) 输入。当输入为矩阵时，你的函数应当对矩阵中的每一个元素执行 S 型函数运算。

```python
print ("sigmoid([ -1, 0, 1, 2]) = " + str(sigmoid(np.array([-1, 0, 1, 2]))))

# UNIT TESTS
from public_tests import *
sigmoid_test(sigmoid)
```

```
sigmoid([ -1, 0, 1, 2]) = [0.26894142 0.5        0.73105858 0.88079708]
All tests passed!
```

### 2.4 逻辑回归的代价函数

在这一节中，你将实现逻辑回归的代价函数 (cost function) 。

### 练习 2

请利用下方的数学方程，补全 `compute_cost` 函数。

回顾一下，逻辑回归的代价函数形式为：

$$ J(\mathbf{w},b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \tag{1}$$

其中：
* m 是数据集中的训练样本数量。
* $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ 是单个数据点的损失 (loss) ，其计算公式为：$$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}$$
* $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ 是模型给出的预测值，而 $y^{(i)}$ 是真实标签 (actual label) 。

* $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x^{(i)}} + b)$ ，其中函数 $g$ 是 S 型函数。
    * 在计算 $f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(z_{\mathbf{w},b}(\mathbf{x}^{(i)}))$ 之前，先计算一个中间变量 $z_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x^{(i)}} + b = w_0x^{(i)}_0 + ... + w_{n-1}x^{(i)}_{n-1} + b$ (其中 $n$ 为特征 features 的数量) 会让代码逻辑更加清晰。

注意：
* 在编写代码时请记住，变量 `X_train` 和 `y_train` 并不是单个标量，而是形状分别为 ($m, n$) 和 ($𝑚$,) 的矩阵/数组，其中 $𝑛$ 是特征数量，$𝑚$ 是训练样本数量。
* 你可以在这部分直接调用刚才在上方实现的 `sigmoid` 函数。

如果遇到困难，可以查看下方代码单元格后面的提示来协助你完成代码实现。

```python
# UNQ_C2
# GRADED FUNCTION: compute_cost
def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost 
    """

    m, n = X.shape
    
    ### START CODE HERE ###
    loss_sum = 0
    for i in range(m):
        z = np.dot(X[i], w) + b
        f = sigmoid(z)
        loss_sum += (-y[i]) * np.log(f) - (1 - y[i]) * np.log(1 - f)
    total_cost = loss_sum / m

    ### END CODE HERE ###

    return total_cost
```


运行下方的代码单元格，使用两组不同的参数 $w$ 和 $b$ 初始值来检验你实现的 `compute_cost` 函数是否正确。


```python
m, n = X_train.shape

# Compute and display cost with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w and b (zeros): {:.3f}'.format(cost))
```

```
Cost at initial w and b (zeros): 0.693
```


```python
# Compute and display cost with non-zero w and b
test_w = np.array([0.2, 0.2])
test_b = -24.
cost = compute_cost(X_train, y_train, test_w, test_b)

print('Cost at test w and b (non-zeros): {:.3f}'.format(cost))


# UNIT TESTS
compute_cost_test(compute_cost)
```

```
Cost at test w and b (non-zeros): 0.218
[92mAll tests passed!
```

### 2.5 逻辑回归的梯度

在这一节中，你将实现逻辑回归的梯度计算。

回顾一下，梯度下降算法 (gradient descent algorithm) 的流程为：

$$\begin{align*}& \text{repeat until convergence:} \; \lbrace \newline \; & b := b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \newline       \; & w_j := w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{for j := 0..n-1}\newline & \rbrace\end{align*}$$

其中，参数 $b$ 和所有的 $w_j$ 都是同步更新的。

### 练习 3

请根据下方的方程 (2) 和 (3) ，补全 `compute_gradient` 函数，用来计算 $\frac{\partial J(\mathbf{w},b)}{\partial w}$ 和 $\frac{\partial J(\mathbf{w},b)}{\partial b}$ 。

$$
\frac{\partial J(\mathbf{w},b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)}) \tag{2}
$$
$$
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)})x_{j}^{(i)} \tag{3}
$$
* m 是数据集中的训练样本数量。
* $f_{\mathbf{w},b}(x^{(i)})$ 是模型的预测值，而 $y^{(i)}$ 是真实标签。


- **注意**：虽然这个梯度公式表面上看起来与线性回归的梯度公式一模一样，但由于线性回归和逻辑回归对模型函数 $f_{\mathbf{w},b}(x)$ 的定义不同，二者的实际计算公式其实是不同的。

和前面一样，你可以直接调用上方已经实现的 `sigmoid` 函数。如果遇到困难，可以查看下方代码单元格后面的提示来协助你完成代码实现。

```python
# UNQ_C3
# GRADED FUNCTION: compute_gradient
def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    ### START CODE HERE ###
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)

        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i

        for j in range(n):
            dj_dw[j] += dj_db_i * X[i][j]

    dj_dw /= m
    dj_db /= m
    ### END CODE HERE ###

        
    return dj_db, dj_dw
```

运行下方的代码单元格，使用两组不同的参数 $w$ 和 $b$ 初始值来检验你实现的 `compute_gradient` 函数是否正确。


```python
# Compute and display gradient with w and b initialized to zeros
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw.tolist()}' )
```

```
dj_db at initial w and b (zeros):-0.1
dj_dw at initial w and b (zeros):[-12.00921658929115, -11.262842205513591]
```


```python
# Compute and display cost and gradient with non-zero w and b
test_w = np.array([ 0.2, -0.5])
test_b = -24
dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)

print('dj_db at test w and b:', dj_db)
print('dj_dw at test w and b:', dj_dw.tolist())

# UNIT TESTS    
compute_gradient_test(compute_gradient)
```

```
dj_db at test w and b: -0.5999999999991071
dj_dw at test w and b: [-44.831353617873795, -44.37384124953978]
[92mAll tests passed!
```

### 2.6 使用梯度下降学习参数 

与上一次作业类似，现在你将使用梯度下降法来寻找逻辑回归模型的最优参数 (optimal parameters) 。
- 这部分不需要你编写任何代码，直接运行下方的单元格即可。

- 检验梯度下降是否正常工作的一个好方法，是观察代价函数 $J(\mathbf{w},b)$ 的数值，检查它是否随着每一步迭代而稳步下降。

- 假设你已经正确实现了梯度计算和代价函数，那么 $J(\mathbf{w},b)$ 的值应当绝不会上升，并且会在算法结束时收敛 (converge) 到一个稳定的数值。

```python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value 
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent
      lambda_ : (scalar, float)    regularization constant
      
    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing
```

现在，让我们运行上方的梯度下降算法，为我们的数据集学习模型参数。

**注意**
下方的代码块运行起来需要几分钟时间，尤其是当我们采用非向量化 (non-vectorized) 的实现版本时。你可以先调小 `iterations` (迭代次数) 来快速测试你的代码实现。等之后有空时，再尝试运行 100,000 次迭代以获得更好的拟合效果。

```python
np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w, b, J_history, _ = gradient_descent(
    X_train,
    y_train,
    initial_w,
    initial_b,
    compute_cost,
    compute_gradient,
    alpha,
    iterations,
    0,
)
```

```
Iteration    0: Cost     0.96   
Iteration 1000: Cost     0.31   
Iteration 2000: Cost     0.30   
Iteration 3000: Cost     0.30   
Iteration 4000: Cost     0.30   
Iteration 5000: Cost     0.30   
Iteration 6000: Cost     0.30   
Iteration 7000: Cost     0.30   
Iteration 8000: Cost     0.30   
Iteration 9000: Cost     0.30   
Iteration 9999: Cost     0.30
```

### 2.7 绘制决策边界

现在，我们将使用梯度下降求出的最终参数来绘制线性拟合边界 (决策边界，decision boundary) 。如果你正确实现了前面的各个部分，应该会看到一张与下图相似的图表：
![[c1_w3_logistic_regression_zh-1793716726729.webp|333]]

我们将调用 `utils.py` 文件中的一个辅助函数来生成这张图表。

```python
plot_decision_boundary(w, b, X_train, y_train)
# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()
```
![[c1_w3_logistic_regression_zh-1793716752446.webp|461]]

### 2.8 评估逻辑回归模型

我们可以通过观察学到的模型在训练集上的预测表现，来评估我们所找到的参数的质量。
你将在下方实现 `predict` 函数来完成这一步。

### 练习 4

请补全 `predict` 函数，使其在给定数据集以及学到的参数向量 $w$ 和 $b$ 的情况下，输出 `1` 或 `0` 的预测结果。
- 首先，你需要为每一个样本计算模型的预测输出 $f(x^{(i)}) = g(w \cdot x^{(i)} + b)$ 
    - 你在前面的部分中已经实现过这一步了。
- 我们将模型的输出 ($f(x^{(i)})$) 解释为：在给定输入特征 $x^{(i)}$ 和参数 $w$ 的条件下，标签 $y^{(i)}=1$ 的概率。
- 因此，要想从逻辑回归模型中得到最终的分类预测 ($y^{(i)}=0$ 或 $y^{(i)}=1$) ，可以使用以下启发式规则 (heuristic) ——

  如果 $f(x^{(i)}) >= 0.5$ ，则预测 $y^{(i)}=1$
  如果 $f(x^{(i)}) < 0.5$ ，则预测 $y^{(i)}=0$
    
如果遇到困难，可以查看下方代码单元格后面的提示来协助你完成代码实现。

```python
# UNQ_C4
# GRADED FUNCTION: predict

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
   
     ### START CODE HERE ###
    # Loop over each example
    for i in range(m):
        z_wb = np.dot(X[i], w) + b

        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        p[i] = 1 if (f_wb >= 0.5) else 0

    ### END CODE HERE ### 
    return p
```


完成 `predict` 函数的编写后，让我们运行下方的代码，通过计算分类器 (classifier) 预测正确的样本百分比，来报告模型在训练集上的准确率 (training accuracy) 。

```python
# Test your predict code
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5 #为什么这么写先不管了

tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')

# UNIT TESTS        
predict_test(predict)
```

```
Output of predict: shape (4,), value [0. 1. 1. 1.]
All tests passed!
```

现在，让我们利用这个函数计算模型在训练集上的准确率：

```python
#Compute accuracy on our training set
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))
```

```
Train Accuracy: 92.000000
```

## 3 - 正则化逻辑回归

在练习的这一部分，你将实现正则化逻辑回归 (regularized logistic regression) ，用来预测晶圆厂生产的微芯片 (microchips) 能否通过质量保证测试 (quality assurance，简称 QA) 。在 QA 环节中，每一块微芯片都要经过多项测试，以确保其功能正常。

### 3.1 问题描述

假设你是这家工厂的产品经理，手头有一批微芯片在两项不同测试中的测试结果。
- 你希望根据这两项测试的结果，决定每一块微芯片是应该被接收 (accepted) 还是拒收 (rejected) 。
- 为了辅助你做出决策，你拥有一份过往微芯片的测试结果数据集，可以基于它构建一个逻辑回归模型。

### 3.2 加载并可视化数据

与本练习的前面几部分类似，我们先来加载这项任务的数据集并将其可视化。

- 下方展示的 `load_dataset()` 函数会将数据加载到变量 `X_train` 和 `y_train` 中：
  - `X_train` 包含了微芯片在两项测试中的测试结果。
  - `y_train` 包含了 QA 的最终判定结果：
      - 若 `y_train = 1` ，表示该微芯片被接收。
      - 若 `y_train = 0` ，表示该微芯片被拒收。
  - `X_train` 和 `y_train` 均为 numpy 数组。


```python
# load dataset
X_train, y_train = load_data("data/ex2data2.txt")
```

#### 查看变量内容

下方的代码会打印出 `X_train` 和 `y_train` 的前五个元素以及变量的数据类型。

```python
# print X_train
print("X_train:", X_train[:5])
print("Type of X_train:",type(X_train))

# print y_train
print("y_train:", y_train[:5])
print("Type of y_train:",type(y_train))
```

```
X_train: [[ 0.051267  0.69956 ]
 [-0.092742  0.68494 ]
 [-0.21371   0.69225 ]
 [-0.375     0.50219 ]
 [-0.51325   0.46564 ]]
Type of X_train: <class 'numpy.ndarray'>
y_train: [1. 1. 1. 1. 1.]
Type of y_train: <class 'numpy.ndarray'>
```

#### 检查变量的维度

熟悉数据的另一个实用方法是查看它的维度。让我们打印出 `X_train` 和 `y_train` 的形状，看看数据集中一共有多少个训练样本。

```python
print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))
```

```
The shape of X_train is: (118, 2)
The shape of y_train is: (118,)
We have m = 118 training examples
```

#### 可视化你的数据

我们使用辅助函数 `plot_data` (来自 `utils.py`) 来生成如下所示的图3，其中横纵坐标轴分别是两项测试的得分，正样本 (y = 1，即被接收) 和负样本 (y = 0，即被拒收) 用不同的标记符号加以区分。

![[Pasted image 20261103225504.png|400]]

```python
# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")

# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()
```
![[c1_w3_logistic_regression_zh-1793717743264.webp|457]]

图3显示，==我们的数据集无法用一条穿过图表的直线将正负样本划分开来。==因此，如果直接生搬硬套普通的逻辑回归，在这个数据集上的表现会很差，因为普通逻辑回归只能找到一条线性的决策边界。

### 3.3 特征映射

要想更好地拟合这些数据，一种有效的方法是从每个数据点中构造出更多的特征。在已提供的 `map_feature` 函数中，我们将==**原始特征映射 (map) 为 $x_1$ 和 $x_2$ 直至六次方的所有多项式项 (polynomial terms) 。**==

$$\mathrm{map\_feature}(x) = 
\left[\begin{array}{c}
x_1\\
x_2\\
x_1^2\\
x_1 x_2\\
x_2^2\\
x_1^3\\
\vdots\\
x_1 x_2^5\\
x_2^6\end{array}\right]$$

经过这次特征映射 (feature mapping) ，原本只有两个特征的向量 (即两项 QA 测试的得分) 被转换为了一个 27 维的特征向量。

- 在这样一个更高维度的特征向量上训练出来的逻辑回归分类器，将能够形成更加复杂的决策边界；当我们在二维图表上将其绘制出来时，它会呈现为一条非线性的曲线。
- 我们已经在 `utils.py` 中为你提供好了 `map_feature` 函数。

```python
print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)
```

```
Original shape of data: (118, 2)
Shape after feature mapping: (118, 27)
```

让我们打印出 `X_train` 和 `mapped_X` 的第一个元素，直观地看看转换前后的变化。

```python
print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])
```

```
X_train[0]: [0.051267 0.69956 ]
mapped X_train[0]: [5.12670000e-02 6.99560000e-01 2.62830529e-03 3.58643425e-02
 4.89384194e-01 1.34745327e-04 1.83865725e-03 2.50892595e-02
 3.42353606e-01 6.90798869e-06 9.42624411e-05 1.28625106e-03
 1.75514423e-02 2.39496889e-01 3.54151856e-07 4.83255257e-06
 6.59422333e-05 8.99809795e-04 1.22782870e-02 1.67542444e-01
 1.81563032e-08 2.47750473e-07 3.38066048e-06 4.61305487e-05
 6.29470940e-04 8.58939846e-03 1.17205992e-01]
```

虽然特征映射让我们能够构建出表达能力更强 (more expressive) 的分类器，但它也更容易陷入过拟合 (overfitting) 。在接下来的练习中，你将实现正则化逻辑回归来拟合这些数据，并亲眼见证正则化是如何帮助我们克服过拟合问题的。

### 3.4 正则化逻辑回归的代价函数

在这一部分，你将实现正则化逻辑回归的代价函数。

回顾一下，正则化逻辑回归的代价函数形式为：
$$J(\mathbf{w},b) = \frac{1}{m}  \sum_{i=0}^{m-1} \left[ -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \right] + \frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$

将其与不带正则化的代价函数 (你已经在上方实现过) 进行对比，后者的形式为：

$$ J(\mathbf{w}.b) = \frac{1}{m}\sum_{i=0}^{m-1} \left[ (-y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)\right]$$

二者的区别在于正则化项 (regularization term) ，即 $$\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$ 
请注意，偏差参数 $b$ 不参与正则化。

### 练习 5

请补全下方的 `compute_cost_reg` 函数，对 $w$ 中的每一个元素计算以下正则化项：
$$\frac{\lambda}{2m}  \sum_{j=0}^{n-1} w_j^2$$

随后，初始代码模板会将这一项与不带正则化的代价 (即你在上方通过 `compute_cost` 计算出的结果) 相加，从而得出带有正则化的总代价。

如果遇到困难，可以查看下方代码单元格后面的提示来协助你完成代码实现。


```python
# UNQ_C5
def compute_cost_reg(X, y, w, b, lambda_ = 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar, float) Controls amount of regularization
    Returns:
      total_cost : (scalar)     cost 
    """

    m, n = X.shape
    
    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b) 
    
    # You need to calculate this value
    reg_cost = 0.
    
    ### START CODE HERE ###
    reg_cost = lambda_ / 2 * m * np.dot(w,w)
    ### END CODE HERE ### 
    
    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + reg_cost

    return total_cost
```

运行下方的代码单元格，检验你实现的 `compute_cost_reg` 函数是否正确。


```python
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost)

# UNIT TEST    
compute_cost_reg_test(compute_cost_reg)
```

```
Regularized cost : 0.6618252552483948
All tests passed!
```

### 3.5 正则化逻辑回归的梯度

在这一节中，你将实现正则化逻辑回归的梯度计算。


正则化代价函数的梯度包含两个部分。第一部分 $\frac{\partial J(\mathbf{w},b)}{\partial b}$ 是一个标量，另一部分则是与参数 $\mathbf{w}$ 形状相同的向量，其中第 $j$ 个元素的定义如下：

$$\frac{\partial J(\mathbf{w},b)}{\partial b} = \frac{1}{m}  \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})  $$

$$\frac{\partial J(\mathbf{w},b)}{\partial w_j} = \left( \frac{1}{m}  \sum_{i=0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)} \right) + \frac{\lambda}{m} w_j  \quad\, \mbox{for $j=0...(n-1)$}$$

将其与不带正则化的代价函数梯度 (你已经在上方实现过) 进行对比，后者的形式为：
$$
\frac{\partial J(\mathbf{w},b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)}) \tag{2}
$$
$$
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - \mathbf{y}^{(i)})x_{j}^{(i)} \tag{3}
$$


如你所见，$\frac{\partial J(\mathbf{w},b)}{\partial b}$ 的计算完全相同，区别在于 $\frac{\partial J(\mathbf{w},b)}{\partial w}$ 中多出了如下这一项：$$\frac{\lambda}{m} w_j  \quad\, \mbox{for $j=0...(n-1)$}$$ 
### 练习 6

请补全下方的 `compute_gradient_reg` 函数，编写代码计算以下正则化梯度项：

$$\frac{\lambda}{m} w_j  \quad\, \mbox{for $j=0...(n-1)$}$$

初始代码模板会将这一项加到上方 `compute_gradient` 返回的 $\frac{\partial J(\mathbf{w},b)}{\partial w}$ 上，从而得到正则化代价函数的总梯度。


如果遇到困难，可以查看下方代码单元格后面的提示来协助你完成代码实现。


```python
# UNQ_C6
def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    """
    Computes the gradient for logistic regression with regularization
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    m, n = X.shape
    
    dj_db, dj_dw = compute_gradient(X, y, w, b)

    ### START CODE HERE ###     
    dj_dw += lambda_ / m * w
    ### END CODE HERE ###         
        
    return dj_db, dj_dw
```


运行下方的代码单元格，检验你实现的 `compute_gradient_reg` 函数是否正确。

```python
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1) 
initial_w  = np.random.rand(X_mapped.shape[1]) - 0.5 
initial_b = 0.5
 
lambda_ = 0.5
dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print(f"dj_db: {dj_db}", )
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )

# UNIT TESTS    
compute_gradient_reg_test(compute_gradient_reg)
```

```
dj_db: 0.07138288792343662
First few elements of regularized dj_dw:
 [-0.010386028450548701, 0.011409852883280122, 0.0536273463274574, 0.003140278267313462]
[92mAll tests passed!
```

### 3.6 使用梯度下降学习参数

与前面的部分类似，现在你将使用上方实现的梯度下降函数来学习最优参数 $w$ 和 $b$ 。
- 如果你已经正确实现了正则化逻辑回归的代价函数和梯度计算，现在就可以顺利运行下一个单元格来学习参数 $w$ 了。
- 完成参数训练后，我们将使用学到的参数来绘制决策边界。

**注意**

下方的代码块运行起来需要较长时间，尤其是采用非向量化的实现版本时。你可以先调小 `iterations` (迭代次数) 来快速测试你的代码实现。等之后有空时，再运行 100,000 次迭代以观察更好的拟合效果。

```python
# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ (you can try varying this)
lambda_ = 0.01    

# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                    compute_cost_reg, compute_gradient_reg, 
                                    alpha, iterations, lambda_)
```

```

```
### 3.7 绘制决策边界
为了帮助你直观地看到该分类器学到的模型，我们将调用 `plot_decision_boundary` 函数，绘制出用于划分正负样本的 (非线性) 决策边界。

- 在该函数中，我们通过计算分类器在均匀网格 (evenly spaced grid) 上的预测值，然后绘制出预测值从 y = 0 变为 y = 1 的等高线图 (contour plot) ，从而描绘出这条非线性的决策边界。

- 学到参数 $w$ 和 $b$ 后，下一步就是绘制出如下所示与图4相似的决策边界。
![[c1_w3_logistic_regression_zh-1793720098632.webp|432]]


```python
plot_decision_boundary(w, b, X_mapped, y_train)
# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()
```
![[c1_w3_logistic_regression_zh-1793720111826.webp|456]]

### 3.8 评估正则化逻辑回归模型

你将调用上方实现的 `predict` 函数，计算正则化逻辑回归模型在训练集上的准确率。

```python
#Compute accuracy on the training set
p = predict(X_mapped, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))
```

```
Train Accuracy: 82.203390
```

**恭喜你完成了本课程的最后一个实验！希望在第二门课程 (Course 2) 中继续见到你，届时你将学习并使用神经网络 (neural networks) 和决策树 (decision trees) 等更高级的机器学习算法。继续加油学习吧！**
