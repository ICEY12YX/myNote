# 实践实验：机器学习 (Machine Learning) 应用建议
在本次实验中，你将探索用于评估和改进机器学习模型的各项实用技术。

# 目录大纲
-  1 - 软件包 
-  2 - 评估学习算法 (多项式回归)
  -  2.1 划分数据集
  -  2.2 用于模型评估的误差计算：线性回归
    -  练习 1
  -  2.3 对比训练集与测试集上的表现
-  3 - 偏差与方差&lt;img align=&quot;Right&quot; src=&quot;./images/C2_W3_BiasVarianceDegree.png&quot;  style=&quot; width:500px; padding: 10px 20px ; &quot;&gt; 
  -  3.1 绘制训练集、交叉验证集与测试集
  -  3.2 寻找最优多项式阶数
  -  3.3 调优正则化参数
  -  3.4 获取更多数据：增加训练集规模 (m)
-  4 - 评估学习算法 (神经网络)
  -  4.1 数据集
  -  4.2 通过计算分类误差评估分类模型
    -  练习 2
-  5 - 模型复杂度
  -  练习 3
  -  5.1 简单模型
    -  练习 4
-  6 - 正则化
  -  练习 5
-  7 - 迭代寻找最优正则化参数值
  -  7.1 测试集评估


_**注意：** 为了防止自动评分器 (Autograder) 出错，请勿编辑或删除本笔记本中的非评分单元格。同时也请不要添加任何新的单元格。
**当你通过本次作业后**，如果想对任何非评分代码进行修改和实验，可以按照本笔记本底部的说明进行操作。_

&lt;a name=&quot;1&quot;&gt;&lt;/a&gt;
## 1 - 软件包 

首先，让我们运行下面的代码单元格，导入本次作业所需的所有软件包。
- numpy 是使用 Python 进行科学计算的基础软件包。
- matplotlib 是 Python 中广受欢迎的绘图库。
- scikitlearn 是用于数据挖掘 (Data Mining) 与机器学习的基础库。
- tensorflow 是一个主流的机器学习平台。


```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from public_tests_a1 import * 

tf.keras.backend.set_floatx('float64')
from assigment_utils import *

tf.autograph.set_verbosity(0)
```

&lt;a name=&quot;2&quot;&gt;&lt;/a&gt;
## 2 - 评估学习算法 (多项式回归 (Polynomial Regression) )

&lt;img align=&quot;Right&quot; src=&quot;./images/C2_W3_TrainingVsNew.png&quot;  style=&quot; width:350px; padding: 10px 20px ; &quot;&gt; 假设你已经创建了一个机器学习模型，并且发现它对训练数据的*拟合 (fits) *效果非常好。这样就大功告成了吗？还差得远呢。我们创建模型的真正目标，是希望它能够对&lt;span style=&quot;color:blue&quot;&gt;*新*&lt;/span&gt;样本做出准确预测。

在将模型正式部署上线之前，你该如何测试它在新数据上的表现呢？  
答案包含两个部分：
* 将原始数据集 (Data set) 划分为“训练集 (Training Set) ”和“测试集 (Test Set) ”。
    * 使用训练数据来拟合模型的参数 (Parameters) 
    * 使用测试数据来评估模型在*新*数据上的表现
* 编写一个误差函数 (Error Function) 来量化评估你的模型。

&lt;a name=&quot;2.1&quot;&gt;&lt;/a&gt;
### 2.1 划分数据集
课堂讲义建议留出数据集的 20%-40% 用于测试。下面让我们使用 `sklearn` 中的 train_test_split 函数来执行划分。运行下面的单元格后，请仔细核对一下各个数组的形状 (Shapes) 。


```python
# Generate some data
X,y,x_ideal,y_ideal = gen_data(18, 2, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)

#split the data using sklearn routine 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)
```

#### 2.1.1 绘制训练集与测试集
从下图中你可以看到，将用于训练的数据点 (红色) 与不参与训练的数据点 (测试集，蓝色) 交错分布在一起。这个特定的数据集是由一个二次函数加上随机噪声生成的。图中还画出了“理想 (ideal) ”的无噪声曲线以供参考。


```python
fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color = "red",           label="train")
ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
plt.show()
```

&lt;a name=&quot;2.2&quot;&gt;&lt;/a&gt;
### 2.2 用于模型评估的误差计算：线性回归 (Linear Regression)
在*评估*线性回归模型时，我们需要计算预测值与目标值之间平方误差的平均值 (即均方误差 (Mean Squared Error) ) 。

$$ J_\text{test}(\mathbf{w},b) = 
            \frac{1}{2m_\text{test}}\sum_{i=0}^{m_\text{test}-1} ( f_{\mathbf{w},b}(\mathbf{x}^{(i)}_\text{test}) - y^{(i)}_\text{test} )^2 
            \tag{1}
$$

&lt;a name=&quot;ex01&quot;&gt;&lt;/a&gt;
### 练习 1

请在下方编写一个函数，用于计算线性回归模型在给定数据集上的均方误差。


```python
# UNQ_C1
# GRADED CELL: eval_mse
def eval_mse(y, yhat):
    """ 
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)             
    """
    m = len(y)
    err = 0.0
    for i in range(m):
    ### START CODE HERE ### 
       
        
    
    ### END CODE HERE ### 
    
    return(err)
```


```python
y_hat = np.array([2.4, 4.2])
y_tmp = np.array([2.3, 4.1])
eval_mse(y_hat, y_tmp)

# BEGIN UNIT TEST
test_eval_mse(eval_mse)   
# END UNIT TEST
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;

    
```python
def eval_mse(y, yhat):
    """ 
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)             
    """
    m = len(y)
    err = 0.0
    for i in range(m):
        err_i  = ( (yhat[i] - y[i])**2 ) 
        err   += err_i                                                                
    err = err / (2*m)                    
    return(err)
``` 

&lt;a name=&quot;2.3&quot;&gt;&lt;/a&gt;
### 2.3 对比训练集与测试集上的表现
让我们构建一个高阶多项式模型，来尽可能减小训练误差。这里将使用 `sklearn` 中的线性回归函数。如果你想查看具体实现细节，可以在导入的工具函数文件中找到相关代码。下面的执行步骤包括：
* 创建并拟合模型 (“拟合 (fit) ”就是训练或运行梯度下降 (Gradient Descent) 的另一种叫法) 。
* 计算模型在训练数据上的误差。
* 计算模型在测试数据上的误差。


```python
# create a model in sklearn, train on training data
degree = 10
lmodel = lin_model(degree)
lmodel.fit(X_train, y_train)

# predict on training data, find training error
yhat = lmodel.predict(X_train)
err_train = lmodel.mse(y_train, yhat)

# predict on test data, find error
yhat = lmodel.predict(X_test)
err_test = lmodel.mse(y_test, yhat)
```

计算结果表明，训练集上的误差远小于测试集上的误差。


```python
print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")
```

下面的图像直观地解释了其中的原因：该模型对训练数据拟合得实在太好了，为了穿过每一个训练样本点，它构造出了一个极其复杂的剧烈波动函数。而测试数据并未参与训练，模型在这些新数据上的预测表现一塌糊涂。  
对于这样的模型，我们会说它：1) 发生了过拟合 (Overfitting) ，2) 具有高方差 (High Variance) ，3) “泛化 (Generalizes) ”能力很差。


```python
# plot predictions over data range 
x = np.linspace(0,int(X.max()),100)  # predict values for plot
y_pred = lmodel.predict(x).reshape(-1,1)

plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)
```

测试集误差清楚地表明，这个模型无法在新数据上正常工作。但是，如果你直接根据测试集误差来指导模型的改进和参数选择，那么改进后的模型固然会在测试数据上表现良好……可别忘了，测试数据原本是用来代表未知的*新*数据的！一旦你用它来做决策，它就不再能客观代表新数据了。
因此，你还需要引入第三个独立的数据集，专门用来评估模型在新数据上的最终表现。

课堂上提出的方案是将数据划分为三组。下表展示的训练集、交叉验证集 (Cross-validation Set) 和测试集的划分比例是一种典型的分配方案，在实际应用中也可以根据可用数据量的大小进行适当调整。

| 数据集 | 占总数据的比例 (%) | 说明 |
|------------------|:----------:|:---------|
| 训练集 (training) | 60 | 在训练或拟合过程中，用于调优模型参数 $w$ 和 $b$ 的数据 |
| 交叉验证集 (cross-validation) | 20 | 用于调优其他模型超参数 (如多项式阶数、正则化参数或神经网络架构) 的数据 |
| 测试集 (test) | 20 | 在完成所有调优之后，用于测试模型并衡量其在新数据上表现的数据 |


下面让我们生成这三个数据集。我们将再次使用 `sklearn` 中的 `train_test_split`，通过连续调用它两次来得到三个数据子集：


```python
# Generate  data
X,y, x_ideal,y_ideal = gen_data(40, 5, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)

#split the data using sklearn routine 
X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.40, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.50, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_cv.shape", X_cv.shape, "y_cv.shape", y_cv.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)
```

&lt;a name=&quot;3&quot;&gt;&lt;/a&gt;
## 3 - 偏差 (Bias) 与方差 (Variance)&lt;img align=&quot;Right&quot; src=&quot;./images/C2_W3_BiasVarianceDegree.png&quot;  style=&quot; width:500px; padding: 10px 20px ; &quot;&gt; 
 在上面的例子中，很明显多项式模型的阶数 (Degree) 设得太高了。那么，你该如何挑选一个合适的阶数值呢？如右图所示，训练集和交叉验证集的表现可以为我们提供明确的指导。通过尝试一系列不同的阶数值，我们可以分别评估模型在训练集和交叉验证集上的表现。当阶数变得过大时，交叉验证集上的表现就会开始相对于训练集出现恶化。下面让我们在自己的例子上试一试。

&lt;a name=&quot;3.1&quot;&gt;&lt;/a&gt;
### 3.1 绘制训练集、交叉验证集与测试集
从下图中可以看到，参与训练的数据点 (红色) 与不参与训练的数据点 (测试集和交叉验证集) 交错分布在一起。


```python
fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, CV, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color = "red",           label="train")
ax.scatter(X_cv, y_cv,       color = dlc["dlorange"], label="cv")
ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
plt.show()
```

&lt;a name=&quot;3.2&quot;&gt;&lt;/a&gt;
### 3.2 寻找最优多项式阶数
在之前的实验中，你已经发现可以利用多项式来构建能够拟合复杂曲线的模型 (参见第 1 门课程第 2 周的特征工程与多项式回归实验) 。此外，你还亲自验证了通过不断增加多项式的*阶数*，可以人为*制造*出过拟合现象 (参见第 1 门课程第 3 周的过拟合实验) 。现在，让我们把这些知识结合起来，检验一下自己区分过拟合与欠拟合 (Under-fitting) 的能力。

让我们反复训练模型，在每次迭代中将多项式的阶数加 1。为了追求更快的速度和更简洁的代码，这里我们将直接使用 scikit-learn 的线性回归模型。


```python
max_degree = 9
err_train = np.zeros(max_degree)    
err_cv = np.zeros(max_degree)      
x = np.linspace(0,int(X.max()),100)  
y_pred = np.zeros((100,max_degree))  #columns are lines to plot

for degree in range(max_degree):
    lmodel = lin_model(degree+1)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[degree] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[degree] = lmodel.mse(y_cv, yhat)
    y_pred[:,degree] = lmodel.predict(x)
    
optimal_degree = np.argmin(err_cv)+1
```

&lt;font size=&quot;4&quot;&gt;让我们把结果绘制出来：&lt;/font&gt;


```python
plt.close("all")
plt_optimal_degree(X_train, y_train, X_cv, y_cv, x, y_pred, x_ideal, y_ideal, 
                   err_train, err_cv, optimal_degree, max_degree)
```

上图清楚地证明，将数据分为模型训练过的数据和未训练过的数据两组，可以用来准确判断模型究竟是处于欠拟合还是过拟合状态。在我们的例子中，通过逐步增加所使用的多项式阶数，我们创建了一系列从欠拟合过渡到过拟合的模型。
- 在左图中，实线代表这些模型给出的预测曲线。阶数为 1 的多项式模型生成了一条直线，几乎没有穿过几个数据点；而最高阶数的模型则紧紧贴合着每一个训练数据点。
- 在右图中：
    - 正如预期的那样，随着模型复杂度 (Model Complexity) 的增加，训练数据上的误差 (蓝色) 持续下降
    - 交叉验证数据上的误差起初随着模型逐渐贴合数据规律而下降，但随后当模型开始在训练数据上发生过拟合 (即未能良好*泛化*) 时，交叉验证误差反而开始大幅上升。  
    
值得一提的是，这些实际例子中的误差曲线并不像理论课讲义里画的示意图那样平滑。很显然，具体哪些数据点被随机分配到了哪一组，会对你的具体数值结果产生显著影响。但最关键的是把握住整体的变化趋势。

&lt;a name=&quot;3.3&quot;&gt;&lt;/a&gt;
### 3.3 调优正则化 (Regularization) 参数
在之前的实验中，你曾利用*正则化*来减轻过拟合。与挑选多项式阶数类似，我们完全可以使用相同的方法来调优正则化参数 lambda ($\lambda$) 。

下面让我们从一个高阶多项式开始，通过不断改变正则化参数的大小来演示这一过程。


```python
lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4,1e-3,1e-2, 1e-1,1,10,100])
num_steps = len(lambda_range)
degree = 10
err_train = np.zeros(num_steps)    
err_cv = np.zeros(num_steps)       
x = np.linspace(0,int(X.max()),100) 
y_pred = np.zeros((100,num_steps))  #columns are lines to plot

for i in range(num_steps):
    lambda_= lambda_range[i]
    lmodel = lin_model(degree, regularization=True, lambda_=lambda_)
    lmodel.fit(X_train, y_train)
    yhat = lmodel.predict(X_train)
    err_train[i] = lmodel.mse(y_train, yhat)
    yhat = lmodel.predict(X_cv)
    err_cv[i] = lmodel.mse(y_cv, yhat)
    y_pred[:,i] = lmodel.predict(x)
    
optimal_reg_idx = np.argmin(err_cv) 
```


```python
plt.close("all")
plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range)
```

上方的图表显示，随着正则化强度的增加，模型逐渐从一个高方差 (过拟合) 模型转变为一个高偏差 (欠拟合) 模型。右图中的垂直虚线标示出了 lambda 的最优取值。在这个例子中，多项式的阶数被固定为 10。

&lt;a name=&quot;3.4&quot;&gt;&lt;/a&gt;
### 3.4 获取更多数据：增加训练集规模 (m)
当一个模型出现过拟合 (高方差) 时，收集更多的数据往往能够有效提升其性能。下面让我们在这里验证一下。


```python
X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range,degree = tune_m()
plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree)
```

上方的图表表明，当模型具有高方差并发生过拟合时，增加样本数量能够显著改善模型表现。请留意左图中的各条曲线：对应于最大样本量 $m$ 的那条最终曲线是一条非常平滑的曲线，恰好处在数据分布的中心位置。而在右图中，随着样本数量的增加，训练集和交叉验证集上的误差表现逐渐收敛到相近的水平。当然，实际曲线并不像讲义示意图那样绝对平滑，这是完全正常的；但核心趋势依然十分明确：更多的数据能够提升模型的泛化能力。

> 请注意，如果模型本身处于高偏差 (欠拟合) 状态，那么单纯增加更多训练样本并不能提升模型的性能。


&lt;a name=&quot;4&quot;&gt;&lt;/a&gt;
## 4 - 评估学习算法 (神经网络 (Neural Network) )
在上面的部分中，你调优了多项式回归模型的各项参数。接下来，你将使用神经网络模型来进行实验。让我们先从创建一个分类数据集开始。

&lt;a name=&quot;4.1&quot;&gt;&lt;/a&gt;
### 4.1 数据集
运行下面的代码单元格，生成一个数据集并将其划分为训练集、交叉验证集 (CV) 和测试集。在这个例子中，为了突出展示效果，我们特意提高了交叉验证集数据点所占的比例。  


```python
# Generate and split data set
X, y, centers, classes, std = gen_blobs()

# split the data. Large CV population for demonstration
X_train, X_, y_train, y_ = train_test_split(X,y,test_size=0.50, random_state=1)
X_cv, X_test, y_cv, y_test = train_test_split(X_,y_,test_size=0.20, random_state=1)
print("X_train.shape:", X_train.shape, "X_cv.shape:", X_cv.shape, "X_test.shape:", X_test.shape)
```


```python
plt_train_eq_dist(X_train, y_train,classes, X_cv, y_cv, centers, std)
```

在上图左侧，你可以看到生成的数据分布。图中共有六个用不同颜色标识的簇 (Clusters) ，同时展示了训练样本点 (圆点) 和交叉验证样本点 (三角形) 。最耐人寻味的是那些落在交界模糊地带的样本点——相邻的两个簇都有可能认为这些点属于自己。你觉得神经网络模型会如何处理这些点呢？什么样的表现算是过拟合？什么样又算是欠拟合？  
上图右侧展示了一个“理想 (ideal) ”模型的示例，也就是当我们预先知道数据生成的真实分布规律时所能构建出的最优模型。图中的分界线代表“等距 (equal distance) ”决策边界 (Decision Boundaries) ，边界上的点到相邻簇中心的距离相等。值得注意的是，即便是这样一个理想模型，也会对整个数据集中大约 8% 的样本产生“错误分类 (misclassify) ”。

&lt;a name=&quot;4.2&quot;&gt;&lt;/a&gt;
### 4.2 通过计算分类误差评估分类模型
这里使用的分类模型评估函数非常简单，就是预测错误的样本占总样本的比例：  
$$ J_{cv} =\frac{1}{m}\sum_{i=0}^{m-1} 
\begin{cases}
    1, & \text{if $\hat{y}^{(i)} \neq y^{(i)}$}\\
    0, & \text{otherwise}
\end{cases}
$$

&lt;a name=&quot;ex02&quot;&gt;&lt;/a&gt;
### 练习 2

请在下方补全计算分类误差的子程序。请注意，在本次实验中，目标值直接就是类别的整数索引，并没有采用独热编码 (One-hot Encoded) 。


```python
# UNQ_C2
# GRADED CELL: eval_cat_err
def eval_cat_err(y, yhat):
    """ 
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      cerr: (scalar)             
    """
    m = len(y)
    incorrect = 0
    for i in range(m):
    ### START CODE HERE ### 
       
            
    
    ### END CODE HERE ### 
    
    return(cerr)
```


```python
y_hat = np.array([1, 2, 0])
y_tmp = np.array([1, 2, 3])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.333" )
y_hat = np.array([[1], [2], [0], [3]])
y_tmp = np.array([[1], [2], [1], [3]])
print(f"categorization error {np.squeeze(eval_cat_err(y_hat, y_tmp)):0.3f}, expected:0.250" )

# BEGIN UNIT TEST  
test_eval_cat_err(eval_cat_err)
# END UNIT TEST
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
```python
def eval_cat_err(y, yhat):
    """ 
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      cerr: (scalar)             
    """
    m = len(y)
    incorrect = 0
    for i in range(m):
        if yhat[i] != y[i]:    # @REPLACE
            incorrect += 1     # @REPLACE
    cerr = incorrect/m         # @REPLACE
    return(cerr)                                    
``` 

&lt;a name=&quot;5&quot;&gt;&lt;/a&gt;
## 5 - 模型复杂度
在下面的部分中，你将构建两个模型：一个复杂模型和一个简单模型。你将通过对这两个模型进行评估，来判断它们是否容易发生过拟合或欠拟合。

###  5.1 复杂模型

&lt;a name=&quot;ex03&quot;&gt;&lt;/a&gt;
### 练习 3
请在下方构建一个三层神经网络模型：
* 包含 120 个单元 (Units) 、使用 relu 激活函数 (Activation Function) 的全连接层 (Dense Layer) 
* 包含 40 个单元、使用 relu 激活函数的全连接层
* 包含 6 个单元、使用线性激活函数 (`linear`，而非 softmax) 的全连接层  
使用以下配置进行编译 (`compile`) ：
* 使用 `SparseCategoricalCrossentropy` 作为损失函数 (Loss) ，请务必记得设置 `from_logits=True`
* 使用学习率 (Learning Rate) 为 0.01 的 Adam 优化器 (Optimizer) 。


```python
# UNQ_C3
# GRADED CELL: model
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.random.set_seed(1234)
model = Sequential(
    [
        ### START CODE HERE ### 
        
        
        
        ### END CODE HERE ### 

    ], name="Complex"
)
model.compile(
    ### START CODE HERE ### 
    loss=None,
    optimizer=None,
    ### END CODE HERE ### 
)
```


```python
# BEGIN UNIT TEST
model.fit(
    X_train, y_train,
    epochs=1000
)
# END UNIT TEST
```


```python
# BEGIN UNIT TEST
model.summary()

model_test(model, classes, X_train.shape[1]) 
# END UNIT TEST
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
模型摘要应当如下所示 (层实例的名称编号可能会有所递增) ：
```
Model: "Complex"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
L1 (Dense)                   (None, 120)               360       
_________________________________________________________________
L2 (Dense)                   (None, 40)                4840      
_________________________________________________________________
L3 (Dense)                   (None, 6)                 246       
=================================================================
Total params: 5,446
Trainable params: 5,446
Non-trainable params: 0
_________________________________________________________________
```
  &lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看更多提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
  
```python
tf.random.set_seed(1234)
model = Sequential(
    [
        Dense(120, activation = 'relu', name = "L1"),      
        Dense(40, activation = 'relu', name = "L2"),         
        Dense(classes, activation = 'linear', name = "L3")  
    ], name="Complex"
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),          
    optimizer=tf.keras.optimizers.Adam(0.01),   
)

model.fit(
    X_train,y_train,
    epochs=1000
)                                  
``` 


```python
#make a model for plotting routines to call
model_predict = lambda Xl: np.argmax(tf.nn.softmax(model.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict,X_train,y_train, classes, X_cv, y_cv, suptitle="Complex Model")
```

可以看到，这个复杂模型煞费苦心地去迎合和捕捉每个类别中的异常值 (Outliers) 。结果适得其反，导致它在交叉验证集上把不少正常样本都给分错了类。让我们来计算一下它的分类误差。


```python
training_cerr_complex = eval_cat_err(y_train, model_predict(X_train))
cv_cerr_complex = eval_cat_err(y_cv, model_predict(X_cv))
print(f"categorization error, training, complex model: {training_cerr_complex:0.3f}")
print(f"categorization error, cv,       complex model: {cv_cerr_complex:0.3f}")
```

&lt;a name=&quot;5.1&quot;&gt;&lt;/a&gt;
### 5.1 简单模型
现在，让我们来试试一个结构简单的模型。

&lt;a name=&quot;ex04&quot;&gt;&lt;/a&gt;
### 练习 4

请在下方构建一个两层神经网络模型：
* 包含 6 个单元、使用 relu 激活函数的全连接层
* 包含 6 个单元、使用线性激活函数的全连接层
使用以下配置进行编译：
* 使用 `SparseCategoricalCrossentropy` 作为损失函数，请务必记得设置 `from_logits=True`
* 使用学习率为 0.01 的 Adam 优化器。


```python
# UNQ_C4
# GRADED CELL: model_s

tf.random.set_seed(1234)
model_s = Sequential(
    [
        ### START CODE HERE ### 
        
        
        ### END CODE HERE ### 
    ], name = "Simple"
)
model_s.compile(
    ### START CODE HERE ### 
    loss=None,
    optimizer=None,
    ### START CODE HERE ### 
)

```


```python
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# BEGIN UNIT TEST
model_s.fit(
    X_train,y_train,
    epochs=1000
)
# END UNIT TEST
```


```python
# BEGIN UNIT TEST
model_s.summary()

model_s_test(model_s, classes, X_train.shape[1])
# END UNIT TEST
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
模型摘要应当如下所示 (层实例的名称编号可能会有所递增) ：
```
Model: "Simple"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
L1 (Dense)                   (None, 6)                 18        
_________________________________________________________________
L2 (Dense)                   (None, 6)                 42        
=================================================================
Total params: 60
Trainable params: 60
Non-trainable params: 0
_________________________________________________________________
```
  &lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看更多提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
  
```python
tf.random.set_seed(1234)
model_s = Sequential(
    [
        Dense(6, activation = 'relu', name="L1"),            # @REPLACE
        Dense(classes, activation = 'linear', name="L2")     # @REPLACE
    ], name = "Simple"
)
model_s.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),     # @REPLACE
    optimizer=tf.keras.optimizers.Adam(0.01),     # @REPLACE
)

model_s.fit(
    X_train,y_train,
    epochs=1000
)                                   
``` 


```python
#make a model for plotting routines to call
model_predict_s = lambda Xl: np.argmax(tf.nn.softmax(model_s.predict(Xl)).numpy(),axis=1)
plt_nn(model_predict_s,X_train,y_train, classes, X_cv, y_cv, suptitle="Simple Model")
```

这个简单模型的表现相当不错！让我们来计算一下它的分类误差。


```python
training_cerr_simple = eval_cat_err(y_train, model_predict_s(X_train))
cv_cerr_simple = eval_cat_err(y_cv, model_predict_s(X_cv))
print(f"categorization error, training, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
print(f"categorization error, cv,       simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )
```

与刚才那个复杂模型相比，我们的简单模型在训练数据上的分类误差略高一点点，但在交叉验证数据上的表现却明显更好。

&lt;a name=&quot;6&quot;&gt;&lt;/a&gt;
## 6 - 正则化
与多项式回归中的情况一样，我们也可以对神经网络应用正则化，以此来缓和复杂模型带来的过拟合影响。下面让我们动手试一试。

&lt;a name=&quot;ex05&quot;&gt;&lt;/a&gt;
### 练习 5

请重新构建你刚才的复杂模型，但这一次加入 L2 正则化 (L2 Regularization) 。
请在下方构建一个三层模型：
* 包含 120 个单元、使用 relu 激活函数的全连接层，并设置 `kernel_regularizer=tf.keras.regularizers.l2(0.1)`
* 包含 40 个单元、使用 relu 激活函数的全连接层，并设置 `kernel_regularizer=tf.keras.regularizers.l2(0.1)`
* 包含 6 个单元、使用线性激活函数的全连接层
使用以下配置进行编译：
* 使用 `SparseCategoricalCrossentropy` 作为损失函数，请务必记得设置 `from_logits=True`
* 使用学习率为 0.01 的 Adam 优化器。


```python
# UNQ_C5
# GRADED CELL: model_r

tf.random.set_seed(1234)
model_r = Sequential(
    [
        ### START CODE HERE ### 
        
        
        
        ### START CODE HERE ### 
    ], name= None
)
model_r.compile(
    ### START CODE HERE ### 
    loss=None,
    optimizer=None,
    ### START CODE HERE ### 
)

```


```python
# BEGIN UNIT TEST
model_r.fit(
    X_train, y_train,
    epochs=1000
)
# END UNIT TEST
```


```python
# BEGIN UNIT TEST
model_r.summary()

model_r_test(model_r, classes, X_train.shape[1]) 
# END UNIT TEST
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
模型摘要应当如下所示 (层实例的名称编号可能会有所递增) ：
```
Model: "ComplexRegularized"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
L1 (Dense)                   (None, 120)               360       
_________________________________________________________________
L2 (Dense)                   (None, 40)                4840      
_________________________________________________________________
L3 (Dense)                   (None, 6)                 246       
=================================================================
Total params: 5,446
Trainable params: 5,446
Non-trainable params: 0
_________________________________________________________________
```
  &lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看更多提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
  
```python
tf.random.set_seed(1234)
model_r = Sequential(
    [
        Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name="L1"), 
        Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.1), name="L2"),  
        Dense(classes, activation = 'linear', name="L3")  
    ], name="ComplexRegularized"
)
model_r.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    optimizer=tf.keras.optimizers.Adam(0.01),                             
)

model_r.fit(
    X_train,y_train,
    epochs=1000
)                                   
``` 


```python
#make a model for plotting routines to call
model_predict_r = lambda Xl: np.argmax(tf.nn.softmax(model_r.predict(Xl)).numpy(),axis=1)
 
plt_nn(model_predict_r, X_train,y_train, classes, X_cv, y_cv, suptitle="Regularized")
```

现在的决策边界看起来与之前的“理想”模型非常接近了！让我们来检查一下分类误差。


```python
training_cerr_reg = eval_cat_err(y_train, model_predict_r(X_train))
cv_cerr_reg = eval_cat_err(y_cv, model_predict_r(X_cv))
test_cerr_reg = eval_cat_err(y_test, model_predict_r(X_test))
print(f"categorization error, training, regularized: {training_cerr_reg:0.3f}, simple model, {training_cerr_simple:0.3f}, complex model: {training_cerr_complex:0.3f}" )
print(f"categorization error, cv,       regularized: {cv_cerr_reg:0.3f}, simple model, {cv_cerr_simple:0.3f}, complex model: {cv_cerr_complex:0.3f}" )
```

可以看到，简单模型在训练集上的表现比正则化模型略好一点，但在交叉验证集上的表现则不如正则化后的复杂模型。

&lt;a name=&quot;7&quot;&gt;&lt;/a&gt;
## 7 - 迭代寻找最优正则化参数值
就像在线性回归中所做的那样，你也可以在神经网络中尝试多个不同的正则化参数值。下面这段代码需要几分钟的时间来运行。如果你有时间，可以运行它并查看结果；如果没有时间，其实你此时已经完成了本次作业所有计分部分的任务！


```python
tf.random.set_seed(1234)
lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
models=[None] * len(lambdas)

for i in range(len(lambdas)):
    lambda_ = lambdas[i]
    models[i] =  Sequential(
        [
            Dense(120, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(40, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(lambda_)),
            Dense(classes, activation = 'linear')
        ]
    )
    models[i].compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.01),
    )

    models[i].fit(
        X_train,y_train,
        epochs=1000
    )
    print(f"Finished lambda = {lambda_}")

```


```python
plot_iterate(lambdas, models, X_train, y_train, X_cv, y_cv)
```

随着正则化强度的增加，模型在训练集和交叉验证集上的表现逐渐靠拢。对于当前的数据集和模型来说，选择 lambda > 0.01 似乎是一个相当合理的决定。

&lt;a name=&quot;7.1&quot;&gt;&lt;/a&gt;
### 7.1 测试集评估
最后，让我们把优化好的模型放到测试集上检验一下，并将它们的表现与“理想”性能进行对比。


```python
plt_compare(X_test,y_test, classes, model_predict_s, model_predict_r, centers)
```

由于我们的测试集规模较小，且恰好包含了一些偏离中心的异常样本点，因此测试集上的分类误差数值看起来偏高。不过横向对比来看，我们优化后的模型在测试集上的表现已经与理论上的理想模型不相上下了。

## 恭喜！ 
你已经掌握了在评估和改进机器学习模型时不可或缺的重要工具与方法，主要包括：  
* 将数据划分为参与训练的数据集和不参与训练的数据集，使你能够准确区分欠拟合与过拟合
* 划分出训练集、交叉验证集和测试集三个独立的数据子集，使你能够：
    * 使用训练集来训练模型参数 $W,B$
    * 使用交叉验证集来调优模型复杂度、正则化参数以及样本规模等超参数
    * 使用测试集来客观评估模型在“真实世界”未知数据上的表现
* 通过对比训练集与交叉验证集上的表现，深入洞察模型究竟是倾向于过拟合 (高方差) 还是欠拟合 (高偏差) 

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;2&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;如果你想对任何非评分代码进行修改和实验，请点击这里。&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    &lt;p&gt;&lt;i&gt;&lt;b&gt;重要提示：为了避免自动评分器出现问题，请务必在已经通过本次作业之后再进行以下操作。&lt;/b&gt;&lt;/i&gt;
    &lt;ol&gt;
        &lt;li&gt; 在笔记本的菜单栏上，依次点击 “View” &gt; “Cell Toolbar” &gt; “Edit Metadata”&lt;/li&gt;
        &lt;li&gt; 点击你想要锁定 / 解锁的代码单元格旁边的 “Edit Metadata” 按钮&lt;/li&gt;
        &lt;li&gt; 将 “editable” 属性的值设置为：
            &lt;ul&gt;
                &lt;li&gt; 如果你想解锁它，设为 “true” &lt;/li&gt;
                &lt;li&gt; 如果你想锁定它，设为 “false” &lt;/li&gt;
            &lt;/ul&gt;
        &lt;/li&gt;
        &lt;li&gt; 在笔记本的菜单栏上，依次点击 “View” &gt; “Cell Toolbar” &gt; “None” &lt;/li&gt;
    &lt;/ol&gt;
    &lt;p&gt; 下面是演示如何执行上述步骤的简短动图： 
        &lt;br&gt;
        &lt;img src=&quot;https://lh3.google.com/u/0/d/14Xy_Mb17CZVgzVAgq7NCjMVBvSae3xO1&quot; align=&quot;center&quot; alt=&quot;unlock_cells.gif&quot;&gt;
&lt;/details&gt;
