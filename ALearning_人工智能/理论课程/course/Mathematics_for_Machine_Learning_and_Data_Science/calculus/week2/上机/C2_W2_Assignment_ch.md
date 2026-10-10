___
# 使用梯度下降法进行优化：线性回归

在本作业中，你将构建一个简单的线性回归 (Linear Regression) 模型，用于根据电视营销支出预测销售额。你将探究解决这个问题的三种不同方法。你将分别使用 `NumPy` 和 `Scikit-Learn` 的线性回归模型，并从头开始使用梯度下降 (Gradient Descent) 构建和优化平方和代价函数 (Sum of Squares Cost Function)。

___

___

## 依赖包



加载所需的依赖包：

```python
import numpy as np
# A library for programmatic plot generation.
import matplotlib.pyplot as plt
# A library for data manipulation and analysis.
import pandas as pd
# LinearRegression from sklearn.
from sklearn.linear_model import LinearRegression

```

导入为此笔记本定义的单元测试。

```python
import w2_unittest

```

## 1 - 打开数据集并陈述问题



在本实验中，你将为一个简单的 Kaggle 数据集构建线性回归模型，该数据集保存在文件 `data/tvmarketing.csv` 中。该数据集仅包含两个字段：电视营销支出 ( `TV` ) 和销售额 ( `Sales` )。

### 练习 1



使用 `pandas` 函数 `pd.read_csv` 从 `path` 路径打开 .csv 文件。

```python
path = "data/tvmarketing.csv"

### START CODE HERE ### (~ 1 line of code)
adv = None.None(None)
### END CODE HERE ###

```

```python
# Print some part of the dataset.
adv.head()

```

##### **预期输出**

```Python
	TV	Sales
0	230.1	22.1
1	44.5	10.4
2	17.2	9.3
3	151.5	18.5
4	180.8	12.9

```

```python
w2_unittest.test_load_data(adv)

```

`pandas` 有一个可以根据 DataFrame 字段绘制图表的内置函数。默认情况下，它的绘图后端使用的是 matplotlib。让我们在这里尝试使用它：

```python
adv.plot(x='TV', y='Sales', kind='scatter', c='black')

```

你可以使用这个数据集通过线性回归解决一个非常直观的问题：给定一个电视营销预算，预测对应的销售额。

## 2 - 使用 `NumPy` 和 `Scikit-Learn` 在 Python 中进行线性回归



将 DataFrame 所需的字段保存到变量 `X` 和 `Y` 中：

```python
X = adv['TV']
Y = adv['Sales']

```

### 2.1 - 使用 `NumPy` 的线性回归



你可以使用函数 `np.polyfit(x, y, deg)` 将阶数为 `deg` 的多项式拟合到数据点 $(x, y)$ 上，从而使平方误差和 (Sum of Squared Errors) 最小化。你可以在官方文档中阅读了解更多相关细节。当设定 `deg = 1` 时，你可以获得线性回归线的斜率 (Slope) `m` 和截距 (Intercept) `b` ：

```python
m_numpy, b_numpy = np.polyfit(X, Y, 1)

print(f"Linear regression with NumPy. Slope: {m_numpy}. Intercept: {b_numpy}")

```

*注意*：`NumPy` 文档建议将 `Polynomial.fit` 类方法作为编写新代码时的推荐方法，因为它在数值计算上更加稳定。但是在这个简单的例子中，为了简便起见，你可以放心继续使用 `np.polyfit` 函数。

你可以通过运行以下代码来绘制出这条线性回归线。回归线在图中以红色显示。

```python
def plot_linear_regression(X, Y, x_label, y_label, m, b, X_pred=np.array([]), Y_pred=np.array([])):
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    ax.plot(X, Y, 'o', color='black')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.plot(X, m*X + b, color='red')
    # Plot prediction points (empty arrays by default - the predictions will be calculated later).
    ax.plot(X_pred, Y_pred, 'o', color='blue', markersize=8)
    
plot_linear_regression(X, Y, 'TV', 'Sales', m_numpy, b_numpy)

```

### 练习 2



给定一个包含 $X$ 值的数组，将刚刚获得的斜率和截距系数代入方程 $Y = mX + b$ 中来进行预测。

```python
# This is organised as a function only for grading purposes.
def pred_numpy(m, b, X):
    ### START CODE HERE ### (~ 1 line of code)
    Y = None
    ### END CODE HERE ###
    
    return Y

```

```python
X_pred = np.array([50, 120, 280])
Y_pred_numpy = pred_numpy(m_numpy, b_numpy, X_pred)

print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales using NumPy linear regression:\n{Y_pred_numpy}")

```

##### **预期输出**

```Python
TV marketing expenses:
[ 50 120 280]
Predictions of sales using NumPy linear regression:
[ 9.40942557 12.7369904  20.34285287]

```

```python
w2_unittest.test_pred_numpy(pred_numpy)

```

现在你可以将预测出的数据点（蓝点）添加到图表中了。

```python
plot_linear_regression(X, Y, 'TV', 'Sales', m_numpy, b_numpy, X_pred, Y_pred_numpy)

```

### 2.2 - 使用 `Scikit-Learn` 的线性回归



`Scikit-Learn` 是一个强大的开源机器学习 (Machine Learning) 库，它广泛支持监督学习 (Supervised Learning) 和无监督学习 (Unsupervised Learning) 任务。它还提供了丰富多样的工具，用于模型拟合 (Model Fitting) 、数据预处理 (Data Preprocessing) 、模型选择 (Model Selection) 、模型评估 (Model Evaluation) 以及许多其他实用功能。`Scikit-learn` 提供了数十种内置的机器学习算法和模型，这些在框架中被称为估计器 (Estimators) 。每个估计器都可以通过调用它的 `fit` 方法来拟合各种数据。你可以在这里找到它的完整文档。

让我们为线性回归模型创建一个估计器对象：

```python
lr_sklearn = LinearRegression()

```

该估计器可以通过调用 `fit` 函数直接从数据中进行学习。然而，如果在尝试运行以下代码时你会得到一个错误提示，这是因为模型要求输入的数据必须被重塑为二维数组的格式：

```python
print(f"Shape of X array: {X.shape}")
print(f"Shape of Y array: {Y.shape}")

try:
    lr_sklearn.fit(X, Y)
except ValueError as err:
    print(err)

```

你可以使用 `reshape` 函数将数组的维度提升一维，或者也可以采用以下这种直观的方法来实现：

```python
X_sklearn = X[:, np.newaxis]
Y_sklearn = Y[:, np.newaxis]

print(f"Shape of new X array: {X_sklearn.shape}")
print(f"Shape of new Y array: {Y_sklearn.shape}")

```

### 练习 3



将处理好的 `X_sklearn` 和 `Y_sklearn` 数组传递给 `lr_sklearn.fit` 函数，以此来拟合这个线性回归模型。

```python
### START CODE HERE ### (~ 1 line of code)
None.None(None, None)
### END CODE HERE ###

```

```python
m_sklearn = lr_sklearn.coef_
b_sklearn = lr_sklearn.intercept_

print(f"Linear regression using Scikit-Learn. Slope: {m_sklearn}. Intercept: {b_sklearn}")

```

##### **预期输出**

```Python
Linear regression using Scikit-Learn. Slope: [[0.04753664]]. Intercept: [7.03259355]

```

```python
w2_unittest.test_sklearn_fit(lr_sklearn)

```

请注意，你已经得到了与 `NumPy` 函数 `polyfit` 完全相同的结果。现在，为了进行下一步预测，直接使用 `Scikit-Learn` 提供的 `predict` 函数会非常方便。

### 练习 4



使用 `np.newaxis` 函数（请参考上文示例）增加 $X$ 数组的维度，并将处理后的结果传递给 `lr_sklearn.predict` 函数来进行销售额预测。

```python
# This is organised as a function only for grading purposes.
def pred_sklearn(X, lr_sklearn):
    ### START CODE HERE ### (~ 2 lines of code)
    X_2D = None[None, None.None]
    Y = None.None(None)
    ### END CODE HERE ###
    
    return Y

```

```python
Y_pred_sklearn = pred_sklearn(X_pred, lr_sklearn)

print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales using Scikit_Learn linear regression:\n{Y_pred_sklearn.T}")

```

##### **预期输出**

```Python
TV marketing expenses:
[ 50 120 280]
Predictions of sales using Scikit_Learn linear regression:
[[ 9.40942557 12.7369904  20.34285287]]

```

```python
w2_unittest.test_sklearn_predict(pred_sklearn, lr_sklearn)

```

最终预测出的数值也是相同的。

## 3 - 使用梯度下降的线性回归



能够自动拟合模型的现成函数使用起来固然非常方便，但为了能更深入地理解模型本身及其背后隐藏的数学原理，最好还是由你自己动手从零实现一个核心算法。让我们尝试通过最小化原始值 $y^{(i)}$ 和预测值 $\hat{y}^{(i)}$ 之间的差异来逐步寻找最优的线性回归系数 $m$ 和 $b$ ，对于每个训练样本，衡量这种差异的损失函数 (Loss Function) 被定义为 $L\left(w, b\right)  = \frac{1}{2}\left(\hat{y}^{(i)} - y^{(i)}\right)^2$。在公式中除以 $2$ 仅仅是为了方便数学上的缩放计算，在下文中计算偏导数 (Partial Derivatives) 时你就会立刻明白这样设计的巧妙之处了。

为了从整体上比较预测值向量 $\hat{Y}$ 和包含了原始值 $y^{(i)}$ 的向量 $Y$ ，你可以对所有训练示例的损失函数值求取一个平均值：

$$E\left(m, b\right) = \frac{1}{2n}\sum_{i=1}^{n} \left(\hat{y}^{(i)} - y^{(i)}\right)^2 =  \frac{1}{2n}\sum_{i=1}^{n} \left(mx^{(i)}+b - y^{(i)}\right)^2,\tag{1}$$

这里的 $n$ 代表了数据点的总数量。这个数学函数在学术上被称为平方和代价函数 (Cost Function) 。为了能够使用梯度下降算法进行优化，你需要按照以下公式计算出它的偏导数：

\begin{align}
\frac{\partial E }{ \partial m } &=
\frac{1}{n}\sum_{i=1}^{n} \left(mx^{(i)}+b - y^{(i)}\right)x^{(i)},\
\frac{\partial E }{ \partial b } &=
\frac{1}{n}\sum_{i=1}^{n} \left(mx^{(i)}+b - y^{(i)}\right),
\tag{2}\end{align}

并使用下面的一组表达式来迭代地更新参数值：

\begin{align}
m &= m - \alpha \frac{\partial E }{ \partial m },\
b &= b - \alpha \frac{\partial E }{ \partial b },
\tag{3}\end{align}

这里的 $\alpha$ 被称为学习率 (Learning Rate) ，它决定了每次更新的步长大小。

需要注意的是，原始数据数组 `X` 和 `Y` 具有完全不同的度量单位。为了让梯度下降算法能够更加高效地运行，你需要先将它们统一转换到相同的无量纲尺度下。实现这一目标的一种常用技术手段叫做归一化 (Normalization) ：从数组的每个元素中减去该数组的整体平均值 (Mean) ，然后再除以它的标准差 (Standard Deviation) （标准差是一种用来衡量一组数值分散或集中程度的统计指标）。如果你对平均值和标准差这些概念还不太熟悉，现在大可不必担心——我们将在随后的专项课程中对此进行详细探讨。

其实归一化并不是算法运行的强制要求——即使完全不加处理，梯度下降法同样可以工作。但是由于 `X` 和 `Y` 的尺度差异过大，这会导致代价函数的表面变得异常扭曲且陡峭。如果不进行归一化，你就必须被迫设定一个极小极小的学习率 $\alpha$ ，这样一来算法可能将需要耗费数以千计的迭代次数才能缓慢收敛 (Converge) ，而不是短短的几十次。因此，归一化就像是给算法铺平了道路，有助于显著提升梯度下降算法的收敛效率。

归一化操作在以下代码中被简洁地实现了：

```python
X_norm = (X - np.mean(X))/np.std(X)
Y_norm = (Y - np.mean(Y))/np.std(Y)

```

根据上文的方程 $(1)$ 来在代码中定义代价函数：

```python
def E(m, b, X, Y):
    return 1/(2*len(Y))*np.sum((m*X + b - Y)**2)

```

### 练习 5



定义函数 `dEdm` 和 `dEdb` ，以根据数学方程 $(2)$ 准确计算出偏导数。这里可以巧妙地利用输入数据 `X` 和 `Y` 的向量化形式来进行高效计算。

```python
def dEdm(m, b, X, Y):
    ### START CODE HERE ### (~ 1 line of code)
    # Use the following line as a hint, replacing all None.
    res = 1/len(None)*np.dot(None*None + None - None, None)
    ### END CODE HERE ###
    
    return res
    

def dEdb(m, b, X, Y):
    ### START CODE HERE ### (~ 1 line of code)
    # Replace None writing the required expression fully.
    res = None
    ### END CODE HERE ###
    
    return res


```

```python
print(dEdm(0, 0, X_norm, Y_norm))
print(dEdb(0, 0, X_norm, Y_norm))
print(dEdm(1, 5, X_norm, Y_norm))
print(dEdb(1, 5, X_norm, Y_norm))

```

##### **预期输出**

```Python
-0.7822244248616067
5.098005351200641e-16
0.21777557513839355
5.000000000000002

```

```python
w2_unittest.test_partial_derivatives(dEdm, dEdb, X_norm, Y_norm)

```

### 练习 6



使用之前的表达式 $(3)$ 来实现梯度下降的核心循环：
\begin{align}
m &= m - \alpha \frac{\partial E }{ \partial m },\
b &= b - \alpha \frac{\partial E }{ \partial b },
\end{align}

在这里 $\alpha$ 就对应着代码中的 `learning_rate` 变量。

```python
def gradient_descent(dEdm, dEdb, m, b, X, Y, learning_rate = 0.001, num_iterations = 1000, print_cost=False):
    for iteration in range(num_iterations):
        ### START CODE HERE ### (~ 2 lines of code)
        m_new = None
        b_new = None
        ### END CODE HERE ###
        m = m_new
        b = b_new
        if print_cost:
            print (f"Cost after iteration {iteration}: {E(m, b, X, Y)}")
        
    return m, b

```

```python
print(gradient_descent(dEdm, dEdb, 0, 0, X_norm, Y_norm))
print(gradient_descent(dEdm, dEdb, 1, 5, X_norm, Y_norm, learning_rate = 0.01, num_iterations = 10))

```

##### **预期输出**

```Python
(0.49460408269589495, -3.489285249624889e-16)
(0.9791767513915026, 4.521910375044022)

```

```python
w2_unittest.test_gradient_descent(gradient_descent, dEdm, dEdb, X_norm, Y_norm)

```

现在，让我们从初始点 $\left(m_0, b_0\right)=\left(0, 0\right)$ 开始，正式运行梯度下降优化方法。

```python
m_initial = 0; b_initial = 0; num_iterations = 30; learning_rate = 1.2
m_gd, b_gd = gradient_descent(dEdm, dEdb, m_initial, b_initial, 
                              X_norm, Y_norm, learning_rate, num_iterations, print_cost=True)

print(f"Gradient descent result: m_min, b_min = {m_gd}, {b_gd}") 

```

请一定记住，我们在之前对初始的训练数据集进行了归一化处理。为了能够输出符合真实业务场景的最终预测结果，你需要先对待预测的 `X_pred` 数组按照相同的标准进行归一化，然后利用刚才学习到的线性回归系数 `m_gd` 、 `b_gd` 计算出 `Y_pred` ，最后最重要的一步是，对计算得出的结果进行反归一化 (Denormalize) （也就是执行与归一化完全相反的数学还原过程）：

```python
X_pred = np.array([50, 120, 280])
# Use the same mean and standard deviation of the original training array X
X_pred_norm = (X_pred - np.mean(X))/np.std(X)
Y_pred_gd_norm = m_gd * X_pred_norm + b_gd
# Use the same mean and standard deviation of the original training array Y
Y_pred_gd = Y_pred_gd_norm * np.std(Y) + np.mean(Y)

print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales using Scikit_Learn linear regression:\n{Y_pred_sklearn.T}")
print(f"Predictions of sales using Gradient Descent:\n{Y_pred_gd}")

```

你应该已经发现，你得到了与前面几个小节中使用现成库几乎完全相同的结果。

干得漂亮！现在你不仅知道了什么是梯度下降算法，还亲手将其应用于了真实模型参数的训练过程当中。能够在一个简洁的基础案例中不借助外部库手动复现出这些结果，绝对能让你对自己深入掌握了那些常用机器学习函数底层的真实运作原理而感到更加自信。

```python


```