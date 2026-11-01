# 选修实验：使用 Scikit-Learn 实现线性回归 (Linear Regression)

有一个名为 scikit-learn 的开源且可用于商业用途的机器学习 (Machine Learning) 工具包。该工具包内置了你在本课程中将要接触到的许多经典算法的实现。

## 实验目标

在本次实验中，你将：
- 利用 scikit-learn，通过基于正规方程 (Normal Equation) 的闭式解 (Closed-form Solution) 来实现线性回归

## 工具函数

你将使用来自 scikit-learn 以及 matplotlib 和 NumPy 的函数。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lab_utils_multi import load_house_data
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)
```

# 线性回归——闭式解

Scikit-learn 提供了线性回归模型 (linear regression model) ，它实现了基于闭式解的线性回归。

让我们继续使用前面实验中的数据——一套面积为 1000 平方英尺的房子售价为 \$300,000，另一套面积为 2000 平方英尺的房子售价为 \$500,000。

| 面积 (1000 平方英尺) | 价格 (千美元) |
| :---- | :---- |
| 1 | 300 |
| 2 | 500 |

### 加载数据集 (Data set)

```python
X_train = np.array([1.0, 2.0])   #features
y_train = np.array([300, 500])   #target value
```

### 创建并拟合模型

下面的代码使用 scikit-learn 执行回归分析。 
第一步是创建一个回归对象。  
第二步是调用与该对象关联的 `fit` 方法。
这一步会执行回归计算，将模型参数 (Parameters) 拟合到输入数据上。
请注意，该工具包==**要求输入的特征矩阵 (Matrix) X 必须是二维 (Two-dimensional) 的。**==
    而且应该最好要搞成下面的形式  一整行是一个数据, 每列对应不同维度的数据

```python
linear_model = LinearRegression()
#X must be a 2-D Matrix
linear_model.fit(X_train.reshape(-1, 1), y_train) 
```
我们上面干的事情, 其实就是在求闭式解, 和上一节 归一化+ 基于梯度下降的回归不一样

### 查看参数

在 scikit-learn 中，参数 $\mathbf{w}$ 和 $\mathbf{b}$ 分别被称为“系数 (coefficients) ”和“截距 (intercept) ”。

```python
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"'manual' prediction: f_wb = wx+b : {1200*w + b}")
```

```
w = [200.], b = 100.00
'manual' prediction: f_wb = wx+b : [240100.]
```

### 进行预测 (Make Predictions)

调用 `predict` 函数即可生成预测结果。

```python
y_pred = linear_model.predict(X_train.reshape(-1, 1))

print("Prediction on training set:", y_pred)

X_test = np.array([[1200]])
print(f"Prediction for 1200 sqft house: ${linear_model.predict(X_test)[0]:0.2f}")
```

```
Prediction on training set: [300. 500.]
Prediction for 1200 sqft house: $240100.00
```

## 第二个示例

第二个示例来自前面包含多个特征 (Features) 的实验。最终求出的参数值和预测结果，与之前实验中未进行归一化 (Normalization) 时通过“长时间运行”梯度下降 (Gradient Descent) 所得到的结果非常接近。
那次未归一化的梯度下降运行花费了好几个小时才得出结果，而这里的计算几乎是瞬间完成的。==闭式解在像这样规模较小的数据集上表现非常出色，但在面对大规模数据集时，其计算开销可能会变得非常大。==

> ==**闭式解不需要进行特征归一化。**==

```python
# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
```

```python
linear_model = LinearRegression()
linear_model.fit(X_train, y_train) 
```

```python
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
```

```
w = [  0.27 -32.62 -67.25  -1.47], b = 220.42
```

```python
print(f"Prediction on training set:\n {linear_model.predict(X_train)[:4]}" )
print(f"prediction using w,b:\n {(X_train @ w + b)[:4]}")
print(f"Target values \n {y_train[:4]}")

x_house = np.array([1200, 3,1, 40]).reshape(-1,4)
x_house_predict = linear_model.predict(x_house)[0]
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.2f}")
```

```
Prediction on training set:
 [295.18 485.98 389.52 492.15]
prediction using w,b:
 [295.18 485.98 389.52 492.15]
Target values 
 [300.  509.8 394.  540. ]
 predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = $318709.09
```

## 恭喜！

在本次实验中，你：

- 使用了开源机器学习工具包 scikit-learn  
- 利用该工具包中的闭式解实现了线性回归


