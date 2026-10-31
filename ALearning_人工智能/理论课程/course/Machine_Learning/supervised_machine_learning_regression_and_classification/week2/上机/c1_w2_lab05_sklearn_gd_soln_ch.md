# 选修实验：使用 Scikit-Learn 实现线性回归 (Linear Regression using Scikit-Learn) 

在机器学习 (machine learning) 领域，有一个开源且支持商用落地的知名工具包叫做 scikit-learn。这个工具包内置了大量你在本课程中将会接触到的经典算法的成熟实现。



## 实验目标
在本次实验中，你将：
- 利用 scikit-learn 工具包，基于梯度下降 (Gradient Descent) 算法实现线性回归 (linear regression) 

## 实验工具
你将综合使用来自 scikit-learn 的内置函数，以及 matplotlib 和 NumPy 库。


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from lab_utils_multi import  load_house_data
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')
```

# 梯度下降 (Gradient Descent) 
Scikit-learn 提供了一个基于随机梯度下降的回归模型 (regression model) `sklearn.linear_model.SGDRegressor`。与你之前亲手编写的梯度下降算法一样，这个模型在输入特征经过归一化 (normalized) 处理时表现最佳。而 `sklearn.preprocessing.StandardScaler` 则可以帮我们自动完成与之前实验完全相同的 Z-score 标准化 (z-score normalization) ——在 scikit-learn 文档中，它通常被称为“标准分数” ('standard score') 。

### 加载数据集 (data set) 


```python
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
```

### 对训练数据进行特征缩放/归一化


```python
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")
```

### 创建并拟合回归模型


```python
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")
```

### 查看模型参数 (parameters) 
请注意，这里学到的参数对应的是*归一化之后*的输入数据。可以看到，模型拟合出的参数数值与我们在前面实验中针对同一批数据手动算出的结果高度吻合。


```python
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")
```

### 进行预测 (Make predictions) 
接下来，我们对训练数据的目标值 (targets) 进行预测。这里我们将同时演示两种方式：一是直接调用模型自带的 `predict` 方法，二是利用学到的参数 $w$ 和 $b$ 手动进行点积计算。


```python
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")
```

### 绘制预测结果对比图
最后，让我们把模型给出的预测值与真实目标值画在图表上做个直观对比。


```python
# plot predictions and targets vs original features    
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
```

## 恭喜！
在本次实验中，你已经成功：
- 掌握了如何使用工业界广泛采用的开源机器学习工具包 scikit-learn
- 调用该工具包中的梯度下降与特征归一化模块，快速构建并训练了一个多变量线性回归模型


```python

```
