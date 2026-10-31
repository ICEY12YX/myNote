# 非计分实验：使用 Scikit-Learn 实现逻辑回归 (Logistic Regression) 




## 实验目标
在本实验中，你将：
-  使用 scikit-learn 训练一个逻辑回归模型 (logistic regression model) 。


## 数据集 (Dataset) 
让我们继续从之前使用过的同一个数据集开始。


```python
import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
```

## 拟合模型 (Fit the model) 

下方的代码从 scikit-learn 中导入了逻辑回归模型。你可以通过调用 `fit` 函数，在训练数据 (training data) 上完成该模型的拟合。


```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X, y)
```

## 进行预测 (Make Predictions) 

你可以通过调用 `predict` 函数，查看该模型所给出的预测结果 (predictions) 。


```python
y_pred = lr_model.predict(X)

print("Prediction on training set:", y_pred)
```

## 计算准确率 (Calculate accuracy) 

你可以通过调用 `score` 函数，计算该模型在训练集 (training set) 上的准确率 (accuracy) 。


```python
print("Accuracy on training set:", lr_model.score(X, y))
```
