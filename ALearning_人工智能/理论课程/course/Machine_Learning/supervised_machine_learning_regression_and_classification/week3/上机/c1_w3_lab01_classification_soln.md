___
- [[#可选实验：分类 (Classification)|可选实验：分类 (Classification)]]
	- [[#可选实验：分类 (Classification)#分类问题 (Classification Problems)|分类问题 (Classification Problems)]]
	- [[#可选实验：分类 (Classification)#线性回归方法 (Linear Regression approach)|线性回归方法 (Linear Regression approach)]]
	- [[#可选实验：分类 (Classification)#恭喜！|恭喜！]]

___
# 可选实验：分类 (Classification)

在本实验中，你将对比回归 (regression) 与分类的异同。

```
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import dlc, plot_data
from plt_one_addpt_onclick import plt_one_addpt_onclick
plt.style.use('./deeplearning.mplstyle')

```

## 分类问题 (Classification Problems)

![[c1_w3_lab01_classification_soln-1793641257685.webp|424]]
 常见的分类问题包括：识别一封电子邮件是垃圾邮件还是正常邮件，或者判断一个肿瘤是恶性的还是良性的。具体来说，这些都属于**二分类 (binary classification)** 问题的典型例子，即结果只有两种可能。
 我们通常用成对的“正类 (positive) ”/“负类 (negative) ”来描述这些结果，比如“是”/“否”、“真”/“假”或者“1”/“0”。

在绘制分类数据集 (data sets) 的图表时，我们常常用不同的符号来代表样本 (example) 的结果。在下面的图表中，“X”用来代表正类的值，而“O”则用来代表负类的结果。

```python
x_train = np.array([0.0, 1, 2, 3, 4, 5])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_train2 = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])

```

```python
pos = y_train == 1
neg = y_train == 0

fig,ax = plt.subplots(1,2,figsize=(8,3))
# 图1，单变量 (single variable) 
ax[0].scatter(x_train[pos], y_train[pos], marker="x", s=80, c="red", label="y=1")
ax[0].scatter(
    x_train[neg],
    y_train[neg],
    marker="o",
    s=100,
    label="y=0",
    facecolors="none",
    edgecolors=dlc["dlblue"],
    lw=3,
)

ax[0].set_ylim(-0.08, 1.1)
ax[0].set_ylabel("y", fontsize=12)
ax[0].set_xlabel("x", fontsize=12)
ax[0].set_title("one variable plot")
ax[0].legend()

# 图2，双变量 (two variables) 
plot_data(X_train2, y_train2, ax[1])
ax[1].axis([0, 4, 0, 4])
ax[1].set_ylabel('$x_1$', fontsize=12)
ax[1].set_xlabel('$x_0$', fontsize=12)
ax[1].set_title('two variable plot')
ax[1].legend()
plt.tight_layout()
plt.show()
```
![[c1_w3_lab01_classification_soln-1793644585664.webp]]


观察上方的图表，请注意以下几点：

* 在单变量图中，正类结果既用红色的“X”表示，也对应纵轴上的 y=1。负类结果则用蓝色的“O”表示，位于 y=0 处。
    - 回想一下，在线性回归 (linear regression) 中，y 的取值并不会被限制在两个数值上，而是可以取任意数值。

* 在双变量图中，纵轴不再表示 y。正类结果直接用红色的“X”表示，而负类结果则使用蓝色的“O”符号来表示。
    - 回想一下，在多变量线性回归 (linear regression with multiple variables) 中，y 的取值同样不会局限于两个数值，因此类似的图表会呈现出三维 (three-dimensional) 结构。

## 线性回归方法 (Linear Regression approach)

在上一周的学习中，你曾运用线性回归构建了一个预测模型 (prediction model) 。下面让我们沿用课程讲座中介绍的那个简单例子，在这里尝试一下这种方法。该模型将根据肿瘤的大小来预测肿瘤是良性还是恶性。请尝试按以下步骤操作：

* 点击“Run Linear Regression” (运行线性回归) ，为给定的数据寻找最佳的线性回归模型。
    - 请注意，生成的线性模型 (linear model) 并**不能**很好地拟合数据。
        改进结果的一种方法是引入一个\*阈值 (threshold) \*。
![[c1_w3_lab01_classification_soln-1793645016439.webp|574]]

* 勾选“Toggle 0.5 threshold” (切换 0.5 阈值) 复选框，查看应用阈值之后的预测效果。
    - 这些预测看起来很不错，预测结果与实际数据完全吻合。
![[c1_w3_lab01_classification_soln-1793645026981.webp]]

* *重要提示*：现在，在图表最右侧较大的肿瘤尺寸范围内 (靠近 10 的位置) ，继续添加几个“恶性”数据点 (data points) ，然后重新运行线性回归。
  * 此时，模型虽然能正确预测较大的肿瘤，但位于 x=3 处的数据点却被错误预测了！
![[c1_w3_lab01_classification_soln-1793645088294.webp|582]]

* 如需清空或重置图表，请重新运行包含绘图命令的代码单元格。

```python
w_in = np.zeros((1))
b_in = 0
plt.close('all') 
addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=False)

```

上面的例子清晰地表明，==线性模型不足以对分类数据 (categorical data) 进行建模==。接下来的实验将介绍如何对该模型进行扩展。

## 恭喜！

在本实验中，你已经完成了以下内容：

* 探索了分类数据集及其图表绘制方法
* 明确了线性回归并不适用于解决分类问题。
