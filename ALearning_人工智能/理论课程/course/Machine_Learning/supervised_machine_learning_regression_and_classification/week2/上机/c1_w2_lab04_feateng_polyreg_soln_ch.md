
___

- [[#选修实验：特征工程与多项式回归 (Feature Engineering and Polynomial Regression)|选修实验：特征工程与多项式回归 (Feature Engineering and Polynomial Regression)]]
	- [[#选修实验：特征工程与多项式回归 (Feature Engineering and Polynomial Regression)#实验目标|实验目标]]
	- [[#选修实验：特征工程与多项式回归 (Feature Engineering and Polynomial Regression)#实验工具|实验工具]]
- [[#特征工程与多项式回归概览|特征工程与多项式回归概览]]
	- [[#特征工程与多项式回归概览#多项式特征 (Polynomial Features)|多项式特征 (Polynomial Features)]]
		- [[#多项式特征 (Polynomial Features)#如何筛选合适的特征|如何筛选合适的特征]]
		- [[#多项式特征 (Polynomial Features)#换个视角看问题|换个视角看问题]]
		- [[#多项式特征 (Polynomial Features)#对构造出的特征进行缩放|对构造出的特征进行缩放]]
		- [[#多项式特征 (Polynomial Features)#拟合更加复杂的函数|拟合更加复杂的函数]]
	- [[#特征工程与多项式回归概览#恭喜！|恭喜！]]

___

# 选修实验：特征工程与多项式回归 (Feature Engineering and Polynomial Regression) 

![[c1_w2_lab04_feateng_polyreg_soln_ch-1793525935015.webp]]

## 实验目标

在本次实验中，你将：
- 深入探索特征工程 (feature engineering) 与多项式回归 (polynomial regression) ——借助这两项技术，你可以直接沿用线性回归 (linear regression) 的底层机制，去拟合非常复杂甚至高度非线性 (non-linear) 的函数。

## 实验工具

你将继续使用在之前实验中开发好的各个函数，以及 Matplotlib 和 NumPy 库。

```python
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
```

# 特征工程与多项式回归概览

在默认情况下，线性回归只能帮我们构建形如下式的数学模型 (models) ：
$$f_{\mathbf{w},b} = w_0x_0 + w_1x_1+ ... + w_{n-1}x_{n-1} + b \tag{1}$$ 
但如果你的特征 (features) 与数据本质上是非线性的，或者是多个特征的组合，那该怎么办呢？
举个例子，房屋价格往往并不会随着居住面积的增加而呈简单的直线增长，而是==会对面积过小或过大的房子产生一定的“折价惩罚”==，从而呈现出如上方图示那样的弯曲曲线。我们怎样才能用手头现成的线性回归机制去拟合这样一条曲线呢？回想一下，我们手头所谓的“机制”，其实就是通过调整公式 (1) 中的参数 (parameters) $\mathbf{w}$ 和 $\mathbf{b}$，让方程尽可能去“拟合” (fit) 训练数据 (training data) 。然而，如果死守着公式 (1) 原有的特征形式，无论你怎么调整 $\mathbf{w}$ 和 $\mathbf{b}$，都永远不可能拟合出一条非线性的曲线。

## 多项式特征 (Polynomial Features) 

上面我们提到了数据呈现非线性规律的场景。现在，不妨试着用我们已经学过的知识来拟合一条非线性曲线。让我们先从一个最简单的二次函数 (quadratic) 开始：$y = 1+x^2$
    假设输入是 0-19 每个自然整数
    y输出就是 1+x平方

这里用到的所有辅助函数你都已经很熟悉了，随时可以在 `lab_utils.py` 文件中查阅它们的源码。此外，💛我们还会用到 `np.c_[..]`，这是 NumPy 中一个专门用来沿着列边界对数组进行横向拼接的函数。

```python
# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2
X = x.reshape(-1, 1) #变列向量

model_w,model_b = run_gradient_descent_feng(X,y,iterations=1000, alpha = 1e-2)

plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("no feature engineering")
plt.plot(x,X@model_w + model_b, label="Predicted Value")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```
![[c1_w2_lab04_feateng_polyreg_soln_ch-1793527435715.webp|519]]

不出所料，==直接用直线去拟合的效果非常糟糕。==我们真正需要的是形如 $y= w_0x_0^2 + b$ 的模型，也就是引入一个**多项式特征**。
为了做到这一点，你可以==直接对输入数据 (input data) 动手改造==，从中构造 (engineer) 出所需的特征。只要把原始的输入数据替换成对 $x$ 求平方之后的新数据，你就能轻松实现 $y= w_0x_0^2 + b$ 的拟合效果。下面让我们动手试一试，在下方代码中把 `X` 替换成 `X**2`：


```python
# create target data
x = np.arange(0, 20, 1)
y = 1 + x**2

# Engineer features 
X = x**2      #<-- added engineered feature
```

```python
X = X.reshape(-1, 1)  #X should be a 2-D Matrix
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha = 1e-5)

plt.scatter(x, y, marker="x", c="r", label="Actual Value")
plt.title("Added x**2 feature")
plt.plot(x, np.dot(X, model_w) + model_b, label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```
![[c1_w2_lab04_feateng_polyreg_soln_ch-1793527800593.webp|466]]

太棒了！拟合出的曲线几乎与真实数据完美重合。
请留意图表正上方打印出的 $\mathbf{w}$ 和 b 的数值：`w,b found by gradient descent: w: [1.], b: 0.0490`。
梯度下降 (gradient descent) 自动将 $\mathbf{w},b$ 的初始值调整到了 (1.0, 0.049) ，
对应着模型 $y=1*x_0^2+0.049$，这已经非常接近我们的真实目标函数 $y=1*x_0^2+1$ 了。
如果让它继续迭代更长时间，拟合精度还会进一步提升。

### 如何筛选合适的特征

在刚才的例子里，我们事先就已经知道需要引入一个 $x^2$ 项。==但在实际问题中，究竟需要哪些特征往往并不那么一目了然。==这时，一种常见的策略是先==把各种潜在的多项式特征都加进去，再看看哪些特征最管用。==
比如，要是我们刚才尝试的是这样一个模型：$y=w_0x_0 + w_1x_1^2 + w_2x_2^3+b$，结果会怎样呢？

请运行下方的代码单元格来一探究竟。

```python
# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
```

```python
model_w,model_b = run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

plt.scatter(x, y, marker="x", c="r", label="Actual Value")
plt.title("x, x**2, x**3 features")
plt.plot(x, X @ model_w + model_b, label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```
![[c1_w2_lab04_feateng_polyreg_soln_ch-1793528263132.webp|433]]
解释
    ![[c1_w2_lab04_feateng_polyreg_soln_ch-1793528200762.webp|534]]
    ![[c1_w2_lab04_feateng_polyreg_soln_ch-1793528243427.webp|324]]
    当然对应的还有`np.r_`      沿一维末尾直接延展或上下拼

请观察打印出的参数值：$\mathbf{w}$ 为 `[0.08 0.54 0.03]`，而 b 为 `0.0106`。这意味着经过拟合/训练之后，我们得到的模型是：
$$ 0.08x + 0.54x^2 + 0.03x^3 + 0.0106 $$
可以看到，梯度下降通过==大幅提高 $w_1$ 项相对于其他权重的数值，自动突出了最契合真实规律的 $x^2$ 特征数据。==
如果你让梯度下降继续运行更长的时间，它还会进一步削弱其余两项的影响力。
>梯==**度下降能够通过调高对应参数的权重，自动帮我们挑出那些“正确的”特征！**==

不妨再来梳理一下这个非常深刻的规律：
- 某个特征对应的==权重 (weight) 数值越小，说明该特征越不重要== (或者越偏离真实规律) ；
    在极端情况下，当某个权重变成 0 或极其接近 0 时，就意味着它所对应的那个特征对拟合数据毫无帮助。
- 在上面的例子中，完成拟合后，$x^2$ 特征所对应的权重明显远大于 $x$ 或 $x^3$ 的权重，因为它在拟合真实数据时发挥的作用最大。

### 换个视角看问题

在上面的讨论中，我们是根据多项式特征与目标数据 (target data) 的契合程度来挑选特征的。其实还可以换一个角度来理解：一旦我们构造出了新的特征，接下来执行的依然是标准的线性回归。
既然如此，==最理想的新特征自然就是与目标值之间呈现标准线性关系 (linear) 的那个特征。==
    解释 [[最理想的新特征呈现标准线性关系]]
通过下面的例子，你可以一目了然地看清这一点：

```python
# create target data
x = np.arange(0, 20, 1)
y = x**2

# engineer features .
X = np.c_[x, x**2, x**3]   #<-- added engineered feature
X_features = ['x','x^2','x^3']
```

```python
fig,ax=plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")
plt.show()
```
![[c1_w2_lab04_feateng_polyreg_soln_ch-1793528487124.webp]]

从上方的三张对比图中可以清楚地看到：当把构造出的 $x^2$ 特征作为横轴、目标值 $y$ 作为纵轴作图时，两者呈现出了笔直的线性关系！这样一来，线性回归就能轻而易举地利用这个特征构建出精准的模型了。

### 对构造出的特征进行缩放

正如我们在上一个实验中所讲的，如果数据集中的各个特征在数值尺度上相差悬殊，我们就应当先进行**特征缩放 (feature scaling)** ，以便加快梯度下降的收敛速度。在上面的例子中，我们同时引入了 $x$、$x^2$ 和 $x^3$，这三项的数值跨度天生就会相差成百上千倍。下面，就让我们对这个例子应用 Z-score 标准化 (Z-score normalization) ：

```python
# create target data
x = np.arange(0,20,1)
X = np.c_[x, x**2, x**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X,axis=0)}")

# add mean_normalization 
X = zscore_normalize_features(X)     
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X,axis=0)}")
```

```
Peak to Peak range by column in Raw        X:[  19  361 6859]
Peak to Peak range by column in Normalized X:[3.3  3.18 3.28]
```

完成归一化之后，我们就可以放心地设置一个大得多的学习率 (learning rate) alpha，重新运行一次梯度下降：

```python
x = np.arange(0,20,1)
y = x**2

X = np.c_[x, x**2, x**3]
X = zscore_normalize_features(X) 

model_w, model_b = run_gradient_descent_feng(X, y, iterations=100000, alpha=1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value")
plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
```
![[c1_w2_lab04_feateng_polyreg_soln_ch-1793528723597.webp|590]]

特征缩放让算法的收敛 (converge) 速度实现了质的飞跃！   
请再次留意打印出的权重 $\mathbf{w}$ 数值：对应着 $x^2$ 项的 $w_1$ 依然得到了最大程度的凸显，而==梯度下降几乎已经把无关的 $x^3$ 项和 $x$ 项彻底剔除掉了。==

### 拟合更加复杂的函数

有了特征工程这一利器，哪怕是形态相当复杂的函数，线性回归也照样能够轻松建模：

```python
x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X) 

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

plt.scatter(x, y, marker="x", c="r", label="Actual Value")
plt.title("Normalized x x**2, x**3 feature")
plt.plot(x, X @ model_w + model_b, label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

```
![[c1_w2_lab04_feateng_polyreg_soln_ch-1793529349624.webp]]
![[c1_w2_lab04_feateng_polyreg_soln_ch-1793529366705.webp|471]]
    类似泰勒展开
    就是用幂次函数拟合各种式子

## 恭喜！

在本次实验中，你已经掌握了以下重要技能：
- 学会了如何借助特征工程，让线性回归也能游刃有余地拟合复杂甚至高度非线性的函数
- 深刻认识到了在进行特征工程时配合使用特征缩放的重要性

