# 选修实验：特征缩放与学习率 (多变量) (Feature scaling and Learning Rate (Multi-variable) ) 

## 实验目标
在本次实验中，你将：
- 调用在上一个实验中开发好的多变量 (multiple variables) 线性回归函数
- 在包含多个特征 (multiple features) 的数据集上运行梯度下降 (Gradient Descent) 算法
- 探索*学习率 (learning rate) alpha* 的大小对梯度下降收敛过程的具体影响
- 利用 Z-score 标准化 (z-score normalization) 进行*特征缩放 (feature scaling) *，从而大幅提升梯度下降的运行表现

## 实验工具
你将综合使用在上一个实验中编写的各个函数，以及 Matplotlib 和 NumPy 库。


```python
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data, run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')
```

## 符号说明 (Notation) 

|通用 &lt;br /&gt;  符号  | 含义说明| Python 变量名 (如适用) |
|: ------------|: ------------------------------------------------------------||
| $a$ | 标量 (scalar) ，非粗体                                                      ||
| $\mathbf{a}$ | 向量 (vector) ，小写粗体                                                 ||
| $\mathbf{A}$ | 矩阵 (matrix) ，大写粗体                                         ||
| **回归 (Regression) ** |         |    |     |
|  $\mathbf{X}$ | 训练样本矩阵                  | `X_train` |   
|  $\mathbf{y}$  | 训练样本目标值 (targets)                | `y_train` 
|  $\mathbf{x}^{(i)}$, $y^{(i)}$ | 第 $i$ 个训练样本 | `X[i]`, `y[i]`|
| m | 训练样本的总数量 | `m`|
| n | 每个样本中包含的特征数量 | `n`|
|  $\mathbf{w}$  |  参数 (parameter) ：权重 (weight)                       | `w`    |
|  $b$           |  参数：偏置 (bias)                                           | `b`    |     
| $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ | 以 $\mathbf{w},b$ 为参数的模型在 $\mathbf{x}^{(i)}$ 处的计算结果：$f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)}+b$  | `f_wb` | 
|$\frac{\partial J(\mathbf{w},b)}{\partial w_j}$| 代价 (cost) 关于参数 $w_j$ 的梯度或偏导数 (partial derivative) |`dj_dw[j]`| 
|$\frac{\partial J(\mathbf{w},b)}{\partial b}$| 代价关于参数 $b$ 的梯度或偏导数| `dj_db`|

# 问题背景

和前面的实验一样，我们继续以“房价预测”作为研究案例。如下表所示，本次使用的训练数据集 (training data set) 包含了大量样本，每个样本都具有 4 个特征 (房屋面积、卧室数量、楼层数以及房龄) 。请注意，在本实验中，房屋面积特征直接以“平方英尺” (sqft) 为单位，而早期实验使用的单位则是“1000 平方英尺”。同时，本次的数据集规模也比上一个实验大得多。

我们希望利用这些数据构建一个线性回归模型 (linear regression model) ，进而用它去预测其他房子的价格——比如一套面积为 1200 平方英尺、拥有 3 间卧室、1 层楼高、房龄 40 年的房子。

## 数据集 (Dataset) ：
| 面积 (平方英尺) | 卧室数量  | 楼层数 | 房龄 | 价格 (千美元)  |   
| ----------------| ------------------- |----------------- |--------------|----------------------- |  
| 952             | 2                   | 1                | 65           | 271.5                  |  
| 1244            | 3                   | 2                | 64           | 232                    |  
| 1947            | 3                   | 2                | 17           | 509.8                  |  
| ...             | ...                 | ...              | ...          | ...                    |



```python
# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
```

下面，我们先将每个特征分别与房屋价格绘制成散点图，直观地看一看整个数据集及其各个特征的分布情况。


```python
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()
```

把每一个特征与目标值 (房屋价格) 分别作图对比，能帮我们初步判断出哪些特征对房价的影响力最大。从上方的图表中可以看出：随着房屋面积增大，房价也随之明显走高；卧室数量和楼层数对房价似乎没有特别强烈的影响；而新建成的房子普遍比老房子的价格更高。

&lt;a name=&quot;toc_15456_5&quot;&gt;&lt;/a&gt;
## 多变量梯度下降
下面是你在上一个实验中推导并实现的多变量梯度下降核心公式：

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j := w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{1}  \; & \text{for j = 0..n-1}\newline
&b\ \ := b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
\end{align*}$$

其中，n 是特征的数量，参数 $w_j$ 和 $b$ 会同时完成更新；而对应的偏导数计算公式分别为：

$$
\begin{align}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{2}  \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3}
\end{align}
$$
* m 是数据集中训练样本的总数

    
* $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ 是模型给出的预测值 (prediction) ，而 $y^{(i)}$ 则是真实的目标值


## 学习率 (Learning Rate) 
&lt;figure&gt;
    &lt;img src=&quot;./images/C1_W2_Lab06_learningrate.PNG&quot; style=&quot;width:1200px;&quot; &gt;
&lt;/figure&gt;
理论课上已经讨论过不少关于如何设置学习率 $\alpha$ 的问题。如上方公式 (1) 所示，学习率直接控制着每一轮参数更新的步幅大小，而且所有参数都共用同一个学习率 $\alpha$。  

下面，就让我们在这个数据集上运行梯度下降，并分别尝试几种不同的 $\alpha$ 取值。

### $\alpha$ = 9.9e-7


```python
#set alpha to 9.9e-7
_, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = 9.9e-7)
```

看起来这个学习率设得太高了，求解过程根本无法收敛 (converge) ，总代价非但没有下降，反而在*不断飙升*。让我们把运行结果画出来看看：


```python
plot_cost_i_w(X_train, y_train, hist)
```

右侧的图表展示了其中一个参数 $w_0$ 的变化轨迹。在每一次迭代 (iteration) 中，它的更新步幅都冲过了头，直接越过了最优值 (optimal value) ，结果导致总代价不仅没有逼近最低点 (minimum) ，反而在*一步步增大*。需要注意的是，这张图展示的并不是最完整的全貌，因为每一轮迭代其实都在同时修改全部 4 个权重参数，而不仅仅是一个参数。这张图只是在将其余参数固定为合理数值的前提下，单独展示了 $w_0$ 的变化情况。因此在当前图表及后续图表中，你可能会注意到蓝色曲线与橙色折线之间存在轻微的偏差。


### $\alpha$ = 9e-7
接下来，让我们把学习率稍微调小一点点，看看会发生什么。


```python
#set alpha to 9e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 9e-7)
```

在整个运行过程中，代价一直在稳步下降，这说明当前的 alpha 取值不算太大。


```python
plot_cost_i_w(X_train, y_train, hist)
```

从左侧图表中可以看到，代价正如我们所期望的那样持续下降。而在右侧图表中，虽然参数 $w_0$ 依然在最低点两侧来回震荡 (oscillating) ，但每一次迭代后的代价都在不断减小，而不再是发散增大。请留意上方的打印日志：随着 `w[0]` 在最优值两侧来回跳跃，偏导数 `dj_dw[0]` 在每一轮迭代时都会改变正负号。
使用这个 alpha 值最终是能够收敛的，你也可以试着修改迭代次数，观察它的后续表现。

### $\alpha$ = 1e-7
现在，让我们把 $\alpha$ 再进一步调小一些，看看又会有怎样的表现。


```python
#set alpha to 1e-7
_,_,hist = run_gradient_descent(X_train, y_train, 10, alpha = 1e-7)
```

在整个运行过程中，代价同样保持持续下降，说明此时的 $\alpha$ 也不算太大。


```python
plot_cost_i_w(X_train,y_train,hist)
```

在左侧图表中，代价一如既往地平稳下降。而在右侧图表中可以看到，这一次 $w_0$ 不再出现来回震荡，而是沿着单侧平稳地逼近最低点。在整个运行期间，偏导数 `dj_w0` 始终保持为负数。这个方案最终同样能够顺利收敛。

## 特征缩放 (Feature Scaling) 
&lt;figure&gt;
    &lt;img src=&quot;./images/C1_W2_Lab06_featurescalingheader.PNG&quot; style=&quot;width:1200px;&quot; &gt;
&lt;/figure&gt;
理论课上专门强调了对数据集进行重新缩放 (rescaling) 的重要性——这样可以让各个特征都处于相近的数值范围之内。
如果你对背后的深层原因感兴趣，可以点击下方的“Details”标题展开阅读；如果你想直接看具体做法，下面的章节将带你一步步动手实现特征缩放。

&lt;details&gt;
&lt;summary&gt;
    &lt;font size=&#39;3&#39;, color=&#39;darkgreen&#39;&gt;&lt;b&gt;Details&lt;/b&gt;&lt;/font&gt;
&lt;/summary&gt;

让我们回过头再看一看 $\alpha$ = 9e-7 时的情况。这个数值已经非常接近在不引发发散 (diverging) 的前提下我们所能设置的 $\alpha$ 上限了。下图展示的是运行最初几轮迭代时的短期表现：

&lt;figure&gt;
    &lt;img src=&quot;./images/C1_W2_Lab06_ShortRun.PNG&quot; style=&quot;width:1200px;&quot; &gt;
&lt;/figure&gt;

从上图中不难发现，虽然总代价确实在不断下降，但由于 $w_0$ 对应的梯度幅值远大于其他参数，因此 $w_0$ 的更新进展比其他几个参数快得多。

而下面这张图则展示了在 $\alpha$ = 9e-7 下耗时数小时进行超长迭代后的结果：

&lt;figure&gt;
    &lt;img src=&quot;./images/C1_W2_Lab06_LongRun.PNG&quot; style=&quot;width:1200px;&quot; &gt;
&lt;/figure&gt;
    
可以看到，总代价在经历了最初的快速下降之后，随后的下降速度变得极其缓慢。请对比观察一下 `w0` 与 `w1`、`w2`、`w3` 之间的差异，以及 `dj_dw0` 与 `dj_dw1-3` 之间的差异：`w0` 很快就逼近了它的最终收敛值，同时 `dj_dw0` 也迅速缩小到了一个微乎其微的数值，说明 `w0` 已经基本就位；然而其余几个参数的调整速度却慢如蜗牛。

为什么会这样呢？我们有什么办法可以改进吗？请看下图：
&lt;figure&gt;
    &lt;center&gt; &lt;img src=&quot;./images/C1_W2_Lab06_scale.PNG&quot;   &gt;&lt;/center&gt;
&lt;/figure&gt;   

上方这张图揭示了各个权重参数 $w$ 更新速度极不均衡的根本原因：
- 所有的参数更新 ($w$ 和 $b$) 都共用同一个学习率 $\alpha$。
- 在更新各个权重 $w$ 时 (不包括 $b$) ，公共的误差项都会分别乘以对应的特征值。
- 偏偏各个特征在数值量级 (magnitude) 上相差悬殊，导致某些特征对应的参数更新得飞快，而另一些则慢得惊人。在这个例子里，$w_0$ 乘以的是通常大于 1000 的“房屋面积 (平方英尺) ”，而 $w_1$ 乘以的则是通常只有 2 到 4 的“卧室数量”。
    
解决这一难题的利器就是特征缩放。

理论课上一共介绍了三种不同的缩放方法：
- 特征缩放：最简单的做法是将每个正数特征直接除以它的最大值；或者更通用地，同时利用每个特征的最小值和最大值，按照 `(x-min)/(max-min)` 进行重缩放。这两种方式都能将特征归一化 (normalizes) 到 -1 至 1 的区间内——前者形式简单，适合处理全为正数的特征 (正如理论课上的例子) ，而后者则适用于任意特征。
- 均值归一化 (Mean normalization) ：$x_i := \dfrac{x_i - \mu_i}{max - min} $ 
- Z-score 标准化：也就是我们接下来要重点探索的方法。


### Z-score 标准化
经过 Z-score 标准化处理后，所有特征的均值 (mean) 都会变成 0，标准差 (standard deviation) 都会变成 1。

要实现 Z-score 标准化，只需按照下方公式对你的输入特征值进行转换即可：
$$x^{(i)}_j = \dfrac{x^{(i)}_j - \mu_j}{\sigma_j} \tag{4}$$ 
其中，$j$ 代表在矩阵 $\mathbf{X}$ 中选定的某一个特征 (即某一列) 。$\mu_j$ 是第 (j) 个特征下所有样本值的均值，而 $\sigma_j$ 则是第 (j) 个特征的标准差：
$$
\begin{align}
\mu_j &= \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j \tag{5}\\
\sigma^2_j &= \frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^2  \tag{6}
\end{align}
$$

>**实现注意事项**：在对特征进行归一化处理时，务必把计算过程中用到的均值和标准差妥善保存下来。因为当模型训练好参数之后，我们往往需要用它去预测那些从未见过的新房子的价格。当拿到一组新的输入值 x (比如某套房子的客厅面积和卧室数量) 时，我们必须先用之前从训练集里算出的均值和标准差对这个新样本 x 进行同样的归一化转换，然后才能输入模型进行预测。

**代码实现**


```python
def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
      X (ndarray (m,n))     : input data, m examples, n features
      
    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
 
#check our work
#from sklearn.preprocessing import scale
#scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)
```

下面让我们一步步拆解 Z-score 标准化的具体过程。下方的三张子图清晰展示了数据一步步完成转换的轨迹：


```python
mu     = np.mean(X_train,axis=0)   
sigma  = np.std(X_train,axis=0) 
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma      

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:,0], X_mean[:,3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()
```

上方的图表展示了训练集中两个特征——“房龄” (age) 与“房屋面积 (平方英尺) ” (size(sqft)) 之间的分布关系。*这三张图均采用相同的横纵坐标比例尺 (equal scale) 绘制*。
- 左图：未归一化 (Unnormalized) ：可以看到，“房屋面积 (平方英尺) ”这一特征的数值跨度或方差 (variance) 比“房龄”大得多。
- 中图：第一步先从每个特征中减去各自的平均值，使得所有特征都以 0 为中心分布。由于比例尺的缘故，纵轴“房龄”的变化肉眼不太容易看出来，但横轴“房屋面积 (平方英尺) ”已经明显以 0 为中心对称分布了。
- 右图：第二步再除以各自的标准差。这样处理后，两个特征不仅都以 0 为中心，而且拥有了非常相近的数值尺度。

现在，让我们正式对训练数据进行归一化，并将它与原始数据做个对比：


```python
# normalize the original features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")
```

经过归一化处理后，每一列特征的峰峰值跨度 (Peak to Peak range) 从原本相差成千上万倍，瞬间缩小到了只有 2 到 3 倍的紧凑范围。


```python
fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count"); 
fig.suptitle("distribution of features after normalization")

plt.show()
```

请观察上方图表：归一化之后的数据范围 (横轴) 全部以 0 为中心，大致分布在 +/- 2 之间。最重要的是，现在每一个特征的数值跨度都变得高度一致。

接下来，让我们在归一化后的数据集上重新运行梯度下降算法。
请特别留意这里设置的**大得多的学习率 alpha**——它将大幅加快梯度下降的收敛速度。


```python
w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )
```

经过缩放的特征让我们**以快得多的速度**拿到了非常精准的拟合结果！请看打印日志，在这段相当简短的迭代运行结束时，每一个参数的梯度都已经变得微乎其微。对于经过特征归一化的回归问题而言，将学习率设为 0.1 通常是一个非常不错的起点。
下面让我们把模型的预测结果与真实目标值画在同一张图上做个对比。请注意，在计算预测值时使用的是归一化后的特征，而在绘制散点图时，横坐标展示的则是原始的特征数值。


```python
#predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()
```

拟合效果看起来非常棒！这里有两点值得注意：
- 当模型包含多个特征时，我们无法再用单独一张二维平面图把预测结果与所有特征的关系同时画出来了。
- 在生成上方图表的预测值时，我们输入的是归一化后的特征。凡是使用从归一化训练集上学到的参数来进行预测，输入的样本特征也必须先做同样的归一化处理。

**房价预测 (Prediction) **
我们构建模型的最终目的，是用它去预测那些不在现有数据集中的房屋价格。现在，就让我们来预测一套面积为 1200 平方英尺、3 间卧室、1 层楼高、房龄 40 年的房子的价格。请牢记：你必须先用训练数据归一化时算出的均值和标准差，对这组新房屋数据进行归一化转换。


```python
# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")
```

**代价等高线 (Cost Contours) **  
&lt;img align=&quot;left&quot; src=&quot;./images/C1_W2_Lab06_contours.PNG&quot;   style=&quot;width:240px;&quot; &gt;我们还可以从“代价等高线图 (contour plot) ”的视角来理解特征缩放的意义。当不同特征的数值尺度不匹配时，代价关于各个参数的等高线图会显得极不对称 (asymmetric) 。

在下方的对比图中，横纵坐标的参数比例尺已经对齐。左侧展示的是特征归一化之前，权重 w[0] (对应房屋面积) 与 w[1] (对应卧室数量) 的代价等高线图。由于两者的尺度极度不对称，等高线被拉成了细长的条状，甚至连完整的闭合曲线都看不见。相比之下，右侧经过特征归一化之后，代价等高线变成了非常对称 (symmetric) 的同心圆环。这样一来，在执行梯度下降时，各个参数的更新就能齐头并进，以相同的步伐快速奔向谷底。



```python
plt_equal_scale(X_train, X_norm, y_train)
```


## 恭喜！
在本次实验中，你已经完成了以下实践：
- 调用了你在之前实验中开发的多特征线性回归核心函数
- 深入探索了学习率 $\alpha$ 对算法收敛过程的影响
- 亲眼验证了使用 Z-score 标准化进行特征缩放在加速梯度下降收敛方面的巨大价值

## 致谢
本实验所使用的房屋数据源自 Dean De Cock 为数据科学教育整理编纂的艾姆斯房价数据集 (Ames Housing dataset) 。
