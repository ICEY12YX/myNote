# 选修实验 - 多分类 (Multi-class Classification)


## 1.1 实验目标
在本次实验中，你将探索一个使用神经网络 (Neural Networks) 解决多分类问题的实例。
![[c2_w2_multiclass_tf-1794089060117.webp]]


## 1.2 工具函数

你将使用一些绘图子程序，它们保存在当前目录下的 `lab_utils_multiclass_TF.py` 文件中。

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib widget
from sklearn.datasets import make_blobs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
np.set_printoptions(precision=2)
from lab_utils_multiclass_TF import *
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
```

# 2.0 多分类

神经网络常被用于对数据进行分类。典型的例子包括：
- 接收照片作为输入，并将照片中的主体分类为 {狗, 猫, 马, 其他}
- 接收一个句子作为输入，并对其中的各个词汇进行“词性”分类：{名词, 动词, 形容词等}  

这类网络的==最后一层 (Layer) 会包含多个单元 (Units) ==，每一个输出都与一个特定的类别相对应。当把一个输入样本送入网络时，输出值最高的那个单元所对应的类别，就是网络给出的预测类别。如果将输出进一步送入 **Softmax 函数 (Softmax Function)** ，那么 Softmax 的输出就会给出该输入样本属于各个类别的概率 (Probabilities) 。

在本次实验中，你将看到一个在 Tensorflow 中搭建多分类网络的实例。随后，我们还将深入观察神经网络究竟是如何做出这些预测判定的。

让我们先从创建一个包含四个类别的数据集 (Dataset) 开始。

## 2.1 准备并可视化数据

我们将使用 Scikit-Learn 的 `make_blobs` 函数来生成一个包含 4 个类别的训练数据集，如下方图表所示。

```python
# make 4-class dataset for classification
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)
```

```python
plt_mc(X_train,y_train,classes, centers, std=std)
```
![[c2_w2_multiclass_tf-1794089148007.webp|348]]

图中的每一个圆点都代表一个训练样本。坐标轴 (x0,x1) 是输入特征 (Features) ，而颜色则代表该样本所属的类别。一旦模型训练完成，当向它输入一个新的样本 (x0,x1) 时，模型就会预测出它所属的类别。  

虽然这是一个人工生成的数据集，但它非常具有代表性，很好地反映了许多现实世界中的分类问题：既有多个输入特征 (x0,...,xn) ，又有多个输出类别，而训练模型的目标就是利用这些输入特征来准确预测出对应的输出类别。

```python
# show classes in data set
print(f"unique classes {np.unique(y_train)}")
# show how classes are represented
print(f"class representation {y_train[:10]}") #前10个样本的类别
# show shapes of our dataset
print(f"shape of X_train: {X_train.shape}, shape of y_train: {y_train.shape}")
```

```
unique classes [0 1 2 3]
class representation [3 3 3 0 3 3 3 3 2 0]
shape of X_train: (100, 2), shape of y_train: (100,)
```

## 2.2 模型

![[c2_w2_multiclass_tf-1794089209625.webp|453]]
本次实验将使用如右图所示的两层网络。
与二分类 (Binary Classification) 网络不同，这个网络有四个输出，分别对应四个类别。给定一个输入样本，输出值最高的那个单元就是网络对该输入所预测的类别。  

下面是如何在 Tensorflow 中构建该网络的示例代码。请留意，==输出层使用的是线性 (`linear`)== 激活函数 (Activation Function) ，而不是 `softmax`。虽然我们也可以直接把 Softmax 放在输出层中，但在训练过程中，将线性输出直接传递给损失函数 (Loss Function) 在数值计算上会更加稳定。如果后续需要使用模型来预测概率，只需在预测时再应用 Softmax 即可。

```python
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        Dense(2, activation = 'relu',   name = "L1"),
        Dense(4, activation = 'linear', name = "L2")
    ]
)
```

下面的语句用于编译和训练该网络。将 `from_logits=True` 作为参数传递给损失函数，即明确告知系统输出层使用的是线性激活函数而非 Softmax。

```python
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

model.fit(
    X_train,y_train,
    epochs=200
)
```

模型训练完成后，我们就可以直观地看到它是如何对训练数据进行分类的。

```python
plt_cat_mc(X_train, y_train, model, classes)
```
![[c2_w2_multiclass_tf-1794089278444.webp]]
可能由于一些偏差 上面那个训练的不好, 再跑一次就好了
![[c2_w2_multiclass_tf-1794089352887.webp]]

在上图中，决策边界 (Decision Boundaries) 展示了模型是如何划分输入空间的。这个非常简单的模型毫不费力地就完成了对训练数据的分类。那么，它究竟是如何做到这一点的呢？让我们更深入地剖析一下这个网络的内部细节。

在下面的代码中，我们将从训练好的模型中提取出权重 (Weights) ，并利用它们绘制出网络中==每一个单元所表示的函数图像==。再往下看，还会有一段对这些结果的详细解析。其实你并不需要掌握这些底层细节就能成功地使用神经网络，但了解它们有助于你建立起更敏锐的直觉，明白各个层究竟是如何协同配合来解决分类问题的。

```python
# gather the trained parameters from the first layer
l1 = model.get_layer("L1")
W1,b1 = l1.get_weights()
```

```python
# plot the function of the first layer
plt_layer_relu(X_train, y_train.reshape(-1,), W1, b1, classes)
```
![[c2_w2_multiclass_tf-1794089424400.webp]]

```python
# gather the trained parameters from the output layer
l2 = model.get_layer("L2")
W2, b2 = l2.get_weights()
# create the 'new features', the training examples after L1 transformation
Xl2 = np.maximum(0, np.dot(X_train,W1) + b1)

plt_output_layer_linear(Xl2, y_train.reshape(-1,), W2, b2, classes,
                        x0_rng = (-0.25,np.amax(Xl2[:,0])), x1_rng = (-0.25,np.amax(Xl2[:,1])))
```
![[c2_w2_multiclass_tf-1794089436847.webp]]

## 原理解析
#### 第 1 层
![[c2_w2_multiclass_tf-1794089458848.webp]]
这些图像展示了网络第一层中单元 0 和单元 1 的功能。坐标轴代表输入特征 ($x_0,x_1$) ，而单元的输出大小则由背景颜色的深浅来表示，具体数值可参考每张图右侧的颜色条。请注意，==由于这些单元使用的是 ReLU (修正线性单元) ，因此它们的输出并不局限于 0 到 1 之间==——在这个例子里，它们的峰值甚至超过了 20。
图中的等高线 (Contour Lines) 标示出了输出 $a^{[1]}_j$ 从零变为非零的转折分界线。回想一下 ReLU 的函数图像：
![[c2_w2_multiclass_tf-1794089543785.webp|238]]
==图中的这条等高线，正是 ReLU 函数图像上的那个拐点 (Inflection Point) 。==

单元 0 成功地将类别 ==0 和 1 与类别 2 和 3 分隔开来==：位于分界线左侧的点 (类别 0 和 1) 输出为零，而位于分界线右侧的点则会输出大于零的值。  
单元 1 则将类别 ==0 和 2 与类别 1 和 3 分隔开来==  ：位于分界线上方的点 (类别 0 和 2) 输出为零，而位于分界线下方的点则会输出大于零的值。接下来，让我们看看这些输出在下一层中是如何发挥作用的！

#### 第 2 层 (输出层)  
![[c2_w2_multiclass_tf-1794089597816.webp]]
这些图中的圆点，是经过第一层变换映射后的训练样本。你可以这样来理解：==第一层其实是为第二层创造出了一组全新的特征，供第二层做进一步评估。== 这些==图表的坐标轴正是上一层的两个输出 $a^{[1]}_0$ 和 $a^{[1]}_1$。==正如我们在上面所分析的那样，类别 0 和 1 (蓝色和绿色) 的 $a^{[1]}_0 = 0$，而类别 0 和 2 (蓝色和橙色) 的 $a^{[1]}_1 = 0$。  
同样地，背景颜色的深浅代表了输出数值的高低。  
单元 0 在 (0,0) 附近产生最大值，而类别 0 (蓝色) 恰好就被映射到了这个位置。  
单元 1 在左上角区域产生最高值，从而精准选中了类别 1 (绿色) 。  
单元 2 则瞄准了类别 2 (橙色) 所在的右下角区域。  
单元 3 在右上角区域产生最高值，从而选中了我们的最后一个类别 (紫色) 。  

还有一个从图像上不太容易直接看出来的关键细节：==各个单元之间的输出数值其实是经过全局协调的。==对于某个特定类别的样本点来说，==负责该类别的单元仅仅在自己的图像上取得最大值是不够的，它的输出值还必须同时高于所有其他单元在该点上的输出值。==
    求是上图坐标轴同一个区域 , 只有四张图里的一张 让这点达到最大值
这一协调工作正是通过包含在==损失函数 (`SparseCategoricalCrossEntropy`) 内部的隐式 Softmax 函数来完成的。==与其他各自独立计算的激活函数不同，Softmax 能够横跨所有输出单元进行全局协同。

==即使你完全不清楚每个单元在底层究竟做了什么，也照样能够成功地应用神经网络。==但希望这个例子能为你带来一些直观的启发，让你看清神经网络“引擎盖下”的工作奥秘。

## 恭喜！
你已经学会了如何构建并运行一个用于多分类任务的神经网络。

