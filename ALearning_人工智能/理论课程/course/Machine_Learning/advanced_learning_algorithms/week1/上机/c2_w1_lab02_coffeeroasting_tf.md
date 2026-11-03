# 选修实验 - 简单神经网络 (Simple Neural Network)
在本次实验中，我们将使用 Tensorflow 搭建一个小型的神经网络。
   &lt;center&gt; &lt;img  src=&quot;./images/C2_W1_CoffeeRoasting.png&quot; width=&quot;400&quot; /&gt;   &lt;center/&gt;



```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from lab_utils_common import dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

```

## 数据集 (DataSet)


```python
X,Y = load_coffee_data();
print(X.shape, Y.shape)
```

下面让我们把咖啡烘焙的数据绘制出来。这里的两个特征 (Features) 分别是摄氏温度 (Temperature) 和以分钟为单位的持续时间 (Duration) 。家庭咖啡烘焙指南建议，持续时间最好控制在 12 到 15 分钟之间，而温度则应当保持在 175 到 260 摄氏度之间。当然，随着温度升高，烘焙的持续时间就应该相应缩短。


```python
plt_roast(X,Y)
```

### 数据归一化 (Normalize Data)
如果先对数据进行归一化 (Normalization) 处理，那么将数据拟合到权重 (Weights) 的过程 (也就是下周课程将讲到的反向传播 (Back-propagation) ) 就会进行得更快。这与你在第 1 门课程中使用的流程完全一致，即将数据中的各个特征分别归一化到相似的数值范围内。
下面的流程使用了 Keras 的归一化层 (Normalization Layer) 。它包含以下几个步骤：
- 创建一个“归一化层”。请注意，在这里的应用方式中，它并不是你模型内部的一层。
- 对数据进行“适配 (adapt) ”。这一步会学习数据集的均值 (Mean) 和方差 (Variance) ，并将这些数值保存在内部。
- 对数据进行归一化。  
将来任何要使用该训练模型的新数据，都必须先进行同样的归一化处理，这一点非常重要。


```python
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")
```

接下来，我们通过平铺 / 复制数据来扩大训练集 (Training Set) 的规模，从而减少训练所需的轮数 (Epochs) 。


```python
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1))   
print(Xt.shape, Yt.shape)   
```

## Tensorflow 模型

### 模型
   &lt;center&gt; &lt;img  src=&quot;./images/C2_W1_RoastingNetwork.PNG&quot; width=&quot;200&quot; /&gt;   &lt;center/&gt;  
让我们来搭建课堂上介绍过的“咖啡烘焙网络”。如下所示，它包含两个带有 Sigmoid 激活函数 (Activation Function) 的层：


```python
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)
```

>**注 1：** `tf.keras.Input(shape=(2,)),` 指定了期望的输入形状。这使得 Tensorflow 能够在此时就确定权重和偏置 (Bias) 参数 (Parameters) 的维度大小。在探索 Tensorflow 模型时，这非常有用。而在实际应用中，这一行语句可以省略，Tensorflow 会在执行 `model.fit` 语句并传入输入数据时自动确定网络参数的大小。  
>**注 2：** 在最后一层中直接包含 Sigmoid 激活函数并不是最佳工程实践。更好的做法是将其合并到损失函数 (Loss Function) 中进行计算，这样可以提高数值稳定性 (Numerical Stability) 。我们会在后续的实验中详细介绍这一点。

`model.summary()` 提供了对该网络结构的简要说明：


```python
model.summary()
```

摘要中显示的参数数量对应于权重和偏置数组中的元素总数，如下所示。


```python
L1_num_params = 2 * 3 + 3   # W1 parameters  + b1 parameters
L2_num_params = 3 * 1 + 1   # W2 parameters  + b2 parameters
print("L1 params = ", L1_num_params, ", L2 params = ", L2_num_params  )
```

让我们来检查一下 Tensorflow 已经实例化的权重和偏置。权重 $W$ 的维度大小应当是 (输入中的特征数量, 该层中的单元数量) ，而偏置 $b$ 的大小则应当与该层中的单元 (Units) 数量相匹配：
- 在包含 3 个单元的第一层中，我们期望 W 的大小为 (2,3) ，且 $b$ 应该包含 3 个元素。
- 在包含 1 个单元的第二层中，我们期望 W 的大小为 (3,1) ，且 $b$ 应该包含 1 个元素。


```python
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print(f"W1{W1.shape}:\n", W1, f"\nb1{b1.shape}:", b1)
print(f"W2{W2.shape}:\n", W2, f"\nb2{b2.shape}:", b2)
```

下面这两条语句将在第 2 周进行详细讲解。目前你只需要了解：
- `model.compile` 语句用于定义损失函数，并指定编译时的优化器 (Optimizer) 。
- `model.fit` 语句用于运行梯度下降 (Gradient Descent) ，使权重拟合到数据上。


```python
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)

model.fit(
    Xt,Yt,            
    epochs=10,
)
```

#### 训练轮数 (Epochs) 与批次 (Batches)
在上面的 `fit` 语句中，`epochs` 的数量被设置为 10。这意味着整个数据集在训练过程中将被完整地使用 10 次。在训练期间，你会看到描述训练进度的输出信息，形式如下：
```
Epoch 1/10
6250/6250 [==============================] - 6s 910us/step - loss: 0.1782
```
第一行 `Epoch 1/10` 描述了模型当前正在运行第几轮训练。为了提高计算效率，训练数据集会被拆分成一个个“批次 (Batches) ”。在 Tensorflow 中，一个批次的默认大小是 32。我们扩展后的数据集中共有 200000 个样本，也就是大约 6250 个批次。第二行中的符号 `6250/6250 [====` 描述的就是当前已经执行到了哪一个批次。

#### 更新后的权重
完成拟合后，模型的权重已经得到了更新：


```python
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)
```

你可以看到，这些数值与调用 `model.fit()` 之前打印出来的数值已经完全不同了。借助这些更新后的参数，模型现在应该能够分辨出什么样的咖啡烘焙是好是坏。

为了方便接下来的讨论，我们不直接使用你刚才训练出来的权重，而是先加载一组我们从之前的训练运行中保存下来的权重。这样做是为了保证本实验笔记本在 Tensorflow 未来的版本更新中依然保持稳定。不同的训练运行可能会产生略微不同的结果，而接下来的讨论内容是基于下面即将加载的这组特定权重展开的。

等你稍后完成了本次实验，可以随时把下面这个单元格的代码注释掉并重新运行整个笔记本，看看结果是否有什么差异。如果你在上面的训练结束后得到了很低的损失值 (例如 0.002) ，那么你大概率会得到完全相同的结果。


```python
# After finishing the lab later, you can re-run all 
# cells except this one to see if your trained model
# gets the same results.

# Set weights from a previous run. 
W1 = np.array([
    [-8.94,  0.29, 12.89],
    [-0.17, -7.34, 10.79]] )
b1 = np.array([-9.87, -9.28,  1.01])
W2 = np.array([
    [-31.38],
    [-27.86],
    [-32.79]])
b2 = np.array([15.54])

# Replace the weights from your trained model with
# the values above.
model.get_layer("layer1").set_weights([W1,b1])
model.get_layer("layer2").set_weights([W2,b2])
```


```python
# Check if the weights are successfully replaced
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
print("W1:\n", W1, "\nb1:", b1)
print("W2:\n", W2, "\nb2:", b2)
```

### 预测 (Predictions)
&lt;img align=&quot;left&quot; src=&quot;./images/C2_W1_RoastingDecision.PNG&quot;     style=&quot; width:380px; padding: 10px 20px; &quot; &gt;

一旦你拥有了一个训练好的模型，就可以用它来进行预测。回想一下，我们这个模型的输出是一个概率 (Probability) ——在这个例子里，就是咖啡烘焙品质良好的概率。为了做出最终的分类判定，我们必须将该概率与一个阈值 (Threshold) 进行比较。在这里，我们将使用 0.5 作为阈值。

让我们先从创建输入数据开始。模型期望接收一个或多个样本，其中每个样本对应矩阵 (Matrix) 中的一行。在这个例子中，我们有两个特征，因此矩阵的形状将是 (m,2) ，其中 m 是样本的数量。
请记住，由于我们已经对输入特征进行了归一化处理，因此我们也必须对测试数据 (Test Data) 进行同样的归一化。  
要进行预测，只需调用 `predict` 方法即可。


```python
X_test = np.array([
    [200,13.9],  # positive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)
```

为了将概率转化为最终的分类决策，我们需要应用一个阈值：


```python
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")
```

这一步也可以用更简洁的代码来实现：


```python
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")
```

## 层的功能解析 (Layer Functions)
让我们深入检查一下各个单元所代表的函数功能，看看它们在咖啡烘焙决策中分别扮演了什么角色。我们将针对输入 (持续时间, 温度) 的所有可能取值，绘制出每个节点 (Node) 的输出结果。每个单元都是一个逻辑函数 (Logistic Function) ，其输出范围在 0 到 1 之间。图中的阴影深浅就代表了输出值的大小。
> 注：在实验课中，我们通常从 0 开始编号，而在理论课讲义中则可能从 1 开始编号。


```python
plt_layer(X,Y.reshape(-1,),W1,b1,norm_l)
```

图中的阴影表明，第一层中的每个单元分别负责识别一个不同的“烘焙失败 (bad roast) ”区域：当温度过低时，单元 0 的输出值较大；当持续时间过短时，单元 1 的输出值较大；而当时间与温度的组合不合适时，单元 2 的输出值较大。值得注意的是，这个网络完全是通过梯度下降的过程自己学会这些函数的。而这些函数，与人类在面对同样问题时可能会采取的判断逻辑惊人地相似。

最后一层的函数图像要稍微难可视化一些。它的输入是第一层的输出。我们已经知道第一层使用的是 Sigmoid 函数，因此它们的输出范围都在 0 到 1 之间。我们可以创建一个三维 (3-D) 图像，计算出这三个输入在所有可能组合下的输出结果，如下所示。在上面的第一层图像中，高输出值对应的是“烘焙失败”的区域；而在下面的图像中，最大输出值出现在三个输入都比较小的区域，这恰好对应了“烘焙良好 (good roast) ”的区域。


```python
plt_output_unit(W2,b2)
```

最后这张图展示了整个网络协同工作的全貌。  
左图是最后一层输出的原始概率值，用蓝色阴影的深浅来表示，并叠加显示了由 X 和 O 代表的训练数据。  
右图则是经过决策阈值过滤后的网络输出结果，这里的 X 和 O 对应的是网络最终做出的分类判定。  
下面这段代码需要一点时间来运行。


```python
netf= lambda x : model.predict(norm_l(x))
plt_network(X,Y,netf)
```

## 恭喜！
你已经在 Tensorflow 中成功搭建了一个小型神经网络。
这个网络生动地展示了神经网络是如何通过将复杂决策拆解并分配给多个单元，从而协同处理复杂判定任务的。
