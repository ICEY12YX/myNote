___
- [[#选修实验 - ReLU 激活函数 (ReLU Activation)|选修实验 - ReLU 激活函数 (ReLU Activation)]]
	- [[#选修实验 - ReLU 激活函数 (ReLU Activation)#2 - ReLU 激活函数|2 - ReLU 激活函数]]
		- [[#2 - ReLU 激活函数#为什么需要非线性激活函数？|为什么需要非线性激活函数？]]
	- [[#选修实验 - ReLU 激活函数 (ReLU Activation)#恭喜！|恭喜！]]

___
# 选修实验 - ReLU 激活函数 (ReLU Activation)

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.activations import linear, relu, sigmoid
%matplotlib widget
from matplotlib.widgets import Slider
from lab_utils_common import dlc
from autils import plt_act_trio
from lab_utils_relu import *
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

```

## 2 - ReLU 激活函数

本周我们引入了一个新的激活函数：修正线性单元 (Rectified Linear Unit, ReLU) 。
![[c2_w2_relu-1794084422177.webp|300]]

```python
plt_act_trio()
```
![[c2_w2_relu-1794084440730.webp]]

![[c2_w2_relu-1794084401148.webp|465]]

右侧来自课堂讲义的例子展示了 ReLU 的一个应用场景。在这个例子中，推导出的“认知度 (awareness) ” 特征 (Feature) 并不是非黑即白的二值变量，而是具有连续的取值范围。==Sigmoid 函数最适合处理“开 / 关”或二分类 (Binary) 的情况==，而 ==ReLU 则能够提供连续的线性关系==。此外，它还拥有一个==输出恒为零的“关闭 (off) ”区间==。  
正是这个“关闭”特性，使得 ReLU 成为了一种**非线性激活函数 (Non-Linear Activation)** 。为什么我们需要这一特性呢？下面让我们一探究竟。

### 为什么需要非线性激活函数？  

![[c2_w2_relu-1794084514505.webp|332]]
左图所示的函数是由若干段线性部分拼接而成的 (即分段线性 (Piecewise Linear) 函数) 。==在线性区间内，斜率 (Slope) 保持恒定==；
而在转折点 (Transition Points) 处，斜率会发生突变。==在每一个转折点上，都会有一个新的线性函数加入进来==，当它与原有的函数叠加在一起时，就会产生新的斜率。这个新函数是在转折点处被添加进来的，但在到达该转折点之前，它对输出没有任何贡献。非线性激活函数的作用，恰恰就是在==到达转折点之前 (有时也包括在转折点之后) 将输入“屏蔽 / 关闭”掉。==下面的练习将为你提供一个更加直观具体的例子。

本练习将在一个回归 (Regression) 问题中使用下方的神经网络 (Neural Network) ，你需要用它来拟合一个分段线性的目标函数：
![[c2_w2_relu-1794086696939.webp]]
该网络在第一层 (Layer) 中包含 3 个单元 (Units) ，必须让每一个单元都发挥作用才能共同拼出目标函数。
其中，单元 0 已经预先设定并固定好了，专门用来映射第一段线段。你需要修改单元 1 和单元 2 中的权重 (Weights) 和偏置 (Biases) ，以此来拟合第 2 段和第 3 段线段。输出单元也是固定不变的，它只是简单地将第一层各个单元的输出相加。  

请使用下方的滑块来调整权重和偏置，使网络输出与目标曲线相匹配。
提示：请先从调整 `w1` 和 `b1` 开始，在拟合好第 2 段线段之前，保持 `w2` 和 `b2` 为零。直接点击滑块轨道比拖动滑块要更快一些。如果你在调整过程中遇到困难，也不用担心，下面的正文会对此进行更详细的解析。

```python
_ = plt_relu_ex()
```
![[c2_w2_relu-1794086872217.webp|655]]

 
本次练习的目标，是让你切身体会 ReLU 的非线性行为是如何提供关键的“开关控制”能力的——它能够==让特定的函数保持关闭状态，直到真正需要它们发挥作用的那一刻==。让我们来看看在刚才的例子中，这一机制具体是如何运作的。

![[c2_w2_relu-1794086916474.webp|551]]
右侧的图表展示了第一层中各个单元的输出图像。  
从最上方开始看，单元 0 负责拟合标有数字 1 的第一段线段。图中同时展示了线性函数 $z$ 以及经过 ReLU 之后的激活输出 $a$。你可以清楚地看到，==ReLU 在区间 [0,1] 之后将函数截断为零。==这一点至关重要，因为==它防止了单元 0 对后续的线段产生干扰。==

单元 1 负责第 2 段线段。在这里，ReLU 让该单元在 x 达到 1 之前一直保持“静默 (输出为零) ”。由于此时第一个单元已经不再产生贡献，因此单元 1 的斜率 $w^{[1]}_1$ 就直接等于目标线段在这一段(指左边的2线段)的斜率。同时，我们需要调整偏置，使z在 x 达到 1 之前一直保持为负数。(这样经过relu以后, 这段才会一直是0)  请留意，单元 1 的贡献会一直延续到第 3 段线段所在的区间。

单元 2 负责第 3 段线段。==ReLU 同样将该单元的输出一直置零，直到 x 达到对应的转折点位置。==此时，单元 2 的斜率 $w^{[1]}_2$ 必须设置得恰到好处，==使得单元 1 和单元 2 的斜率之和正好等于第 3 段线段所需的目标斜率。== 同样地，我们通过调整偏置，使输入在 x 达到 2 之前一直保持为负值。

正是 ReLU 激活函数的这种“关闭”或屏蔽特性，赋予了==模型将一段段简单的线性片段“缝合”在一起的能力==，从而能够==拟合出复杂的非线性函数。==


## 恭喜！
你现在已经更加熟悉 ReLU 激活函数，并深刻理解了其非线性行为的重要意义。

