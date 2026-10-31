# 非计分实验：过拟合 (Overfitting) 

&lt;img align=&quot;left&quot; src=&quot;./images/C1_W3_Overfitting_a.png&quot;     style=&quot; width:250px; padding: 10px; &quot; &gt;
&lt;img align=&quot;left&quot; src=&quot;./images/C1_W3_Overfitting_b.png&quot;     style=&quot; width:250px; padding: 10px; &quot; &gt;
&lt;img align=&quot;left&quot; src=&quot;./images/C1_W3_Overfitting_c.png&quot;     style=&quot; width:250px; padding: 10px; &quot; &gt;

## 实验目标
在本实验中，你将探索：
- 容易发生过拟合的具体场景
- 应对过拟合的一些常用解决方案


```python
%matplotlib widget
import matplotlib.pyplot as plt
from ipywidgets import Output
from plt_overfit import overfit_example, output
plt.style.use('./deeplearning.mplstyle')
```

# 过拟合
本周的课程讲座介绍了容易出现过拟合现象的几种典型情况。运行下方的代码单元格将生成一幅交互式图表，供你深入探索过拟合问题。代码单元格下方附有进一步的操作说明。


```python
plt.close("all")
display(output)
ofit = overfit_example(False)
```

在上方的交互式图表中，你可以执行以下操作：
- 在回归 (Regression) 与分类 (Categorization) 示例之间自由切换
- 添加新的数据点
- 选择模型的阶数 (degree) 
- 将模型拟合 (fit) 到数据上  

建议你尝试以下几项实验操作：
- 将阶数设置为 degree = 1 来拟合数据；观察什么是“欠拟合 (underfitting) ”。
- 将阶数设置为 degree = 6 来拟合数据；观察什么是“过拟合”。
- 调整模型阶数，寻找能够实现“最佳拟合 (best fit) ”的阶数
- 添加新数据：
    - 添加极端的样本点会加剧过拟合现象 (假设这些点属于异常值 (outliers) ) 。
    - 添加符合整体趋势的常规样本点则有助于缓解过拟合
- 在 `Regression` (回归) 和 `Categorical` (分类) 模式之间切换，分别尝试这两种示例。

如需重置图表，请重新运行该代码单元格。点击鼠标时请放慢速度，留出足够的时间让图表完成刷新后再进行下一次点击。

关于底层实现的说明：
- 图中的“ideal” (理想) 曲线代表用于生成该数据集的真实底层模型，我们在其基础上叠加了随机噪声 (noise) 从而构建出当前的数据集
- 为了提升交互响应速度，“fit”操作并未采用纯粹的梯度下降 (gradient descent) 算法。这类优化求解方法非常适用于规模较小的数据集。

## 恭喜！
你已经对过拟合的成因及其解决方案建立起了直观的认识。在下一个实验中，你将深入探索一种最常用的过拟合解决方案——正则化 (Regularization) 。


```python

```
