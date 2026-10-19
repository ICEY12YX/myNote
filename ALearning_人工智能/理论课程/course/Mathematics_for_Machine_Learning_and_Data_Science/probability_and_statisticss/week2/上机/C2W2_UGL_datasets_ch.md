___
# 不计分实验 - 汇总统计量 (Summary Statistics)

在本实验笔记本 (notebook) 中，你将探索两个截然不同的数据集 (datasets)。你很快会发现，单纯依赖均值 (mean) 、方差 (variance)  (或标准差 (standard deviation) ) 以及相关性 (correlation) 等核心统计指标，往往无法真正描绘出数据的全貌。因此，在分析数据时，强烈建议结合数据可视化 (visualization) 技术或其他统计度量方法，从而对数据建立更深刻的理解。

___

___

你将使用两个极具代表性的经典数据集：安斯库姆四重奏 (Anscombe's quartet) 与数据恐龙十二图 (Datasaurus Dozen) 数据集。
这两个数据集均为人工构建，专门用来直观展示某些统计指标在==捕捉关键数据特征时的局限性。==更确切地说，它们有力地揭示了盲目迷信单一指标可能会带来怎样的误导。如果你感兴趣，可以查阅 Wikipedia 对应词条以及 Autodesk Research 发布的专题文章，深入了解有关安斯库姆四重奏与数据恐龙十二图的更多背景知识。

```python
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import utils
%matplotlib widget

```

# 1.1 第一个数据集 - 安斯库姆四重奏 (Anscombe's quartet)

第一个数据集最初由统计学家弗朗西斯·安斯库姆 (Francis Anscombe) 设计，旨在强调数据分析中==绘制图表的重要性==，并展示==异常值 (outliers) 及其他高影响力观测点==对统计特性的显著==影响==。 (来自 Wikipedia) 

要加载以 `.csv` 格式存储的数据集，可以使用 pandas 提供的 read_csv 函数。该函数可以直接将数据读取为数据帧 (DataFrame) 。如需了解该函数的更多详细用法，可在代码编辑器中输入 `help(pd.read_csv)` 进行查看。

```python
# 此行代码用于读取与当前 notebook 位于同一目录下的数据集 'df_anscombe.csv'
df_anscombe = pd.read_csv('df_anscombe.csv')

```

调用 `df_anscombe.head()` 将展示数据集的前 5 行，方便你初步浏览数据结构。

```python
df_anscombe.head()

```
![[C2W2_UGL_datasets_ch-1792436215950.webp|173]]

```python
# 统计该数据集中包含的不同数据组数量
df_anscombe.group.nunique()
#df_anscombe.group  访问 DataFrame 中名为 `group` 的列（返回一个 Series）
#.nunique() 统计该列中不重复（唯一）值的个数

# 返回4
```

该数据集由 4 个数据组构成，每组均包含两个分量——`x` 与 `y`。在分析这些数据时，可以分别计算各组的均值与方差，以及组内 x 与 y 之间的相关性。
    数据组理解为实验 就是4个不同实验 得到4组数据
Pandas 内置了 `DataFrame.describe` 函数，能够便捷地呈现各变量的常用统计量。若想==按照 group 列对数据进行拆分==，可以使用💛 `DataFrame.groupby` 函数。

下一段代码首先基于 group 列对 `DataFrame` 进行分组，随后调用💛 describe 函数计算每组中各个变量的常用统计指标。

```python
df_anscombe.groupby('group').describe()

```
![[C2W2_UGL_datasets_ch-1792436485160.webp]]

从输出结果可以看出，各组数据在统计指标上极为相似——无论 `x` 还是 `y`，各组计算出的均值与标准差==几乎完全一致==。

此外，你还可以进一步分析各组内 `x` 与 `y` 之间的相关性。

获取各组的相关系数矩阵 (correlation matrix) 的方法与此前类似：先使用 `DataFrame.groupby` 依据 `group` 列完成分组，再调用💛 `.corr` 函数计算相关系数。

```python
df_anscombe.groupby('group').corr()

```
![[C2W2_UGL_datasets_ch-1792436568622.webp|258]]
解释
    ![[C2W2_UGL_datasets_ch-1792436660294.webp]]
    主对角线 上的0.816 是x,y的相关系数 #correlation_coefficient 
    x和y的 y和x的相关系数 没有区别 所以是一样的

继续解释
    ![[C2W2_UGL_datasets_ch-1792436981647.webp]]

正如你所看到的，各组中 `x` 与 `y` 的相关性在保留三位小数时==完全相同==。不仅如此，极高的相关系数值还表明，各组中的 `x` 与 `y` 之间均呈现出==强烈的线性相关性==。

尽管这四个组在各项统计数值上表现得如出一辙，但我们==仍然必须将数据可视化，才能真正看清它们潜在的真实差异。==

```python
utils.plot_anscombes_quartet()

```
![[C2W2_UGL_datasets_ch-1792436853944.webp]]
这差距不是一点半点

一旦将数据绘制成图，这四个组的形态差异立刻显露无遗：

1. 第一组数据在 `x` 与 `y` 之间展现出清晰纯粹的**线性关系 (linear relationship)** 。

2. 第二组数据呈现出鲜明的**非线性 (non-linear)** 趋势，这说明通用的==皮尔逊相关系数 (Pearson correlation) 并不适合用来刻画这种数据分布==。

3. 第三组数据本应是完美的线性分布，但==受到单个异常值 (outlier) 的显著干扰==。

4. 第四组数据表明，在 `x` 保持恒定的情况下 `y` 可以取到多个不同数值，意味着==两者之间不存在明确的关联==；同时该组中也存在==一个极端的异常值==。

这四个数据组生动地证明：==**仅凭汇总统计量远远不足以全面透视数据。**==通过**数据可视化**、辨析潜在的异常值并挖掘更深层的复杂关联，对于真正洞察数据背后的底层规律至关重要。

# 2 - 第二个数据集 - 数据恐龙十二图 (Datasaurus Dozen)

安斯库姆四重奏的诞生启发了后来的学者与研究者，他们尝试生成更多==点集形态各异、却共享完全相同汇总统计量的数据集==。由 Autodesk 设计的“数据恐龙十二图” (Datasaurus Dozen) 便是其中的经典之作。

这一次，我们将尝试不同的探索流程。不再先看统计量再画图，而是直接从这十二组数据中挑选两组进行对比，并亲自计算它们的统计指标。

```python
df_datasaurus = pd.read_csv("datasaurus.csv")

```

接下来的代码单元将启动一个交互式小部件，方便你自由探索该数据集中包含的不同组别。

```python
utils.display_widget()

```
![[C2W2_UGL_datasets_ch-1792437179185.webp|538]]
![[C2W2_UGL_datasets_ch-1792437275016.webp|578]]
![[C2W2_UGL_datasets_ch-1792437297949.webp|565]]
妈的 你不管上面选什么数据集 statistics算出来都是差不多的


正如你所见，不同形态、不同分布的数据集完全可能产生一模一样的统计指标。因此在==**数据分析工作中必须时刻警惕：统计摘要绝不能代替对数据全貌的真实洞察。**==

祝贺你！你已经顺利完成了本次实验，并深刻领悟到了汇总统计量在描述数据完整特征时的局限性。请牢记，在未来的分析中，唯有结合可视化与更深入的细致分析，才能真正读懂数据背后的规律。