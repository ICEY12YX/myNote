___
# 探索性数据分析 - 数据可视化与汇总统计量 (Summary Statistics)



欢迎来到探索性数据分析 (EDA) 系列的第三个实验笔记本 (notebook)。本篇是上周网约车分析实验的延续。

在本次实验中，你将使用芝加哥市 2022 年的网约车出行数据，该数据集的原始数据源自 芝加哥开放数据门户。上周你已经初步接触了该数据集，本周将继续基于你在上一实验中清理并抽样缩减后的版本开展深入分析。

### 学习目标：



在本实验中，你将在实际数据分析场景中运用课程中学习到的核心概念：

* 概率 (Probability)


* 描述性统计量 (均值、中位数、标准差和四分位数)


* 箱线图 (Box plots)


* 联合分布 (Joint distribution)


* 边缘分布 (Marginal distribution)


* 相关性 (Correlation)



# 1. 导入 Python 库



同往常一样，第一步是导入本次分析所需的核心库。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 用于展示交互式地图的库
import folium 
from folium.plugins import FastMarkerCluster

```

# 2. 加载数据集



接下来加载数据集。这是你在上一篇实验中预处理完成的精简版数据集。

```python
# 读取数据集
# 注意 parse_dates 参数，它会自动将指定列解析为日期时间格式。
df = pd.read_csv("data/rideshare_2022_cleaned.csv", parse_dates=['trip_start_timestamp', 'date'])

# 展示数据集的前几行
df.head()

```

# 3. 探究汇总统计量



要深入透视数据，掌握每一列中数值的分布特征至关重要。在之前的实验中，你已经绘制过单列特征的直方图 (histograms) 来观察其分布规律。现在，让我们采用更系统的方法。我们首先关注数值型特征：Pandas 提供了一个非常实用的内置函数 `.describe()`，它可以直接返回包含各数值列汇总统计量的新数据帧 (DataFrame)。运行下方代码单元，计算并呈现当前数据集的统计摘要。输出的数据帧中将包含各列的样本计数、均值 (mean) 、标准差 (standard deviation) 、最小值、最大值以及 25%、50% (中位数 (median) ) 和 75% 的四分位数 (quartiles)。此时你应该已经对这些统计概念十分熟悉，如果需要复习，建议回顾第 2 周第 1 课的教学视频。

```python
# 计算并展示汇总统计量
df.describe()

```

在上方生成的统计摘要中蕴含着非常丰富的业务与统计洞察。仔细观察表格内容有助于深化对数据的理解，以下几个问题可以引导你进行深入思考：

1. 观察每一列的极值：各列的最小值和最大值是多少？两者跨度有多大？例如：最短与最长行程距离分别是多少，极差是多少？


2. 观察各列的均值：例如平均行程距离是多少？均值是更靠近最小值还是更靠近最大值？


3. 观察各列的标准差：例如行程距离的波动程度与离散度如何？


4. 对比四分位数与均值：均值位于中位数的上方还是下方？该如何合理解释这种差异？ (结合上一篇实验中看到的直方图形状，思考是否存在长尾或厚尾现象)。



请注意，上述前三个思考题均以 `trip_miles` (行程英里数) 为例，你完全可以将同样的问题迁移到其他任意字段上，例如：平均小费、中位数小费、最高与最低小费分别是多少？

## 3.1 使用箱线图可视化汇总统计量



理解数据的绝佳手段是进行数据可视化 (visualization) ！箱线图 (boxplot) 是直观展现汇总统计量的标准利器。幸运的是，`Pandas` 已经原生集成了该功能，直接调用 `DataFrame.boxplot()` 即可轻松绘制。请记住，箱线图直观整合了所有四分位数、极值范围以及潜在离群点的信息，其经典形态如下图所示：

与其他绘图操作类似，如果不显式指定绘制哪一列，Pandas 会尝试将所有数值列绘制在同一张图上。由于各列的数值量级与物理意义差异巨大，逐列单独绘制能够更清晰地捕获图表传递的信息。

```python
# 指定需要绘制的列。如果你想探究其他变量，可替换为其他列名
column_to_plot = 'fare'

# 展示箱线图
plt.figure()
df.boxplot(column_to_plot)

```

图中的蓝色矩形框代表四分位距 (IQR, Interquartile Range) ，框内的水平蓝线分别对应第一四分位数 (Q1) 、中位数 (Q2) 和第三四分位数 (Q3)。这些数值与此前 `describe()` 输出中的 25%、50% 和 75% 完全吻合，你可以比对验证。箱体上下两侧延伸出的水平黑线分别代表 +/- 1.5 倍 IQR 的范围边界，这是统计学中识别异常值 (outliers) 的经典默认阈值。落在这些截断线之外的离散数据点均被标记为异常点。

在 `fare` (车费) 这一变量中，你可以看到上方密集分布着大量异常值点，这是为什么呢？不妨回顾一下该变量的分布形态：

```python
df.hist(column_to_plot, density=True)

```

显而易见，该变量的概率密度高度集中在较小的金额区间内，并在右侧延伸出极其显著的长尾 (long tail) ，而这些落在长尾深处的超高金额正是箱线图中被判定出的离群点。

现在你可以尝试更改 `column_to_plot` 变量，绘制其他特征的箱线图。它们的异常点是否同样只出现在单侧？你能推测出背后的原因吗？

## 3.2 可视化不同星期维度的分布差异



如果你希望按不同子集 (例如按星期几) 拆分数据并分别为每个子集绘制箱线图，只需使用 `by` 参数即可轻松搞定。将该参数指定为你想要分组的列名即可。假设我们希望按星期几来分析 `tip` (小费) 变量。从直觉上看，这是根据 `weekday` 的不同取值划分样本类别，并在各类中分别探索数据特征。从概率论的角度来看，你正在探究该变量的条件分布 (conditional distributions) ——即在已知今天是周一的条件下的小费分布、已知是周二条件下的小费分布，依此类推。

```python
df.boxplot(column='tip', by='weekday')

# 限制 y 轴展示区间。注释掉此行可查看完整纵坐标跨度
plt.ylim(-2, 52)

```

细心观察会发现，生成的这组箱线图有些反常。例如，周日 (Sunday) 的小费数据到底发生了什么？那里似乎根本没有画出箱体。再仔细看，其他星期的箱体内部似乎也找不到代表中位数的横线。为什么会出现这种现象？思考一下上一篇实验中小费数据的直方图特征能为你提供线索。该分布是否存在某些极端特性？如果记忆模糊，可以新建一个代码单元重新绘制直方图确认。

另一个更直接的线索藏在统计摘要中。运行下方代码单元，再次调用描述性统计，但这次仅针对按 `weekday` 分组后的 `tip` 列进行计算：

```python
# 按星期分组，选取小费列并计算描述性统计量
df.groupby('weekday')['tip'].describe()

```

可以看到数据帧中充斥着大量的 0。这是因为在周日，有超过 75% 的乘客完全没有支付小费！这就导致前三个四分位数 (极小值、25% 分位数、50% 中位数以及 75% 分位数) 的数值全部为 0。由于箱体的高度正是 25% 与 75% 分位数之间的间距 (IQR) ，两端均为 0 时，箱体的高度自然坍缩为 0，因而在图上完全看不到矩形框。

而在其他工作日，不给小费的人数占比略低于 75%，因此第三四分位数 (75% 列) 出现了非零数值。反映在图表上就是矩形框重新显现。然而中间的横线依然不可见，原因在于中位数 (50% 分位点) 依然是 0，与第一四分位数 (25% 分位点) 发生完全重合，两条线被压在了底部的 0 刻度线上。

这引出了一个极其耐人寻味的统计结论：在周日，所有给出的小费在箱线图规则下全部被判定为异常值 (outliers)。

现在，让我们剔除不给小费的样本，仅对真实给出小费的乘客群体重复上述分析。这将帮助我们更纯粹地看清实际打赏金额的分布规律。

```python
df_tippers = df[df['tip'] > 0]
df_tippers.boxplot(column='tip', by='weekday')

```

该图清晰揭示了实际发生小费行为时的金额分布，但它忽略了一个关键的背景信息：究竟有多少人会选择给小费？你在上一篇实验中已经计算过这一比例。

设想你是一名全职司机，是否会因为某些星期几的小费更高而选择在特定日子多出车？从按星期分组的结果来看，各天的差异似乎并不足以驱动这一决策。那么不妨将视角转向一天中的不同小时时段：乘客在特定时间段内给小费的概率是否会明显提升？让我们运行下方代码一探究竟。为此，我们需要从 `trip_start_timestamp` 字段中提取出一天中的具体小时数，并将其作为新列添加到数据帧中。

```python
# 向数据帧添加小时数列
df["hour"] = df["trip_start_timestamp"].apply(lambda x: x.hour)
# 仅筛选给小费的样本
df_tippers = df[df['tip'] > 0]

# 绘制各小时下的小费箱线图
plt.figure()
df_tippers.boxplot(column='tip', by='hour')
plt.ylim(-2, 52)

# 计算各小时给小费乘客的比例 (百分比)
percentage_of_tippers_hourly = df_tippers.groupby(["hour"])["tip"].count() / df.groupby(["hour"])["tip"].count() * 100

# 绘制小费支付率趋势折线图
plt.figure()
percentage_of_tippers_hourly.plot(marker="o", title="Percentage of Tippers")

```

数据呈现出清晨时段的小费金额略有偏高的趋势。但为什么清晨的小费会更丰厚？是因为人们在清晨具有更强的同理心吗？在草率得出这一主观结论之前，让我们先探讨是否存在其他更符合客观逻辑的潜在解释。

观察行程的实际距离是否能带来补充信息？接下来绘制行程距离在一整天中的动态分布箱线图：

```python
df.boxplot(column='trip_miles', by='hour')
plt.ylim(-10, 210)

```

正如你所看到的，清晨时段的行程距离明显更长！这个发现非常关键：这是否正是平均小费更高的根本原因？运行下一个单元格，绘制 `trip_miles` 与 `tip` 的散点图 (scatter plot)。请牢记此时我们仍然只关注给小费的样本群体 (纳入零小费样本只会在 tip=0 的横轴上堆积海量噪点)。

```python
df_tippers.plot(kind='scatter', x='trip_miles', y='tip', marker=".")

```

你能从这张图总结出两变量之间的相关性 (correlation) 特征吗？请记住，相关系数本质上是将协方差 (covariance) 进行尺度归一化后的度量，其数值被规范在 -1 到 1 之间：


散点图直观显示：随着路程拉长，乘客往往支付更高的小费，这意味着二者存在正相关 (positive correlation) 趋势。然而这种关联并非严苛的线性锁定，因此相关性为正但不会紧贴 1。这一相关性完全符合常理，因为长途出行的基础车费更高，随之按比例给出的打赏自然更高。

利用 Pandas 可以精确量化两个连续变量之间的相关系数，让我们在下方单元格中执行计算：

```python
(df_tippers.tip).corr(df_tippers.trip_miles)

```

计算得出的相关系数为 0.637。回顾统计定义：0 代表不存在线性相关，而 1 代表斜率为正的完美线性关系。在这里，计算结果落在二者之间且略高于 0.5，属于中等程度的正相关 (moderate correlation)。

现在你可以尝试将散点图中的变量进行替换以探索其他特征。例如，绘制 `tip` 与 `fare` (车费) 的散点图。它与 `tip` 对比 `trip_miles` 的表现相比有何不同？尝试计算它们的相关系数并展开对比。

# 4. 探索出行起始地理坐标



另一个极具商业价值的问题是：行程通常都在哪些区域开始？数据集中包含了“Pickup Centroid Latitude” (上车点质心纬度) 、“Pickup Centroid Longitude” (上车点质心经度) 以及对应的下车点经纬度字段，分别指明了行程起止的地理坐标。

运行下方代码，绘制上车起点的空间地理分布：

```python
# 筛选待绘制的经纬度数据
latitude = df.dropna()["pickup_centroid_latitude"].to_numpy()
longitude = df.dropna()["pickup_centroid_longitude"].to_numpy()
# 绘制二维直方图 (2D histogram)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
hist = ax.hist2d(longitude, latitude, bins=50, density=True)
ax.set_aspect(1.3, "box")
fig.colorbar(hist[3])
ax.set_xlabel("Longitude (degrees)")
ax.set_ylabel("Latitude (degrees)")

```

你刚刚绘制的图表展示的正是联合分布 (joint distribution) ！联合分布描述的是单一变量 (或事件) 同时跨越多个不同维度 (在本例中为经度与纬度) 的概率分布状态。

审视图表中的密度特征：在画布的中右侧聚集了海量的出行起点，这很可能对应着芝加哥市中心繁华商业区。但在左上角区域同样存在着一个显著的密集聚集点，该点远离主要城区集群。那个位置究竟坐落着什么？

为了解开这个谜题，我们可以借助交互式地图进行精确空间定位。运行下方代码，将这些坐标点投射到真实的芝加哥电子地图上。由于在地图上渲染海量点位非常消耗计算资源，代码仅抽取了部分样本进行渲染，以确保浏览器可以流畅交互。请在地图中核验点位最密集的区域，并重点查看左上角的高密度孤立点到底对应何处。

注意：渲染交互式地图需要消耗较多系统资源，偶发加载失败。若短时间内未能正常渲染，可重新运行该代码单元。

```python
# 定义用于渲染交互式地图的函数
def interactive_map(df, n_samples=4000):
    
    points = df[["pickup_centroid_longitude", "pickup_centroid_latitude"]].dropna()[0:n_samples]
    
    latitude = points.iloc[0]["pickup_centroid_latitude"]
    longitude = points.iloc[0]["pickup_centroid_longitude"]
    
    map3 = folium.Map(location=[latitude, longitude], zoom_start=9)

    marker_cluster = FastMarkerCluster([]).add_to(map3)
    
    for index, row in points.iterrows():
        latitude = row["pickup_centroid_latitude"]
        longitude = row["pickup_centroid_longitude"]
        folium.Marker((latitude, longitude), icon=folium.Icon(color="green")).add_to(marker_cluster)

    return map3

# 执行地图渲染
# 若地图未显示，请尝试重新运行该单元格；若仍无效，可重启内核 (Kernel) 后重新运行 notebook
interactive_map(df)

```

细致查验地图后，你很可能已经发现：左上角的集中出行点正是芝加哥奥黑尔国际机场 (Chicago O'Hare International Airport)！运行下方代码，按经纬度边界精准锁定来自机场区域的订单并展开深入剖析：

```python
# 筛选所有从机场出发的网约车订单
airport_rides = df[
    (df["pickup_centroid_longitude"] < -87.9) &
    (df["pickup_centroid_latitude"] > 41.97) &
    (df["pickup_centroid_latitude"] < 41.99)
]

airport_df_tippers = airport_rides[airport_rides['tip'] > 0]

# 绘制机场订单各时段小费箱线图
plt.figure()
airport_df_tippers.boxplot(column='tip', by='hour')

# 计算机场订单各时段给小费乘客的比例 (百分比)
airport_tippers_hourly = airport_df_tippers.groupby(["hour"])["tip"].count() / airport_rides.groupby(["hour"])["tip"].count() * 100

# 绘制机场订单小费支付率趋势折线图
plt.figure()
airport_tippers_hourly.plot(marker="o", title="Percentage of Tippers on Rides From the Airport")

```

数据表明，从机场出发的订单中，乘客支付小费的比例明显大幅高于全城平均水平。对于网约车司机而言，这无疑是一个极具吸引力的优质接单集聚地！

**祝贺你顺利完成本次实验！** 在本次学习中，你亲手实操了本课程涵盖的多项核心统计与分析概念：概率、均值/中位数/标准差/四分位数等描述性统计量；使用箱线图与二维直方图刻画单变量及联合分布，并探索了条件与边缘分布的性质。在此基础上，你还充分锤炼了 Pandas 数据处理与图表可视化技能。如果你喜欢这种动手实战，敬请期待下周更精彩的分析实验！


