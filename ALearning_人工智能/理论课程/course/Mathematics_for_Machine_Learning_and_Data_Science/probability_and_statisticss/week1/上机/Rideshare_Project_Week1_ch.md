___
# 探索性数据分析 (Exploratory Data Analysis) - 读懂你的数据集

欢迎来到探索性数据分析 (Exploratory Data Analysis, EDA) 系列的第二个交互式实验 (Notebook)！在这里，你将亲自动手，将课程学到的分析技能运用到解决贴近现实生活的真实数据问题中。本教程属于全套五期系列专题之一，各期内容分别对应课程不同周次的学习规划。这些 Notebook 避开了繁复抽象的纯数学推导，而是侧重于教你如何运用各类 Python 工具库，将统计学概念转化为实战代码。

在本次实验中，我们将探索 ==**2022 年芝加哥市网约车行程数据集**==，原始数据可在芝加哥数据门户网站的这里获取。

我们已经预先下载好了数据，并与本 Notebook 一并放置在工程目录下。如果你查阅上方提及的官方来源，会发现完整数据包含数以亿计的行程记录，体积达数十吉字节 (GB) ，显然超出了一般交互式运行环境的承载能力。因此，我们对数据集进行了前置清洗与**下采样 (Downsampling)** 处理：仅保留 2022 年度的记录，并将采样率缩减至原来的 1/100，确保数据能轻巧载入内存，带来流畅愉悦的实战体验。

本篇教程主要依赖 Pandas 库进行数据处理。如果对其基础操作尚不熟悉，可先回顾此前的 Pandas 基础入门教程。

___
- [[#学习目标 (Learning Objectives) ：|学习目标 (Learning Objectives) ：]]
___

### 学习目标 (Learning Objectives) ：

在本次实战中，你将在具体业务场景中应用以下课程统计学概念：

* 概率 (Probability)
* 条件概率 (Conditional probability)
* 概率分布 (Distributions)

# 1. 导入 Python 库 (Import the Python Libraries)

按照惯例，第一步是导入实战所需的工具库。`pandas` 用于数据的高效读取与清洗处理，`matplotlib` 则用于可视化图表的绘制。

```python
import pandas as pd
import matplotlib.pyplot as plt

```

# 2. 加载数据集 (Load the Dataset)

接下来开始加载数据。该数据已经过 1/100 倍率的下采样稀疏处理，以确保在当前环境中平稳运行。

```python
# 读取数据集，并将行程起止时间解析为日期时间格式
# 这个文件太大了不好git上传 所以保存到绝对路径这里
df = pd.read_csv(
    r"F:\University\LAB\data\rideshare_2022.csv",
    parse_dates=["Trip Start Timestamp", "Trip End Timestamp"],
)

# 查看数据集的前五行
df.head()

```
![[Rideshare_Project_Week1_ch-1792086741710.webp]]

解释
    ![[Rideshare_Project_Week1_ch-1792087335995.webp]]
    ![[Rideshare_Project_Week1_ch-1792087372427.webp]]
    ![[Rideshare_Project_Week1_ch-1792087393064.webp]]
    ![[Rideshare_Project_Week1_ch-1792087412550.webp]]
    ![[Rideshare_Project_Week1_ch-1792087420222.webp]]

# 3. 探索数据集 (Explore the Dataset)

在上面的单元格中，我们加载了数据并展示了前五行。仔细观察打印出的表格内容，该数据集包含以下字段维度：

* `Trip ID`：行程的全局唯一标识标识符。
* `Trip Start Timestamp`：行程开始时间 (按 15 分钟粒度四舍五入聚合)。
* `Trip End Timestamp`：行程结束时间 (按 15 分钟粒度四舍五入聚合)。
* `Trip Seconds`：行程耗时 (单位：秒)。
* `Trip Miles`：行程距离 (单位：英里)。


* `Pickup Census Tract`：出发地上车点所在的人口普查区 (Census Tract) ；芝加哥市以外的区域此项通常留空。
* `Dropoff Census Tract`：目的地下车点所在的人口普查区；芝加哥市以外的区域此项通常留空。

* `Pickup Community Area`：出发地上车点所属的社区规划区 (Community Area) ；芝加哥市以外的区域留空。
* `Dropoff Community Area`：目的地下车点所属的社区规划区；芝加哥市以外的区域留空。


* `Fare`：行程基础车费 (按 2.50 美元粒度四舍五入取整)。
* `Tip`：乘客支付的小费金额 (按 1.00 美元粒度四舍五入取整) ，线下现金小费不纳入统计。
* `Additional Charges`：行程附加费，包括税费、过桥通行费及平台附加费等。
* `Trip Total`：行程总费用；由上述多项费用汇总计算得出 (包含取整规则)。

* `Shared Trip Authorized`：乘客是否勾选同意拼车选项 (无论后续实际是否成功匹配拼车乘客)。
* `Trips Pooled`：若成功拼车，累计汇入的总拼车单数 (含本行程) ；从车辆首次搭载乘客至整车完全放空期间的所有乘客订单均计入此项，哪怕部分乘客在时间线上未曾同乘一车；构成该次总拼车行程的每个单程订单在数据集中均独立记录，且该列数值相同。

* `Pickup Centroid Latitude`：上车点普查区或社区区域中心的纬度坐标 (为保护隐私脱敏至中心点) ；芝加哥市以外区域通常留空。
* `Pickup Centroid Longitude`：上车点普查区或社区区域中心的经度坐标 (脱敏至中心点) ；芝加哥市以外区域通常留空。

* `Pickup Centroid Location`：上车点地理中心位置信息；芝加哥市以外区域通常留空。
* `Dropoff Centroid Latitude`：下车点普查区或社区区域中心的纬度坐标 (脱敏至中心点) ；芝加哥市以外区域通常留空。
* `Dropoff Centroid Longitude`：下车点普查区或社区区域中心的经度坐标 (脱敏至中心点) ；芝加哥市以外区域通常留空。
* `Dropoff Centroid Location`：下车点地理中心位置信息；芝加哥市以外区域通常留空。


运行下方单元格，打印列名、核验非空值数量并审查各列的数据类型。

```python
df.info()

```
![[Rideshare_Project_Week1_ch-1792086924929.webp|445]]

## 3.1 筛选核心目标列 (Select columns of interest)

至此，你已经对数据的整体轮廓有了基本了解。不妨思考下一步的数据策略：哪些字段值得深挖？是否存在缺失值比例过高而影响分析的特征？哪些字段与当前分析目标无关？

在探索性数据分析中，精准提取与业务强相关的核心列并剔除冗余字段是非常标准的操作。这样不仅能大幅简化数据结构，还可以削减内存占用，让数据处理变得更为轻快高效。

在下方代码中，我们==筛选==出本次实验重点关注的特征子集。仅保留这些目标列可以将数据体积压缩近一半。在处理大规模数据时，这种瘦身往往决定了数据集能否顺利加载至可用内存中。此外，代码还会==对列名进行标准化处理，去除所有空格==。

```python
columns_of_interest = [
    "Trip Start Timestamp",
    "Trip Seconds",
    "Trip Miles",
    "Fare",
    "Tip",
    "Additional Charges",
    "Trip Total",
    "Shared Trip Authorized",
    "Trips Pooled",
    "Pickup Centroid Latitude",
    "Pickup Centroid Longitude",
    "Dropoff Centroid Latitude",
    "Dropoff Centroid Longitude",
]

df = df[columns_of_interest]

# 批量重命名列名，剔除空格并转为小写下划线形式
df = df.rename(columns={i: "_".join(i.split(" ")).lower() for i in df.columns})

# 检查清洗精简后的数据集信息
df.info()

```
![[Rideshare_Project_Week1_ch-1792087159035.webp|398]]

批量重命名列名 解释
    [[intro-to-pandas-world-happiness_ch]]


# 4. 数据可视化 (Visualize the data)

要洞察数据规律，进行可视化探索往往是最直观有效的手段，它能帮助我们把握变量的数据分布 (Distribution)。我们可以先从“每日行程订单量”的走势入手。为此，最好新建一个专门存储日期的列。下方代码从 `trip_start_timestamp` 中提取出纯日期信息并生成新列：

```python
# 新建一个专门存储日期的列
df['date'] = pd.to_datetime(df['trip_start_timestamp'].dt.date)

df.head()

```
![[Rideshare_Project_Week1_ch-1792087578717.webp]]

解释
    ![[Rideshare_Project_Week1_ch-1792087501219.webp]]
    ![[Rideshare_Project_Week1_ch-1792087531507.webp]]
    ![[Rideshare_Project_Week1_ch-1792087552870.webp]]


```python
# 指定需要可视化的目标列
column_to_plot = 'date'

# 绘制目标列的直方图，开启密度归一化
df.hist(column_to_plot, density=True)

```
![[Rideshare_Project_Week1_ch-1792087804965.webp|423]]

解释
    ![[Rideshare_Project_Week1_ch-1792087725866.webp]]
    ![[Rideshare_Project_Week1_ch-1792087732262.webp]]
    就是 #probability_density_function #PDF 

上图绘制出了全年网约车订单在时间维度上的分布情况。注意代码中设置了参数 `density=True`，这样能对直方图进行==**缩放归一化**==，使其表现为类似于课程第 2 节中所学到的概率密度函数 (probability density function) ，即==所有柱形条的总面积等于 1==。
    就是变成 #PDF 图了

观察此图，这种分布呈现出什么特征？它是否与视频课程中介绍过的某种典型分布相契合？

尽管在年初阶段的频次略有偏低，但总体而言，全年的订单发生率非常接近**均匀分布 (Uniform Distribution)**。需要明确的是，这并非全市实际所有订单在绝对时间上的完整分布，而是==基于采样样本得出的经验估计值==；现实业务数据必然存在一定的随机波动。

现在，你可以将上方代码中的 `column_to_plot` 变量修改为其他列名，观察不同维度的分布形态。例如 `fare` (车费) 、`tip` (小费) 或 `trip_length` (行程长度) 都极具分析价值，它们能反映出司机的平均接单里程以及对应的报酬水平。

接下来，让我们聚焦于 `tip` (小费) 列：

```python
# 选定待绘制的目标列
column_to_plot = 'tip'

# 绘制小费分布直方图，设置分箱数量为 100
df.hist(column_to_plot, density=True, bins = 100);

```
解释
    ![[Rideshare_Project_Week1_ch-1792087963603.webp|363]]
![[Rideshare_Project_Week1_ch-1792088588063.webp|466]]

观察小费的分布形态，你会得出什么结论？这里的图像显得有些异样，对吧？是什么原因导致了如此奇特的数据形态？最左侧那一根极高的垂直柱条究竟代表了什么？

实际上，==现实中绝大多数乘客在网约车行程中是不支付小费的，因此在 tip = 0 处聚集了极大规模的数据点，形成了极为突兀的高峰。==

基于这份数据，我们可以计算乘客支付小费的经验概率 (Probability)。计算方式非常直接：统计实际给过小费的订单数量，除以全部订单总量即可。

```python
# 创建布尔序列，区分支付小费与未付小费的样本
tippers = df['tip'] > 0
# 统计支付小费的总订单数
number_of_tippers = tippers.sum()
# 统计行程总记录数
total_rides = len(df)

# 计算支付小费的比例
fraction_of_tippers = number_of_tippers / total_rides
print(f'The percentage of riders who tip is {fraction_of_tippers*100:.0f}%.')

```
![[Rideshare_Project_Week1_ch-1792088628464.webp]]

在下一个单元格中，我们将构建一个新的子数据集，过滤掉未给小费的乘客记录 (即排除小费为 0 的样本)。
随后重新绘制直方图，看看去除了零值尖峰后的小费分布呈现何种形态。

```python
# 基于布尔掩码筛选仅包含付小费订单的子数据框 (执行条件过滤)
# df[df['tip'] > 0] 完整写法是
df_tippers = df[tippers]

# 针对支付小费的样本重新绘制分布直方图
df_tippers.hist('tip', density=True, bins = 100);

```
![[Rideshare_Project_Week1_ch-1792088700701.webp|423]]

可以看到，过滤后的分布展现出了清晰且极富规律的几何形态。从统计学本质来看，这实际上是在对原始变量 `tip` 进行条件限制 (Conditioning)。我们绘制出的是在“==乘客确定支付了小费”这一先验发生前提下的小费条件分布==，用数学语言表达即为给定 `tip > 0` 约束下的条件分布。换言之，我们剔除了 `tip = 0` 的零值冗余，转而研究非零有效数据的概率结构。

# 5. 划分关键数据子集 (Split the Data Into Interesting Subsets)

接下来，我们可以探索如何划分业务子集，并研究这些子集之上的条件概率分布。
例如：周末的乘车需求是否明显多于工作日？乘客在周末是否更倾向于慷慨给出小费？解答这些问题有助于精细化剖析周中与周末的运力需求差异，进而辅助平台科学调配司机供给。

为此，我们首先衍生出一个名为 `weekday` 的新列，用于标识当前行程发生在星期几。

```python
# 当字段具备标准日期时间格式时，提取星期名称非常简便
df['weekday'] = df["date"].dt.day_name()

df.head()

```
解释
    ![[Rideshare_Project_Week1_ch-1792088829111.webp]]

![[Rideshare_Project_Week1_ch-1792088840695.webp|279]]

现在，我们可以按星期几聚合统计各自的订单总数。

```python
# 统计每周各天的总订单量
# 返回Series 隐藏的index是'weekday'
daily_ride_counts = df['weekday'].value_counts()

# 定义一周标准的星期序列，用于规整默认无序排列的聚合结果
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# 按照自然星期顺序重排序列
daily_ride_counts = daily_ride_counts.reindex(WEEKDAYS)

daily_ride_counts

```
解释
    ![[Rideshare_Project_Week1_ch-1792089000220.webp]]
    ![[Rideshare_Project_Week1_ch-1792089036445.webp]]

![[Rideshare_Project_Week1_ch-1792089050077.webp|279]]

按照同样的逻辑，我们统计每周各天实际给出小费的订单量。

```python
# 返回Series 隐藏的index是'weekday'
df_tippers = df[df['tip'] > 0]
# 统计每周各天支付小费的订单量
daily_tippers_counts = df_tippers['weekday'].value_counts()

# 按照自然星期顺序重排序列
daily_tippers_counts = daily_tippers_counts.reindex(WEEKDAYS)

daily_tippers_counts

```
![[Rideshare_Project_Week1_ch-1792089173439.webp]]

接着，便能计算出每周各天乘客支付小费的发生比例。

```python
# 合并2个Series
df_daily_aggregation = pd.concat(
    [daily_ride_counts, daily_tippers_counts],
    axis=1,
    keys=["ride_count", "tippers_count"],
)
df_daily_aggregation["tips_percentage"] = (
    df_daily_aggregation["tippers_count"] / df_daily_aggregation["ride_count"] * 100
)

df_daily_aggregation

```
![[Rideshare_Project_Week1_ch-1792089686497.webp|375]]

解释
    ![[Rideshare_Project_Week1_ch-1792089665296.webp]]


我们刚刚完成计算的指标，本质上正是**条件概率 (Conditional Probability)** ：即在==“已知处于某特定星期几”的条件下，乘客给小费的概率是多少？==用数学公式表示即为：

$$P(tip\vert{}weekday)$$


仔细审视输出的数据指标，我们能提炼出极具价值的业务洞察！

可以清楚地发现：周五与周六的打车出行需求大幅超越其他工作日，呈现明显的周末脉冲；
然而，各天乘客支付小费的百分比却保持在极为接近的稳定水平，==并未随周末出行潮而产生明显剧变。==

若有需要，你可以通过下方单元格将清洗加工后的数据导出保存。不执行导出也没关系，因为后续实验已经为你准备好了配套的标准数据集。

```python
# 如需保存修改后的数据框，可取消下方代码注释执行导出：
# df.to_csv("data/rideshare_2022_user.csv", index=False)

```

**祝贺你顺利完成本次实战实验！**在这份教程中，你已经成功运用代码实现了课程中的核心概念：概率、概率分布以及条件概率。与此同时，这也进一步巩固了你的 Pandas 数据操作硬实力。如果你对这种动手分析感到意犹未尽，敬请期待下周更深入的探索性实战演练！