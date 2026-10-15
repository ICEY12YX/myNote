___
# 探索性数据分析 (Exploratory Data Analysis) - Pandas 入门

欢迎来到 Pandas 实战教程。这是探索性数据分析 (Exploratory Data Analysis, EDA) 系列的第一份交互式编程笔记本 (Notebook) ，在这里，你将学以致用，亲自动手将课程学到的技能应用到贴近现实生活的真实数据问题中！本文将带你体验 Pandas 的基础操作，熟悉后续实验与课后作业中经常用到的核心实用函数。如果你已经对 Pandas 驾轻就熟，完全可以跳过这篇教程。

在本次演示中，我们将使用《世界幸福感报告》 (World Happiness Report) 数据集。该数据集包含 2199 行数据，每一行记录了特定国家在特定年份中与幸福感相关的多项指标。目前我们先借助这份数据来掌握 Pandas 的基础操作。在第 3 周的学习中，你还会再次与这份数据相遇，届时我们将展开更深入的探索，分析各项特征之间的关联，看看究竟哪些因素最能预测幸福感。

这份 Notebook 并非详尽无遗的 Pandas 全面指南，而是侧重于演示和讲解本门课程中最为关键的核心函数。如果你想更全面深入地学习 Pandas，可以参考官方教程或查阅其官方文档。

___


___

# 1. 导入库 (Importing the Libraries)

在这份教程中，最不可或缺的库正是我们今天的主角——`Pandas`。此外，我们还会借助 `Seaborn` 库来进行数据可视化绘图。运行下方单元格即可导入这些库：

```python
# 导入 Pandas 库
import pandas as pd
# 导入用于绘图的 Seaborn 库
#!pip install seaborn
import seaborn as sns

```

# 2. 导入数据 (Importing the Data)

成功导入 Pandas 库后，下一步就是加载数据集。这里使用的数据以 `.csv` 格式保存，加载它只需调用 `pd.read_csv(filename)` 函数即可。如果你的数据是其他格式，Pandas 也提供了非常丰富的专用加载函数，具体细节可查阅相关文档。
数据读取完成后，会被存储为💛 `DataFrame` 类型 (可查阅其参考文档) 。这是 Pandas 中最核心且最常用的数据结构，后续的所有实战都会围绕它展开。

```python
# 加载数据集并存入 df 变量
df = pd.read_csv('data/world_happiness.csv')

```

# 3. DataFrame 的基础操作 (Basic Operations With a Dataframe)

## 3.1 查看数据 (View the Dataframe)

你可以使用 💛`DataFrame.head()` 和 `DataFrame.tail()` 分别查看数据的前几行或后几行。
    默认情况下它们会展示 5 行，不过你也可以通过传入参数来自由指定想看的行数。
    严格来说，这两个函数本身并不直接负责屏幕渲染，而是==返回一个新的 DataFrame==。
我们能直接看到精美的表格，是因为Jupyter 会自动将最后一行的表达式结果渲染出来。
你也可以直接输入 `df` 来查看整个数据框的内容；当数据过长时，它会自动截取前几行与后几行进行展示。

需要注意的是，==这种自动显示机制只在代码位于单元格最后一行时生效==。
    Jupyter老生常谈的特点了
如果你希望在一个单元格中同时展示多个 DataFrame，或者在展示之后还想执行其他操作，建议将其包裹在 `print()` 或 `display()` 函数中。
💛`display()` 函数能以网页富文本表格的形式美观输出 (效果与直接输入 `df` 相同) ，而 `print()` 则会将其格式化为纯文本打印。

不妨尝试在下方代码中取消或添加注释，看看实际的输出效果，并试着观察不同的行数组合。

```python
# 如果后面没有其他代码行，该行将直接展示数据框的前几行
df.head()

# 尝试对下方代码的不同组合进行取消注释操作：
# print("Cats are cool.")
# print(df.head())
# print(df)
# print("Some more text about cats being cool.")
# display(df)

```
`df.head()`
![[intro-to-pandas-world-happiness_ch-1792064973990.webp]]

`print(df.head())`
![[intro-to-pandas-world-happiness_ch-1792065133790.webp]]

`print(df)`
![[intro-to-pandas-world-happiness_ch-1792065190690.webp]]

`display(df)`
![[intro-to-pandas-world-happiness_ch-1792065211409.webp]]

现在，让我们来查看数据框的最后几行。注意观察用来指定行数的额外参数。

```python
# 该行代码将仅展示数据框的最后两行
df.tail(2)

```
![[intro-to-pandas-world-happiness_ch-1792065249930.webp]]

## 3.2 索引与列名 (Index and Column Names)

在 `DataFrame` 中，数据是以==二维网格 (包含行与列) ==的形式组织存储的。每==一行==都有唯一的**索引 (Index)** ，每==一列==都有对应的**列名 (Columns)**。若想查看索引或列名，可以直接访问 💛`DataFrame.index` 或 `DataFrame.columns`。

```python
df.index

```
![[intro-to-pandas-world-happiness_ch-1792065304245.webp]]

正如你所看到的，该索引是一个数值范围，包含从 0 (闭区间) 到 2199 (开区间) 的连续数字。

运行下方代码单元格以查看所有列名。

```python
df.columns

```
![[intro-to-pandas-world-happiness_ch-1792065314860.webp]]

这些列名均以字符串格式存储。
可以看到，==部分列名中含有空格。== 这在后续通过属性访问列时可能会带来不便 (稍后你就会看到) ，
==因此将列名重命名、剔除空格是一个非常好的编程习惯。==业界的普遍做法是将空格==**替换为下划线**==。你可以调用💛 `DataFrame.rename()` 方法，并传入一个映射字典来批量修改列名。

在接下来的示例中，我们将演示如何自动将所有列名中的空格一键替换为下划线：

```python
# 构建字典，将旧列名映射为新列名。除了将空格替换为下划线，还将所有字母转为小写
columns_to_rename = {i: "_".join(i.split(" ")).lower() for i in df.columns}
# 注意：该字典是通过遍历列名自动生成的
# 你也可以手动构建字典，仅对需要修改的列进行重命名
# 例如参考下方注释的代码：
# columns_to_rename = {"Country name": "country_name", "Life Ladder": "life_ladder"}

#上面也是语法糖

# 用上面处理过的新列名 执行重命名
df = df.rename(columns=columns_to_rename)
# 查看修改后的数据框
df.head()

```
![[intro-to-pandas-world-happiness_ch-1792074160467.webp]]

解释
    ![[intro-to-pandas-world-happiness_ch-1792073934971.webp]]
    ![[intro-to-pandas-world-happiness_ch-1792074032515.webp]]
    结果类似这样
    `{"User ID": "user_id", "Total Score": "total_score"}`

## 3.3 数据类型 (Data Types)

DataFrame 的一大亮点在于：==不同列之间可以拥有各自独立的数据类型 (`dtypes`)==。这是传统的 NumPy 数组无法直接做到的。
    就类似excel表格
你可以随时查看各列的数据类型，并在必要时进行转换。

```python
df.dtypes

```
![[intro-to-pandas-world-happiness_ch-1792074209391.webp]]

从输出中可以看出各列具有不同的类型；结合实际数据内容来看，目前的类型推断完全正确。不过在真实项目中，若数据源格式存在瑕疵，自动解析出的类型可能会发生偏差。遇到这种情况时，就需要在开展后续分析前==手动对列类型进行校正==。下方代码展示了具体的转换方法 (注：由于当前数据类型本就正确，运行下方代码不会改变现有类型) 。

```python
# 筛选出所有应该为浮点数 (float) 的列
float_columns = [i for i in df.columns if i not in  ["country_name", "year"]]
# 将所有目标列的数据类型统一转换为 float
df = df.astype({i: float for i in float_columns})
# 打印所有列的数据类型
df.dtypes

```
![[intro-to-pandas-world-happiness_ch-1792074301484.webp]]

💛`df.info()` 能够提供更为全面的元数据。除了数据类型外，它还会详细列出每一列的非空值 (non-null) 数量。

```python
df.info()

```
![[intro-to-pandas-world-happiness_ch-1792074333285.webp]]

## 3.4 选取列 (Selecting Columns)

选取单列数据的一种方式是使用属性语法 💛`DataFrame.column_name`。此时你就能体会到此前清除列名空格的明智之处了。这种方式返回的是 Pandas 的==一维序列 (`Series`) 对象==，它与二维的 `DataFrame` 是不同的数据结构。稍后我们会讲解如何以 `DataFrame` 的形式返回列数据。

```python
# 提取 life_ladder 列并赋值给 x
x = df.life_ladder

print(f"type(x):\n {type(x)}\n")
print(f"x:\n{x}")

```
![[intro-to-pandas-world-happiness_ch-1792074609483.webp]]
    发现对Series类型的x
    还会自动打印每行的序号(这个应该不是x的一部分 只是系统自动打印出来了)

另一种常见方式是使用方括号搭配带引号的列名，这与在 Python 字典中按键取值十分类似。与字典相同，使用单引号或双引号均可。

```python
x = df["life_ladder"]

print(f"type(x):\n {type(x)}\n")
print(f"x:\n{x}")

```
(输出同上)

如果传入的不是单个字符串标签，而是一个标签列表，Pandas 就会返回仅包含这些指定列的 `DataFrame` (而非 `Series`) 。你可以用这种方法灵活提取单列或多列数据。

```python
x = df[["life_ladder"]]
# x = df[["life_ladder", "year"]]

print(f"type(x):\n {type(x)}\n")
print(f"x:\n{x}")

```
注意区别
![[intro-to-pandas-world-happiness_ch-1792074760800.webp|386]]

## 3.5 选取行 (Selecting Rows)

传入切片表达式 `:` 可以筛选出匹配的行切片，并返回保留原数据框所有列的 `DataFrame`。

```python
df[2:5]

```
![[intro-to-pandas-world-happiness_ch-1792075062585.webp]]

## 3.6 遍历行 (Iterating Over Rows)

如果需要对每一行进行循环遍历，可以使用 `.iterrows()` 方法。它会在每次迭代中产出一个 `(index, row)` 元组，其中 `row` 是承载该行数据的 `Series` 对象。
需要特别注意的是，按行迭代==无法跨行保留各字段原始的数据类型== **(在 DataFrame 中，数据类型是按列进行统一维护的) 。**

```python
index, row = next(df.iterrows())
row

```
![[intro-to-pandas-world-happiness_ch-1792075149055.webp]]

解释
    ![[intro-to-pandas-world-happiness_ch-1792076264735.webp]]
    ![[intro-to-pandas-world-happiness_ch-1792076292971.webp]]
    ![[intro-to-pandas-world-happiness_ch-1792076319418.webp]]
    ![[intro-to-pandas-world-happiness_ch-1792076466306.webp]]

## 3.7 布尔索引 (Boolean Indexing)

现在来到更有趣的数据筛选环节。细心的你可能已经发现，上述数据集包含了不同年份的观测记录。如果你只想聚焦于某一年的数据该怎么办？或者想按特定国家筛选？抑或是挑出某一列数值大于预设阈值的记录？这时就可以借助**布尔索引 (Boolean Indexing)**。

运行下方代码即可筛选出年份为 2022 年的所有行。你也可以尝试取消注释其他行，观察不同的过滤效果。

```python
df[df["year"] == 2022]
# df[df["life_ladder"] > 5] # 筛选出 life_ladder 大于 5 的行
# df[df["life_ladder"] > 11] # 该条件将返回一个空的 DataFrame

```
![[intro-to-pandas-world-happiness_ch-1792076544291.webp]]

需要注意，在筛选出特定行之后，==**原有的行索引会出现大量断号和跳跃，不再连续。**==虽然这并不影响计算，但在某些业务场景下，我们可能希望行索引能严格对应从 0 开始的真实连续行号。此时可以调用💛 `reset_index()` 来重置索引；而在另外一些场景下，保留跳跃的原始索引反而更方便溯源到原数据框。具体采用哪种方式，取决于实际分析需求。运行下方代码重置索引并观察输出：

```python
new_df = df[df["year"] == 2022]
new_df = new_df.reset_index(drop=True)
new_df

```
![[intro-to-pandas-world-happiness_ch-1792076633568.webp]]
解释
    ![[intro-to-pandas-world-happiness_ch-1792078834474.webp]]
    ![[intro-to-pandas-world-happiness_ch-1792078858788.webp]]

# 4. 描述性统计 (Summary Statistics)

在后续的课程中，我们将系统学习**描述性统计量**。这里先为你展示 Pandas 的便捷之处：只需调用 `describe()`，就能==一键计算出各类核心统计指标==。运行下方代码来快速总览数据的统计全貌；即使暂时不理解每一项指标的具体含义也无需担心，接下来的几周会逐一展开讲解。

```python
df.describe()

```
![[intro-to-pandas-world-happiness_ch-1792078901020.webp]]

不过，盲目套用描述性统计并不总是合乎逻辑的。以当前数据为例，面对各列计算出的汇总数值，我们是否真正理解其实际含义？这份数据涵盖了许多不同的国家，但每个国家所贡献的数据点数量或统计年份跨度完全一致吗？此外，不同国家的人口规模差异巨大，==直接进行简单算术平均真的合理吗？==

# 5. 数据绘图 (Plotting)

若想进行数据可视化，可以直接调用 💛`DataFrame.plot()`。在默认情况下，它会将数据索引作为横坐标 (x 轴) ，并将所有数值型列作为纵坐标 (y 轴) 进行折线绘制。运行下方代码查看绘图结果：

```python
# 如果图表未正常渲染，请先尝试重新运行该单元格
# 若仍无法显示，可以从上方菜单栏的 Kernel 中重启内核并重新运行整个 Notebook
df.plot()

```
![[intro-to-pandas-world-happiness_ch-1792084328422.webp|507]]

显而易见，默认生成的图表并没有提供太多有效信息。此处的==行索引并没有特定的物理意义==，而且==不同列之间的数值量级悬殊== (例如年份都在 2000 左右，而其他指标的数值则小得多) ，导致大量关键曲线被压缩在底部无法分辨。你可以尝试通过调整 `.plot()` 方法的参数来进行定制，详细参数设置可参阅官方 API 文档。

运行下方代码，我们来绘制一张明确指定横纵坐标的散点图 (Scatter Plot)。
横坐标为衡量国家富裕程度的人均 GDP 对数值 (`log_gdp_per_capita`)，纵坐标为生活阶梯评分 (`life_ladder`)，该列数值反映了公众对自身生活质量的主观评估打分 (满分为 1 到 10 分) 。

```python
df.plot(kind='scatter', x='log_gdp_per_capita', y='life_ladder')

```
![[intro-to-pandas-world-happiness_ch-1792084387899.webp]]

观察散点图可以发现，国家富裕程度与国民幸福感之间呈现出明显的正向趋势，直观表明富裕群体在某种程度上往往更具幸福感。在第 3 周的学习中，我们将更深入地量化剖析这种关系。

此外，通过给数据点赋予不同的颜色来区分特征或突出我们重点关注的对象，往往能带来更深刻的洞察。请参考以下示例：

```python
# 构建字典，将国家名称映射为不同颜色
cmap = {
    'Brazil': 'Green',
    'Slovenia': 'Orange',
    'India': 'purple'
}

df.plot(
    kind='scatter',
    x='log_gdp_per_capita',
    y='life_ladder',
    c=[cmap.get(c, 'yellow') for c in df.country_name], # 设置数据点颜色
    s=2 # 设置散点大小
    )

```
![[intro-to-pandas-world-happiness_ch-1792085038819.webp|491]]

从图中可以看出，尽管整体上 GDP 越高对应生活阶梯评分越高，但这并不是放之四海而皆准的绝对规律。对比斯洛文尼亚 (橙色) 和巴西 (绿色) 可以发现，==虽然巴西的经济水平相对较低，但在统计年份中，巴西居民的平均幸福感却普遍高于斯洛文尼亚居民。==

数据绘图的另一大核心用途是直观呈现数据的**分布情况 (Distribution)**。后续课程会进行详细讲解，不过在 Pandas 中，我们已经可以非常轻松地通过**直方图 (Histogram)** 来实现这一目的。只需在目标数据框上调用💛 `DataFrame.hist()` 即可；需要注意，如果数据框包含多个数值列，该方法会为每一列分别绘制一张直方图。如果只想观察某一个特定列，单独提取该列进行绘制即可。

```python
df.hist("life_ladder")
df.hist()
```
![[intro-to-pandas-world-happiness_ch-1792085143823.webp|403]]

![[intro-to-pandas-world-happiness_ch-1792085156072.webp|415]]

这张直方图展现了 `life_ladder` 列的数据分布规律。初看之下，你对这种分布形态有何感想？大多数人对自己的生活质量普遍感到满意吗？要严谨回答这个问题，还需要更深入地挖掘数据背景：必须厘清每个数值的具体统计口径——因为这些数据并非个人的单个受访回答，而是在==不同时间跨度下按国家汇总聚合后的结果。==

除了 Pandas 自带的绘图功能，我们还可以利用**第三方高级可视化库**来打造更具表现力的图表，其中最经典的代表便是 **Seaborn**。在实验开头，我们已经通过 `import seaborn as sns` 完成了导入。运行下方代码来体验其强大而高效的可视化能力 (该图表在后续的 Notebook 中也会频繁使用) 。由于数据特征较多，生成该图可能需要几秒钟的运算时间：

```python
# 如果图表未正常渲染，请先尝试重新运行该单元格
# 若仍无法显示，可以从上方菜单栏的 Kernel 中重启内核并重新运行整个 Notebook
sns.pairplot(df)

```
![[outputdjfo.webp]]

利用这种成对**关系图 (Pairplot，即散点图矩阵)** ，我们可以全方位观察==任意两列指标之间的两两散点相关图==。而在主对角线上 (两列相同的位置) ，它不会绘制成单调的对角线，而是==自动切换为直方图来展现该指标自身的分布状态==。

可以看到，不同指标组合呈现出的散点形态与分布特征大相径庭。不妨结合这些可视化结果，思考能从中提炼出哪些潜在的数据洞察。

# 6. 列间运算 (Operations on Columns)

在实际分析中，现有的单列数据有时无法直接提供所需指标，但我们可以通过对已有字段进行衍生计算来获取新信息。

例如，我们可以创建一个新列，其数值等于两列相加之和：

```python
# 创建一个新列，将年份与生活阶梯评分直接相加
df["this_column_makes_no_sense"] = df["year"] + df["life_ladder"]
# 创建一个新列，计算积极情绪与消极情绪的净差值
df["net_affect_difference"] = df["positive_affect"] - df["negative_affect"]

df.head()

```
![[intro-to-pandas-world-happiness_ch-1792085571746.webp]]

从上方的结果中可以看到新增的两列数据。
第一列并没有明确的实际业务含义，仅仅是把年份和生活阶梯数值简单相加；
而第二列则计算出了积极情绪与消极情绪的净差值。基于这个新衍生出的特征，我们便能进一步探索它与其他指标之间是否存在有趣的关联规律。你还能想到哪些值得计算的衍生字段？总的来说，基于现有列衍生计算出新特征，是探索性分析中极其强大的利器。

如果需要对列执行更为复杂的定制化运算，可以使用 `DataFrame.apply()` 方法，它==允许我们将任意 Python 函数逐行应用到指定列上==。下方代码展示了 `DataFrame.apply()` 的多种用法，你也可以尝试修改 `my_function`，赋予其自定义的计算逻辑：

```python
# 结合 Lambda 匿名函数使用 df.apply()
# 将 life_ladder 列归一化缩放到 0 至 1 之间，并存入新列
df['life_ladder_rescaled'] = df['life_ladder'].apply(lambda x: x / 10)
#apply应该是传入一个函数 而lambda是匿名函数 也算函数

# 结合自定义普通函数使用 df.apply()
# 首先定义函数逻辑 (函数内部可以实现任意数据处理，这里示例将其数值翻倍) 
def my_function(x):
    # 对 x 执行处理操作
    y = x * 2
    return y
# 应用该函数
df['my_function'] = df['life_ladder'].apply(my_function)

# 展示包含新衍生列的数据框
df.head()

```
![[intro-to-pandas-world-happiness_ch-1792085717017.webp]]

**祝贺你顺利完成本次实战实验！** 如果你已经完全掌握了上述代码的核心思想，那么你已经打下了坚实的基础，能够从容应对本周的编程作业以及后续所有涉及 Pandas 的实验与任务。在后续的探索性数据分析实验中，若遇到 Pandas 用法上的疑问，随时可以回顾这篇教程进行温故知新。