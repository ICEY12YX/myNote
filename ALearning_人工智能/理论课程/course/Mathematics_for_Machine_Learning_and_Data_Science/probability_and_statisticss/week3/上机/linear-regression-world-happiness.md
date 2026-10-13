___
# 探索性数据分析 - 线性回归 (Linear Regression)



欢迎来到探索性数据分析 (EDA) 系列的第四个实验笔记本 (notebook)！在本实验中，你将使用在 Pandas 入门教程中见过的《世界幸福感报告》 (World Happiness Report) 数据集。该数据集共包含 2,199 行，每一行都记录了特定国家在某一年份的各项幸福感相关指标。

在上一段视频中，你学习了极大似然估计 (Maximum Likelihood Estimation, MLE) 最常见的应用之一——线性回归。线性回归是一种用于估计两个或多个变量之间线性关系的统计模型。在一元线性回归 (simple linear regression) 中，包含一个自变量 (解释变量) 和一个因变量 (响应变量) ；而在多元线性回归 (multiple linear regression) 中，则包含多个解释变量。如果你想了解更多背景知识，可以查阅 Wikipedia 对应词条。

在本实验中，你将构建自己的线性回归模型，并通过拟合单个及多个解释变量来预测响应变量。我们将使用开源且商用友好的机器学习工具包 scikit-learn。作为一名数据科学家或机器学习从业者，你会在日常工作中广泛接触到该工具包实现的各类机器学习与统计算法。

# 1. 导入所需库



按照惯例，首先导入本实验需要用到的所有库。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 从 scikit-learn 中导入用于构建模型的各类函数
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 导入用于构建交互式组件的函数
import ipywidgets as widgets
from ipywidgets import interact_manual, fixed

# 引入封装好的辅助函数，保持 notebook 整洁
import utils

```

# 2. 读取与预处理数据



你将处理此前在“Pandas 快速入门”实验中接触过的“世界幸福感”数据集。第一步是加载数据并进行清洗。下方代码会重命名各列以去除空格，并删除所有缺失值。

```python
# 读取数据
df = pd.read_csv('data/world_happiness.csv')

# 重命名列名，将其中的空格替换为下划线并转为小写
df = df.rename(columns={i: "_".join(i.split(" ")).lower() for i in df.columns})

# 删除包含缺失值的行，避免对模型拟合产生负面影响
df = df.dropna()

# 展示数据帧
df.head()

```

仔细观察上方代码单元的输出结果。该数据集包含以下字段：

* `country_name`：采集数据的国家名称。


* `year`：数据采集年份。


* `life_ladder`：生活阶梯指数，即问卷调查中对生活质量评估的平均分 (评分区间为 1 至 10)。调查中受访者对自身的生活质量进行主观评估。


* `log_gdp_per_capita`：按购买力平价 (PPP) 计算的人均国内生产总值对数 (log GDP)。


* `social_support`：社会支持度，针对二元选择题“若你身处困境，是否有随时能够依靠并提供帮助的亲友？”的全国平均回答。


* `healthy_life_expectancy_at_birth`：出生时的健康预期寿命。


* `freedom_to_make_life_choices`：选择生活方式的自由度，针对二元选择题“你对自己选择生活方式的自由度感到满意还是不满意？”的全国平均回答。


* `generosity`：慷慨度，根据问题“过去一个月内你是否向慈善机构捐过款？”的回答结合 GDP 计算得出。


* `perceptions_of_corruption`：腐败感知度，关于腐败相关问题回答的综合平均值。


* `positive_affect`：积极情绪，针对微笑、享受生活以及做有趣事情等三道积极情绪问题的平均回答得分。


* `negative_affect`：消极情绪，针对担忧、悲伤与愤怒等三道消极情绪问题的平均回答得分。



关于数据字段的详细解释，可查阅 官方技术附录。

# 3. 数据探查



在正式将线性回归模型拟合到数据之前，先进行数据可视化以掌握整体分布特征是非常明智的。Seaborn 的 `pairplot` 是一个极其实用的工具，能自动绘制数据帧中每对特征之间的散点图，并在对角线上展示各列数值的直方图。运行下方代码单元以实现数据可视化：

```python
# 若图表未正常渲染，请先尝试重新运行此单元格；若仍无法显示，
# 可以重启内核 (从上方的 Kernel 菜单中选择) 并重新运行 notebook
sns.pairplot(df)

```

可以看到，部分变量组合对应的散点图呈现出明显的狭长条状分布，这暗示着两个变量之间可能存在显著的相关性。这些变量对往往是自变量与因变量组合的极佳候选。然而，单纯观察散点分布有时具有欺骗性。你必须明确自己的预测目标，并合理挑选具备解释意义的自变量。重新浏览列名，思考一下在这个数据集中，哪些字段适合作为解释变量，你又最希望预测哪个指标。你也可以从 官方报告 中汲取灵感。

# 4. 一元线性回归 (Simple Linear Regression)



对数据有了清晰认识后，就可以选定自变量 (X) 与因变量 (y)。我们从最直观的组合切入：使用人均 GDP 来解释生活阶梯指数 (衡量居民幸福感的核心指标)。

## 4.1 定义变量



在机器学习领域，通常不会使用同一批数据既用于训练模型又用于评估模型，因为我们需要检验模型对未知 (未参与训练) 新数据的泛化能力 (generalization)。运行下方代码单元将数据拆分为两部分：生成训练集 `X_train`、`y_train` 以及测试集 `X_test`、`y_test`。请注意，自变量使用大写 X，而因变量使用小写 y。这样设计是为了强调 X 通常是二维矩阵，而 y 是一维向量，以便未来无缝扩展到引入多个解释变量的场景。在后文使用更多变量预测幸福感时，你将亲身体验这一点。

```python
# 从数据帧中提取特征与标签
X = df[['log_gdp_per_capita']]
y = df['life_ladder']

# 划分训练集与测试集
# 注意 test_size=0.2 表示测试集占总数据量的 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 查看 X_train 与 y_train 的数据形态
print(f"Independent (X) variable:\n{X_train.values}\n")
print(f"Dependent (y) variable:\n{y_train.values}")

```

## 4.2 构建与拟合线性回归模型



有了训练集与测试集后，就可以着手构建线性回归模型了。借助于 `LinearRegression`，仅需一行代码即可完成构建与训练：

```python
lr = LinearRegression().fit(X_train, y_train)

```

代码的前半部分 `LinearRegression()` 用于实例化线性回归模型，你也可以在此处传入特定超参数；后半部分的 `fit(X_train, y_train)` 方法执行实际的模型训练过程，根据你的训练数据完成模型参数的最优拟合。

## 4.3 查看模型参数



我们可以将拟合出的直线方程表示为 $\hat{y}=Wx+b$，其中 $x$ 代表解释变量，$\hat{y}$ 代表预测的响应变量，而 $W$ 和 $b$ 则是模型学习出的参数。运行下方单元格，可以查看刚刚训练完成的模型中的参数 $W$ 与 $b$：

```python
b = lr.intercept_
w = lr.coef_
print(f"Model parameters:\nw: {w},\nb: {b}")

```

## 4.4 模型预测与评估



拟合好模型后，就可以使用它来进行预测了。在机器学习中，此类模型不仅用于描述现有数据，更关键的用途在于针对全新数据样本生成预测。为了客观评估模型效果，我们在前文已将数据集划分为训练集和测试集，借此检验模型在未知数据上的预测能力。

在测试集上计算预测值的常规方法是调用 `.predict()`。你将一次性对整个测试集执行 `.predict()`。为了加深对模型底层计算机制的理解，我们还将使用上一步提取的参数 W 和 b 进行“手动计算”，并将其与 `.predict()` 的输出进行比对：

```python
# 使用 lr.predict() 进行预测
y_test_preds = lr.predict(X_test)

# 使用参数 w 和 b 进行手动矩阵乘法预测
y_pred = np.dot(X_test, w) + b

# 验证两种预测方式结果是否完全一致
print(f"prediction using np.dot() and predictions using lr.predict are the same: {(y_pred == y_test_preds).all()}\n")

# 将部分预测结果与真实目标值进行比对
print(f"Last four predictions on the test set:\n{y_pred[:4]}\n" )
print(f"Target values \n{y_test[:4].values}\n")

mae = metrics.mean_absolute_error(y_test, y_test_preds)
print(f"Mean Absolute Error on the test set: {mae:.2f}")

```

比对结果表明，至少对于前四个样本，预测值与真实值相当接近。整个测试集上的平均绝对误差 (Mean Absolute Error, MAE) 约为 0.57，这说明尽管该一元模型无法完美预测幸福感，但已经给出了相当不错的近似估计。

## 4.5 结果可视化



现在，你可以将预测结果与训练数据绘制在同一张图表中，直观感受模型的拟合表现。图中的蓝色点是来自训练集的真实数据样本；橙色点的横坐标来自测试集，纵坐标则是模型生成的预测值。

```python
plt.figure()
plt.scatter(X_train, y_train, label = 'Training Data')
plt.scatter(X_test, y_pred, label = 'Predictions on the Test Set')
plt.legend()

```

可以看到，所有的预测点均严格落在拟合程度最佳的那条直线上。然而，真实数据的离散方差远大于这条直线的表现力，无法单凭一条直线完全解释。正如你所知，该数据集中还包含许多其他维度的特征，引入其他变量或许能够解释这些波动。在下一节中，我们将通过多元线性回归来探索这一点。

# 5. 多元线性回归 (Multiple Linear Regression)



在实际场景中，往往需要多个变量共同解释目标变量的变化。此时就可以应用多元线性回归。下方代码单元定义了一个封装函数，以简化后续分析流程：

* `fit_and_plot_linear_model`：该函数的逻辑与前文非常相似。核心区别在于它能同时接收多个特征，使你能够自由尝试使用两个或更多解释变量来构建模型。此外，该函数还会计算特征重要性 (feature importance)。在此我们不深入探讨其数学细节，你只需知道特征重要性得分代表了解释变量之间的相对权重——得分越高，该变量对最终预测结果的贡献度就越大。



阅读下方代码单元以深入理解执行逻辑。绘图与特征重要性计算已被封装至 `utils` 文件中，避免代码过于冗长。

```python
def fit_and_plot_linear_model(data, features):
    
    # 将输入的特征构造成列表作为解释变量
    features = list(features)
    
    # 构建线性回归模型
    
    # 选取目标变量与特征矩阵
    y = data['life_ladder']
    X = data[features]
    # 划分训练集与测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # 拟合线性回归模型
    lr = LinearRegression().fit(X_train, y_train)

    # 计算特征重要性
    most_important, feature_importance_df = utils.calculate_feature_importance(features, lr, X_test, y_test)

    # 准备绘图数据
    X_plot = data[most_important]
    Y_real = data["life_ladder"]
    X_test_plot = X_test[most_important]
    y_test_preds = lr.predict(X_test)
    
    # 绘制真实数据与模型预测结果
    utils.plot_happiness(most_important, X_plot, Y_real, X_test_plot, y_test_preds)

    # 若特征数大于 1，则绘制特征重要性柱状图
    if feature_importance_df is not None:
        utils.plot_feature_importance(feature_importance_df)
    
    # 计算并打印平均绝对误差 (MAE)
    mae = metrics.mean_absolute_error(y_test, y_test_preds)
    print(f"Mean Absolute Error: {mae:.2f}\n")

```

函数定义完毕后即可运行。下方单元格结合 `widgets` 与 `interact_manual` 实现了交互式分析，无需手动修改代码即可通过组件交互多次运行实验。

代码第一行定义了所有可选的预测变量列表，其余部分负责配置交互式调用。运行该单元格后，你可以从列表中多选特征进行回归分析。

该函数将计算多元线性回归，并在二维图表中呈现数据与拟合效果。图表的横轴会自动选取特征重要性得分最高的变量。

**交互组件使用说明**

* 从菜单中多选变量：按住 `Ctrl` 键并点击目标变量即可多选；单次点击只会选中当前项。


* 连续多选：先点击起始变量，再按住 `Shift` 键点击结束变量，两者之间的所有变量均会被选中；也可先点击第一个变量，然后使用 `Shift + 向下箭头` 逐项选中。



```python
# 所有可选预测变量列表
predictors = ['year', 'log_gdp_per_capita', 'social_support', 'healthy_life_expectancy_at_birth', 'freedom_to_make_life_choices', 'generosity', 'perceptions_of_corruption', 'positive_affect', 'negative_affect']

# 特征选择交互组件
feature_selection = widgets.SelectMultiple(
    options=predictors, value=predictors,
    description="Features", disabled=False,
)

# 交互式调用函数
interact_manual(fit_and_plot_linear_model, data=fixed(df), features=feature_selection);

```

你会发现，当选择多个解释变量时，二维图上的预测点不再整齐排列在一条直线上。这是因为二维图表中未展示的其他变量同样对预测值施加了影响，导致投影点偏离了单条直线。但在对应的高维特征空间中，这些点依然严格落在拟合出的超平面上。

**祝贺你顺利完成本次实验！** 如果你已经理解了上述全套流程，说明你已经能够熟练使用 scikit-learn 完成线性回归建模与评估。在后续课程中，你还将接触到使用其他 Python 库实现的线性回归。

