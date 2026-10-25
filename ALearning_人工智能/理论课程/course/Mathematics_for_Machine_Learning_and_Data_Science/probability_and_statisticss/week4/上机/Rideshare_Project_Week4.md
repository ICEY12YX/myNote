___
# 探索性数据分析 - 置信区间与假设检验 (Confidence Intervals and Hypothesis Testing)

欢迎来到探索性数据分析 (EDA) 系列的最后一个实验笔记本 (notebook)！在本实验中，你将继续使用芝加哥市 2022 年的网约车出行数据，该数据最初获取自 这里。在系列的前两篇实验中你已经初步接触过该数据，本次我们将继续基于在第 1 周清理并精简后的数据集展开分析。

### 学习目标：

在本实验中，你将在实际应用中实践课程涵盖的以下统计概念：

* 描述性统计量 (均值与标准差)
* 置信区间 (Confidence intervals)
* 双样本 t 检验 (Two sample t-test)
* 线性回归 (Linear regression)

___


___

# 1. 导入 Python 库



按照惯例，第一步依然是导入本实验中需要调用的核心库。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.formula.api as smf

```

# 2. 加载数据集



接下来加载数据集。为了保证在该计算环境中流畅运行，数据已预先进行了下采样处理。

```python
# 读取数据集
# 注意 parse_dates 参数，它会自动将指定列解析并保存为日期格式。
df = pd.read_csv("data/rideshare_2022_cleaned.csv", parse_dates=['trip_start_timestamp', 'date'])

# 展示数据集的前几行
df.head()

```

# 3. 探究全年每日乘车总量的波动趋势



首先观察全年每日订单量的变化规律。我们在这里的核心目标是计算 (总体) 均值的置信区间，这有助于预测下一年同期的订单需求规模。

```python
# 计算全年每一天的订单总量
daily_rides = df.groupby('date').size().reset_index(name='daily_rides')

# 展示数据帧
daily_rides

```

在统计出每一天的订单总数后，就可以计算样本均值 (sample mean) 和样本标准差 (sample standard deviation) 了。在之前的实验中，“均值”和“标准差”这两个术语的使用较为泛化，但现在你需要更加严谨地区分它们。鉴于当前处理的仅是芝加哥网约车真实数据的一个抽样样本，我们从中所计算出的指标全都是对真实未知总体的估计。换言之，我们当前计算的是样本均值与样本方差，而非总体均值。

```python
# 计算每日订单量的均值与标准差
mean_rides_per_day = daily_rides['daily_rides'].mean()
std_rides_per_day = daily_rides['daily_rides'].std()

print(f'Mean number of rides per day: {mean_rides_per_day:.2f}')
print(f'Standard deviation: {std_rides_per_day:.2f}')

```

在下一个代码单元中，我们将绘制每日订单量的分布图，并叠加热加一条水平红线代表刚刚算出的样本均值：

```python
plt.figure(figsize=(18,6))
# 绘制每日订单柱状图
plt.bar(daily_rides['date'], daily_rides['daily_rides'], label='Rides per Day')

# 绘制均值参考水平线
plt.axhline(y=mean_rides_per_day, c='r', label=f'Mean Rides per Day')

plt.ylabel('Rides per Day', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.xlim(min(daily_rides['date']), max(daily_rides['date']))
plt.legend(fontsize=14)
plt.show()

```

上面我们计算出了平均每天的订单量，但我们对这个均值估计有多大把握呢？

## 3.1 置信区间 (Confidence Intervals)



正如本周课程所学，回答上述问题的一种核心统计工具便是置信区间。接下来让我们求取每日订单均值的 95% 置信区间。由于总体的真实标准差未知 (实际上我们对总体分布形式一无所知) ，我们需要借助 $t$ 分布 (t-distribution) 来构造置信区间。


```python
# 设定目标置信水平
confidence = 0.95

# 使用 scipy 计算临界值 (critical value)
critical_value = scipy.stats.t.ppf(1 - (1 - confidence)/2, df=len(daily_rides)-1)

print(f"For the confidence interval of {confidence}, the critical value is {critical_value}")

# 计算置信区间半宽 (误差界限)
total_days = daily_rides['date'].count()
confidence_interval = critical_value * std_rides_per_day / np.sqrt(total_days)

print(f"With a {100 * confidence}% confidence you can say that your error will be no more than {confidence_interval:.4f} rides per day.")

```

求出平均订单量与置信区间后，我们将它们统一呈现在图表中以观察相对范围。

```python
plt.figure(figsize=(18,6))
# 绘制每日订单柱状图
plt.bar(daily_rides['date'], daily_rides['daily_rides'], label='Rides per Day')

# 绘制样本均值线
plt.axhline(y=mean_rides_per_day, c='r', label=f'Mean Rides per Day +/- {confidence}% Confidence Interval')
# 在均值线周围绘制置信区间带
plt.fill_between(daily_rides['date'], mean_rides_per_day-confidence_interval,
                 mean_rides_per_day+confidence_interval, color='r', alpha=0.2)

plt.ylabel('Rides per Day', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.xlim(min(daily_rides['date']), max(daily_rides['date']))
plt.legend(fontsize=14)
plt.show()

```

可以看到，估算出的置信区间带宽非常窄，这表明我们对总体均值的估计具有极高的精度。在 95% 的置信度下，未知的总体真实均值与我们样本均值的偏差非常微小。初看之下，这种极窄的区间可能反直觉，因为原始数据每天的波动幅度显然大得多。但请牢记：置信区间刻画的不是数据本身的波动离散度，而是对总体均值估计的不确定性。由于我们拥有充足的数据点 (全年的每一天) ，即使单日数据的标准差较大 (比均值置信区间的宽度大出约 10 倍) ，均值的估计依然能达到很高的精度。

观察图表还能捕捉到一个有趣的现象：在每年的最后两周 (圣诞与新年假期) ，订单总量出现了极其显著的下挫。我们完全可以单独截取这部分假期数据，单独估算仅针对该时段总体均值的置信区间。

```python
# 仅筛选假期时段数据
daily_rides_holidays = daily_rides[daily_rides["date"] > "2022-12-17"]

# 计算假期的样本均值与标准差
mean_rides_per_day_holidays = daily_rides_holidays['daily_rides'].mean()
std_rides_per_day_holidays = daily_rides_holidays['daily_rides'].std()

print(f'Mean number of rides per day: {mean_rides_per_day_holidays:.2f} +/- {std_rides_per_day_holidays:.2f}')

# 计算假期总体均值的置信区间
critical_value_holidays =  scipy.stats.t.ppf(1 - (1 - confidence)/2, df=len(daily_rides_holidays)-1)
total_days_holidays = daily_rides_holidays['date'].count()
confidence_interval_holidays = critical_value_holidays * std_rides_per_day_holidays / np.sqrt(total_days_holidays)
print(f"With a {100 * confidence}% confidence you can say that your error will be no more than {confidence_interval_holidays} rides per day.")

```

此时你会发现：虽然假期的均值明显走低且具有相近的标准差，但其均值的置信区间明显膨胀变宽了。这是因为此时我们仅仅使用了 14 个数据点 (两周样本) ，样本容量远低于全年的规模。

在下方代码单元中，我们再次绘制对比图，将假期的统计带与全年数据重叠呈现：

```python
plt.figure(figsize=(18,6))

# 绘制全年每日订单、均值与置信区间
plt.bar(daily_rides['date'], daily_rides['daily_rides'], label='Rides per Day')
plt.axhline(y=mean_rides_per_day, color='C0', label=f'Mean Rides per Day +/- {confidence}% Confidence Interval')
plt.fill_between(daily_rides['date'], mean_rides_per_day-confidence_interval,
                 mean_rides_per_day+confidence_interval, color='C0', alpha=0.3)

# 叠加绘制假期每日订单、均值与置信区间
plt.bar(daily_rides_holidays['date'], daily_rides_holidays['daily_rides'], label='Rides per Day (Holidays)')
plt.axhline(y=mean_rides_per_day_holidays, color='C1', label='Mean Rides per Day (Holidays) +/- {confidence}% Confidence Interval')

plt.fill_between(daily_rides_holidays['date'], mean_rides_per_day_holidays-confidence_interval_holidays,
                 mean_rides_per_day_holidays+confidence_interval_holidays, color='C1', alpha=0.5)

plt.ylabel('Rides per Day', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.xlim(min(daily_rides['date']), max(daily_rides_holidays['date']))
plt.legend(fontsize=14)
plt.show()

```

这正是置信区间的魅力所在——它为评估估计结果的可靠性提供了严谨依据。观察上图可以发现，蓝线 (全年均值) 落在了橙色阴影区域 (假期订单均值的 95% 置信区间) 内。如果收集更多年份的数据，样本量增加会进一步收窄同一置信水平下的区间宽度，从而使均值估计更加确定。

## 3.2 双样本 t 检验 (Two Sample t-test)



图表中的另一个显眼特征是周期性峰值。细看之下，这些尖峰大约每隔 7 天规律出现一次，这暗示着星期几对出行频次有显著影响。运行下方代码单元按星期聚合数据，并计算各星期的订单均值与标准差：

```python
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

daily_rides['weekday'] = daily_rides['date'].apply(lambda x: WEEKDAYS[x.weekday()])

weekday_summary = daily_rides.groupby('weekday')['daily_rides'].describe()
# 按星期自然顺序重排
weekday_summary = weekday_summary.reindex(WEEKDAYS)
weekday_summary

```

数据清晰表明，周五和周六的订单量明显高于其他工作日与周日 (很可能是周末聚会外出需求激增所致)。但我们能否确定有足够的统计学证据支持这一结论？让我们通过假设检验 (hypothesis test) 来验证！

在此处，我们可以使用双样本 t 检验来验证这一假设。回顾该检验所采用的检验统计量公式：

借助 `scipy` 库可以极其便捷地完成该计算。只需调用 `scipy.stats.ttest_ind()` 函数并传入两组样本即可。若令 $\mu_{FS}$ 代表周五和周六订单量的总体均值，$\mu_{other}$ 代表其他日子的总体均值，则可以设立如下假设：

* 零假设 (Null hypothesis, $H_0$)：$\mu_{FS} \leq \mu_{other}$ (周五和周六组的总体均值小于或等于其他组)


* 备择假设 (Alternative hypothesis, $H_1$)：$\mu_{FS} > \mu_{other}$ (周五和周六组的总体均值严格大于其他组)



由于我们期望证实周五与周六的总体均值更高，因此将其设定为单尾备择假设。

该函数会输出检验统计量以及对应的 p 值 (p-value)。随后将 p 值与预设的显著性水平 (significance level，例如 $\alpha=0.05$) 进行对比，以决定是否拒绝零假设。

```python
# 构造两组序列：一组为每周五和周六的订单量，另一组为其余日期的订单量
fridays_and_saturdays = daily_rides[daily_rides["weekday"].isin(["Friday", "Saturday"])]["daily_rides"]
other_days = daily_rides[daily_rides["weekday"].isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Sunday"])]["daily_rides"]

# 注意：这些序列包含各天的全量单日数据，而非上一单元格聚合后的均值
print(f"Number of datapoints for Fridays and Saturdays: {len(fridays_and_saturdays)}")
print(f"Number of datapoints for other days: {len(other_days)}")

# 执行单尾双样本 t 检验
scipy.stats.ttest_ind(a=fridays_and_saturdays, b=other_days, alternative='greater')

```

函数返回了一个极小极其接近 0 的 p 值，这意味着无论在多么严苛的显著性水平下，我们都有充分理由拒绝零假设，确凿得出乘客在周五和周六打车频次更高的结论。

# 4. 基于行程距离与行驶时间计算车费



在本节中，我们将尝试根据数据估算司机能够向乘客收取的车费金额。网约车平台通常会依据单位行驶时间以及覆盖里程进行综合计费，我们可以合理假设当前数据同样遵循该定价机制。首先绘制散点图，观察这些特征之间的相关性：

```python
fig, ax = plt.subplots(1,2, figsize=(12,4))
df.plot.scatter('fare','trip_seconds', ax=ax[0])
df.plot.scatter('fare','trip_miles', ax=ax[1])

```

如图所示，这些变量之间表现出高度相关性，非常适合用于拟合线性回归模型。在前一周的“世界幸福感”实验中，你使用的是面向机器学习的 `scikit-learn` 库；而本周我们将体验专注于严谨统计建模的 `statsmodels` 库。如需查阅其接口细节可访问 官方文档。虽然该库更偏向传统统计学，但调用同样简洁，只需寥寥数行代码便能完成模型拟合并输出详尽完备的统计报告。

在下方代码中，我们将以 `trip_seconds` (行程秒数) 和 `trip_miles` (行程英里数) 为解释变量，以 `fare` (车费) 为响应变量拟合模型：

```python
# 构建公式模型
model = smf.ols(formula='fare ~ trip_seconds + trip_miles', data=df)

# 拟合模型
result = model.fit()

# 输出回归摘要报告
print(result.summary())

```

使用 StatsModels 的一大优势在于，它能一次性输出海量的统计指标，帮助我们全面审视模型的质量与拟合优度。表格中的细节非常充实，其中部分指标超出了本课程的基础范围，但大部分核心概念你完全能够透彻理解。不要被信息量所吓倒，让我们逐一拆解：

第一部分呈现的是模型的基本设定信息：

```
Dep. Variable:                   fare  
Model:                            OLS  
Method:                 Least Squares  
Date:                Thu, 18 Jan 2024  
Time:                        20:45:54  
No. Observations:              689925  
Df Residuals:                  689922  
Df Model:                           2                                         

```

可以看到，因变量是 `fare` 列；采用的模型是普通最小二乘法 (Ordinary Least Squares, OLS) 模型。此外还记录了模型训练的日期、时间以及参与训练的样本观测总量 (No. Observations)。残差自由度 (Df Residuals) 对应于模型相对于训练数据的误差自由度，而模型自由度 (Df Model) 则反映了用于训练的解释变量数量。报告右侧的一组指标主要用于跨模型对比评估，在此暂不展开。

接下来，关注报表的核心中间区域：

```
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
Intercept        6.4621      0.020    329.648      0.000       6.424       6.501
trip_seconds     0.0056   2.51e-05    222.342      0.000       0.006       0.006
trip_miles       0.8693      0.003    343.709      0.000       0.864       0.874

```

这里列出了回归系数的核心参数，让我们逐列解析其统计学含义：

* `coef`：每个特征系数的点估计值 (即截距与斜率)。


* `std_err`：对应回归系数估计值的标准误 (Standard Error)。
后续各列用于衡量该参数估计的显著性程度：


* `t`：针对该系数计算出的 t 检验统计量。其原理与你在假设检验与置信区间中使用的 t 统计量一致。当估计的标准误相对于系数值极小时，该 t 统计量数值会极其庞大。


* `P>|t|`：这一列至关重要，它代表在“该特征真实系数为 0”的零假设下计算出的 p 值。换言之，它衡量了该系数纯粹由随机噪声偶然产生的概率。在本例中，所有 p 值均为 0.000，表明截距以及所引入的两个变量在模型中均发挥着高度显著的作用。


* `[0.025    0.975]`：这两列给出了每个系数的 95% 置信区间的下限与上限。本例中由于样本体量极大且拟合良好，这些区间极其紧凑狭窄。



最后看一下剩余的诊断指标：`Skew` (偏度) 与 `kurtosis` (峰度) 用于衡量残差 (residuals) 的分布对称性与尖锐程度。请记住，残差即真实观测值与模型预测输出之间的差值平方分析基础，是审视误差分布特征的窗口。

除了这些基础指标外，表中还包含了 (`Omnibus` 与 `Prob(Omnibus)`) 以及 (`Jarque-Bera (JB)` 与 `Prob(JB)`) 检验。它们是专用于检验误差正态性的假设检验，其备择假设均为“残差不服从正态分布”。尽管其统计量构造与常规 t 检验不同，但逻辑完全相通：极小的 p 值 (Prob) 表明残差分布偏离了严格的正态分布假定。

通过训练得到的 `result` 对象，我们可以方便地提取这些拟合参数。运行下方单元格查看模型的回归系数：

```python
result.params

```

我们也可以单独提取每一个系数：

```python
starting_fare = result.params["Intercept"]
price_per_second = result.params["trip_seconds"]
price_per_mile = result.params["trip_miles"]

print(f"The starting fare is {starting_fare:.3} USD. In addition the ride costs {price_per_second*60:.3} USD per minute and {price_per_mile:.3} USD per mile.")

```

明确了这些系数后，我们就可以构建一个简易的车费估算器，输入预估路程和行驶时间即可输出预测价格：

```python
def fare_calculator(trip_time, trip_distance):
    return starting_fare + price_per_second * trip_time + price_per_mile * trip_distance

sample_trip_duration = 10 * 60 # 10 分钟
sample_trip_distance = 10 # 10 英里

sample_fare = fare_calculator(sample_trip_duration, sample_trip_distance)

print(f"For a {sample_trip_distance} mile trip that takes {sample_trip_duration/60} minutes, you would pay around {sample_fare:.3} USD.")

```

最后，将原始真实数据点与模型的预测结果绘制在同一张散点图上，直观检验模型的拟合表现：

```python
# 提取拟合模型所用的 x 与 y 数据并剔除空值
x_y = df[["trip_miles", "trip_seconds", "fare"]].dropna()

# 若想切换横轴变量 (例如切换为 trip_miles 或 trip_seconds) ，可修改此行
x_variable = "trip_seconds"

# 准备绘图数据
x_plot =  x_y[x_variable]
y_plot =  x_y["fare"]
y_result = result.predict()

# 绘制散点图
plt.scatter(x_plot, y_plot, label="Original Data")
plt.scatter(x_plot, y_result, label="Prediction")
plt.xlabel(" ".join(x_variable.split("_")).title(), fontsize=14)
plt.ylabel("Fare", fontsize=14)
plt.legend(fontsize=14)

```

尽管真实数据中仍然存在部分未被模型完全解释的残差波动，但对于仅使用两个解释变量的简约模型而言，这一拟合表现已经相当出色！

**祝贺你顺利完成本次实验！** 在本篇中，你亲手实操了本课程涵盖的诸多核心统计方法：均值与标准差等描述性统计量、置信区间构造、双样本 t 检验以及多元线性回归分析。此外，你还进一步熟练了使用 Pandas 和 StatsModels 进行统计建模与绘图的技能。希望你充分享受了这一系列的探索性数据分析实验！