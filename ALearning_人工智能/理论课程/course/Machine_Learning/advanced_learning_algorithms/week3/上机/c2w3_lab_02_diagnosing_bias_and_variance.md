# 选修实验：诊断偏差 (Bias) 与方差 (Variance)

在上一个选修实验中，你学习了如何通过测量训练误差 (Training Error) 和交叉验证误差 (Cross Validation Error) 来评估学习算法的性能。借助这些数值，你能够量化一个模型的表现好坏，从而决定在特定应用中应该选用哪一个模型。在本次实验中，你将在此基础上更进一步，探索一些提升模型性能的实用技巧。事实上，训练误差和交叉验证误差能够为你指明下一步的改进方向——具体来说，它们能直接告诉你模型当前面临的是高偏差 (High Bias，即欠拟合 (Underfitting) ) 问题，还是高方差 (High Variance，即过拟合 (Overfitting) ) 问题。下面这张课堂讲义幻灯片给出了一个直观的例子：

&lt;img src=&#39;images/C2_W3_BiasVariance.png&#39; width=75%&gt;

最左侧的图展示的是高偏差问题：模型未能捕捉到训练数据中的潜在规律，因此训练误差和交叉验证误差都很高。相反，最右侧的图展示的是高方差问题：模型对训练集 (Training Set) 发生了过拟合，因此尽管它的训练误差很低，但在面对新样本时表现却很差，表现为交叉验证误差很高。最理想的模型是中间那张图：它既成功地从训练集中学到了规律，又能很好地泛化 (Generalize) 到未见过的数据上。课堂上针对下一步该如何达到这种“恰到好处”的模型给出了一些实用建议。

为了解决高偏差问题，你可以：
* 尝试添加多项式特征 (Polynomial Features) 
* 尝试获取更多额外特征 (Features) 
* 尝试减小正则化参数 (Regularization Parameter) 

为了解决高方差问题，你可以：
* 尝试增大正则化参数
* 尝试使用更精简的特征子集
* 获取更多的训练样本

在本次实验中，你将逐一尝试所有这些技巧。让我们开始吧！

## 建立基准性能水平 (Baseline Level of Performance)

在诊断模型究竟是存在高偏差还是高方差之前，先弄清楚一个模型在合理范围内大约能达到什么样的误差水平，通常会非常有帮助。正如课堂上所提到的，你可以利用以下任何一种参考标准来设定基准性能水平：

* 人类水平的表现 (Human level performance) 
* 竞争算法的表现
* 基于经验的合理估计

现实世界中的数据往往包含大量噪声 (Noise) ，想要达到 0% 的误差通常是不切实际的。举个例子，假设你在某个计算机视觉 (Computer Vision) 应用中得到了 10% 的训练误差和 15% 的交叉验证误差，你最初可能会以为自己遇到了高偏差问题。然而，后来你发现就连人类专家在这个任务上的误差也无法低于 10%。如果你将这 10% 视为基准性能水平，那么你的模型实际上并不存在高偏差，反而是存在高方差问题——因为此时你应该优先设法缩小交叉验证误差与训练误差之间的那 5% 差距。

牢记这一点后，下面让我们开始探索解决这些问题的具体技术。

## 导入软件包与实验环境配置

除了来自 scikit-learn 的几个线性回归器之外，本次实验中使用的所有其他函数都保存在本笔记本外部的 `utils.py` 文件中。由于大部分代码与上一个实验完全相同，因此我们无需在这里把每一行代码都重复展示出来。该文件主要包含了用于划分数据集的函数，以及遍历一系列超参数 (Hyperparameters，例如多项式阶数、正则化参数) 并绘制各自训练误差与交叉验证误差的绘图函数。如果你想查看具体实现细节，欢迎随时翻阅该文件中的代码。


```python
# for building linear regression models
from sklearn.linear_model import LinearRegression, Ridge

# import lab utility functions in utils.py
import utils 
```

## 解决高偏差问题

首先，让我们来看看当模型出现欠拟合——也就是说，当训练误差远差于基准性能水平时，可以尝试哪些改进措施。

### 尝试添加多项式特征

你在上一个实验中已经见识过这种方法了。添加多项式特征可以帮助模型学习到数据中更加复杂的模式。下面再次通过一张图表，展示随着你不断添加更高阶的多项式特征，训练误差和交叉验证误差会如何变化。在这里，你将使用一个只包含单个特征和单个目标值的人工合成回归数据集，并人为设定一个基准性能水平绘制在图表中。


```python
# Split the dataset into train, cv, and test
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data1.csv')

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")

# Preview the first 5 rows
print(f"first 5 rows of the training inputs (1 feature):\n {x_train[:5]}\n")

# Instantiate the regression model class
model = LinearRegression()

# Train and plot polynomial regression models
utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=400)
```

正如你所看到的，添加的多项式特征越多，模型对训练数据的拟合效果就越好。在这个例子中，模型的表现甚至超越了设定的基准线。此时你可以说，阶数大于 4 的模型已经属于低偏差模型，因为它们的表现已经接近甚至优于基准水平。

然而，如果基准误差被设定得更低一些 (例如你咨询了领域专家，了解到可接受的误差标准其实更加严格) ，那么这些模型就依然会被视为具有高偏差。这时，你就需要尝试其他方法来进一步改进模型了。


```python
# Train and plot polynomial regression models. Bias is defined lower.
utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=10, baseline=250)
```

### 尝试获取更多额外特征

你可以尝试的另一个方向是收集其他维度的特征。假设在看到上面的结果后，你决定启动新一轮数据采集工作，把另一个关键特征也记录下来。现在，你的数据集在输入特征部分就变成了包含 2 列的数据，如下所示。


```python
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data2.csv')

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")

# Preview the first 5 rows
print(f"first 5 rows of the training inputs (2 features):\n {x_train[:5]}\n")
```

现在来看看，在完全相同的训练流程下，新增特征会带来怎样的改变。你会发现，现在的训练误差已经非常接近 (甚至优于) 新的基准线了。


```python
# Instantiate the model class
model = LinearRegression()

# Train and plot polynomial regression models. Dataset used has two features.
utils.train_plot_poly(model, x_train, y_train, x_cv, y_cv, max_degree=6, baseline=250)
```

### 尝试减小正则化参数

到了这一步，你可能会想要引入正则化 (Regularization) 来避免过拟合。但需要小心的一点是：如果你把正则化参数设得太大，反而会导致模型出现欠拟合。下面的代码单元格使用 Ridge 类 (岭回归) 训练了一个 4 阶多项式模型，该类允许你设置正则化参数 (即 lambda 或 $\lambda$) 。你将尝试几个不同的取值，并对比它们的结果。


```python
# Define lambdas to plot
reg_params = [10, 5, 2, 1, 0.5, 0.2, 0.1]

# Define degree of polynomial and train for each value of lambda
utils.train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)
```

生成的图表显示，当初始 $\lambda$ 设为 `10` 时，训练误差明显比基准线要差。这意味着此时对参数 `w` 施加了过大的惩罚，阻碍了模型去学习数据中更复杂的模式。而随着你逐步减小 $\lambda$，模型放宽了这一限制，训练误差也随之顺利逼近基准性能水平。

## 解决高方差问题

接下来，我们将看看当模型对训练集发生过拟合时，可以采取哪些应对措施。此时的主要目标是让模型能够良好地泛化到新样本上，因此我们希望尽可能降低交叉验证误差。

### 尝试增大正则化参数

与上面最后一个练习恰好相反，如果把正则化参数设得太小，虽然能让模型保持低偏差，但对于改善高方差却无济于事。如下所示，你可以通过适当增大 $\lambda$ 的值来显著降低交叉验证误差。


```python
# Define lambdas to plot
reg_params = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

# Define degree of polynomial and train for each value of lambda
utils.train_plot_reg_params(reg_params, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)
```

### 尝试使用更精简的特征子集

在上一个实验中你已经看到，包含过多的多项式项会导致过拟合。你可以减少这类多项式项的数量，观察在哪个位置能够使训练误差和交叉验证误差达到最佳平衡。另一种适合精简特征数量的场景，是数据集中混入了无关特征。例如，医院分配给病人的就诊 ID 号对于诊断肿瘤没有任何帮助，因此你应当确保在训练前将这类无关特征从训练数据中剔除。

为了直观演示剔除无关特征是如何提升模型性能的，你将在两个数据集上分别运行多项式回归：一个是你刚才使用过的数据集 (包含 2 个特征) ，另一个则是额外添加了一列随机 ID 的数据集 (包含 3 个特征) 。你可以通过下面的代码单元格预览这两个数据集。请注意，后一个数据集中的后两列与前一个数据集完全相同，只是在第一列新增了一列随机数。


```python
# Prepare dataset with randomID feature
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data2.csv')

# Preview the first 5 rows
print(f"first 5 rows of the training set with 2 features:\n {x_train[:5]}\n")

# Prepare dataset with randomID feature
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data3.csv')

# Preview the first 5 rows
print(f"first 5 rows of the training set with 3 features (1st column is a random ID):\n {x_train[:5]}\n")
```

现在，你将分别在这两个数据集上训练模型并绘制对比结果。图中的实线表示包含 2 个特征的数据集上的误差，而虚线则表示包含 3 个特征的数据集上的误差。正如你所见，包含 3 个特征的数据集具有更高的交叉验证误差，尤其是在引入更多高阶多项式项之后。这是因为模型正试图从那列毫无意义的随机 ID 中强行寻找规律，哪怕它与目标值之间根本没有任何关系。

另一个观察角度是重点对比 `degree=4` 处的点。你会注意到，尽管使用 3 个特征时的*训练误差*更低，但它的*训练误差与交叉验证误差之间的差距*却比只使用 2 个特征时大得多。这也是模型正在发生过拟合的一个明确警告信号。


```python
# Define the model
model = LinearRegression()

# Define properties of the 2 datasets
file1 = {'filename':'data/c2w3_lab2_data3.csv', 'label': '3 features', 'linestyle': 'dotted'}
file2 = {'filename':'data/c2w3_lab2_data2.csv', 'label': '2 features', 'linestyle': 'solid'}
files = [file1, file2]

# Train and plot for each dataset
utils.train_plot_diff_datasets(model, files, max_degree=4, baseline=250)
```

### 获取更多的训练样本

最后，你还可以通过收集更多训练样本来设法降低交叉验证误差。在下面的代码单元格中，你将训练一个 4 阶多项式模型，并绘制出模型的*学习曲线 (Learning Curve) *，直观观察随着样本数量的增加，训练误差和交叉验证误差会如何变化。


```python
# Prepare the dataset
x_train, y_train, x_cv, y_cv, x_test, y_test = utils.prepare_dataset('data/c2w3_lab2_data4.csv')
print(f"the shape of the entire training set (input) is: {x_train.shape}")
print(f"the shape of the entire training set (target) is: {y_train.shape}\n")
print(f"the shape of the entire cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the entire cross validation set (target) is: {y_cv.shape}\n")

# Instantiate the model class
model = LinearRegression()

# Define the degree of polynomial and train the model using subsets of the dataset.
utils.train_plot_learning_curve(model, x_train, y_train, x_cv, y_cv, degree= 4, baseline=250)
```

从结果中可以看出，随着数据集规模的增大，交叉验证误差逐渐下降并逼近训练误差。从这张图中你还能得出一个重要启示：单纯增加训练样本数量通常无法解决高偏差问题，因为即使数据集不断扩大，训练误差本身依然保持相对平缓，并不会因此下降。

## 总结

在本次实验中，你亲手实践了如何解决学习算法中的高偏差与高方差问题。通过学会如何识别和诊断这些症状，你已经培养出了敏锐的直觉，知道在开发机器学习模型时下一步应当朝哪个方向努力。在接下来的课程中，你将更深入地探讨机器学习系统的开发全流程，并探索在实际项目落地时需要考量的更多关键因素。我们下节课见！
