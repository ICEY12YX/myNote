# 非计分实验 - 树集成模型 (Trees Ensemble)

在本笔记本中，你将：

 - 使用 Pandas 对数据集 (Dataset) 进行独热编码 (One-hot Encoding) 
 - 使用 scikit-learn 实现决策树 (Decision Tree) 、随机森林 (Random Forest) 以及 XGBoost 模型

首先，让我们导入将要使用的各个库。


```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

RANDOM_STATE = 55 ## We will pass it to every sklearn call so we ensure reproducibility
```

# 1. 简介

#### 数据集
- 本数据集来自 Kaggle：Heart Failure Prediction Dataset

#### 背景介绍
- 心血管疾病 (Cardiovascular diseases, CVDs) 是全球排名第一的死因，每年估计夺走 1790 万人的生命，占全球死亡总数的 31%。在心血管疾病导致的死亡中，五分之四死于心脏病发作和中风，且其中三分之一的死亡过早地发生在 70 岁以下人群中。心力衰竭 (Heart failure) 是由心血管疾病引发的一种常见病症。
- 患有心血管疾病或处于高心血管风险中的人群 (由于存在高血压、糖尿病、高脂血症或既往已确诊疾病等一种或多种危险因素) ，迫切需要早期检测与健康管理。  
- 该数据集包含 11 个可用于预测潜在心脏病的特征 (Features) 。
- 下面让我们训练一个机器学习 (Machine Learning) 模型，来辅助诊断这一疾病。

#### 属性说明 (Attribute Information)
- Age：患者年龄 [岁]
- Sex：患者性别 [M: 男性, F: 女性]
- ChestPainType：胸痛类型 [TA: 典型心绞痛, ATA: 非典型心绞痛, NAP: 非心绞痛性疼痛, ASY: 无症状]
- RestingBP：静息血压 [mm Hg]
- Cholesterol：血清胆固醇 [mm/dl]
- FastingBS：空腹血糖 [1: 若 FastingBS > 120 mg/dl, 0: 其他情况]
- RestingECG：静息心电图结果 [Normal: 正常, ST: 存在 ST-T 波异常 (T 波倒置和 / 或 ST 段抬高或压低 > 0.05 mV) , LVH: 根据 Estes 标准显示可能或明确的左心室肥大]
- MaxHR：达到的最大心率 [60 到 202 之间的数值]
- ExerciseAngina：运动诱发的心绞痛 [Y: 是, N: 否]
- Oldpeak：oldpeak = ST [测得的压低数值]
- ST_Slope：运动高峰期 ST 段的斜率 [Up: 向上倾斜, Flat: 平坦, Down: 向下倾斜]
- HeartDisease：输出目标类别 [1: 患有心脏病, 0: 正常]

现在让我们加载该数据集。从上面的属性说明中可以看到，以下变量：

- Sex
- ChestPainType
- RestingECG
- ExerciseAngina
- ST_Slope

属于*类别变量 (Categorical) *，因此我们必须对它们进行独热编码。


```python
# Load the dataset using pandas
df = pd.read_csv("heart.csv")
```


```python
df.head()
```

在将数据送入模型之前，我们需要先做一些数据工程处理。这里共有 5 个类别特征，下面我们将使用 Pandas 对它们进行独热编码。

## 2. 使用 Pandas 进行独热编码

在进行编码前，我们可以检查各个类别变量中包含多少个不同的取值。


```python
cat_variables = ['Sex',
'ChestPainType',
'RestingECG',
'ExerciseAngina',
'ST_Slope'
]
```

作为回顾，独热编码的目标是将一个具有 `n` 种可能取值的类别变量转换为 `n` 个二值变量 (Binary Variables) 。

Pandas 提供了一个内置的方法来对变量进行独热编码，即 `pd.get_dummies` 函数。该函数支持多个参数，但在这里我们只需要用到其中的几个：

 - data：要处理的 DataFrame
 - prefix：新列名的前缀列表，方便我们清楚地知道每个二值特征原先对应的是哪一个类别变量
 - columns：需要进行独热编码的列名列表。'prefix' 和 'columns' 的长度必须一致。
 
如果你想了解更多细节，可以随时输入 `help(pd.get_dummies)` 来阅读该函数的完整文档。


```python
# This will replace the columns with the one-hot encoded ones and keep the columns outside 'columns' argument as it is.
df = pd.get_dummies(data = df,
                         prefix = cat_variables,
                         columns = cat_variables)
```


```python
df.head()
```

接下来，让我们选出将作为模型输入特征的变量。
- 目标变量 (Target) 是 `HeartDisease`。
- 其余所有变量都是可用于预测目标变量 `HeartDisease` 的候选特征。


```python
features = [x for x in df.columns if x not in 'HeartDisease'] ## Removing our target variable
```

一开始我们有 11 个原始特征。让我们看看经过独热编码之后，现在一共有多少个特征变量。


```python
print(len(features))
```

# 3. 划分数据集

在本节中，我们将把数据集划分为训练集 (Train Dataset) 和验证集 (Validation Dataset) 。我们将使用来自 Scikit-learn 的 `train_test_split` 函数。让我们先查看一下它的参数说明。


```python
help(train_test_split)
```


```python
X_train, X_val, y_train, y_val = train_test_split(df[features], df['HeartDisease'], train_size = 0.8, random_state = RANDOM_STATE)

# We will keep the shuffle = True since our dataset has not any time dependency.
```


```python
print(f'train samples: {len(X_train)}')
print(f'validation samples: {len(X_val)}')
print(f'target proportion: {sum(y_train)/len(y_train):.4f}')
```

# 4. 构建模型

## 4.1 决策树 (Decision Tree)

在本节中，让我们使用之前学过的决策树算法，但这次直接调用 Scikit-learn 的实现。

Scikit-learn 的决策树对象中包含许多超参数 (Hyperparameters) 。在这里我们只会用到其中的一部分，并且在本次实验中不会进行专门的特征选择或系统性的网格超参数调优 (不过非常鼓励你在课后自己动手尝试并对比结果 😄 ) 。

我们在这里将要使用并重点探究的超参数包括：

 - min_samples_split：分裂一个内部节点 (Internal Node) 所需的最少样本数。
   - 选择较大的 `min_samples_split` 可以减少分裂次数，有助于减轻过拟合 (Overfitting) 。
 - max_depth：树的最大深度。
   - 选择较小的 `max_depth` 可以减少分裂次数，同样有助于减轻过拟合。


```python
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,
max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.
```


```python
accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
```

请留意增大 `min_samples_split` 的数值是如何减轻过拟合的：
- 将 `min_samples_split` 从 10 增大到 30，再从 30 增大到 50，尽管验证集准确率 (Validation Accuracy) 本身并未显著提升，但训练集准确率与验证集准确率之间的差距明显缩小了，这表明过拟合程度得到了有效降低。

下面让我们对 `max_depth` 做同样的实验。


```python
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
```

我们可以看到，总体而言，减小 `max_depth` 能够有效减轻过拟合：
- 将 `max_depth` 从 8 减小到 4，在显著降低训练集准确率的同时，提升了验证集准确率，使两者更加接近。
- 验证集准确率在树深度 `max_depth=4` 时达到最高点。
- 当 `max_depth` 小于 3 时，训练集和验证集准确率双双下降。此时决策树无法进行足够多次数的分裂来区分正负样本 (即模型对训练集发生了欠拟合 (Underfitting) ) 。
- 而当 `max_depth` 过大 (>= 5) 时，验证集准确率下降而训练集准确率飙升，表明模型对训练集发生了过拟合。

因此，我们可以为当前模型选定这两个超参数的最佳取值组合：
- `max_depth = 4`
- `min_samples_split = 50` 


```python
decision_tree_model = DecisionTreeClassifier(min_samples_split = 50,
                                             max_depth = 4,
                                             random_state = RANDOM_STATE).fit(X_train,y_train)
```


```python
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}")
print(f"Metrics validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_val),y_val):.4f}")
```

尽管整体准确率指标还不算特别高，但模型已经完全没有过拟合的迹象了。

## 4.2 随机森林 (Random Forest)

现在，让我们同样使用 Scikit-learn 的实现来尝试一下随机森林算法。
- 决策树模型中的所有超参数在随机森林算法中也同样存在，因为随机森林本质上就是由许多棵决策树组成的集成模型 (Ensemble) 。
- 随机森林还多出了一个额外的超参数，叫做 `n_estimators`，它代表构成随机森林的决策树数量。

请记住，在随机森林中，为了训练每一棵独立的树，我们既会随机抽取一部分特征子集，又会随机抽取一部分训练样本子集。
- 按照课堂上所讲的规则，如果特征总数为 $n$，我们会在每次分裂时随机选取其中的 $\sqrt{n}$ 个特征来训练单棵决策树。
- 请注意，你可以通过设置 `max_features` 参数来修改这一行为。

你还可以利用另一个参数 `n_jobs` 来加速训练任务：
- 由于每一棵树的拟合过程彼此完全独立，因此我们可以并行地同时训练多棵树。
- 将 `n_jobs` 设得更高，就会调用更多的 CPU 核心。不过请注意，如果将该数值设得过于接近你电脑 CPU 的最大核心数，可能会影响整机的响应性能，甚至导致画面卡顿。
- 修改这个参数不会对最终的模型结果产生任何影响，但能够显著缩短训练耗时。

下面我们将再次运行相同的实验脚本，并新加入对参数 `n_estimators` 的探索 (对比 10、50、100 和 500，其默认值为 100) 。


```python
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]  ## If the number is an integer, then it is the actual quantity of samples,
                                             ## If it is a float, then it is the percentage of the dataset
max_depth_list = [2, 4, 8, 16, 32, 64, None]
n_estimators_list = [10,50,100,500]
```


```python
accuracy_list_train = []
accuracy_list_val = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list) 
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
```

请留意，尽管在 `min_samples_split = 2` 和 `min_samples_split = 10` 处达到的验证集准确率几乎相同，但在后者处，训练集与验证集之间的准确率差距明显缩小了，说明过拟合程度更低。


```python
accuracy_list_train = []
accuracy_list_val = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
```


```python
accuracy_list_train = []
accuracy_list_val = []
for n_estimators in n_estimators_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_val = model.predict(X_val) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_val = accuracy_score(predictions_val,y_val)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_val.append(accuracy_val)

plt.title('Train x Validation metrics')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_val)
plt.legend(['Train','Validation'])
```

那么，让我们使用以下参数组合来拟合一个随机森林模型：

 - max_depth: 16
 - min_samples_split: 10
 - n_estimators: 100


```python
random_forest_model = RandomForestClassifier(n_estimators = 100,
                                             max_depth = 16, 
                                             min_samples_split = 10).fit(X_train,y_train)
```


```python
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_val),y_val):.4f}")
```

请注意，在上面的实验中，我们在搜寻某一个超参数的最佳值时，把其余超参数都保持在了默认状态。
- 而在最理想的情况下，我们其实希望遍历并检验所有待调优超参数的每一种可能取值组合。
- 举例来说，如果我们有 3 个超参数，每个超参数有 4 个候选值想要尝试，那么一共就有 4 x 4 x 4 = 64 种组合需要测试。
- 而当我们每次只修改一个超参数、保持其余超参数不变时，我们实际上只尝试了 4 + 4 + 4 = 12 种情况。
- 如果想要自动遍历测试所有可能的组合，我们可以使用 sklearn 提供的 GridSearchCV (网格搜索交叉验证) 。GridSearchCV 设有一个 `refit` 参数，可以在找到最佳参数组合后自动使用该组合重新拟合整个模型，省去了我们手动编写循环的麻烦。关于 GridSearchCV 的更多信息，请参考其官方文档。

## 4.3 XGBoost

接下来登场的是梯度提升 (Gradient Boosting) 模型，即大名鼎鼎的 XGBoost。提升 (Boosting) 方法同样会训练多棵决策树，但与随机森林中各棵树彼此独立、互不相关不同，XGBoost 中的树是一棵接一棵顺序拟合出来的，每一棵新树都致力于最小化前面所有树累积下来的残差错误。

该模型拥有与普通决策树相同的参数，此外还增加了一个学习率 (Learning Rate) 参数：
- 学习率决定了 XGBoost 内部在使用梯度下降 (Gradient Descent) 方法最小化每一步训练误差时所迈出的步长大小。

XGBoost 有一个非常实用的特性：在调用 `fit` 进行拟合时，它可以接收一个形如 `(X_val,y_val)` 的评估数据集 (Evaluation Dataset) 。
- 在每一次迭代 (即每新增一棵树) 时，它都会在这个评估数据集上测量代价 (Cost) 或评估指标。
- 一旦该代价 (或评估指标) 在连续若干轮迭代 (由参数 `early_stopping_rounds` 指定) 中不再继续下降，训练就会自动提前终止 (Early Stopping) 。
- 迭代次数越多，生成的基学习器 (树) 就越多，而过多的树很容易导致过拟合。  
- 通过在验证集指标不再改善时及时喊停，我们可以有效限制所创建的决策树数量，从而大幅减轻过拟合。

首先，让我们从训练集中再划分出一小部分子集作为早停评估集 (注意：这里绝不能偷用测试集 / 验证集 `X_val`) 。


```python
n = int(len(X_train)*0.8) ## Let's use 80% to train and 20% to eval
```


```python
X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]
```

有了早停机制后，我们就可以放心地将最大树数量 (`n_estimators`) 设得很大，因为只要代价函数停止下降，模型就会自动停下来。

请留意 `.fit()` 方法中的几个关键参数：
- `eval_set = [(X_train_eval,y_train_eval)]`：这里必须向 `eval_set` 传入一个列表，因为你可以同时传入多个不同的评估集元组。
- `early_stopping_rounds`：当评估集上的指标不再改善时，该参数可以帮助模型提前终止训练。这里我们将其设置为 10。
  - 模型会持续记录下迄今为止表现最好 (即评估误差最低) 的那一轮迭代。例如，假设到目前为止第 16 轮的评估误差最低。
  - 随后的每一轮迭代都会将其评估指标与这个历史最佳指标进行比较。如果模型连续运行了 10 轮，却没有任何一轮的指标优于那个历史最佳值，模型就会立刻停止训练。
  - 需要注意的是，训练终止时模型保留的是最后一轮的状态，而不是历史最佳那一轮的状态。例如，如果模型在第 26 轮停止，而最佳轮次是第 16 轮，那么模型最终处于第 26 轮的训练状态，而非第 16 轮 (但可以通过最佳迭代属性进行调用或设置) 。


```python
xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1,verbosity = 1, random_state = RANDOM_STATE)
xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)], early_stopping_rounds = 10)
```

可以看到，尽管我们在初始化模型时允许它最多构建 500 棵树，但算法实际上只拟合了 26 棵树 (即经过了 26 轮训练，索引从 0 到 25) 就停下来了。

为了弄清楚为什么会停在这里，让我们找出训练过程中表现最好 (即评估损失最低) 的那一轮。你既可以直接查看上方输出日志中的验证集对数损失 (Log Loss) 指标，也可以直接查看模型的 `.best_iteration` 属性：


```python
xgb_model.best_iteration
```

表现最好的一轮训练是第 16 轮 (`best_iteration = 16`) ，此时的对数损失低至 0.3948。  
- 在此之后的连续 10 轮训练中 (从第 17 轮到第 26 轮) ，对数损失全都高于这个值。
- 由于我们将 `early_stopping_rounds` 设为了 10，因此到了连续第 10 轮对数损失仍未刷新最佳纪录时，训练便自动终止了。
- 你可以尝试修改 `early_stopping_rounds` 的数值来亲自验证这一点。例如，如果你将它设为 20，模型就会在第 36 轮 (16 + 20) 停止训练。


```python
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_val),y_val):.4f}")
```

在这个例子中，随机森林和 XGBoost 取得了旗鼓相当的出色表现 (验证集准确率几乎一致，且远高于单棵决策树) 。  

恭喜！你已经学会了如何使用来自 scikit-learn 库的决策树、随机森林以及强大的 XGBoost 模型！
