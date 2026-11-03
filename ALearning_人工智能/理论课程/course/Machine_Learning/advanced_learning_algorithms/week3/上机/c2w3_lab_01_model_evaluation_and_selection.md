# 选修实验：模型评估与选择 (Model Evaluation and Selection)

在将机器学习 (Machine Learning) 应用于现实世界的问题时，量化学习算法的性能并对比不同的模型是非常常见的任务。在本次实验中，你将运用课堂上分享的实用技巧来练习这些操作。具体来说，你将：

* 将数据集 (Datasets) 划分为训练集 (Training Set) 、交叉验证集 (Cross Validation Set) 和测试集 (Test Set) 
* 评估回归 (Regression) 与分类 (Classification) 模型
* 添加多项式特征 (Polynomial Features) 以提升线性回归 (Linear Regression) 模型的性能
* 对比几种不同的神经网络架构 (Neural Network Architectures) 

本次实验还将帮助你提前熟悉本周编程作业中将要看到的代码。让我们开始吧！

## 导入软件包与实验环境配置

首先，你将导入完成本次实验任务所需的软件包。我们还加入了一些命令，通过降低输出详细程度并屏蔽非关键警告，让后续的输出结果更加清晰易读。


```python
# for array computations and loading data
import numpy as np

# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# for building and training neural networks
import tensorflow as tf

# custom functions
import utils

# reduce display precision on numpy arrays
np.set_printoptions(precision=2)

# suppress warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
```

## 回归问题

首先，你的任务是为一个回归问题开发模型。下方提供的数据集包含 50 个样本，每个样本由一个输入特征 (Feature) `x` 及其对应的目标值 (Target) `y` 组成。


```python
# Load the dataset from the text file
data = np.loadtxt('./data/data_w3_ex1.csv', delimiter=',')

# Split the inputs and outputs into separate arrays
x = data[:,0]
y = data[:,1]

# Convert 1-D arrays into 2-D because the commands later will require it
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

print(f"the shape of the inputs x is: {x.shape}")
print(f"the shape of the targets y is: {y.shape}")
```

你可以把数据集绘制出来，直观地了解目标值是如何随输入特征变化的。如果你想查看具体的绘图代码，可以在本笔记本外部的 `utils.py` 文件中找到 `plot_dataset()` 函数。


```python
# Plot the entire dataset
utils.plot_dataset(x=x, y=y, title="input vs. target")
```

## 将数据集划分为训练集、交叉验证集和测试集

在之前的实验中，你可能使用了整个数据集来训练模型。然而在实际应用中，最好的做法是留出一部分数据，专门用来衡量模型对新样本的泛化 (Generalize) 能力。这样你就能清楚地知道模型是否在训练集上发生了过拟合 (Overfit) 。

正如课堂上所提到的，通常我们会将数据划分为三个部分：

* ***训练集 (training set) *** - 用于训练模型
* ***交叉验证集 (cross validation set，也称为验证集 (validation set) 、开发集 (development set / dev set) ) *** - 用于评估你正在对比挑选的不同模型配置。例如，你可以利用它来决定应该向数据集中添加多少阶的多项式特征。
* ***测试集 (test set) *** - 用于客观评估你最终选定的模型在新样本上的表现。在模型仍在开发和调优阶段时，绝不能使用测试集来做决策。

Scikit-learn 提供了一个 `train_test_split` 函数，可以将数据划分为上述几个部分。在下面的代码单元格中，你将把整个数据集按照 60% 训练集、20% 交叉验证集和 20% 测试集的比例进行划分。


```python
# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Delete temporary variables
del x_, y_

print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")
```

你可以在下方再次绘制数据集，看看哪些数据点被分到了训练集、交叉验证集或测试集中。


```python
utils.plot_train_cv_test(x_train, y_train, x_cv, y_cv, x_test, y_test, title="input vs. target")
```

## 拟合线性模型

既然已经划分好了数据，你可以首先尝试拟合一个线性模型。接下来你将在下面的几个小节中完成这一步。

### 特征缩放 (Feature scaling)

在本专项课程的前一门课中，你已经了解到进行特征缩放通常是一个好主意，它能帮助模型更快地收敛 (Converge) 。当你的输入特征取值范围相差悬殊时，这一点尤为重要。在本次实验的后续部分，你将添加多项式项，届时各个输入特征的取值范围确实会产生巨大差异。例如，$x$ 的范围大约在 1600 到 3600 之间，而 $x^2$ 的范围则会达到 256 万到 1296 万之间。

在第一个模型中你只会用到 $x$，但现在先练习一下特征缩放很有好处，方便后续直接应用。为此，你将使用 scikit-learn 中的 `StandardScaler` 类。它会计算输入数据的 z-score (标准分数) 。作为回顾，z-score 的计算公式如下：

$$ z = \frac{x - \mu}{\sigma} $$

其中，$\mu$ 是特征值的均值 (Mean) ，$\sigma$ 是标准差 (Standard Deviation) 。下面的代码展示了如何使用该类来预处理训练集。你可以再次绘制结果图像，检查它是否依然保持了与之前相同的分布形态。在新图像中，`x` 的取值范围应该明显缩小了。


```python
# Initialize the class
scaler_linear = StandardScaler()

# Compute the mean and standard deviation of the training set then transform it
X_train_scaled = scaler_linear.fit_transform(x_train)

print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}")
print(f"Computed standard deviation of the training set: {scaler_linear.scale_.squeeze():.2f}")

# Plot the results
utils.plot_dataset(x=X_train_scaled, y=y_train, title="scaled input vs. target")
```

### 训练模型

接下来，你将创建并训练一个回归模型。在本次实验中，你将使用 LinearRegression 类，但也请留意，scikit-learn 中还有其他经典的线性回归器可供选择。


```python
# Initialize the class
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train_scaled, y_train )
```

### 评估模型

为了评估模型的性能，你需要分别测量模型在训练集和交叉验证集上的误差。对于训练误差，回想一下计算均方误差 (Mean Squared Error, MSE) 的公式：

$$J_{train}(\vec{w}, b) = \frac{1}{2m_{train}}\left[\sum_{i=1}^{m_{train}}(f_{\vec{w},b}(\vec{x}_{train}^{(i)}) - y_{train}^{(i)})^2\right]$$

Scikit-learn 也提供了一个内置的 `mean_squared_error()` 函数供你调用。不过请注意，根据官方文档说明，scikit-learn 的实现只除以了样本数量 `m`，而没有除以 `2*m`。正如本专项课程第 1 门课 (代价函数 (Cost Function) 章节) 中所提到的，除以 `2m` 是我们遵循的课程惯例，但无论是否包含这个系数 2，背后的优化计算逻辑都是相通的。因此，为了与上面的公式保持一致，你可以调用 scikit-learn 的函数后再除以 2，如下所示。我们还同时提供了一个使用 for 循环的实现，方便你验证两者的计算结果完全相等。

另外还有一点需要注意：既然你是使用缩放后的数值 (即 z-score) 来训练模型的，那么在预测时你也应当向模型传入缩放后的训练集，而不是原始数值。


```python
# Feed the scaled training set and get the predictions
yhat = linear_model.predict(X_train_scaled)

# Use scikit-learn's utility function and divide by 2
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

# for-loop implementation
total_squared_error = 0

for i in range(len(yhat)):
    squared_error_i  = (yhat[i] - y_train[i])**2
    total_squared_error += squared_error_i                                              

mse = total_squared_error / (2*len(yhat))

print(f"training MSE (for-loop implementation): {mse.squeeze()}")
```

接着，你可以使用基本相同的公式来计算交叉验证集的均方误差：

$$J_{cv}(\vec{w}, b) = \frac{1}{2m_{cv}}\left[\sum_{i=1}^{m_{cv}}(f_{\vec{w},b}(\vec{x}_{cv}^{(i)}) - y_{cv}^{(i)})^2\right]$$

与训练集一样，你也需要对交叉验证集进行特征缩放。在使用 z-score 时，有一个*至关重要*的注意事项：在对交叉验证集进行缩放时，你必须使用**训练集**的均值和标准差。这样做是为了确保输入特征按照模型所期望的同一标准进行变换。我们可以通过下面这个场景来直观理解其中的道理：

* 假设你的训练集中有一个等于 `500` 的输入特征，利用训练集的 z-score 将其缩放后变成了 `0.5`。
* 训练完成后，你的模型能够准确地将这个缩放后的输入 `x=0.5` 映射到目标输出 `y=300`。
* 现在假设你部署了这个模型，某位用户向它输入了一个同样等于 `500` 的新样本。
* 如果你使用任何其他的均值和标准差来计算这个输入样本的 z-score，那么它缩放后的结果可能就不再是 `0.5` 了，而你的模型十有八九会给出错误的预测 (即不等于 `y=300`) 。

在下面的代码中，你将继续使用刚才创建的那个 `StandardScaler` 对象来缩放交叉验证集，但这次只调用它的 `transform()` 方法，而不是 `fit_transform()`。


```python
# Scale the cross validation set using the mean and standard deviation of the training set
X_cv_scaled = scaler_linear.transform(x_cv)

print(f"Mean used to scale the CV set: {scaler_linear.mean_.squeeze():.2f}")
print(f"Standard deviation used to scale the CV set: {scaler_linear.scale_.squeeze():.2f}")

# Feed the scaled cross validation set
yhat = linear_model.predict(X_cv_scaled)

# Use scikit-learn's utility function and divide by 2
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")
```

## 添加多项式特征

从前面的图表中你可能已经注意到，当 `x` 较小时，目标值 `y` 上升得比较陡峭；而随着 `x` 增大，目标值 `y` 似乎逐渐趋于平缓。因此，一条简单的直线可能并不是最佳选择。既然你已经拿到了线性模型在训练集和交叉验证集上的 MSE 数值，接下来不妨尝试添加多项式特征，看看能否获得更好的性能表现。整体代码结构与之前大体相同，只是多出了几个额外的预处理步骤。让我们在下面一看究竟。

### 创建额外特征

首先，你需要从训练集生成多项式特征。下面的代码演示了如何使用 `PolynomialFeatures` 类来完成这一步。它会创建一个包含输入 `x` 的平方值的新输入特征 (即阶数 degree=2) 。


```python
# Instantiate the class to make polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)

# Compute the number of features and transform the training set
X_train_mapped = poly.fit_transform(x_train)

# Preview the first 5 elements of the new training set. Left column is `x` and right column is `x^2`
# Note: The `e+<number>` in the output denotes how many places the decimal point should 
# be moved. For example, `3.24e+03` is equal to `3240`
print(X_train_mapped[:5])
```

随后，你将像之前一样对输入特征进行缩放，以收窄数值范围。


```python
# Instantiate the class
scaler_poly = StandardScaler()

# Compute the mean and standard deviation of the training set then transform it
X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

# Preview the first 5 elements of the scaled training set.
print(X_train_mapped_scaled[:5])
```

接着你就可以开始训练模型了。训练完成后，再用交叉验证集来衡量模型的表现。像之前一样，请务必对交叉验证集执行与训练集完全相同的变换流程：先添加相同阶数的多项式特征，然后再利用训练集的统计量对其数值范围进行缩放。


```python
# Initialize the class
model = LinearRegression()

# Train the model
model.fit(X_train_mapped_scaled, y_train )

# Compute the training MSE
yhat = model.predict(X_train_mapped_scaled)
print(f"Training MSE: {mean_squared_error(y_train, yhat) / 2}")

# Add the polynomial features to the cross validation set
X_cv_mapped = poly.transform(x_cv)

# Scale the cross validation set using the mean and standard deviation of the training set
X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

# Compute the cross validation MSE
yhat = model.predict(X_cv_mapped_scaled)
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")
```

你会发现，在加入了二阶多项式项之后，训练集和交叉验证集上的 MSE 都得到了显著改善。你可能会想进一步引入更高阶的多项式项，看看哪一个阶数能带来最佳性能。正如课堂上所展示的，你可以构建如下所示的 10 个不同阶数的模型：

&lt;img src=&#39;images/C2_W3_poly.png&#39; width=50%&gt;

你可以编写一个循环，将前面几个代码单元格中的所有步骤都囊括进来。下面这段实现代码依次添加了最高到 10 阶 (`degree=10`) 的多项式特征，并在最后将结果绘制成图表，方便你直观地对比各个模型的表现。


```python
# Initialize lists to save the errors, models, and feature transforms
train_mses = []
cv_mses = []
models = []
polys = []
scalers = []

# Loop over 10 times. Each adding one more degree of polynomial higher than the last.
for degree in range(1,11):
    
    # Add polynomial features to the training set
    poly = PolynomialFeatures(degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    polys.append(poly)
    
    # Scale the training set
    scaler_poly = StandardScaler()
    X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)
    scalers.append(scaler_poly)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train_mapped_scaled, y_train )
    models.append(model)
    
    # Compute the training MSE
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    train_mses.append(train_mse)
    
    # Add polynomial features and scale the cross validation set
    X_cv_mapped = poly.transform(x_cv)
    X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)
    
    # Compute the cross validation MSE
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    cv_mses.append(cv_mse)
    
# Plot the results
degrees=range(1,11)
utils.plot_train_cv_mses(degrees, train_mses, cv_mses, title="degree of polynomial vs. train and CV MSEs")
```

### 选择最佳模型

在选择模型时，你希望挑选一个在训练集和交叉验证集上都表现良好的模型。这意味着它既能够从训练集中学习到潜在规律，又不会发生过拟合。如果你使用的是本实验的默认设置，你会注意到从 1 阶 (`degree=1`) 到 2 阶 (`degree=2`) 模型，交叉验证误差出现了大幅下降；随后一直到 5 阶 (`degree=5`) 都保持着相对平缓的趋势；然而在此之后，随着你继续添加更高阶的多项式特征，交叉验证误差总体上反而越来越糟。基于这些观察，你可以决定选用 `cv_mse` 最低的那个模型，作为最适合你当前应用场景的模型。


```python
# Get the model with the lowest CV MSE (add 1 because list indices start at 0)
# This also corresponds to the degree of the polynomial added
degree = np.argmin(cv_mses) + 1
print(f"Lowest CV MSE is found in the model with degree={degree}")
```

选定模型后，你就可以通过计算测试集的 MSE 来公布模型的泛化误差 (Generalization Error) 了。像往常一样，你必须对测试集数据执行与训练集和交叉验证集完全相同的特征变换流程。


```python
# Add polynomial features to the test set
X_test_mapped = polys[degree-1].transform(x_test)

# Scale the test set
X_test_mapped_scaled = scalers[degree-1].transform(X_test_mapped)

# Compute the test MSE
yhat = models[degree-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Training MSE: {train_mses[degree-1]:.2f}")
print(f"Cross Validation MSE: {cv_mses[degree-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")
```

## 神经网络 (Neural Networks)

这套模型选择流程同样适用于在不同的神经网络架构之间进行挑选。在本节中，你将创建如下所示的几个神经网络模型，并将它们应用于上面的同一个回归任务中。

&lt;img src=&#39;images/C2_W3_NN_Arch.png&#39; width=40%&gt;

### 准备数据

你将继续使用上一节生成的训练集、交叉验证集和测试集。从本课程前面的讲座中你已经了解到，神经网络本身就具备学习非线性关系 (Non-linear Relationships) 的能力，因此你可以选择跳过添加多项式特征这一步。不过下面的代码中依然保留了这一步骤，以备你稍后想要尝试并观察它对结果会产生什么影响。这里默认的阶数 `degree` 被设为 `1`，表示直接使用原始的 `x_train`、`x_cv` 和 `x_test` (即不添加任何额外的多项式特征) 。


```python
# Add polynomial features
degree = 1
poly = PolynomialFeatures(degree, include_bias=False)
X_train_mapped = poly.fit_transform(x_train)
X_cv_mapped = poly.transform(x_cv)
X_test_mapped = poly.transform(x_test)
```

接下来，你将对输入特征进行缩放，以帮助梯度下降 (Gradient Descent) 更快地收敛。请再次留意，在对交叉验证集和测试集进行缩放时，我们只调用了 `transform()` 而不是 `fit_transform()`，以确保使用的是从训练集计算得出的均值和标准差。


```python
# Scale the features using the z-score
scaler = StandardScaler()
X_train_mapped_scaled = scaler.fit_transform(X_train_mapped)
X_cv_mapped_scaled = scaler.transform(X_cv_mapped)
X_test_mapped_scaled = scaler.transform(X_test_mapped)
```

### 构建并训练模型

接下来，你将创建前面展示的那几种神经网络架构。如果你想查看或修改具体实现，可以在 `utils.py` 文件中的 `build_models()` 函数里找到相关代码。你将在下面的循环中调用该函数并依次训练这些模型，同时记录下每个模型在训练集和交叉验证集上的误差。


```python
# Initialize lists that will contain the errors for each model
nn_train_mses = []
nn_cv_mses = []

# Build the models
nn_models = utils.build_models()

# Loop over the the models
for model in nn_models:
    
    # Setup the loss and optimizer
    model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    )

    print(f"Training {model.name}...")
    
    # Train the model
    model.fit(
        X_train_mapped_scaled, y_train,
        epochs=300,
        verbose=0
    )
    
    print("Done!\n")

    
    # Record the training MSEs
    yhat = model.predict(X_train_mapped_scaled)
    train_mse = mean_squared_error(y_train, yhat) / 2
    nn_train_mses.append(train_mse)
    
    # Record the cross validation MSEs 
    yhat = model.predict(X_cv_mapped_scaled)
    cv_mse = mean_squared_error(y_cv, yhat) / 2
    nn_cv_mses.append(cv_mse)

    
# print results
print("RESULTS:")
for model_num in range(len(nn_train_mses)):
    print(
        f"Model {model_num+1}: Training MSE: {nn_train_mses[model_num]:.2f}, " +
        f"CV MSE: {nn_cv_mses[model_num]:.2f}"
        )
```

根据记录下来的误差结果，你就可以决定哪一个模型最适合你的应用了。不妨看看上面的输出结果，看看你是否认同下方代码所选定的 `model_num`。最后，你将计算测试误差，以评估该模型对新样本的泛化能力。


```python
# Select the model with the lowest CV MSE
model_num = 3

# Compute the test MSE
yhat = nn_models[model_num-1].predict(X_test_mapped_scaled)
test_mse = mean_squared_error(y_test, yhat) / 2

print(f"Selected Model: {model_num}")
print(f"Training MSE: {nn_train_mses[model_num-1]:.2f}")
print(f"Cross Validation MSE: {nn_cv_mses[model_num-1]:.2f}")
print(f"Test MSE: {test_mse:.2f}")
```

## 分类问题

在本次实验的最后一部分，你将在一个分类任务上练习模型的评估与选择。整体流程与回归问题非常相似，最主要的区别在于误差的计算方式。你将在接下来的几个小节中看到具体操作。

### 加载数据集

首先，你将加载一个用于二分类 (Binary Classification) 任务的数据集。它包含 200 个样本，每个样本有两个输入特征 (`x1` 和 `x2`) ，以及一个取值为 `0` 或 `1` 的目标标签 `y`。


```python
# Load the dataset from a text file
data = np.loadtxt('./data/data_w3_ex2.csv', delimiter=',')

# Split the inputs and outputs into separate arrays
x_bc = data[:,:-1]
y_bc = data[:,-1]

# Convert y into 2-D because the commands later will require it (x is already 2-D)
y_bc = np.expand_dims(y_bc, axis=1)

print(f"the shape of the inputs x is: {x_bc.shape}")
print(f"the shape of the targets y is: {y_bc.shape}")
```

你可以把数据集绘制出来，观察正负样本在空间中是如何分布的。


```python
utils.plot_bc_dataset(x=x_bc, y=y_bc, title="x1 vs. x2")
```

### 划分并准备数据集

接下来，你将生成训练集、交叉验证集和测试集，依然采用与之前相同的 60/20/20 划分比例，并按照上一节的方法对特征进行缩放。


```python
from sklearn.model_selection import train_test_split

# Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables.
x_bc_train, x_, y_bc_train, y_ = train_test_split(x_bc, y_bc, test_size=0.40, random_state=1)

# Split the 40% subset above into two: one half for cross validation and the other for the test set
x_bc_cv, x_bc_test, y_bc_cv, y_bc_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

# Delete temporary variables
del x_, y_

print(f"the shape of the training set (input) is: {x_bc_train.shape}")
print(f"the shape of the training set (target) is: {y_bc_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_bc_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_bc_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_bc_test.shape}")
print(f"the shape of the test set (target) is: {y_bc_test.shape}")
```


```python
# Scale the features

# Initialize the class
scaler_linear = StandardScaler()

# Compute the mean and standard deviation of the training set then transform it
x_bc_train_scaled = scaler_linear.fit_transform(x_bc_train)
x_bc_cv_scaled = scaler_linear.transform(x_bc_cv)
x_bc_test_scaled = scaler_linear.transform(x_bc_test)
```

### 评估分类模型的误差

在前面关于回归模型的章节中，你使用了均方误差来衡量模型的表现。而对于分类模型，你可以通过计算模型预测错误 (Misclassified) 的样本占总样本的比例，来得到一个类似的评估指标。例如，如果模型在 5 个样本中预测错了 2 个，那么你报告的分类误差就是 `40%` 或 `0.4`。下面的代码分别使用 for 循环以及 NumPy 的 `mean()` 函数演示了这一计算过程。


```python
# Sample model output
probabilities = np.array([0.2, 0.6, 0.7, 0.3, 0.8])

# Apply a threshold to the model output. If greater than 0.5, set to 1. Else 0.
predictions = np.where(probabilities >= 0.5, 1, 0)

# Ground truth labels
ground_truth = np.array([1, 1, 1, 1, 1])

# Initialize counter for misclassified data
misclassified = 0

# Get number of predictions
num_predictions = len(predictions)

# Loop over each prediction
for i in range(num_predictions):
    
    # Check if it matches the ground truth
    if predictions[i] != ground_truth[i]:
        
        # Add one to the counter if the prediction is wrong
        misclassified += 1

# Compute the fraction of the data that the model misclassified
fraction_error = misclassified/num_predictions

print(f"probabilities: {probabilities}")
print(f"predictions with threshold=0.5: {predictions}")
print(f"targets: {ground_truth}")
print(f"fraction of misclassified data (for-loop): {fraction_error}")
print(f"fraction of misclassified data (with np.mean()): {np.mean(predictions != ground_truth)}")
```

### 构建并训练模型

你将使用与上一节相同的那几种神经网络架构，因此可以再次调用 `build_models()` 函数来创建这些模型的新实例。

在这里，你将遵循上周介绍的推荐做法：在输出层使用线性 (`linear`) 激活函数 (Activation Function)  (而不是 `sigmoid`) ，并在声明模型的损失函数 (Loss Function) 时设置 `from_logits=True`。由于这是一个二分类问题，你将使用二元交叉熵损失 (Binary Crossentropy Loss) 。

训练完成后，你将使用 sigmoid 函数把模型的原始输出转换为概率 (Probabilities) 。在此基础上，设定一个分类阈值 (Threshold) ，即可分别求出训练集和交叉验证集上预测错误的样本比例。

你可以在下面的代码单元格中看到上述全部过程。


```python
# Initialize lists that will contain the errors for each model
nn_train_error = []
nn_cv_error = []

# Build the models
models_bc = utils.build_models()

# Loop over each model
for model in models_bc:
    
    # Setup the loss and optimizer
    model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    )

    print(f"Training {model.name}...")

    # Train the model
    model.fit(
        x_bc_train_scaled, y_bc_train,
        epochs=200,
        verbose=0
    )
    
    print("Done!\n")
    
    # Set the threshold for classification
    threshold = 0.5
    
    # Record the fraction of misclassified examples for the training set
    yhat = model.predict(x_bc_train_scaled)
    yhat = tf.math.sigmoid(yhat)
    yhat = np.where(yhat >= threshold, 1, 0)
    train_error = np.mean(yhat != y_bc_train)
    nn_train_error.append(train_error)

    # Record the fraction of misclassified examples for the cross validation set
    yhat = model.predict(x_bc_cv_scaled)
    yhat = tf.math.sigmoid(yhat)
    yhat = np.where(yhat >= threshold, 1, 0)
    cv_error = np.mean(yhat != y_bc_cv)
    nn_cv_error.append(cv_error)

# Print the result
for model_num in range(len(nn_train_error)):
    print(
        f"Model {model_num+1}: Training Set Classification Error: {nn_train_error[model_num]:.5f}, " +
        f"CV Set Classification Error: {nn_cv_error[model_num]:.5f}"
        )
```

根据上面的输出结果，你就可以挑选出表现最好的模型了。如果有多个模型在交叉验证集误差上打成平手，你可以引入其他标准来做出抉择。例如，你可以选择训练误差较低的那一个；而在工程实践中，更常见的做法是选择规模较小的那个模型，因为它能节省计算资源。在我们的例子中，Model 1 规模最小，而 Model 3 规模最大。

最后，你可以计算测试集误差，以此作为模型泛化误差的最终评估报告。


```python
# Select the model with the lowest error
model_num = 3

# Compute the test error
yhat = models_bc[model_num-1].predict(x_bc_test_scaled)
yhat = tf.math.sigmoid(yhat)
yhat = np.where(yhat >= threshold, 1, 0)
nn_test_error = np.mean(yhat != y_bc_test)

print(f"Selected Model: {model_num}")
print(f"Training Set Classification Error: {nn_train_error[model_num-1]:.4f}")
print(f"CV Set Classification Error: {nn_cv_error[model_num-1]:.4f}")
print(f"Test Set Classification Error: {nn_test_error:.4f}")
```

## 总结

在本次实验中，你练习了如何评估模型的性能，并在不同的模型配置之间做出选择。你将数据集划分为训练集、交叉验证集和测试集，并亲身体会了它们在机器学习应用中各自扮演的角色。在接下来的课程章节中，你将学到更多通过诊断偏差 (Bias) 与方差 (Variance) 来改进模型的实用技巧。继续加油！
