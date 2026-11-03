# 实践实验：决策树 (Decision Trees)

在本次练习中，你将从零开始实现一棵决策树，并将其应用于判断野生蘑菇是可食用还是有毒的分类任务中。

# 目录大纲
-  1 - 软件包 
-  2 - 问题描述
-  3 - 数据集
  -  3.1 独热编码后的数据集
-  4 - 决策树核心概念回顾
  -  4.1 计算熵
    -  练习 1
  -  4.2 划分数据集
    -  练习 2
  -  4.3 计算信息增益
    -  练习 3
  -  4.4 获取最佳分裂特征
    -  练习 4
-  5 - 构建决策树


_**注意：** 为了防止自动评分器 (Autograder) 出错，请勿编辑或删除本笔记本中的非评分单元格。同时也请不要添加任何新的单元格。
**当你通过本次作业后**，如果想对任何非评分代码进行修改和实验，可以按照本笔记本底部的说明进行操作。_

&lt;a name=&quot;1&quot;&gt;&lt;/a&gt;
## 1 - 软件包 

首先，让我们运行下面的代码单元格，导入本次作业所需的所有软件包。
- numpy 是使用 Python 进行矩阵 (Matrices) 与科学计算的基础软件包。
- matplotlib 是 Python 中著名的绘图库。
- ``utils.py`` 包含了本次作业所需的辅助函数，你不需要修改该文件中的任何代码。



```python
import numpy as np
import matplotlib.pyplot as plt
from public_tests import *
from utils import *

%matplotlib inline
```

&lt;a name=&quot;2&quot;&gt;&lt;/a&gt;
## 2 - 问题描述

假设你正在创办一家种植并销售野生蘑菇的公司。
- 由于并非所有蘑菇都可以安全食用，你希望能够根据一朵蘑菇的外观物理特征，准确判断它究竟是可食用的还是有毒的。
- 你手头已经有了一些现成的历史数据可用于完成这项任务。

你能利用这些数据，帮助自己识别出哪些蘑菇可以安全出售吗？

注：本实验所使用的数据集仅作教学演示用途，绝不可作为现实生活中鉴别可食用野生蘑菇的依据。



&lt;a name=&quot;3&quot;&gt;&lt;/a&gt;
## 3 - 数据集 (Dataset)

首先，你需要加载本次任务的数据集。你收集到的数据集如下所示：

|                                                     | 菌盖颜色 (Cap Color) | 菌柄形状 (Stalk Shape) | 是否独生 (Solitary) | 是否可食用 (Edible) |
|:---------------------------------------------------:|:---------:|:-----------:|:--------:|:------:|
| &lt;img src=&quot;images/0.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 棕色 (Brown) | 上宽下窄 (Tapering) | 是 (Yes) |    1   |
| &lt;img src=&quot;images/1.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 棕色 (Brown) | 上窄下宽 (Enlarging) | 是 (Yes) |    1   |
| &lt;img src=&quot;images/2.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 棕色 (Brown) | 上窄下宽 (Enlarging) | 否 (No) |    0   |
| &lt;img src=&quot;images/3.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 棕色 (Brown) | 上窄下宽 (Enlarging) | 否 (No) |    0   |
| &lt;img src=&quot;images/4.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 棕色 (Brown) | 上宽下窄 (Tapering) | 是 (Yes) |    1   |
| &lt;img src=&quot;images/5.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 红色 (Red) | 上宽下窄 (Tapering) | 是 (Yes) |    0   |
| &lt;img src=&quot;images/6.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 红色 (Red) | 上窄下宽 (Enlarging) | 否 (No) |    0   |
| &lt;img src=&quot;images/7.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 棕色 (Brown) | 上窄下宽 (Enlarging) | 是 (Yes) |    1   |
| &lt;img src=&quot;images/8.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 红色 (Red) | 上宽下窄 (Tapering) | 否 (No) |    1   |
| &lt;img src=&quot;images/9.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 棕色 (Brown) | 上窄下宽 (Enlarging) | 否 (No) |    0   |


- 你一共有 10 个蘑菇样本。对于每个样本，你记录了：
    - 三个特征 (Features) 
        - 菌盖颜色 (`Brown` 棕色 或 `Red` 红色) ，
        - 菌柄形状 (`Tapering (形如 \/)` 上宽下窄 或 `Enlarging (形如 /\)` 上窄下宽) ，以及
        - 是否独生 (`Yes` 是 或 `No` 否) 
    - 标签 (Label) 
        - 是否可食用 (`1` 表示可食用，`0` 表示有毒) 

&lt;a name=&quot;3.1&quot;&gt;&lt;/a&gt;
### 3.1 独热编码 (One-hot Encoded) 后的数据集
为了方便代码实现，我们已经对这些特征进行了独热编码 (即将它们转换成了取值为 0 或 1 的二值特征) ：

|                                                    | 棕色菌盖 (Brown Cap) | 上宽下窄菌柄 (Tapering Stalk Shape) | 独生 (Solitary) | 可食用 (Edible) |
|:--------------------------------------------------:|:---------:|:--------------------:|:--------:|:------:|
| &lt;img src=&quot;images/0.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           1          |     1    |    1   |
| &lt;img src=&quot;images/1.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           0          |     1    |    1   |
| &lt;img src=&quot;images/2.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           0          |     0    |    0   |
| &lt;img src=&quot;images/3.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           0          |     0    |    0   |
| &lt;img src=&quot;images/4.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           1          |     1    |    1   |
| &lt;img src=&quot;images/5.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     0     |           1          |     1    |    0   |
| &lt;img src=&quot;images/6.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     0     |           0          |     0    |    0   |
| &lt;img src=&quot;images/7.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           0          |     1    |    1   |
| &lt;img src=&quot;images/8.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     0     |           1          |     0    |    1   |
| &lt;img src=&quot;images/9.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           0          |     0    |    0   |


因此：
- `X_train` 为每个样本包含了三个特征：
    - 棕色菌盖 (取值为 `1` 表示“棕色”菌盖，取值为 `0` 表示“红色”菌盖) 
    - 上宽下窄菌柄 (取值为 `1` 表示“上宽下窄”菌柄，取值为 `0` 表示“上窄下宽”菌柄) 
    - 独生 (取值为 `1` 表示“是”，取值为 `0` 表示“否”) 

- `y_train` 表示该蘑菇是否可食用：
    - `y = 1` 表示可食用
    - `y = 0` 表示有毒


```python
X_train = np.array([[1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],[0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])
```

#### 查看变量
让我们进一步熟悉一下你的数据集。  
- 一个很好的起点是直接把每个变量打印出来，看看里面包含了什么内容。

下面的代码打印了 `X_train` 的前几个元素以及该变量的数据类型。


```python
print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:",type(X_train))
```

现在，让我们对 `y_train` 做同样的操作：


```python
print("First few elements of y_train:", y_train[:5])
print("Type of y_train:",type(y_train))
```

#### 检查变量的维度 (Dimensions)

熟悉数据的另一种有效方法是查看它的维度。

请打印出 `X_train` 和 `y_train` 的形状 (Shape) ，看看你的数据集中一共有多少个训练样本。


```python
print ('The shape of X_train is:', X_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(X_train))
```

&lt;a name=&quot;4&quot;&gt;&lt;/a&gt;
## 4 - 决策树核心概念回顾

在本次实践实验中，你将基于所提供的数据集构建一棵决策树。

- 回想一下，构建决策树的基本步骤如下：
    - 从根节点 (Root Node) 开始，此时根节点包含所有的训练样本
    - 计算按照所有候选特征进行分裂所能获得的信息增益 (Information Gain) ，并挑选出信息增益最高的那个特征
    - 根据选定的特征划分数据集，从而创建出决策树的左分支 (Left Branch) 和右分支 (Right Branch) 
    - 不断重复上述分裂过程，直到满足停止条件 (Stopping Criteria) 为止
  
  
- 在本次实验中，你将亲手实现以下几个核心函数，它们将使你能够利用信息增益最高的特征将一个节点分裂为左右两个分支：
    - 计算节点处的熵 (Entropy) 
    - 根据给定的特征，将节点处的数据集划分为左右两个分支
    - 计算按给定特征进行分裂所能获得的信息增益
    - 挑选出能使信息增益最大化的最佳特征
    
- 随后，我们将调用你实现好的这些辅助函数，通过不断重复分裂过程直至满足停止条件，构建出一棵完整的决策树
    - 在本次实验中，我们选定的停止条件是将最大深度 (`max_depth`) 限制为 2

&lt;a name=&quot;4.1&quot;&gt;&lt;/a&gt;
### 4.1 计算熵

首先，你将编写一个名为 `compute_entropy` 的辅助函数，用于计算某个节点处的熵 (即不纯度 (Impurity) 的度量指标) 。
- 该函数接收一个 NumPy 数组 (`y`) 作为输入，数组中的元素标示了该节点内的各个样本究竟是可食用的 (`1`) 还是有毒的 (`0`) 。

请按以下步骤补全下方的 `compute_entropy()` 函数：
* 计算 $p_1$，即该节点中可食用蘑菇样本所占的比例 (也就是在 `y` 中取值为 `1` 的样本比例) 
* 随后按照如下公式计算熵：

$$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$$
* 注意事项：
    * 对数 log 是以 $2$ 为底计算的
    * 在代码实现中，规定 $0\text{log}_2(0) = 0$。也就是说，如果 `p_1 = 0` 或 `p_1 = 1`，请直接将熵设为 `0`
    * 请务必先检查节点处的数据是否为空 (即检查 `len(y) != 0`) ；如果为空，则直接返回 `0`
    
&lt;a name=&quot;ex01&quot;&gt;&lt;/a&gt;
### 练习 1

请按照上述说明补全 `compute_entropy()` 函数。
    
如果你在编写时遇到困难，可以查看下方代码单元格后面的提示来辅助实现。


```python
# UNQ_C1
# GRADED FUNCTION: compute_entropy

def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    # You need to return the following variables correctly
    entropy = 0.
    
    ### START CODE HERE ###
    
        
        
            
        
            
    ### END CODE HERE ###        
    
    return entropy
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
    
   * 如何计算 `p1`：
       * 你可以通过 `y[y == 1]` 获取 `y` 中取值为 `1` 的样本子集
       * 你可以使用 `len(y)` 获取 `y` 中的样本总数
   * 如何计算 `entropy`：
       * np.log2 可以帮你计算 NumPy 数组以 2 为底的对数
       * 如果 `p1` 的值为 0 或 1，请确保直接将 `entropy` 设为 `0` 
     
    <details>
          <summary><font size="2" color="darkblue"><b> 点击查看更多提示</b></font></summary>
        
    * 你可以按照如下整体结构来实现该函数：
    ```python 
    def compute_entropy(y):
        
        # You need to return the following variables correctly
        entropy = 0.

        ### START CODE HERE ###
        if len(y) != 0:
            # Your code here to calculate the fraction of edible examples (i.e with value = 1 in y)
            p1 =

            # For p1 = 0 and 1, set the entropy to 0 (to handle 0log0)
            if p1 != 0 and p1 != 1:
                # Your code here to calculate the entropy using the formula provided above
                entropy = 
            else:
                entropy = 0. 
        ### END CODE HERE ###        

        return entropy
    ```
    
    如果你还是卡住了，可以查看下面的具体代码提示，了解如何计算 `p1` 和 `entropy`。
    
    <details>
          <summary><font size="2" color="darkblue"><b>计算 p1 的提示</b></font></summary>
           &emsp; &emsp; 你可以通过 <code>p1 = len(y[y == 1]) / len(y) </code> 来计算 p1
    </details>

     <details>
          <summary><font size="2" color="darkblue"><b>计算 entropy 的提示</b></font></summary>
          &emsp; &emsp; 你可以通过 <code>entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)</code> 来计算 entropy
    </details>
        
    </details>

&lt;/details&gt;

    


你可以运行下面的测试代码，检验自己的实现是否正确：


```python
# Compute entropy at the root node (i.e. with all examples)
# Since we have 5 edible and 5 non-edible mushrooms, the entropy should be 1"

print("Entropy at root node: ", compute_entropy(y_train)) 

# UNIT TESTS
compute_entropy_test(compute_entropy)
```

**期望输出**：
&lt;table&gt;
  &lt;tr&gt;
    &lt;td&gt; &lt;b&gt;Entropy at root node:&lt;b&gt; 1.0 &lt;/td&gt; 
  &lt;/tr&gt;
&lt;/table&gt;

&lt;a name=&quot;4.2&quot;&gt;&lt;/a&gt;
### 4.2 划分数据集

接下来，你将编写一个名为 `split_dataset` 的辅助函数，它接收某个节点处的数据以及用于分裂的特征索引，并将数据划分为左右两个分支。在稍后的小节中，你还将编写代码来评估这次分裂的效果究竟有多好。

- 该函数接收训练数据 `X`、当前节点所包含的数据点索引列表 `node_indices`，以及要用于分裂的特征索引 `feature`。
- 它会对数据进行划分，并返回落入左分支的索引子集 `left_indices` 和落入右分支的索引子集 `right_indices`。
- 举例来说，假设我们从根节点开始 (因此 `node_indices = [0,1,2,3,4,5,6,7,8,9]`) ，并且选择按照特征 `0` (即样本是否具有棕色菌盖) 进行分裂。
    - 那么该函数的输出结果就应当是：`left_indices = [0,1,2,3,4,7,9]` (具有棕色菌盖的数据点) 和 `right_indices = [5,6,8]` (不具有棕色菌盖的数据点) 。
    
    
| 索引 |                                                    | 棕色菌盖 (Brown Cap) | 上宽下窄菌柄 (Tapering Stalk Shape) | 独生 (Solitary) | 可食用 (Edible) |
|-------|:--------------------------------------------------:|:---------:|:--------------------:|:--------:|:------:|
| 0     | &lt;img src=&quot;images/0.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           1          |     1    |    1   |
| 1     | &lt;img src=&quot;images/1.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           0          |     1    |    1   |
| 2     | &lt;img src=&quot;images/2.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           0          |     0    |    0   |
| 3     | &lt;img src=&quot;images/3.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           0          |     0    |    0   |
| 4     | &lt;img src=&quot;images/4.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           1          |     1    |    1   |
| 5     | &lt;img src=&quot;images/5.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     0     |           1          |     1    |    0   |
| 6     | &lt;img src=&quot;images/6.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     0     |           0          |     0    |    0   |
| 7     | &lt;img src=&quot;images/7.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           0          |     1    |    1   |
| 8     | &lt;img src=&quot;images/8.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     0     |           1          |     0    |    1   |
| 9     | &lt;img src=&quot;images/9.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; |     1     |           0          |     0    |    0   |
    
&lt;a name=&quot;ex02&quot;&gt;&lt;/a&gt;
### 练习 2

请补全下方所示的 `split_dataset()` 函数：

- 遍历 `node_indices` 中的每一个索引
    - 如果 `X` 在该样本索引、该特征列上的取值为 `1`，则将该索引添加到 `left_indices` 中
    - 如果 `X` 在该样本索引、该特征列上的取值为 `0`，则将该索引添加到 `right_indices` 中

如果你遇到困难，可以查看下方单元格后面的提示来辅助实现。


```python
# UNQ_C2
# GRADED FUNCTION: split_dataset

def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """
    
    # You need to return the following variables correctly
    left_indices = []
    right_indices = []
    
    ### START CODE HERE ###
    
        
            
        
            
    ### END CODE HERE ###
        
    return left_indices, right_indices
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
    
   * 你可以按照如下整体结构来实现该函数：
    ```python 
    def split_dataset(X, node_indices, feature):
    
        # You need to return the following variables correctly
        left_indices = []
        right_indices = []

        ### START CODE HERE ###
        # Go through the indices of examples at that node
        for i in node_indices:   
            if # Your code here to check if the value of X at that index for the feature is 1
                left_indices.append(i)
            else:
                right_indices.append(i)
        ### END CODE HERE ###
        
    return left_indices, right_indices
    ```
    <details>
          <summary><font size="2" color="darkblue"><b> 点击查看更多提示</b></font></summary>
        
    判断条件可以写为 <code> if X[i][feature] == 1:</code>。
        
    </details>

&lt;/details&gt;

    


现在，让我们运行下面的代码块来检验你的实现。正如上面所讨论的，我们先尝试在包含所有样本的根节点处按照特征 0 (棕色菌盖) 划分数据集。我们还提供了一个辅助函数，用于将划分结果直观地可视化出来。


```python
# Case 1

root_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Feel free to play around with these variables
# The dataset only has three features, so this value can be 0 (Brown Cap), 1 (Tapering Stalk Shape) or 2 (Solitary)
feature = 0

left_indices, right_indices = split_dataset(X_train, root_indices, feature)

print("CASE 1:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# Visualize the split 
generate_split_viz(root_indices, left_indices, right_indices, feature)

print()

# Case 2

root_indices_subset = [0, 2, 4, 6, 8]
left_indices, right_indices = split_dataset(X_train, root_indices_subset, feature)

print("CASE 2:")
print("Left indices: ", left_indices)
print("Right indices: ", right_indices)

# Visualize the split 
generate_split_viz(root_indices_subset, left_indices, right_indices, feature)

# UNIT TESTS    
split_dataset_test(split_dataset)
```

**期望输出**：
```
CASE 1:
Left indices:  [0, 1, 2, 3, 4, 7, 9]
Right indices:  [5, 6, 8]

CASE 2:
Left indices:  [0, 2, 4]
Right indices:  [6, 8]
```

&lt;a name=&quot;4.3&quot;&gt;&lt;/a&gt;
### 4.3 计算信息增益

接下来，你将编写一个名为 `compute_information_gain` 的函数，它接收训练数据、某个节点处的样本索引列表以及一个候选分裂特征，并返回按该特征分裂所能获得的信息增益。

&lt;a name=&quot;ex03&quot;&gt;&lt;/a&gt;
### 练习 3

请补全下方所示的 `compute_information_gain()` 函数，按照如下公式计算信息增益：

$$\text{Information Gain} = H(p_1^\text{node})- (w^{\text{left}}H(p_1^\text{left}) + w^{\text{right}}H(p_1^\text{right}))$$

其中：
- $H(p_1^\text{node})$ 是分裂前该节点处的熵
- $H(p_1^\text{left})$ 和 $H(p_1^\text{right})$ 分别是分裂后产生的左分支和右分支处的熵
- $w^{\text{left}}$ 和 $w^{\text{right}}$ 分别是左分支和右分支中的样本占当前节点总样本的比例

提示：
- 你可以直接调用刚才在上面实现好的 `compute_entropy()` 函数来计算各个节点的熵
- 我们已经为你提供了一段初始代码，它会调用你刚才实现的 `split_dataset()` 函数对数据集进行划分

如果你遇到困难，可以查看下方单元格后面的提示来辅助实现。


```python
# UNQ_C3
# GRADED FUNCTION: compute_information_gain

def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        feature (int):           Index of feature to split on
   
    Returns:
        cost (float):        Cost computed
    
    """    
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    # You need to return the following variables correctly
    information_gain = 0
    
    ### START CODE HERE ###
    
    
    
    
    
    
    
    
    
    
    
    
    
    ### END CODE HERE ###  
    
    return information_gain
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
    
   * 你可以按照如下整体结构来实现该函数：
    ```python 
    def compute_information_gain(X, y, node_indices, feature):
        # Split dataset
        left_indices, right_indices = split_dataset(X, node_indices, feature)

        # Some useful variables
        X_node, y_node = X[node_indices], y[node_indices]
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # You need to return the following variables correctly
        information_gain = 0

        ### START CODE HERE ###
        # Your code here to compute the entropy at the node using compute_entropy()
        node_entropy = 
        # Your code here to compute the entropy at the left branch
        left_entropy = 
        # Your code here to compute the entropy at the right branch
        right_entropy = 

        # Your code here to compute the proportion of examples at the left branch
        w_left = 
        
        # Your code here to compute the proportion of examples at the right branch
        w_right = 

        # Your code here to compute weighted entropy from the split using 
        # w_left, w_right, left_entropy and right_entropy
        weighted_entropy = 

        # Your code here to compute the information gain as the entropy at the node
        # minus the weighted entropy
        information_gain = 
        ### END CODE HERE ###  

        return information_gain
    ```
    如果你还是卡住了，请查看下方的具体提示。
    
    <details>
          <summary><font size="2" color="darkblue"><b> 计算各个熵值的提示</b></font></summary>
        
    <code>node_entropy = compute_entropy(y_node)</code><br>
    <code>left_entropy = compute_entropy(y_left)</code><br>
    <code>right_entropy = compute_entropy(y_right)</code>
        
    </details>
    
    <details>
          <summary><font size="2" color="darkblue"><b>计算 w_left 和 w_right 的提示</b></font></summary>
           <code>w_left = len(X_left) / len(X_node)</code><br>
           <code>w_right = len(X_right) / len(X_node)</code>
    </details>
    
    <details>
          <summary><font size="2" color="darkblue"><b>计算 weighted_entropy 的提示</b></font></summary>
           <code>weighted_entropy = w_left * left_entropy + w_right * right_entropy</code>
    </details>
    
    <details>
          <summary><font size="2" color="darkblue"><b>计算 information_gain 的提示</b></font></summary>
           <code> information_gain = node_entropy - weighted_entropy</code>
    </details>


&lt;/details&gt;


现在，你可以运行下面的单元格来检验自己的实现，并看看在根节点处分别按照每一个特征进行分裂所能获得的信息增益各是多少：


```python
info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
print("Information Gain from splitting the root on brown cap: ", info_gain0)

info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
print("Information Gain from splitting the root on tapering stalk shape: ", info_gain1)

info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
print("Information Gain from splitting the root on solitary: ", info_gain2)

# UNIT TESTS
compute_information_gain_test(compute_information_gain)
```

**期望输出**：
```
Information Gain from splitting the root on brown cap:  0.034851554559677034
Information Gain from splitting the root on tapering stalk shape:  0.12451124978365313
Information Gain from splitting the root on solitary:  0.2780719051126377
```

在根节点处按照“是否独生 (Solitary) ” (`feature = 2`) 进行分裂能够带来最大的信息增益。因此，它是我们在根节点处进行分裂的最佳特征。

&lt;a name=&quot;4.4&quot;&gt;&lt;/a&gt;
### 4.4 获取最佳分裂特征
现在，让我们编写一个函数来自动找出用于分裂的最佳特征——具体做法就像上面那样，依次计算每个特征的信息增益，并返回能够带来最大信息增益的那个特征索引。

&lt;a name=&quot;ex04&quot;&gt;&lt;/a&gt;
### 练习 4
请补全下方所示的 `get_best_split()` 函数。
- 该函数接收训练数据，以及当前节点处的数据点索引列表。
- 该函数的输出是能够带来最大信息增益的特征索引：
    - 你可以使用 `compute_information_gain()` 函数遍历所有特征，计算出每个特征对应的信息增益。
如果你遇到困难，可以查看下方单元格后面的提示来辅助实现。


```python
# UNQ_C4
# GRADED FUNCTION: get_best_split

def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    
    # Some useful variables
    num_features = X.shape[1]
    
    # You need to return the following variables correctly
    best_feature = -1
    
    ### START CODE HERE ###
    
    
    
        
        
            
            
    ### END CODE HERE ##    
   
    return best_feature
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
    
   * 你可以按照如下整体结构来实现该函数：
    
    ```python 
    def get_best_split(X, y, node_indices):   

        # Some useful variables
        num_features = X.shape[1]

        # You need to return the following variables correctly
        best_feature = -1

        ### START CODE HERE ###
        max_info_gain = 0

        # Iterate through all features
        for feature in range(num_features): 
            
            # Your code here to compute the information gain from splitting on this feature
            info_gain = 
            
            # If the information gain is larger than the max seen so far
            if info_gain > max_info_gain:  
                # Your code here to set the max_info_gain and best_feature
                max_info_gain = 
                best_feature = 
        ### END CODE HERE ##    
   
    return best_feature
    ```
    如果你还是卡住了，请查看下方的具体提示。
    
    <details>
          <summary><font size="2" color="darkblue"><b> 计算 info_gain 的提示</b></font></summary>
        
    <code>info_gain = compute_information_gain(X, y, node_indices, feature)</code>
    </details>
    
    <details>
          <summary><font size="2" color="darkblue"><b>更新 max_info_gain 和 best_feature 的提示</b></font></summary>
           <code>max_info_gain = info_gain</code><br>
           <code>best_feature = feature</code>
    </details>
&lt;/details&gt;


现在，让我们运行下面的代码单元格来检验你编写的函数是否正确。


```python
best_feature = get_best_split(X_train, y_train, root_indices)
print("Best feature to split on: %d" % best_feature)

# UNIT TESTS
get_best_split_test(get_best_split)
```

正如我们在上面所看到的，该函数返回在根节点处用于分裂的最佳特征是特征 2 (“是否独生 Solitary”) 。

&lt;a name=&quot;5&quot;&gt;&lt;/a&gt;
## 5 - 构建决策树

在本节中，我们将利用你在上面实现好的各个函数，通过递归 (Recursive) 方式不断挑选最佳特征进行分裂，直到达到停止条件 (最大深度为 2) ，从而生成一棵完整的决策树。

这一部分你不需要编写任何代码，直接运行即可。


```python
# Not graded
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    """
    Build a tree using the recursive algorithm that split the dataset into 2 subgroups at each node.
    This function just prints the tree.
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
        branch_name (string):   Name of the branch. ['Root', 'Left', 'Right']
        max_depth (int):        Max depth of the resulting tree. 
        current_depth (int):    Current depth. Parameter used during recursive call.
   
    """ 

    # Maximum depth reached - stop splitting
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
   
    # Otherwise, get best split and split the data
    # Get the best feature and threshold at this node
    best_feature = get_best_split(X, y, node_indices) 
    
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    # Split the dataset at the best feature
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    # continue splitting the left and the right child. Increment current depth
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)
```


```python
build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
generate_tree_viz(root_indices, y_train, tree)
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;2&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;如果你想对任何非评分代码进行修改和实验，请点击这里。&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    &lt;p&gt;&lt;i&gt;&lt;b&gt;重要提示：为了避免自动评分器出现问题，请务必在已经通过本次作业之后再进行以下操作。&lt;/b&gt;&lt;/i&gt;
    &lt;ol&gt;
        &lt;li&gt; 在笔记本的菜单栏上，依次点击 “View” &gt; “Cell Toolbar” &gt; “Edit Metadata”&lt;/li&gt;
        &lt;li&gt; 点击你想要锁定 / 解锁的代码单元格旁边的 “Edit Metadata” 按钮&lt;/li&gt;
        &lt;li&gt; 将 “editable” 属性的值设置为：
            &lt;ul&gt;
                &lt;li&gt; 如果你想解锁它，设为 “true” &lt;/li&gt;
                &lt;li&gt; 如果你想锁定它，设为 “false” &lt;/li&gt;
            &lt;/ul&gt;
        &lt;/li&gt;
        &lt;li&gt; 在笔记本的菜单栏上，依次点击 “View” &gt; “Cell Toolbar” &gt; “None” &lt;/li&gt;
    &lt;/ol&gt;
    &lt;p&gt; 下面是演示如何执行上述步骤的简短动图： 
        &lt;br&gt;
        &lt;img src=&quot;https://lh3.google.com/u/0/d/14Xy_Mb17CZVgzVAgq7NCjMVBvSae3xO1&quot; align=&quot;center&quot; alt=&quot;unlock_cells.gif&quot;&gt;
&lt;/details&gt;
