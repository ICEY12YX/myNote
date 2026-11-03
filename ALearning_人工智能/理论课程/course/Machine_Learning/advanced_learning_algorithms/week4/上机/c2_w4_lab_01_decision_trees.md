# 非计分实验：决策树 (Decision Trees)

在本笔记本中，你将通过可视化方式直观地了解决策树是如何利用信息增益 (Information Gain) 进行节点分裂 (Split) 的。

我们将重新审视视频课程中使用过的那个数据集 (Dataset) 。该数据集如下：

正如你在课程视频中所看到的，在决策树中，我们通过考察某次分裂所能带来的**信息增益**，来决定是否对一个节点 (Node) 进行分裂。 (对应视频中的信息增益示意图) 

其中：

$$\text{Information Gain} = H(p_1^\text{node})- \left(w^{\text{left}}H\left(p_1^\text{left}\right) + w^{\text{right}}H\left(p_1^\text{right}\right)\right),$$

而 $H$ 是熵 (Entropy) ，其定义为：

$$H(p_1) = -p_1 \text{log}_2(p_1) - (1- p_1) \text{log}_2(1- p_1)$$

请记住，这里的对数 log 是以 2 为底的。请运行下面的代码块，亲自观察当概率 $p$ 变化时，熵 $H(p)$ 会呈现出怎样的变化规律。

请注意，当 $p = 0.5$ 时，$H$ 达到最大值。这意味着事件发生的概率恰好是 $0.5$ (最不确定) 。而当 $p = 0$ 和 $p = 1$ 时，$H$ 取得最小值，即事件是否发生是完全可以预测的。因此，熵直观地衡量了一个事件的不确定性或可预测程度。


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
```


```python
%matplotlib widget
_ = plot_entropy()

```


```python

```

|                                                     | 耳朵形状 (Ear Shape) | 脸型 (Face Shape) | 胡须 (Whiskers) | 猫 (Cat) |
|:---------------------------------------------------:|:---------:|:-----------:|:---------:|:------:|
| &lt;img src=&quot;images/0.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 尖耳 (Pointy) | 圆脸 (Round) | 有 (Present) |    1   |
| &lt;img src=&quot;images/1.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 垂耳 (Floppy) | 非圆脸 (Not Round) | 有 (Present) |    1   |
| &lt;img src=&quot;images/2.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 垂耳 (Floppy) | 圆脸 (Round) | 无 (Absent) |    0   |
| &lt;img src=&quot;images/3.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 尖耳 (Pointy) | 非圆脸 (Not Round) | 有 (Present) |    0   |
| &lt;img src=&quot;images/4.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 尖耳 (Pointy) | 圆脸 (Round) | 有 (Present) |    1   |
| &lt;img src=&quot;images/5.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 尖耳 (Pointy) | 圆脸 (Round) | 无 (Absent) |    1   |
| &lt;img src=&quot;images/6.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 垂耳 (Floppy) | 非圆脸 (Not Round) | 无 (Absent) |    0   |
| &lt;img src=&quot;images/7.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 尖耳 (Pointy) | 圆脸 (Round) | 无 (Absent) |    1   |
| &lt;img src=&quot;images/8.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 垂耳 (Floppy) | 圆脸 (Round) | 无 (Absent) |    0   |
| &lt;img src=&quot;images/9.png&quot; alt=&quot;drawing&quot; width=&quot;50&quot;/&gt; | 垂耳 (Floppy) | 圆脸 (Round) | 无 (Absent) |    0   |


我们将使用**独热编码 (One-hot Encoding) **对这些类别特征 (Categorical Features) 进行编码，具体规则如下：

- 耳朵形状 (Ear Shape) ：尖耳 (Pointy) = 1，垂耳 (Floppy) = 0
- 脸型 (Face Shape) ：圆脸 (Round) = 1，非圆脸 (Not Round) = 0
- 胡须 (Whiskers) ：有 (Present) = 1，无 (Absent) = 0

因此，我们得到了两个数组：

- `X_train`：对于每个样本，包含 3 个特征：
            - 耳朵形状 (如果是尖耳则为 1，否则为 0) 
            - 脸型 (如果是圆脸则为 1，否则为 0) 
            - 胡须 (如果有胡须则为 1，否则为 0) 
            
- `y_train`：该动物是否为猫的目标标签 (Label) 
            - 如果该动物是猫则为 1
            - 否则为 0


```python

```


```python
X_train = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
```


```python
#For instance, the first example
X_train[0]
```

这意味着第一个样本具有尖耳朵、圆脸型，并且有胡须。

在每一个节点上，我们都会分别计算按各个特征进行分裂所能获得的信息增益——具体做法是将当前节点的熵与分裂后左右两个子节点的加权熵 (Weighted Entropy) 进行比较，然后选择信息增益最高的那个特征来分裂该节点。

因此，根节点 (Root Node) 包含了我们数据集中的所有动物。请记住，$p_1^{node}$ 是根节点中正类 (猫) 所占的比例。所以：

$$p_1^{node} = \frac{5}{10} = 0.5$$

现在，让我们编写一个函数来计算熵。


```python
def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1- p)*np.log2(1 - p)
    
print(entropy(0.5))
```

为了具体演示这一过程，让我们分别计算按每一个特征分裂节点时所能获得的信息增益。为此，我们先编写几个辅助函数。


```python
def split_indices(X, index_feature):
    """Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have 
    that feature = 1 and the right node those that have the feature = 0 
    index feature = 0 => ear shape
    index feature = 1 => face shape
    index feature = 2 => whiskers
    """
    left_indices = []
    right_indices = []
    for i,x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices
```

因此，如果我们选择按“耳朵形状”进行分裂，那么左子节点中应该包含以下样本索引 (请对照上表进行核对) ：

$$0 \quad 3 \quad 4 \quad 5 \quad 7$$

而右子节点中则包含其余的样本索引。


```python
split_indices(X_train, 0)
```

接下来，我们还需要编写另一个函数来计算分裂后两个子节点的加权熵。正如你在视频课程中所看到的，我们必须先求出：

- $w^{\text{left}}$ 和 $w^{\text{right}}$，即落入**每个子节点**中的动物样本占总样本的比例。
- $p^{\text{left}}$ 和 $p^{\text{right}}$，即在**每个分裂出的子节点内部**，猫所占的比例。

请务必注意这两个定义之间的区别！！举例来说，如果我们按照索引为 0 的特征 (耳朵形状) 对根节点进行分裂，那么在包含动物样本 0、3、4、5 和 7 的左子节点，以及包含其余样本的右子节点中，我们分别有：

$$w^{\text{left}}= \frac{5}{10} = 0.5 \text{ 且 } p^{\text{left}} = \frac{4}{5}$$
$$w^{\text{right}}= \frac{5}{10} = 0.5 \text{ 且 } p^{\text{right}} = \frac{1}{5}$$


```python
def weighted_entropy(X,y,left_indices,right_indices):
    """
    This function takes the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)
    
    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy
```


```python
left_indices, right_indices = split_indices(X_train, 0)
weighted_entropy(X_train, y_train, left_indices, right_indices)
```

可以看到，分裂出的两个子节点的加权熵约为 0.72。为了计算出最终的**信息增益**，我们只需用待分裂节点 (在本例中即根节点) 的熵减去这个加权熵即可。


```python
def information_gain(X, y, left_indices, right_indices):
    """
    Here, X has the elements in the node and y is theirs respectives classes
    """
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X,y,left_indices,right_indices)
    return h_node - w_entropy
```


```python
information_gain(X_train, y_train, left_indices, right_indices)
```

现在，让我们依次计算分别使用每一个特征对根节点进行分裂时所能获得的信息增益：


```python
for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
    left_indices, right_indices = split_indices(X_train, i)
    i_gain = information_gain(X_train, y_train, left_indices, right_indices)
    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")
    
```

由此可见，用于分裂根节点的最佳特征确实是“耳朵形状 (Ear Shape) ”。请运行下面的代码，直观查看这次分裂的实际效果。你不需要理解下面这个代码块的具体底层实现。


```python
tree = []
build_tree_recursive(X_train, y_train, [0,1,2,3,4,5,6,7,8,9], "Root", max_depth=1, current_depth=0, tree = tree)
generate_tree_viz([0,1,2,3,4,5,6,7,8,9], y_train, tree)
```

这一建树过程是**递归 (Recursive) **进行的，也就是说，我们必须对生成的每一个子节点重复执行上述计算，直到满足以下任一停止条件 (Stopping Criteria) 为止：

- 分裂后的树深度 (Tree Depth) 超过了设定的阈值 (Threshold) 
- 分裂出的节点中只包含单一类别 (100% 纯净) 
- 分裂所能带来的信息增益低于设定的阈值

最终构建完成的决策树如下所示：


```python
tree = []
build_tree_recursive(X_train, y_train, [0,1,2,3,4,5,6,7,8,9], "Root", max_depth=2, current_depth=0, tree = tree)
generate_tree_viz([0,1,2,3,4,5,6,7,8,9], y_train, tree)
```

恭喜！你已经完成了本实验笔记本的全部内容！
