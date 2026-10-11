___
# 非计分实验 - 蒙提霍尔问题 (Monty Hall problem)


欢迎！在本次非计分实验中，你将通过探索著名的蒙提霍尔问题 (Monty Hall problem) ，领略概率 (probability) 违背直觉的奇妙特性。这个问题初看之下似乎微不足道，但它生动地展示了一个事实：概率的表现往往与我们的第一直觉大相径庭。

首先，导入实验所需的函数库：

```python
import numpy as np
import utils

%matplotlib widget

```

### 1. 问题介绍

假设你正在参加一档电视节目，有机会通过玩一个游戏赢得一辆汽车。游戏规则非常简单：你需要在三扇紧闭的门中做出选择。其中一扇门后面是一辆汽车，而另外两扇门后面则是山羊。

游戏分为两个阶段进行：

1. 主持人让你在三扇门中任选一扇，但暂时不能打开它。


2. 接着，心知汽车确切位置的主持人会在剩下的两扇门中挑出一扇打开，露出一只山羊。



抉择的时刻到了，假设你挑选了 1 号门。就在 1 号门即将开启前，已知底细的主持人打开了 3 号门，露出一只山羊，此时 1 号门和 2 号门依然紧闭。接着主持人向你发问：

**“你想要换选 2 号门吗？”**

这个问题听起来有些耐人寻味——既然主持人清楚大奖花落谁家，他会不会是在设局诱导你做出糟糕的选择？你会怎么做？是改变主意换门，还是坚持最初选定的 1 号门？

随着你对概率与统计 (Probability and Statistics) 的理解逐渐加深，你可以思考得更深入一步：哪种策略能带给你更高的胜率？是换门还是坚守原选择？或者换不换根本没有区别？

既然我们有 Python 这一强大工具，在本 Notebook 中，你将通过模拟这个游戏亲自揭开谜底！首先，你可以在下方实时体验几局游戏，直观感受一下其中的玄机！

## 2. 亲身体验游戏！

运行下一个代码单元格，你可以亲自试玩一阵，尝试不同的策略。左侧面板会呈现真实的游戏界面，操作指南如下：

* 只需在三扇门中任选其一即可开启一局新游戏。


* 在你做出初步选择后，主持人会打开剩余两扇门中的一扇，且门后必定是山羊。


* 随后你可以决定是坚持原选择还是换选另一扇门。


* 如果你点击了主持人已经打开的那扇门，该局游戏将不作数。


* 如果点击了三扇门以外的区域，游戏将重置且不计入统计。


* 在揭晓奖品 (游戏结束)  后，点击屏幕任意位置即可重新开始游戏。


* 若希望重置计数器，重新运行该单元格即可。



右侧面板会记录总游玩局数以及两种策略的成功率。不妨多玩几次，看看能否发现什么规律！

```python
game = utils.monty_hall_game()

```

在继续往下探索前，请确保你已经试玩了足够多次并形成了自己的假设。换门更明智吗？还是坚守初始选择更好？亦或两者完全没有差异？

## 3. 进行多次迭代模拟

亲自试玩了一段时间后，你或许已经对克敌制胜的优选策略有了初步的设想。现在，你将对游戏进行成千上万次迭代模拟，观察两种策略的胜率是否确有差别。

为此，我们提供了 `monty_hall` 函数。该函数接受一个布尔值 (boolean)  参数，用于控制你是否选择换门。若想了解其具体实现原理，可阅读代码注释。注意，代码中用数值 `0` 代表山羊，用数值 `1` 代表汽车：

```python
def monty_hall(switch):

    # All doors have a goat initially
    doors = np.array([0, 0, 0])

    # Randomnly decide which door will have a car
    winner_index = np.random.randint(0, 3)

    # Place the car in the winner door
    doors[winner_index] = 1

    # Participant selects a door at random
    choice = np.random.randint(0, 3)

    # Get doors that can be opened (host cannot open the door chosen or the one with the car)
    openable_doors = [i for i in range(3) if i not in (winner_index, choice)]

    # Host opens one of the available doors at random
    door_to_open = np.random.choice(openable_doors)

    # Switch to the other available door (the one that is not the original choice or the opened one)
    if switch:
        choice = [i for i in range(3) if i not in (choice, door_to_open)][0]

    # Return 1 if you open a door with a car, 0 otherwise
    return doors[choice]

```

你可以使用上面的函数来模拟单局游戏。但这并不直观，更好的方法是借助该函数批量运行多局模拟并记录结果。这样一来，你就能确切知道长期坚持某种策略是否能够稳操胜券。

你可以将上述函数传入另一个函数中，指定策略并分别模拟 1 次、10 次、100 次以及 1000 次。随着运行次数的增加，你会看到两种策略的胜率逐步收敛至它们的真实概率：

```python
utils.success_rate_plot(monty_hall)

```

## 4. 解析解 (Analytical Solution)

现在你已经熟悉了该问题，并且获得了强有力的实证证据：最佳策略是**换门**，因为换门能让你赢得大约 67% 的对局！

现在我们从数学解析的角度来论证它！为此，我们先给出若干定义。

定义事件 (event) ：

$E_1$ = 汽车在 1 号门后
$E_2$ = 汽车在 2 号门后
$E_3$ = 汽车在 3 号门后

或者用更简洁的形式表示：对于 $i = 1,2,3$，$E_i$ = 汽车在 $i$ 号门后。

请注意，根据游戏规则，这些事件是互斥的 (mutually exclusive) ，换言之，汽车不可能同时出现在两扇门后。这意味着：

$P(E_1 \cap E_2) = 0$，$P(E_1 \cap E_3) = 0$ 且 $P(E_2 \cap E_3) = 0$。也可以写成：

$$P(E_i \cap E_j) = 0  \text{      对于 } i \neq j.$$

另一个由规则决定的事实是，**汽车必定在三扇门中的某一扇之后**，因此：

$$P(E_1 \cup E_2 \cup E_3) = 1.$$

事实上，这就是所有可能结果构成的集合，即样本空间 (sample space) **或**全集 (universe)  $\Omega$。

假设你选择了 **1 号门**。由于汽车出现在三扇门中任一扇的概率均等，我们可知：

$$P(E_1) = \frac{1}{3}.$$

根据补集规则 (complement rule) ，汽车不在 1 号门后的概率为：

$$P(E_1^c) = 1 - P(E_1) = 1 - \frac{1}{3} = \frac{2}{3}$$

因为全集由 $E_1 \cup E_2 \cup E_3$ 给出 (汽车在 1 号门或 2 号门或 3 号门后) ，所以 $E_1^c = E_2 \cup E_3$，进而有 $P(E_2 \cup E_3) = \frac{2}{3}$。你可以从下图中获得直观的认识。

现在，在你选定 1 号门之后，主持人打开了 3 号门，露出一只山羊，并问你是否愿意换门。如果你坚持不换，获胜的概率依然保持为 $\frac{1}{3}$，因为这完全由你的初始选择决定。而如果你选择**换门**，你会发现主持人实际上**为你提供了一条额外的信息**。他向你明确排除了 3 号门，这意味着：

$$P(E_3) = 0.$$

至此，推导已基本完成，因为正如你所知：$\frac{2}{3} = P(E_2 \cup E_3) = P(E_2) + P(E_3) - P(E_2 \cap E_3)$。你已经知道 $P(E_2 \cap E_3) = 0$，因为它们是互斥事件 (汽车只可能位于三扇门中的**某一扇**之后) ，而主持人又为你提供了至关重要的补充信息：$P(E_3) = 0$。据此，你可以轻松得出结论：

$$P(win \vert{} switch) = P(E_2) = \frac{2}{3}.$$

换句话说，**在已知**汽车不在 3 号门的条件下，汽车位于 2 号门后的条件概率 (conditional probability)  为 $\frac{2}{3} \approx 0.67$，这与你在模拟实验中所看到的结果完全吻合！

## 5 广义蒙提霍尔问题 (可选内容)

让我们考虑一个更具一般性的全新游戏设定。

此时的游戏规则如下：

* 共有 $n$ 扇门，你必须选择其中一扇门。


* 主持人会打开 $k$ 扇门，并展示门后的山羊。


* 你可以选择换门，也可以坚持原先的选择。



在这种情况下，换门依然是更优选择吗？它会取决于 $k$ 还是取决于 $n$ 呢？

## 5.1 模拟实验

你可以通过模拟这一问题来建立直觉。

```python
def generalized_monty_hall(switch, n = 3, k = 1):
    if not (0 <= k <= n-2):
        raise ValueError('k must be between 0 and n-2, so the Host can leave at least 1 openable door!')
    
    # All doors have a goat initially
    doors = np.array([0 for _ in range(n)])
    
    # Decide which door will have a car
    winner = np.random.randint(0,n)

    # Place the car in the winner door
    doors[winner] = 1.0
    
    # Participant selects a door at random
    choice = np.random.randint(0,n)
    
    # Get doors that can be opened (host cannot open the door chosen or the one with the car)
    openable_doors = [i for i in range(n) if i not in (winner, choice)]
    
    # Host open k of the available doors at random
    door_to_open = np.random.choice(openable_doors, size = k, replace = False)
    
    # Switch to the other available door (the one that is not the original choice or the opened one)
    if switch:
        choices = [i for i in range(n) if i not in np.array(choice) and i not in np.array(door_to_open)]
        # Player chooses another door at random
        choice = np.random.choice(choices)
    
    # Return 1 if you open a door with a car, 0 otherwise
    return doors[choice]

```

```python
utils.success_rate_plot(generalized_monty_hall)

```

## 5.2 解析解

本节内容相对进阶，如果你愿意，完全可以跳过！

回顾此时的游戏设定：

* 共有 $n$ 扇门，你必须选择其中一扇门。


* 主持人打开 $k$ 扇门，露出门后的山羊。


* 你可以选择换门，也可以坚持原先的选择。



核心问题是：换门在任何情况下都是更优策略吗？这一结论是否取决于 $k$ 的大小？

为了从数学解析上回答这个问题，首先定义以下事件：

$$E_i = \text{ 汽车在第 } i \text{ 扇门后。此时，} i = 1, \ldots, n.$$

同样，由于只有 1 辆汽车，各个 $E_i$ 之间是相互排斥的。

注意，由于主持人既不会打开玩家选择的门，也绝不会打开藏有汽车的门，因此 $k$ 存在一个上限，即 $n-2$，所以：

$$0 \leq k \leq n-2.$$

我们可以设定两个前提假设：

* 玩家选择了 1 号门


* 主持人打开了第 $2, \ldots, k+1$ 号门



之所以能这样设定，是因为我们总能通过重新编号来实现这一对应关系。例如，若玩家选了 10 号门，我们可以将它重新编号为 1 号门，而原先的 1 号门则变为 10 号门。这样做仅仅是为了避免复杂的下标标记。用数学术语来说，这通常被称为不失一般性 (without loss of generality) ，因为重新编号完全不会影响最终的推导结果。

现在面对 $n$ 扇门，汽车位于 1 号门后的概率为 $\frac{1}{n}$，即：

$$P(E_1) = \frac{1}{n}.$$

根据补集规则，汽车**不在** 1 号门后的概率为：

$$P(E_1^c) = 1 - P(E_1) = 1 - \frac{1}{n} = \frac{n-1}{n}.$$

注意到：

$$E_1^c = E_2 \cup E_3 \cup \ldots \cup E_n.$$

为了简化上面等号右侧的表达式，我们可以引入一种紧凑的记号将其写作：

$$\bigcup_{i = 2}^{n} E_i.$$

它的工作原理与求和符号 $\sum$ 类似，只是这里执行的运算是集合的并集 (union) 。

因此，我们可知：

$$P\left(\bigcup_{i = 2}^{n} E_i\right) = \frac{n-1}{n}.$$

现在我们可以回答这个问题：在选择换门的前提下，获胜的概率是多少？

我们来看看下面这张图：

如果玩家随机换选一扇可用的门，那么他必须在第 $k+2, k+3, \ldots, n-1, n$ 号门中做出选择。因此，挑中汽车的概率等于：

在 1 号门中**未选中汽车**的概率 $\left(P(E_1^c) = \frac{n-1}{n}\right)$，乘以**此时**挑中汽车的概率。因为剩余可选的门共有 $n-k-1$ 扇，所以此时挑中的概率为 $\frac{1}{n-k-1}$。

因此，换门后的最终获胜概率为：

$$P(win \vert{} switch) = \frac{n-1}{n} \cdot \frac{1}{n-k-1}.$$

该式可以重新整理写为：

$$P(win \vert{} switch) = \frac{n-1}{n} \cdot \frac{1}{n-k-1} = \frac{1}{n} \cdot \frac{n-1}{n-k-1} \geq \frac{1}{n} = P(E_1) = P(win \vert{} not\ switch).$$

并且，等号仅在 $k = 0$ 时成立。这意味着主持人没有打开任何一扇门。

因此，**换门永远是更好的选择**。这在初听时可能违背直觉，但请仔细想想：选择换门时，你其实是在充分利用主持人赋予你的**全新信息**；而如果你固执地不换门，就相当于直接无视了这份珍贵的新信息。

**祝贺！你已顺利完成了关于蒙提霍尔问题的非计分实验！**