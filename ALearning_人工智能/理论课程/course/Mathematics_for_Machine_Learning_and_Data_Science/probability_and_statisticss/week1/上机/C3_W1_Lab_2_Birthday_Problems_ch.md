___
# 非计分实验：生日问题 (Birthday Problems)

欢迎！在本次实验中，你将通过探讨著名的**生日问题 (birthday problem)** ，重新审视概率 (probability) 的反直觉特性，并理解如何借助概率思维来优化决策、用最小的样本投入换取最大化的判断收益。事实上，你将接触到该问题的 4 种变体。你可以借鉴已知问题的求解逻辑来推导下一个问题，其展现出的计算效率与结果或许会完全颠覆你的直觉！

___
- [[#问题背景介绍|问题背景介绍]]
- [[#体验匹配自身生日的游戏|体验匹配自身生日的游戏]]
- [[#问题一：匹配固定日期|问题一：匹配固定日期]]
- [[#解析解 (Analytical Solution)|解析解 (Analytical Solution)]]
- [[#问题二：匹配指定学生的生日|问题二：匹配指定学生的生日]]
- [[#解析解|解析解]]
- [[#问题三：任意两名学生生日匹配|问题三：任意两名学生生日匹配]]
- [[#解析解|解析解]]
- [[#问题四：跨教室间的双向匹配|问题四：跨教室间的双向匹配]]
- [[#解析解|解析解]]
___


让我们开始吧！

```python
import numpy as np
import matplotlib.pyplot as plt
import utils

%matplotlib widget

```

## 问题背景介绍

所有这些问题都共享相似的设定：假设教室里坐满了学生 ==(人数可变) ==，我们希望评估其中两名学生同一天生日、或任意学生在特定日期过生日的客观概率。正如前文所述，你将探索该问题的 4 种衍生形态。

从实际效用与建模的角度来看，我们可以从两个==互逆的维度==来考量这些问题：

* 为了以预期的目标概率实现生日匹配，教室里==至少需要配置多少名学生== `n`？
* 在既定的人数规模 `n` 下，==出现匹配的整体概率==是多少？

这两种思路分别从“资源需求控制”与“产出效果评估”两个角度进行建模，但它们在本质上是等价的。

## 体验匹配自身生日的游戏

为了更直观地展现这种概率机制在实际决策中的表现，这里提供了一个互动小游戏。你可以运行以下单元格来启动游戏，操作十分便捷：在下拉菜单中选择你的生日 (无需年份) ，然后点击 `Simulate!` 按钮随机生成学生，直到有一名学生的生日与你完全吻合。左侧图表记录了每次模拟所需的试验次数，右侧图表则通过直方图 (histogram)  展现了这一变量的分布情况，直观呈现不同成本区间下的发生频次。

你可以根据需要多次运行该模拟，从而对这一随机过程的概率分布 (probability distribution)  建立起更务实的经验认知 (建议至少测试 30 次以积累有效数据) ：

```python
game = utils.your_bday()

```
![[C3_W1_Lab_2_Birthday_Problems_ch-1791898420915.webp]]
就是每次simulate以后
程序随机生成学生和他们的生日 直到生成到的这个同学和你生日相同
此时停下来
    统计生成的人数  放在左图
    比如 第1,2,3次实验生成500人  4次实验100人 
        那右图就会统计 500人 3次(count)  100人1次

## 问题一：匹配固定日期

第一个问题旨在解决一个典型的收益平衡点问题：如果预先给定一个固定日期，教室人数 `n` 至少需要达到多少，才能==确保至少有一人匹配的成功概率达到或超过 0.5==？
![[C3_W1_Lab_2_Birthday_Problems_ch-1791898086909.webp|305]]

在推导严谨的解析解之前，我们先通过 Python 编写模拟程序来近似评估其效用。为此，可以使用下方单元格中提供的辅助函数 `simulate`。运行该代码以加载后续所需的模拟工具：

```python
def simulate(problem_func, n_students=365, n_simulations=1000):
    
    # 将匹配（重合）的计数器初始化为 0
    matches = 0
    
    # 按照指定的模拟次数运行模拟
    for _ in range(n_simulations):
        
        # 如果班级中存在匹配（生日重合），则将匹配计数器加 1
        if problem_func(n_students):
            matches += 1
    
    # 返回 匹配次数 / 模拟总次数 的比率
    return matches/n_simulations
```
解释
    `for _ in range(n_simulations):`  
    下划线 就是填i的地方 
    这里意思是 就算填了i  后面i也用不到 所以写成下划线

当向该函数传入学生人数与模拟轮数时，它会输出对应情境下的模拟概率，该过程具有以下两项关键性质：

* 投入的学生基数越大，达成匹配的成功概率就越高。
* 增加模拟轮数能有效==摊薄随机波动==，使估算结果更趋近于真实客观概率 (这一性质被称为**中心极限定理 (Central Limit Theorem)** ，我们将在第 3 周深入讨论其在提升估计精度方面的价值)。

这一工具非常实用，那么具体该如何配合业务场景使用呢？
你需要传入一个能够准确模拟具体问题规则的目标函数 就是上面的`problem_func`。为确保 `simulate` 能够正确统计结果，该函数必须满足两个条件：

* 接收学生人数作为输入参数。
* 若出现生日匹配则返回 True，否则返回 False。

运行以下代码单元格，即可构建模拟问题一的模型函数：
    生日反正是365天里的一天,所以下面对于某人的生日之前定义为0-364 中的一个数就好了
```python
def problem_1(n_students):
    
    # 预定义一个特定的生日 (每次匹配的生日也是随机的)
    predef_bday = np.random.randint(0, 365)
    
    # 为每个学生生成生日
    gen_bdays = np.random.randint(0, 365, (n_students))
    
    # 检查预定义的生日是否在学生们的生日列表中 (返回布尔值)
    return predef_bday in gen_bdays

```

现在，将这两个函数结合运行，即可评估特定班级规模下的匹配概率。注意，你可以调整变量 `n` 的取值以模拟不同的人数配置。同时需要注意，本次模拟将迭代次数从默认的 1000 次增加到了 10,000 次，虽然牺牲了一定的计算时间，但换来了显著更高的评估精度与确定性：
    就是此时只有 学生人数 和 n_simulations 是可以改的
    要匹配的生日也是每次随机生成的 不需要手动设置

```python
n = 100 # try changing this value!
simulated_prob = simulate(problem_1, n_students=n, n_simulations=10_000)

print(f"The simulated probability of any student to have a bday equal to a predefined value is {simulated_prob} in a classroom with {n} students")

```
输出 (更改不同的n)
    The simulated probability of any student to have a bday equal to a predefined value is 0.2414 in a classroom with 100 students
    The simulated probability of any student to have a bday equal to a predefined value is 0.3307 in a classroom with 150 students
    The simulated probability of any student to have a bday equal to a predefined value is 0.3935 in a classroom with 180 students
    The simulated probability of any student to have a bday equal to a predefined value is 0.4184 in a classroom with 200 students
    The simulated probability of any student to have a bday equal to a predefined value is 0.4571 in a classroom with 220 students
    The simulated probability of any student to have a bday equal to a predefined value is 0.4754 in a classroom with 230 students
    The simulated probability of any student to have a bday equal to a predefined value is 0.4818 in a classroom with 240 students
    The simulated probability of any student to have a bday equal to a predefined value is 0.4943 in a classroom with 250 students
    The simulated probability of any student to have a bday equal to a predefined value is 0.5057 in a classroom with 255 students
    The simulated probability of any student to have a bday equal to a predefined value is 0.5165 in a classroom with 260 students

这种方法直观有效，但若要寻找最优决策阈值，==逐个尝试 `n` 的取值效率过低==。为了在全局范围内全面权衡投入人数与产出概率的关系，我们可以绘制出模拟概率随班级人数变化的收益曲线：

```python
# Generate the simulated probability for every classroom
simulated_probs_1 = [
    simulate(problem_1, n_students=n) for n in utils.big_classroom_sizes
]
#应该是语法糖写法 可以理解的 (其实就是for循环执行simulate,结果保存在数组里)

# Create a scatterplot of simulated probabilities vs classroom size
utils.plot_simulated_probs(simulated_probs_1, utils.big_classroom_sizes)

```
![[C3_W1_Lab_2_Birthday_Problems_ch-1791900032788.webp]]
发现趋势是250上下的时候 匹配正确率达到0.5左右


请记住，蒙特卡洛模拟本质上是一种通过有限采样逼近真实解的经验估算方法。因此，==绘制出的曲线并非绝对平滑==，每次执行模拟时得到的数值也会存在微小的随机偏差。

## 解析解 (Analytical Solution)

在建立了经验直觉之后，我们从数理逻辑上精确计算教室中==至少有一名学生与预设日期同一天生日的概率== $P$。
    显然，$P = P(n)$，即概率值直接取决于班级人数规模，并且随着样本投入 $n$ 的增长，$P(n)$ 必然单调递增并无限趋近于 $1$。
    一旦推导出 $P(n)$ 的闭式解析表达式，我们就能精确计算出满足 $P(n) \geq 0.5$ ==所需的最小人数==成本 $n$。

此处假设一年固定有 $365$ 天。
设 $D$ 为预设的生日日期，并随机抽取一名学生。

定义**事件 (event)**  $S_i$ 为：
    第 $i$ 名学生的生日落在日期 $D$。
由于一年中每一天作为生日的可能性均等，因此该事件的概率为 $P(S_i) = \frac{1}{365}$。
运用**补集规则 (complement rule)** ，该学生生日不落在日期 $D$ 的概率为 $P(S_i^c) = 1 - P(S_i) = 1 - \frac{1}{365}$。

对任意一名学生而言，该概率都是均等相同的，且合理假定各学生的生日相互**独立 (independence)**。设目标事件为 $\mathcal{S}$，即至少有一名学生的生日在日期 $D$。显而易见：

$\mathcal{S}^c$ 即为该事件的对立事件，即**没有任何一名学生的生日落在日期 $D$**，这等价于一系列相互独立的联合条件：

* 学生 1 的生日不是 $D$，并且
* 学生 2 的生日不是 $D$，并且
...
* 学生 $k$ 的生日不是 $D$。

按照上述定义，这一对立状态可表示为 $S_1^c \cap S_2^c \cap \ldots \cap S_k^c$。因此：

$\begin{equation} \begin{split} P(\mathcal{S}) {} & = 1 - P(\mathcal{S}^c) \\               & = 1 - P(S_1^c \cap S_2^c \cap \ldots \cap S_k^c) \\               & = 1 - P(S_1^c)P(S_2^c) \cdots P(S_k^c) \text{ (独立性)}\\               & = 1 - (1 - \frac{1}{365})^n. \end{split} \end{equation}$
    p(sc) 是求剩下c个人都不在这天的概率
    反一下 p(s)是前求 至少有一个人在这天的概率
    这里只是判断另外人 不在 特定的生日 的概率
    没考虑另外人之间 有同样生日的问题
        所以都乘 364/365

正如推导所示，$P(\mathcal{S}) = 1 - (1 - \frac{1}{365})^n = P(n)$。现在，我们可以精确求解：究竟需要付出多少人数规模 $n$，才能使整体成功概率满足 $P(n) \geq \frac{1}{2}$？

不等式 $P(n) \geq \frac{1}{2}$ 等价于：

$$
\begin{align}
 1 - \left(1 - \frac{1}{365}\right)^n &\geq \frac{1}{2} \textit{, 这个1移项到另一边}\\
\left(1 - \frac{1}{365}\right)^n &\leq \frac{1}{2}\\
\left(\frac{364}{365}\right)^n &\leq 2^{-1} \\
\ln{\left(\frac{364}{365}\right)}^n &\leq \ln{2^{-1}} \\
n \ln{\left(\frac{364}{365}\right)} &\leq -\ln{2}\\
\end{align}
$$

利用对数工具可得 $\ln{2} \approx 0.693$，且 $\ln{\frac{364}{365}} \approx -0.003$，将常数代入上述不等式：

$$n \cdot -0.003 \leq -0.693,$$

两边同除以负数后等价于 $n \geq \frac{0.693}{0.003} = 253$。这意味着，若要针对单一既定目标获得过半的胜率，系统必须承担高达== 253 人==的样本搜寻成本。

## 问题二：匹配指定学生的生日

第二个问题与第一个问题非常相似，唯一的区别在于：比较基准不再是预先外部指定(上面是每次随机生成的,固不固定日期无所谓应该)的日期，而是==从教室里的学生中随机指定一名个体作为基准==。该问题可表述为：在一个拥有 `n` 名学生的教室中，如果随机抽取一名学生，那么 `n` 至少需要达到多少，才能使这名学生与班上其他任意一名学生生日重合的概率达到或超过 0.5？
![[C3_W1_Lab_2_Birthday_Problems_ch-1791900780834.webp]]

我们可以直接复用前面构建的高效测试框架 `simulate`，只需编写针对此情境的模型函数。**但在动手之前，请先评估你的预期：从资源消耗的角度来看，你认为达到相同的成功率门槛，所需的人数 `n` 会与问题一接近，还是需要付出更大或更小的样本代价？**
    我猜其实没差 还是250左右

运行以下单元格即可验证这一推断：

```python
def problem_2(n_students):
    
    # Generate birthdays for every student
    gen_bdays = np.random.randint(0, 365, (n_students))
    
    # Pick one student at random
    rnd_index = np.random.randint(0, len(gen_bdays))
    
    # Get the bday from the selected student
    rnd_bday = gen_bdays[rnd_index]
    
    # Take the bday out of the pool of bdays (otherwise there is always a match)
    remaining_bdays = np.delete(gen_bdays, rnd_index, axis=0)
    #这里应该就删挑出来那一个 不会把这个生日的全删了 因为传进去的是rnd_index
    
    # Check if another student shares the same bday
    return rnd_bday in remaining_bdays

```

```python
# Generate the simulated probability for every classroom
simulated_probs_2 = [simulate(problem_2, n_students=n) for n in utils.big_classroom_sizes]

# Create a scatterplot of simulated probabilities vs classroom size
utils.plot_simulated_probs(simulated_probs_2, utils.big_classroom_sizes)

```
![[C3_W1_Lab_2_Birthday_Problems_ch-1791901390659.webp]]
确实没差

## 解析解

注意到该问题在数学结构上与问题一几乎完全相同。
核心区别在于：我们并非从日历的 365 天中先验抽取某一天 $D$，而是先固定了某位特定学生的生日 $D$。为了将其规约为已知的高效解法，我们可以将该被抽中的学生从样本池中剥离，进而将问题转化为在剩余的 $n-1$ 名学生中寻找匹配。因此，整个模型的数学推导完全一致，==唯一的变量替换是有效样本容量由 $n$ 变为 $n-1$==。由此可直接得出概率公式：

$$P(n) = 1 - \left(1 - \frac{1}{365} \right)^{n-1}.$$

经过简单的代数求解，满足 $P(n) \geq \frac{1}{2}$ 的充分必要条件为 $n \geq {\frac{0.693}{0.003}} + 1 = 254$。
    看吧没差吧
由于该学生自身被作为固定基准而无法参与匹配，系统必须额外增加 1 个人头成本来填补这一基准消耗。

## 问题三：任意两名学生生日匹配

第三个问题是所有生日问题中最著名、也最能体现概率聚合效益的经典场景，课堂上曾以互补的方式对其展开探讨。

这一次，我们的优化目标不再局限于匹配某一特定锚点，而是最大化==全局范围==内的任意配对机会：在拥有若干学生的教室中，究竟只需多少人 `n`，就能使“存在任意两名学生同一天生日”的概率达到或超过 0.5？

注意，在课堂讲授中，通常计算的是**全班所有人均不在同一天过生日**的概率。而在本实验中，我们关注的是其对立事件——即**至少有两人同一天生日**，这正是利用补集规则实现高效求解的经典例证。
![[C3_W1_Lab_2_Birthday_Problems_ch-1791901666850.webp]]

在运行仿真之前，不妨对比前面的问题反思：**你认为达到 50% 概率匹配上生日 所需的人数 `n` 会与前两个问题接近吗？如果让你凭经验判断，为了达成目标，所需的人口规模会大幅增加还是急剧下降？**
    这次应该不一样了 人数n应该减少 就是很少人也可以达到50%

为了帮助你建立经验直觉，运行以下代码单元格体验该问题的交互式演示。操作规则非常清晰：

* 点击上方控制面板==任意区域== (位于 `Figure` 标题下方)  即可启动一轮新的模拟测试
* 上方面板会实时生成随机生日，并在==首次出现两人冲突重合时立即发出提示==
* 左下方图表持续记录每次模拟达成匹配所需的最小样本量
* 右下方图表以直方图形式汇总全部轮次的频数分布
(其实就是一开始我们进行交互的那个程序, 只是这次不是指定生日了)

* **建议至少运行 30 次模拟，以充分观察该机制在小样本下的惊人收益**


```python
game_third_prob = utils.third_bday_problem()

```
![[C3_W1_Lab_2_Birthday_Problems_ch-1791902052313.webp]]
![[C3_W1_Lab_2_Birthday_Problems_ch-1791902060523.webp]]

现在，你应该对产生配对所需的人数规模形成了初步估计。接下来，如同前面的步骤一样，通过批量模拟验证你的直觉判断：

```python
def problem_3(n_students):
    
    # Generate birthdays for every student
    gen_bdays = np.random.randint(0, 365, (n_students))
    
    # Get array containing unique bdays
    unique_bdays = np.array(list(set(gen_bdays))) 
    # 放set里再转list再转array
    
    # Check that both the original and unique arrays have the same length 
    # (if so then no two students share the same bday)
    return len(unique_bdays) != len(gen_bdays)

```
这个程序的逻辑应该和上面的交互程序不一样
因为这里学生人数是函数外面给的
    utils里人数实际上是 `small_classroom_sizes = [*range(1, 80)]`

```python
# Generate the simulated probability for every classroom
simulated_probs_3 = [simulate(problem_3, n_students=n) for n in utils.small_classroom_sizes]

# Create a scatterplot of simulated probabilities vs classroom size
utils.plot_simulated_probs(simulated_probs_3, utils.small_classroom_sizes)

```
![[C3_W1_Lab_2_Birthday_Problems_ch-1791902093350.webp]]

## 解析解

该问题的结构与前两者存在显著差异，需要重新评估各阶段的边际概率。此时最经济的计算策略是分步推进：按顺序依次选入学生，定义 $Q_i$ 为第 $i$ 名学生的生日与此前所有已选入学生均不冲突的条件概率。显然 $Q_1 = 1$，因为首位进入的学生不存在任何既有竞争对象。

当引入第 2 名学生时：

$$Q_2 = \frac{364}{365}.$$

原因在于：在第 1 名学生的生日既定的情况下，第 2 名学生有 364 天可供选择而不发生冲突，避免重叠的成功率为 $\frac{364}{365}$。进入第 3 名学生时，池中已有 2 个被占用的独立日期，因此不与前两人发生碰撞的概率降为：

$$Q_3 = \frac{363}{365}.$$

依此类推，当引入第 $n$ 名学生时，其避免冲突的边际概率为：

$$Q_n = \frac{365 - (n-1)}{365}.$$
    这个就是之前学的那个概率了

基于各个体生日相互独立的假设，选入的 $n$ 名学生**彼此之间完全无生日重合**的联合概率，即为各阶段非冲突概率的累乘积。记该概率为 $Q$：

$$Q = Q_1 \cdot Q_2 \cdot \ldots \cdot Q_{n-1} \cdot Q_n = 1 \cdot \frac{364}{365} \cdot \frac{363}{365} \cdot \ldots \cdot \frac{365 - (n-2)}{365} \cdot \frac{365 - (n-1)}{365}.$$

我们追求的目标是**至少有两人共享生日**，根据补集定理，该目标事件的发生概率定义为 $P:= 1 - Q$。

虽然在实际工程中，编写几行紧凑的 Python 循环即可精准搜寻出使 $P \geq \frac{1}{2}$ 成立的最小临界值 $n$，但为了建立具有通用推广价值的理论模型，我们推导其解析近似公式。

当 $x$ 为极小的正数时，可利用一阶泰勒展开近似公式：$1 - x \approx e^{-x}$。将连乘式 $Q$ 展开改写为：

$$Q = 1 \cdot \frac{364}{365} \cdot \frac{363}{365} \cdot \ldots \cdot \frac{365 - (n-2)}{365} \cdot \frac{365 - (n-1)}{365}      = \left(1 - \frac{1}{365} \right) \cdot \left(1 - \frac{2}{365} \right) \cdot \ldots \cdot \left(1 - \frac{n-2}{365} \right) \cdot \left(1 - \frac{n-1}{365} \right).$$

代入指数==近似关系式==：

$$Q \approx e^{-\frac{1}{365}} \cdot e^{-\frac{2}{365}} \cdot \ldots \cdot e^{-\frac{n-2}{365}} \cdot e^{-\frac{n-1}{365}} = e^{- \frac{1 + 2 + \ldots + (n-1)}{365}}.$$

利用等差数列 (arithmetic progression)  前 $n-1$ 项求和公式 $1 + 2 + \ldots + (n-1) = \frac{n(n-1)}{2}$，我们能够将庞大的配对网络复杂度凝练为紧凑的形式：

$$Q \approx e^{-\frac{n(n-1)}{730}}.$$

从而目标概率可表述为 $P = 1 - Q = 1 - e^{-\frac{n(n-1)}{730}}$。使 $P \geq \frac{1}{2}$ 成立等价于 $1 - e^{-\frac{n(n-1)}{730}} \geq \frac{1}{2}$，整理得 $\frac{n(n-1)}{730} \geq \ln 2$，即 $n(n-1) \geq 730 \cdot \ln 2$。代入 $\ln 2 \approx 0.6931$：

$$n(n-1) \geq 505.96 \geq 505.$$

求解二次方程 $n(n-1) = 505$，其唯一的正实数根为 $n = \frac{1 + \sqrt{2021}}{2} \approx 23$。

结论表明：**仅仅需要 $n \geq 23$ 名同学**，即可达成 $P \geq \frac{1}{2}$ 的目标！ 相比于问题一中必须投入 253 人才能达成单一匹配，网络式全员两两配对所产生的组合数呈现二次方级爆增，以极小的组织规模实现了最大的概率。

## 问题四：跨教室间的双向匹配

第四个也是最后一个问题与第三个问题类似，不同之处在于比较范围被划分到了两个独立的群体之间。我们希望在==两个规模各为 `n` 人的教室==之间建立跨群体的连接：==在两个各容纳 `n` 名学生的教室中，`n` 至少需要达到多少==，才能使==教室 A 与教室 B 之间至少存在一对==学生生日相同的概率达到或超过 0.5？
    应该就是把上面的n除以2就好了
    但是也不一定
    因为是要求A,B之间有  A里面有和B里面有 没算在内

![[C3_W1_Lab_2_Birthday_Problems_ch-1791902531123.webp]]

**在开始仿真实验之前，再次结合投入与产出的网络规模，提出你的假设！**

```python
def problem_4(n_students):
    
    # Generate birthdays for every student in classroom 1
    gen_bdays_1 = np.random.randint(0, 365, (n_students))
    
    # Generate birthdays for every student in classroom 2
    gen_bdays_2 = np.random.randint(0, 365, (n_students))
    
    # Check for any match between both classrooms
    return np.isin(gen_bdays_1, gen_bdays_2).any()

```
解释
    ![[C3_W1_Lab_2_Birthday_Problems_ch-1791902504098.webp]]

```python
# Generate the simulated probability for every classroom
simulated_probs_4 = [simulate(problem_4, n_students=n) for n in utils.small_classroom_sizes]

# Create a scatterplot of simulated probabilities vs classroom size
utils.plot_simulated_probs(simulated_probs_4, utils.small_classroom_sizes)

```
![[C3_W1_Lab_2_Birthday_Problems_ch-1791902555387.webp]]
差不多就是 23 的一半 多

## 解析解

解决这一跨群体匹配问题的逻辑与问题一具有深刻的相通性。此时我们不再是拿单一个体去比对**一个**固定日期，而是==将教室 B 中的每个个体与教室 A 中的 $n$ 个已有生日进行跨组比对==。
回顾问题一，当只存在一个日期 $D$ 时，教室内所有人都**未匹配**该日期的概率为 $Q_1 = (1 - \frac{1}{365})^n$ (即当时 $P(\mathcal{S})$ 的对立事件)。
    这个没问题
沿用问题三中样本彼此==独立==的分析范式，将教室 B 中的每一位学生视作独立的探测单元。==对于抽取的每一名学生，其避开所有 $n$ 个指定日期的概率均为== $Q_i = (1 - \frac{1}{365})^n$，
因此整个教室 B 中没有任何人匹配到教室 A 中任一生日的联合概率为：

$$Q = Q_1 \cdot Q_2 \cdot \ldots \cdot Q_{n-1} \cdot Q_n = (1 - \frac{1}{365})^{n^2}$$

在小量条件下再次运用指数逼近 $1 - x \approx e^{-x}$：

$$Q \approx e^{-\frac{n^2}{365}}$$

由此得出两群之间实现至少一次有效匹配的宏观概率：

$$P(n) \approx 1 - e^{-\frac{n^2}{365}}.$$

若要满足 $P(n) \geq \frac{1}{2}$ 的收益阈值，只需满足 $n \geq \sqrt{\ln 2 \cdot 365} \approx 15.9$，即取整数 $n \geq 16$ (原推导取整下界标注为 $15$)。此时两班双向交互产生的 $n^2$ 对组合关系，使跨群体连接在极低的人均门槛下迅速实现。

**祝贺！你已顺利完成了关于生日问题的非计分实验！**