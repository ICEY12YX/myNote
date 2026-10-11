___
# 朴素贝叶斯 (Naive Bayes)



欢迎来到“机器学习与数据科学中的概率与统计 (Probability and Statistics for Machine Learning and Data Science) ”课程的第一个作业，这也是“机器学习与数据科学数学基础 (Math for Machine Learning and Data Science) ”专项课程的最后一门课！

在本次作业中，你将运用课堂上学到的知识，动手实现用于垃圾邮件检测的朴素贝叶斯算法 (Naive Bayes algorithm)。第 1 至 3 节提供了该问题的必要背景；在第 4 节中，你将编写具体的函数来实现该算法；第 5 节则包含一些有趣且极具启发性的非计分拓展内容。

# 目录

* [ 1 - 简介](https://www.google.com/search?q=%231)

* [ 2 - 必要的库导入](https://www.google.com/search?q=%232)

* [ 3 - 数据集](https://www.google.com/search?q=%233)

* [ 3.1 加载与探索数据集](https://www.google.com/search?q=%233.1)

* [ 3.2 数据集预处理](https://www.google.com/search?q=%233.2)

* [ 3.3 文本预处理](https://www.google.com/search?q=%233.3)

* [ 3.4 划分训练集与测试集](https://www.google.com/search?q=%233.4)



* [ 4 - 实现朴素贝叶斯算法](https://www.google.com/search?q=%234)

* [ 4.1 计算 $P(\text{email} \mid \text{spam})$ 与 $P(\text{email} \mid \text{ham})$](https://www.google.com/search?q=%234.1)

* [ 4.2 计算 $P(\text{spam})$ 与 $P(\text{ham})$](https://www.google.com/search?q=%234.2)

* [ 4.3 综合应用](https://www.google.com/search?q=%234.3)

* [ 练习 1](https://www.google.com/search?q=%23ex01)

* [ 练习 2](https://www.google.com/search?q=%23ex02)

* [ 练习 3](https://www.google.com/search?q=%23ex03)

* [ 练习 4](https://www.google.com/search?q=%23ex04)



* [ 4.4 模型评估](https://www.google.com/search?q=%234.4)



* [ 5 - 附录 (非计分章节) ](https://www.google.com/search?q=%235)

* [ 5.1 朴素贝叶斯模型中隐藏的缺陷](https://www.google.com/search?q=%235.1)

* [ 5.2 提升模型表现：朴素贝叶斯的工程实践](https://www.google.com/search?q=%235.2)




## 1 - 简介

朴素贝叶斯算法 (Naive Bayes algorithm) 是机器学习与数据科学的基石之一，它巧妙利用贝叶斯定理 (Bayes' Theorem) 来判定某个数据点归属于特定类别的可能性。该算法做出了一个“朴素的假设”：假设各个特征之间彼此相互独立。在现实数据中，这种假设几乎不可能完全成立，但正是得益于这种简化，算法的实现难度大幅降低，而且正如你即将见证的那样，它往往能产生令人惊叹的优异效果。需要注意的是，朴素贝叶斯属于监督学习 (supervised learning) 算法，这意味着它必须依赖已经标注好的数据才能有效工作。以本实验为例，我们需要先准备一批已经被标记为“垃圾邮件 (spam) ”或“正常邮件 (ham) ”的邮件集合，以此来训练算法模型。

### 运用朴素贝叶斯进行垃圾邮件检测

本作业聚焦于一个典型的二分类问题 (binary classification problem) ：区分垃圾邮件与非垃圾邮件 (日常俗称为正常邮件 “ham”)。在本任务中，垃圾邮件标记为 $1$，正常邮件标记为 $0$。

对于任意一封给定的邮件，我们最关心的后验概率可表示为：

$$P(\text{spam} \mid \text{email})$$

该概率值越高，这封邮件被判定为垃圾邮件的可能性就越大。借助课堂中讲授的贝叶斯定理，我们可以展开如下计算：

$$P(\text{spam} \mid \text{email}) = \frac{P(\text{email} \mid \text{spam}) \cdot P(\text{spam})}{P(\text{email})}$$

各项含义拆解如下：

* $P(\text{spam})$：随机抽取一封邮件为垃圾邮件的先验概率，等价于数据集中垃圾邮件所占的比例。


* $P(\text{email} \mid \text{spam})$：已知该邮件是垃圾邮件的前提下，出现该特定邮件内容的条件概率。


* $P(\text{email})$：该特定邮件内容在所有邮件中出现的边缘概率。



在具体计算时，我们可以运用一个巧妙的“捷径”：直接忽略分母项 $P(\text{email})$。因为分类的核心目的仅是比较该邮件属于垃圾邮件还是正常邮件的相对概率高低。我们写出 $P(\text{spam} \mid \text{email})$ 与 $P(\text{ham} \mid \text{email})$ 的对应表达式：

$$P(\text{spam} \mid \text{email}) = \frac{P(\text{email} \mid \text{spam}) \cdot P(\text{spam})}{P(\text{email})}$$

$$P(\text{ham} \mid \text{email}) = \frac{P(\text{email} \mid \text{ham}) \cdot P(\text{ham})}{P(\text{email})}$$

由于分母 $P(\text{email}) > 0$ 且在两个表达式中完全一致，比较两者的相对大小时仅需对比各自的分子项，分母可以直接略去不计。

## 2 - 必要的库导入

下方的代码块将导入本作业所需的全部函数库、工具模块以及用于提供实时评测反馈的单元测试组件。

```python
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
import string

```

```python
import w1_unittest

```

## 3 - 数据集

### 3.1 加载与探索数据集

以下代码块将把数据集读入内存。我们将使用 Pandas 库将其加载为其核心数据结构——DataFrame。不过**无需担心**你是否精通 Pandas，在此处它仅用于直观展示数据结构，后续实际算法实现时我们都会统一转换为 NumPy 数组来处理。

```python
dataframe_emails = pd.read_csv('emails.csv')
dataframe_emails.head()

```

让我们对数据集进行初步的探索：

```python
print(f"Number of emails: {len(dataframe_emails)}")
print(f"Proportion of spam emails: {dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")
print(f"Proportion of ham emails: {1-dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")

```

请注意，该数据集属于典型的不平衡数据 (unbalanced dataset) ：正常邮件的数量是垃圾邮件的两倍以上！ 了解这一点对于任何数据分析项目都极为关键，因为样本失衡可能会对包括朴素贝叶斯在内的诸多机器学习算法产生实质影响。

### 3.2 数据集预处理

当前 DataFrame 包含两列：`text` 列包含邮件的正文内容，`spam` 列包含指明邮件类别的数值变量。谨记：$1$ 代表垃圾邮件，$0$ 代表正常邮件。下面的函数将完成两个重要的预处理步骤：

* 每封邮件正文均以 `Subject:` 开头，该函数会剔除每封邮件开头的这一前缀词。


* 对整个数据集进行随机打乱 (shuffle)。目前原始数据中所有垃圾邮件都集中排在前半部分，紧接着是正常邮件。只有先将数据充分打乱，才能合理地切分训练集与测试集。



即使你对函数中的部分 Python 语法不太熟悉也无需焦虑，引入这些处理旨在提醒大家：在投入模型构建之前，充分探索与清洗数据是必不可少的标准流程。

```python
def preprocess_emails(df):
    """
    对包含邮件数据的 DataFrame 进行预处理。

    参数：
    - df (pandas.DataFrame)：包含 'text' 和 'spam' 列的输入 DataFrame。

    返回值：
    - tuple：包含两个元素的元组：
        1. X (numpy.array)：剔除了 "Subject:" 前缀后的邮件文本数组。
        2. Y (numpy.array)：标记每封邮件属于垃圾邮件 (1) 还是正常邮件 (0) 的标签数组。

    该函数会打乱输入的数据集，以避免训练集/测试集划分时产生数据倾斜。
    随后提取邮件文本及标签，并切除每封邮件起始处的 "Subject:" 前缀。
    """
    # 打乱数据集
    df = df.sample(frac = 1, ignore_index = True, random_state = 42)
    # 去除构成邮件前 9 个字符的 "Subject:" 字符串，并转为 numpy 数组
    X = df.text.apply(lambda x: x[9:]).to_numpy()
    # 将标签转换为 numpy 数组
    Y = df.spam.to_numpy()
    return X, Y

```

```python
X, Y = preprocess_emails(dataframe_emails)

```

我们打印出前 5 封邮件的内容：

```python
print(X[:5])

```

以及对应的前 5 个标签：

```python
print(Y[:5])

```

可以看到，NumPy 数组 `X` 是一个字符串数组，数组中的每个元素代表一封邮件，其下标与 `Y` 中的类别标签一一对应。**不妨尝试修改 `email_index` 的数值**，查看不同邮件的文本及其真实类别。请记住，$0$ 代表正常邮件，$1$ 代表垃圾邮件。

```python
email_index = 30
print(f"Email index {email_index}: {X[email_index]}\n\n")
print(f"Class: {Y[email_index]}")

```

### 3.3 文本预处理

本小节虽然不在课程视频的理论考查范围内，也不包含计分函数，但在处理文本数据时不可或缺，掌握这些技能对你在机器学习与数据科学领域的进阶大有裨益！

在自然语言文本中，常常存在大量对文本核心语义贡献极低的高频虚词，例如介词、代词等，这些词被称为停用词 (stopwords) 。由于它们在各种文本中泛滥出现，几乎无法为分类任务提供有价值的区分特征。因此，标准的做法是过滤掉所有停用词以及标点符号，从而提炼出一组紧凑纯净的特征词汇。这正是下面函数的核心功能。

另一个关键环节是邮件的分词 (tokenization) 。分词就是把长文本切分成独立的语素标记 (tokens) ——在此处即切分成单个单词。经过清洗后，每封邮件都将被转化为一个 NumPy 单词数组，其中已剔除全部停用词与标点。

```python
def preprocess_text(X):
    """
    通过剔除停用词与标点符号来预处理文本集合。

    参数：
    - X (str 或 array-like)：待处理的输入文本数据。若传入单条字符串，
      会自动转换为单元素 numpy 数组。

    返回值：
    - numpy.array：预处理后的文本数据数组，每个元素代表一篇已剔除停用词与标点的文本。

    说明：
    - 该函数借助自然语言工具包 (nltk) 库进行分词和停用词过滤。
    - 若输入为单个字符串，返回单篇预处理结果。
    """
    # 构建停用词与标点符号集合
    stop = set(stopwords.words('english') + list(string.punctuation))

    # 处理传入单封邮件字符串而非数组的情形
    if isinstance(X, str):
        X = np.array([X])

    # 用于保存处理结果的列表
    X_preprocessed = []

    for i, email in enumerate(X):
        email = np.array([i.lower() for i in word_tokenize(email) if i.lower() not in stop]).astype(X.dtype)
        X_preprocessed.append(email)
        
    if len(X) == 1:
        return X_preprocessed[0]
    return X_preprocessed

        
        

```

```python
# 该函数运行可能需要耗费数秒至一分钟左右。
X_treated = preprocess_text(X)

```

预处理完成后，每封邮件的内容都已转变为剔除了无用停用词的单词数组。下面的示例展示了随机挑选的邮件 (例如第 989 封)  在清洗前后的对比。你可以随意修改索引测试其他邮件。这组经过净化规整的词汇序列，正是算法后续真正消费的输入特征。

```python
email_index = 989
print(f"Email before preprocessing: {X[email_index]}")
print(f"Email after preprocessing: {X_treated[email_index]}")

```

### 3.4 划分训练集与测试集

现在我们将数据集划分为训练集和测试集。这里采用经典的 80/20 比例，即 80% 的样本用于模型训练，剩余 20% 用于客观评估测试。

```python
TRAIN_SIZE = int(0.80*len(X_treated)) # 80% 的样本用于训练。

X_train = X_treated[:TRAIN_SIZE]
Y_train = Y[:TRAIN_SIZE]
X_test = X_treated[TRAIN_SIZE:]
Y_test = Y[TRAIN_SIZE:]

```

前面我们观察到垃圾邮件大约占整体的 24%。划分后务必检查训练集和测试集中该比例是否保持一致，否则可能导致训练出的模型产生系统性偏差。

```python
print(f"Proportion of spam in train dataset: {sum(Y_train == 1)/len(Y_train):.4f}")
print(f"Proportion of spam in test dataset: {sum(Y_test == 1)/len(Y_test):.4f}")

```

两者的正负样本比例虽有微弱差异，但极为接近，完全符合实验要求！

## 4 - 实现朴素贝叶斯算法

再次明确任务目标：对比 $P(\text{spam} \mid \text{email})$ 与 $P(\text{ham} \mid \text{email})$ 的大小并做出归类决策。在具体判定时，只需分别计算分子项 $P(\text{spam}) \cdot P(\text{email} \mid \text{spam})$ 与 $P(\text{ham}) \cdot P(\text{email} \mid \text{ham})$ 即可完成比较。

### 4.1 计算 $P(\text{email} \mid \text{spam})$ 与 $P(\text{email} \mid \text{ham})$

由于两种类别的计算逻辑完全对称，我们先以垃圾邮件为例。

每封邮件本质上都是一个词汇列表。我们的目标是计算在已知该邮件为垃圾邮件的条件下，观察到该特定词汇序列的概率。实现方式是运用概率乘法法则 (product rule)。将一封邮件表示为词汇集合 $\text{email} = \{\text{word}_1, \text{word}_2, \ldots, \text{word}_n \}$，计算式展开为：

$$P(\text{email} \mid \text{spam}) = P(\text{word}_1 \mid \text{spam}) \cdot P(\text{word}_2 \mid \text{spam}) \cdots P(\text{word}_n \mid \text{spam})$$

这也正是“朴素贝叶斯”中朴素假设 (naive assumption) 的由来！ 我们强行假设每个单词在邮件中出现的概率与其他单词完全独立。显然，这种假设在真实语言环境中并不成立：例如包含 “party” (派对)  的邮件往往大概率伴随 “invitation” (邀请函) ；包含 “prize” (大奖)  的邮件极有可能同时出现 “congratulations” (祝贺)。但正是这种看似不严密的独立性假设，赋予了我们直接将联合条件概率拆解为边缘连乘的数学便利。我们无需估算极其复杂的词间高阶条件依赖，仅需将各个独立的单词条件概率相乘即可。朴素贝叶斯建立在对现实世界的大胆简化之上，但在工程实践中却展现出了非凡的分类效果！

那么在已知为垃圾邮件的情况下，单个单词 $\text{word}_1$ 出现的概率该如何统计呢？

$$P(\text{word}_1 \mid \text{spam}) = \frac{\text{包含 } \text{word}_1 \text{ 的垃圾邮件数量}}{\text{垃圾邮件总数}}$$

其中符号 # 代表样本数量，即分子统计的是包含该单词的垃圾邮件总篇数。

这个统计非常直接：统计包含 $\text{word}_1$ 的垃圾邮件篇数，再除以垃圾邮件的总篇数。遍历数据集中出现的每一个词汇并重复此过程，即可为后续计算任意邮件的先验条件概率打下基础。基于这一思路，**你的首个任务是构建一个名为 `word_frequency` 的字典，记录数据集中每个单词分别在正常邮件与垃圾邮件中出现的频次**。

#### 4.1.1 处理连乘中的零概率陷阱

在实际应用中，如果某个单词从未在垃圾邮件 (或正常邮件)  中出现过，会导致 $P(\text{word} \mid \text{spam}) = 0$。在连乘链条中，只要出现一个 $0$，整个累乘结果就会瞬间归零。这显然不合理——我们绝不能因为一个生僻词就全盘抹杀整封邮件的综合概率特征。为了解决这个问题，我们将**采用加一平滑策略，把所有单词在垃圾邮件和正常邮件中的初始频次均设为 1**。通过人为假定每个单词至少在两种邮件中各出现过一次，即可彻底规避连乘中的零乘积陷阱。

### 4.2 计算 $P(\text{spam})$ 与 $P(\text{ham})$

在运用贝叶斯定理时，我们还需要用到正常邮件与垃圾邮件的基础先验概率。该统计非常直接，即各自在训练集总体中所占的样本比例：

$$P(\text{spam}) = \frac{\text{垃圾邮件总数}}{\text{邮件总数}}$$

$$P(\text{ham}) = \frac{\text{正常邮件总数}}{\text{邮件总数}}$$

### 4.3 综合应用

要判定一封邮件的归属类别，只需将前面计算好的各项分量相乘，并比较两者大小：

* $P(\text{spam}) \cdot P(\text{email} \mid \text{spam})$
* $P(\text{ham}) \cdot P(\text{email} \mid \text{ham})$

### 练习 1

你的任务是实现以下函数，生成一个词频字典，记录数据集中每个单词在垃圾邮件 (1) 和正常邮件 (0) 中出现的频次。

```python
def get_word_frequency(X,Y):
    """
    计算在一组已标注为垃圾邮件 (1) 或正常邮件 (0) 的数据中每个单词的词频。

    参数：
    - X (numpy.array)：邮件数组，其中每封邮件表示为一个单词列表。
    - Y (numpy.array)：与 X 中每封邮件对应的标签数组。1 表示垃圾邮件，0 表示正常邮件。

    返回值：
    - word_dict (dict)：以邮件中出现的独立单词为键，以包含垃圾邮件 (1) 与正常邮件 (0) 对应频次的子字典为值的嵌套字典。
    """
    # 创建空字典
    word_dict = {}

    ### START CODE HERE ###

    num_emails = None

    # 遍历每封预处理后的邮件及其标签
    for i in range(num_emails):
        # 获取第 i 封邮件
        email = X[i] 
        # 获取第 i 个标签，指明该邮件是否为垃圾邮件。1 = None
        # 变量名 cls 是 class 的缩写，因为 class 在 Python 中属于保留关键字。
        cls = Y[i] 
        # 为避免在单封邮件中重复统计同一单词，通过 set 集合去重
        email = set(email) 
        # 遍历该邮件中的每个独立单词
        for word in email:
            # 若单词尚未存在于字典中，则初始化添加。记住初始计数均设为 1（平滑策略）
            if word not in word_dict.keys():
                word_dict[word] = {"spam": None, "ham": None}
            # 根据类别标签累加计数：cls == 0 累加 ham，cls == 1 累加 spam
            if cls == 0:    
                word_dict[None][None] += 1
            if cls == 1:
                word_dict[None][None] += 1
    
    ### END CODE HERE ###
    return word_dict

```

```python
test_output = get_word_frequency([['like','going','river'], ['love', 'deep', 'river'], ['hate','river']], [1,0,0])
print(test_output)

```

##### **预期输出** (词条输出顺序可能略有差异，核心是每个词对应的统计值须一致)



```Python
{'going': {'spam': 2, 'ham': 1}, 'river': {'spam': 2, 'ham': 3}, 'like': {'spam': 2, 'ham': 1}, 'deep': {'spam': 1, 'ham': 2}, 'love': {'spam': 1, 'ham': 2}, 'hate': {'spam': 1, 'ham': 2}}

```

下一个代码块将对你实现的函数进行单元测试。别担心，此处暂不计分，仅用于验证逻辑的正确性。如果测试未通过，会输出详细的排查提示供你调试。

```python
w1_unittest.test_get_word_frequency(get_word_frequency)

```

```python
# 基于训练集构建真实的词频字典
word_frequency = get_word_frequency(X_train,Y_train)

```

我们还需要一个类别频次字典，用以记录训练集中正常邮件 (0) 与垃圾邮件 (1) 的总样本数。下面的代码行将为你自动创建：

```python
# 统计垃圾与正常邮件数量：由于约定 spam = 1, ham = 0，可直接对训练集数组求和与比对
class_frequency = {'ham': sum(Y_train == 0), 'spam': sum(Y_train == 1)}

```

```python
print(class_frequency)

```

若要获取训练集中垃圾邮件的占比，可直接计算：

```python
# 计算公式：(垃圾邮件数量) / (总邮件数量)。
# 每封邮件非黑即白，总邮件数 = 正常邮件数 + 垃圾邮件数。
proportion_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam'])
print(f"The proportion of spam emails in training is: {proportion_spam:.4f}")

```

注意，该结果与此前我们在探索阶段算出的数值完全吻合！

### 练习 2

在接下来的练习中，你将实现计算 $P(\text{word} \mid \text{spam})$ 与 $P(\text{word} \mid \text{ham})$ 的函数。由于两者的数学计算形式完全一致，我们将抽象出一个通用函数 `prob_word_given_class` 来计算 $P(\text{word} \mid \text{class})$，其中 `class` 可以是 `spam` ($1$) 或 `ham` ($0$)。

回顾计算公式：

$$P(\text{word}_i \mid \text{class}) = \frac{\text{该类别中包含 } \text{word}_i \text{ 的邮件篇数}}{\text{该类别邮件总数}}$$

**注意：当前练习无需考虑单词是否存在于字典中的边缘情况，后续顶层函数会统一处理。**

```python
def prob_word_given_class(word, cls, word_frequency, class_frequency):
    """
    计算特定单词在指定类别下出现的条件概率。

    参数：
    - word (str)：目标单词。
    - cls (str)：目标类别，取值为 'spam' 或 'ham'。
    - word_frequency (dict)：词频字典。
    - class_frequency (dict)：类别频数统计字典。

    返回值：
    - float：该单词在给定类别下的条件概率。
    """
    ### START CODE HERE ###
    
    # 获取该单词在对应类别下的出现次数
    amount_word_and_class = word_frequency[None][None]
    p_word_given_class = None/class_frequency[None]

    ### END CODE HERE ###
    return p_word_given_class

    

```

```python
print(f"P(lottery | spam) = {prob_word_given_class('lottery', cls = 'spam', word_frequency = word_frequency, class_frequency = class_frequency)}")
print(f"P(lottery | ham) = {prob_word_given_class('lottery', cls = 'ham', word_frequency = word_frequency, class_frequency = class_frequency)}")
print(f"P(schedule | spam) = {prob_word_given_class('schedule', cls = 'spam', word_frequency = word_frequency, class_frequency = class_frequency)}")
print(f"P(schedule | ham) = {prob_word_given_class('schedule', cls = 'ham', word_frequency = word_frequency, class_frequency = class_frequency)}")

```

##### **预期输出** (最后几位小数可能因计算精度微有差异)



```Python
P(lottery | spam) = 0.00807899461400359
P(lottery | ham) = 0.0002883506343713956
P(schedule | spam) = 0.008976660682226212
P(schedule | ham) = 0.10294117647058823

```

下一个代码块将对你的函数执行单元测试，以验证实现的准确性。

```python
w1_unittest.test_prob_word_given_class(prob_word_given_class, word_frequency, class_frequency)

```

### 练习 3

在本练习中，你将实现计算 $P(\text{email} \mid \text{class})$ 的函数，其中类别为 spam ($1$) 或 ham ($0$)。这里我们将使用*朴素假设*：

$$P(\text{email} \mid \text{class}) = P(\text{word}_1 \mid \text{class}) \cdot P(\text{word}_2 \mid \text{class}) \cdots P(\text{word}_n \mid \text{class})$$

核心思路是遍历邮件中的每个有效单词，并在每一步将当前累积概率乘以对应的 $P(\text{word} \mid \text{class})$。

提示：在 Python 中累乘更新变量时，可以直接使用 `value *= update` 简化语法。

```python
def prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    基于预处理后的邮件内容，计算邮件在特定类别（垃圾或正常）下的似然概率。

    参数：
    - treated_email (list)：邮件预处理后的单词列表。
    - cls (str)：邮件类别标签，'spam' 或 'ham'。
    - word_frequency (dict)：词频字典。
    - class_frequency (dict)：类别频数统计字典。

    返回值：
    - float：给定邮件属于指定类别的条件概率。
    """

    # prob 初始值赋为 1，在每次循环中累乘当前的 P(word | class)
    prob = 1

    ### START CODE HERE ###

    for word in None:
        # 仅对存在于词频字典中的词执行连乘运算
        if word in word_frequency.keys(): 
            # 累乘更新 prob，注意传入 word_frequency 和 class_frequency 参数！
            prob *= None

    return prob

```

```python
example_email = "Click here to win a lottery ticket and claim your prize!"
treated_email = preprocess_text(example_email)
prob_spam = prob_email_given_class(treated_email, cls = 'spam', word_frequency = word_frequency, class_frequency = class_frequency)
prob_ham = prob_email_given_class(treated_email, cls = 'ham', word_frequency = word_frequency, class_frequency = class_frequency)
print(f"Email: {example_email}\nEmail after preprocessing: {treated_email}\nP(email | spam) = {prob_spam}\nP(email | ham) = {prob_ham}")

```

##### **预期输出** (末位小数可能存在微小精度差异)



```Python
Email: Click here to win a lottery ticket and claim your prize!
Email after preprocessing: ['click' 'win' 'lottery' 'ticket' 'claim' 'prize']
P(email | spam) = 5.3884806600117164e-11
P(email | ham) = 1.2428344868918976e-15

```

运行单元测试以检验 `prob_email_given_class` 函数的实现：

```python
w1_unittest.test_prob_email_given_class(prob_email_given_class, word_frequency, class_frequency)

```

### 练习 4

在本练习中，你将结合前面的成果，同时完成以下两项计算以评估邮件的归属类别：

* $P(\text{spam}) \cdot P(\text{email} \mid \text{spam})$
* $P(\text{ham}) \cdot P(\text{email} \mid \text{ham})$

数值较大的一方对应的类别，即为算法预测的分类结果。该函数还提供了一个可选参数 `return_likelihood`，用于在需要时直接返回两者的计算数值，而非仅返回分类类别。

**注意**：最终分类决策输出为一个整数标签 ($1$ 或 $0$)。直接输出分类数值而不是字符串，极大地方便了后续各项性能指标 (例如准确率等)  的向量化统计。

```python
def naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood = False):    
    """
    用于垃圾邮件检测的朴素贝叶斯分类器。

    该函数基于朴素贝叶斯算法计算邮件为垃圾邮件 (1) 或正常邮件 (0) 的概率。
    结合邮件在各类别下的条件似然概率以及各类别先验概率，通过比对两者大小做出最终判定。

    参数：
    - treated_email (list)：预处理后的邮件单词列表。
    - word_frequency (dict)：词频字典。
    - class_frequency (dict)：类别频数统计字典。
    - return_likelihood (bool)：若为 True，则返回垃圾与正常邮件的似然估计值元组。

    返回值：
    若 return_likelihood = False：
        - int：若分类为垃圾邮件返回 1，若为正常邮件返回 0。
    若 return_likelihood = True：
        - tuple：格式为 (spam_likelihood, ham_likelihood) 的元组。
    """

    ### START CODE HERE ###
    
    # 调用上一步实现的函数计算 P(email | spam)。务必传入 word_frequency 和 class_frequency 参数！
    prob_email_given_spam = None

    # 调用上一步实现的函数计算 P(email | ham)。务必传入 word_frequency 和 class_frequency 参数！
    prob_email_given_ham = None

    # 利用 class_frequency 字典计算 P(spam)，公式：垃圾邮件数 / 邮件总数
    p_spam = None

    # 利用 class_frequency 字典计算 P(ham)，公式：正常邮件数 / 邮件总数
    p_ham = None

    # 计算乘积 P(spam) * P(email | spam)，记为 spam_likelihood
    spam_likelihood = None * None

    # 计算乘积 P(ham) * P(email | ham)，记为 ham_likelihood
    ham_likelihood = None * None


    ### END CODE HERE ###
    
    # 若传入 return_likelihood = True，直接返回对应的概率度量元组
    if return_likelihood == True:
        return (spam_likelihood, ham_likelihood)
    
    # 比较两项度量，输出数值较大者对应的分类类别
    elif spam_likelihood >= ham_likelihood:
        return 1
    else:
        return 0

```

```python
example_email = "Click here to win a lottery ticket and claim your prize!"
treated_email = preprocess_text(example_email)

print(f"Email: {example_email}\nEmail after preprocessing: {treated_email}\nNaive Bayes predicts this email as: {naive_bayes(treated_email, word_frequency, class_frequency)}")

print("\n\n")
example_email = "Our meeting will happen in the main office. Please be there in time."
treated_email = preprocess_text(example_email)

print(f"Email: {example_email}\nEmail after preprocessing: {treated_email}\nNaive Bayes predicts this email as: {naive_bayes(treated_email, word_frequency, class_frequency)}")

```

#### **预期输出**

```
Email: Click here to win a lottery ticket and claim your prize!
Email after preprocessing: ['click' 'win' 'lottery' 'ticket' 'claim' 'prize']
Naive Bayes predicts this email as: 1



Email: Our meeting will happen in the main office. Please be there in time.
Email after preprocessing: ['meeting' 'happen' 'main' 'office' 'please' 'time']
Naive Bayes predicts this email as: 0

```

运行单元测试以验证完整朴素贝叶斯分类器的构建：

```python
w1_unittest.test_naive_bayes(naive_bayes, word_frequency, class_frequency)

```

### 4.4 模型评估

本小节不包含计分任务，内容属于课堂理论的工程延伸。强烈建议仔细阅读并理解其中的逻辑，因为准确度量模型表现在机器学习工程中至关重要。

在这一节中，你将全面评估所构建模型的实际泛化能力。回顾前文，我们在 80% 的数据上训练了模型，并保留了 20% 的未见数据作为测试集。显而易见的核心问题是：当面对这些全新的测试数据时，模型的预测准确率究竟如何？衡量这一表现最基础的指标是准确率 (accuracy) ——即模型预测正确的样本占总样本的比例。

计算准确率时，需要分别统计：

* 模型正确判定为垃圾邮件的样本数 (称为真正例 (true positives) )


* 模型正确判定为正常邮件的样本数 (称为真负例 (true negatives) )



最终，将真正例与真负例的总数除以测试集总样本数，即可得到准确率。如果模型绝对完美无误，准确率将达到 1 (即 100%)。下面的代码实现了这些评估指标的统计：

```python
def get_true_positives(Y_true, Y_pred):
    """
    计算二分类任务中的真正例 (True Positives) 数量。

    参数：
    - Y_true (list)：真实标签列表 (0 或 1)。
    - Y_pred (list)：预测标签列表 (0 或 1)。

    返回值：
    - int：真正例数量（真实标签与预测标签均为 1）。
    """
    # 验证真实标签与预测标签长度一致
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_positives = 0
    # 遍历列表中的每个样本
    for i in range(n):
        true_label_i = Y_true[i]
        predicted_label_i = Y_pred[i]
        # 仅当真实为 1 且预测亦为 1 时计数加 1
        if true_label_i == 1 and predicted_label_i == 1:
            true_positives += 1
    return true_positives
        
def get_true_negatives(Y_true, Y_pred):
    """
    计算二分类任务中的真负例 (True Negatives) 数量。

    参数：
    - Y_true (list)：真实标签列表 (0 或 1)。
    - Y_pred (list)：预测标签列表 (0 或 1)。

    返回值：
    - int：真负例数量（真实标签与预测标签均为 0）。
    """
    
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_negatives = 0
    for i in range(n):
        true_label_i = Y_true[i]
        predicted_label_i = Y_pred[i]
        # 仅当真实为 0 且预测亦为 0 时计数加 1
        if true_label_i == 0 and predicted_label_i == 0:
            true_negatives += 1
    return true_negatives
        

```

```python
# 收集测试集上的全部预测结果：

Y_pred = []

# 遍历测试集中的每封邮件
for email in X_test:
    # 执行分类预测
    prediction = naive_bayes(email, word_frequency, class_frequency)
    Y_pred.append(prediction)

# 校验预测列表与真实标签列表长度是否一致：
print(f"Y_test and Y_pred matches in length? Answer: {len(Y_pred) == len(Y_test)}")

```

```python
# 统计真正例数量：
true_positives = get_true_positives(Y_test, Y_pred)

# 统计真负例数量：
true_negatives = get_true_negatives(Y_test, Y_pred)

print(f"The number of true positives is: {true_positives}\nThe number of true negatives is: {true_negatives}")

# 真正例加真负例除以总样本量，计算最终准确率
accuracy = (true_positives + true_negatives)/len(Y_test)

print(f"Accuracy is: {accuracy:.4f}")

```

干得漂亮！基于“词汇间互不相关”这一极为简化的独立性假设，我们从零搭建的朴素贝叶斯模型在测试集上便达到了 84.82% 的优异准确率！ 现在，你可以在下方自定义一段邮件内容，亲自测试模型的分类表现：

```python
email = "Please meet me in 2 hours in the main building. I have an important task for you."
# email = "You win a lottery prize! Congratulations! Click here to claim it"

# 预处理测试邮件
treated_email = preprocess_text(email)
# 获取预测结果：若输出为 1 则映射为 "spam"，否则为 "ham"
prediction = "spam" if naive_bayes(treated_email, word_frequency, class_frequency) == 1 else "ham"
print(f"The email is: {email}\nThe model predicts it as {prediction}.")

```

## 5 - 附录 (非计分章节)

以下各节不计入作业考核成绩，但展现了对现有模型极其关键的工程缺陷分析与优化方案。如果你时间充裕并希望深入探究，强烈推荐继续阅读以下内容。

### 5.1 朴素贝叶斯模型中隐藏的缺陷

当前的朴素模型中潜伏着一个致命隐患，严重拖累了其性能上限。让我们通过一个具体的长文本邮件案例来复现这一问题：

```python
example_index = 4798
example_email = X[example_index]
treated_email = preprocess_text(example_email)
print(f"The email is:\n\t{example_email}\n\nAfter preprocessing:\n\t:{treated_email}")

```

我们计算该邮件对应的 $P(\text{spam}) \cdot P(\text{email} \mid \text{spam})$ 与 $P(\text{ham}) \cdot P(\text{email} \mid \text{ham})$。可以通过传入参数 `return_likelihood = True` 来获取具体数值：

```python
spam_likelihood, ham_likelihood = naive_bayes(treated_email, word_frequency = word_frequency, class_frequency = class_frequency, return_likelihood = True)
print(f"spam_likelihood: {spam_likelihood}\nham_likelihood: {ham_likelihood}")

```

异常情况出现了：两者的似然估算值居然全都是 $0$！ 怎么会发生这种情况？顺便一提，根据此前代码中“当 $\text{spam\_likelihood} \geq \text{ham\_likelihood}$ 时判定为 1”的规则，在 $0 \ge 0$ 的情况下该邮件会被直接误判为垃圾邮件。我们比对一下真实标签与预测标签：

```python
print(f"The example email is labeled as: {Y[example_index]}")
print(f"Naive bayes model classifies it as: {naive_bayes(treated_email, word_frequency, class_frequency)}")

```

这是一封原本正常的邮件，却因为计算错误被粗暴地拦截到了垃圾箱！ 为什么两个非零概率的连乘积最终都变成了 $0$？答案就藏在底层的数值计算精度之中！

回顾朴素贝叶斯的核心公式：

$$P(\text{email} \mid \text{spam}) = P(\text{word}_1 \mid \text{spam}) \cdot P(\text{word}_2 \mid \text{spam}) \cdots P(\text{word}_n \mid \text{spam})$$

这是整封邮件中**每一个**单词条件概率的长链条连乘。

```python
print(f"The example email has: {len(treated_email)} words in the product.")

```

这封邮件在预处理后整整包含 2657 个词汇！ 我们来看看前 3 个词的条件概率分布：

```python
for i in range(3):
    word = treated_email[i]
    p_word_given_ham = prob_word_given_class(word, cls = 'ham', word_frequency = word_frequency, class_frequency = class_frequency)
    print(f"Word: {word}. P({word} | ham) = {p_word_given_ham}")

```

每个概率都是位于 $0$ 到 $1$ 之间的小数。将 2657 个介于 $0$ 和 $1$ 之间的小数连续相乘，即使在最理想的情况下 (每个词的概率都保持在 $10^{-1}$ 的数量级) ，最终的乘积也将骤降至 $10^{-2657}$ 这一微观尺度——这是任何通用计算机在双精度浮点数下都无法表示的极小数值。我们打印 Python 浮点数的极限范围：

```python
import sys

print(sys.float_info)

```

可以看到，Python 浮点数所能表示的最小非零正数大约在 $10^{-308}$ 量级，远大于 $10^{-2657}$。由于精度溢出，Python 在连乘中途便强制将其截断为 $0$，导致信息彻底丢失。换句话说，在原有的实现逻辑下，任何超过一定长度的长文本邮件都会因为数值下溢而无条件被误判为垃圾邮件。鉴于该问题严重削弱了算法稳定性，我们必须予以彻底修复。

#### 5.1.1 数值下溢 (Underflow) 问题的对数化解法

这种由于数值过小超出计算机硬件精度极限而导致的现象被称为算术下溢 (underflow problem) 。其根源在于朴素贝叶斯中过长链条的小数连乘。幸运的是，数学工具为我们提供了一个极其优雅的解决途径。

回想一下，朴素贝叶斯算法的核心诉求仅在于**比对相对大小**，绝对概率数值的具体大小毫无意义。这正是我们在推导初期能够直接舍弃共有分母的根本原因：

$$P(\text{spam} \mid \text{email}) = \frac{P(\text{spam}) \cdot P(\text{email} \mid \text{spam})}{P(\text{email})}$$

$$P(\text{ham} \mid \text{email}) = \frac{P(\text{ham}) \cdot P(\text{email} \mid \text{ham})}{P(\text{email}) }$$

只要能够无损判断下列两项乘积谁大谁小：

$$P(\text{spam}) \cdot P(\text{email} \mid \text{spam})$$

$$P(\text{ham}) \cdot P(\text{email} \mid \text{ham})$$

分类决策就依然完全等价。

数学上，如果存在一个函数作用于上述两项时能够严格保持其保序性 (monotonicity) ，那么通过比较该函数映射后的输出，同样能准确锁定具有最大值的类别。

任何**严格单调递增函数**都天生具备这一特性：严格保持极值点的位置不变。因此，只要找到一个单调递增且能化简长链连乘的数学工具即可。这个利器就是对数函数 ($\log$) ！对数运算法则能够奇迹般地将乘法转换为加法！借助单调性，我们只需比较以下两项对数值的大小：

$$\log \left(P(\text{spam}) \cdot P(\text{email} \mid \text{spam}) \right)$$

$$\log \left(P (\text{ham}) \cdot P(\text{email} \mid \text{ham}) \right)$$

以任意类别统写展开：

$$\log \left(P(\text{class}) \cdot P(\text{email} \mid \text{class}) \right) = \log \left(P(\text{class}) \right) + \log \left( P(\text{email} \mid \text{class}) \right)$$

并且条件似然的展开式变为：

$$\log \left( P(\text{email} \mid \text{class}) \right) = \log  \left(P(\text{word}_1 \mid \text{class}) \cdot P(\text{word}_2 \mid \text{class}) \cdots P(\text{word}_n \mid \text{class}) \right) = \log  \left(P(\text{word}_1 \mid \text{class}) \right) + \log \left(P(\text{word}_2 \mid \text{class})\right) + \cdots + \log \left( P(\text{word}_n \mid \text{class}) \right)$$

通过取对数，原本脆弱的极小浮点数超长连乘，被成功转化为累加求和操作，数值稳定性得到了质的飞跃。现在，我们用这一技巧重构代码。需要升级两个函数：

* `prob_email_given_class` $\rightarrow$ `log_prob_email_given_class`：将单词概率的累乘改为对数概率的累加。


* `naive_bayes` $\rightarrow$ `log_naive_bayes`：将乘积 $P(\text{class}) \cdot P(\text{email} \mid \text{class})$ 替换为其对应的对数求和。



```python
def log_prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    基于预处理邮件内容，计算邮件在指定类别下的对数概率值。

    参数：
    - treated_email (list)：预处理后的邮件单词列表。
    - cls (str)：类别标签 ('spam' 或 'ham')。

    返回值：
    - float：邮件属于该类别的对数条件概率。
    """

    # 累加初始值设为 0，因为后续将累加 log(P(word | class))
    prob = 0

    for word in treated_email: 
        # 仅对存在于词频字典中的词进行处理
        if word in word_frequency.keys(): 
            # 累加 log(P(word | class))
            prob += np.log(prob_word_given_class(word, cls,word_frequency, class_frequency))

    return prob

```

```python
# 观察单个词汇在对数转换前后的表现对比
one_word_email = ['schedule']
word = one_word_email[0]
prob_spam = prob_email_given_class(one_word_email, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency)
log_prob_spam = log_prob_email_given_class(one_word_email, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency)
print(f"For word {word}:\n\tP({word} | spam) = {prob_spam}\n\tlog(P({word} | spam)) = {log_prob_spam}")

```

可以看到，$\log$ 成功将极其微小的浮点数转换为了处在合理范围内的负数，并将乘法改造成了更稳定的累加操作。

接下来实现改进后的 `log_naive_bayes`：

```python
def log_naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood = False):    
    """
    通过比较对数概率实现的垃圾邮件检测朴素贝叶斯分类器。

    参数：
    - treated_email (list)：预处理后的邮件单词列表。
    - return_likelihood (bool)：若为 True，返回对数似然估算元组。

    返回值：
    - int：预测类别，1 为垃圾邮件，0 为正常邮件。
    """
    
    # 基于对数累加计算对数似然 log P(email | spam)
    log_prob_email_given_spam = log_prob_email_given_class(treated_email, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency) 

    # 基于对数累加计算对数似然 log P(email | ham)
    log_prob_email_given_ham = log_prob_email_given_class(treated_email, cls = 'ham',word_frequency = word_frequency, class_frequency = class_frequency) 

    # 计算类别先验概率 P(spam)
    p_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam']) 

    # 计算类别先验概率 P(ham)
    p_ham = class_frequency['ham']/(class_frequency['ham'] + class_frequency['spam']) 

    # 计算 log(P(spam)) + log(P(email | spam))
    log_spam_likelihood = np.log(p_spam) + log_prob_email_given_spam 

    # 计算 log(P(ham)) + log(P(email | ham))
    log_ham_likelihood = np.log(p_ham) + log_prob_email_given_ham 

    if return_likelihood == True:
        return (log_spam_likelihood, log_ham_likelihood)
    
    # 比对两者大小；由于对数单调递增，保序性完全保持
    if log_spam_likelihood >= log_ham_likelihood:
        return 1
    else:
        return 0

```

重新测试之前下溢崩溃的长邮件样本：

```python
log_spam_likelihood, log_ham_likelihood = log_naive_bayes(treated_email,word_frequency = word_frequency, class_frequency = class_frequency,return_likelihood = True)
print(f"log_spam_likelihood: {log_spam_likelihood}\nlog_ham_likelihood: {log_ham_likelihood}")

```

结果一目了然：我们得到了两个非零且差异鲜明的对数值！ 此时 `log_ham_likelihood` 显著大于垃圾邮件的对数值，因此 `log_naive_bayes` 能够精准地将其纠正并归类为正常邮件：

```python
print(f"The example email is labeled as: {Y[example_index]}")
print(f"Log Naive bayes model classifies it as: {log_naive_bayes(treated_email,word_frequency = word_frequency, class_frequency = class_frequency)}")

```

让我们重新在完整测试集上评估这一数学优化带来的整体准确率飞跃：

```python
Y_pred = []

for email in X_test:
    prediction = log_naive_bayes(email,word_frequency = word_frequency, class_frequency = class_frequency)
    Y_pred.append(prediction)

true_positives = get_true_positives(Y_test, Y_pred)
true_negatives = get_true_negatives(Y_test, Y_pred)

print(f"The number of true positives is: {true_positives}\nThe number of true negatives is: {true_negatives}")

accuracy = (true_positives + true_negatives)/len(Y_test)

print(f"The accuracy is: {accuracy:.4f}")

```

这是一个**极其震撼**的飞跃！ 在没有调整任何训练数据的前提下，仅仅通过修复底层的**数学计算方式**，测试集上的分类准确率就从 84.82% 飙升到了 99.21%——性能净提升近 17 个百分点！ 这充分彰显了数值稳定性在机器学习算法工程落地中的巨大威力。

### 5.2 提升模型表现：朴素贝叶斯的工程实践

#### 5.2.1 任务背景

在本节中，你将运用上述两版朴素贝叶斯模型 (原始连乘版与对数平滑版)  来解决一个真实的工业级工程挑战：

假设你需要为一款特定的邮件客户端软件开发垃圾邮件拦截系统。该软件的核心业务诉求是：既要高效拦截垃圾邮件保护用户免受骚扰，**又必须极其严格地避免将正常邮件误判进垃圾箱**，因为丢失重要工作邮件的代价难以承受。相比之下，偶尔漏过一两封垃圾邮件到收件箱的容忍度则要高得多。

#### 5.2.2 准确率的局限性与指标深化

回顾我们在前面定义的准确率 (accuracy) ，在真实非平衡分类任务中它存在明显的指标欺骗性。正如开篇统计所示，本数据集中垃圾邮件仅占 23.88%。如果开发一个毫无智能的摆烂规则——**把所有邮件无脑全部放行到收件箱**，其账面准确率也能轻而易举达到 76.12%！

为了客观测度模型在不同风险维度的表现，我们必须拆解出两个核心问题：

* 在全部垃圾邮件中，算法成功抓取出了多少？这部分被称为真正例 (True Positives) 。


* 在全部**正常邮件**中，算法**错误拦截**了多少？这部分被称为假正例 (False Positives) 。**这是工程上必须重点监控的危险指标。**



第一个问题对应着评估指标——召回率 (recall) 。它统计的是实际所有的垃圾邮件中，被模型准确捕获的比例：

$$\text{recall} = \frac{\text{真正例 (被成功识别的垃圾邮件)}}{\text{实际全部垃圾邮件}}$$

由于垃圾邮件最终只会被判定为垃圾邮件 (真正例)  或漏判为正常邮件 (假负例) ，该指标亦写作：

$$\text{recall} =\frac{\text{true positives}}{\text{true positives} + \text{false negatives}}$$

实现召回率评估函数：

```python
def get_recall(Y_true, Y_pred):
    """
    计算二分类任务的召回率 (Recall)。

    参数：
    - Y_true (array-like)：真实标签。
    - Y_pred (array-like)：预测标签。

    返回值：
    - recall (float)：召回率分数。
    """
    # 统计实际垃圾邮件总数（由于标注为 1，直接对 Y 标签求和即可）
    total_number_spams = Y_test.sum()
    true_positives = get_true_positives(Y_true, Y_pred)
    
    recall = true_positives/total_number_spams
    return recall

```

```python
# 分别收集两版模型对测试集的预测结果
Y_pred_naive_bayes = []
Y_pred_log_naive_bayes = []

for email in X_test:
 prediction = naive_bayes(email,word_frequency = word_frequency, class_frequency = class_frequency)
 log_prediction = log_naive_bayes(email,word_frequency = word_frequency, class_frequency = class_frequency)
 Y_pred_naive_bayes.append(prediction)
 Y_pred_log_naive_bayes.append(log_prediction)

# 分别计算召回率
recall_naive_bayes = get_recall(Y_test, Y_pred_naive_bayes)
recall_log_naive_bayes = get_recall(Y_test, Y_pred_log_naive_bayes)

```

```python
print(f"The proportion of spam emails the standard Naive Bayes model can correctly classify as spam (recall) is: {recall_naive_bayes:.4f}")
print(f"The proportion of spam emails the log Naive Bayes model can correctly classify as spam (recall) is: {recall_log_naive_bayes:.4f}")

```

从数据上看，两款模型在**捕获垃圾邮件**方面都展现出了极高的敏感度 (sensitivity) ，均能侦测出超过 98% 的垃圾邮件！

但关键的第二问随之而来：这关乎另一项核心指标——精确率 (precision) 。

在所有被模型拦截并扔进垃圾箱的邮件中，究竟有多少是**真正**的垃圾邮件？ 这一点至关重要——如果一个模型极端到把所有邮件统统判定为垃圾邮件，其召回率固然是完美的 100%，但在实际使用中却形同灾难。

这直接对应于假正例 (False Positives) 的数量。我们编写函数统计究竟有多少正常邮件被错误丢弃：

```python
def get_false_positives(Y_true, Y_pred):
    """
    计算二分类任务中的假正例 (False Positives) 数量。

    参数：
    - Y_true (list)：真实标签列表 (0 或 1)。
    - Y_pred (list)：预测标签列表 (0 或 1)。

    返回值：
    - int：假正例数量（真实标签为 0 但预测标签为 1）。
    """
    
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)

    false_positives = 0
    for i in range(n):
        true_label_i = Y_true[i]
        predicted_label_i = Y_pred[i]
        # 仅当真实为 0（正常邮件）但模型预测为 1（垃圾邮件）时计数
        if true_label_i == 0 and predicted_label_i == 1:
            false_positives += 1
    return false_positives

```

```python
# 统计两款模型在测试集上产生的假正例数量
 
false_positives_naive_bayes = get_false_positives(Y_test, Y_pred_naive_bayes)
false_positives_log_naive_bayes = get_false_positives(Y_test, Y_pred_log_naive_bayes)

```

```python
print(f"Number of false positives in the standard Naive Bayes model: {false_positives_naive_bayes}")
print(f"Number of false positives in the log Naive Bayes model: {false_positives_log_naive_bayes}")

```

对比极其悬殊！原始版本的模型误杀了 169 封正常邮件，而对数优化后的模型仅仅误拦截了 4 封！

为了更科学地量化这一可靠性，我们计算精确率 (precision) ：即模型识别出的“垃圾邮件”中真正垃圾邮件所占的纯度。注意分母由两部分构成：被正确拦截的垃圾邮件 (真正例)  加上被误伤的正常邮件 (假正例)。

$$\text{precision} = \frac{\text{true positives}}{\text{true positives} + \text{false positives}}$$

这一指标直接衡量了模型报警的**可信度与相关性**。实现精确率函数：

```python
def get_precision(Y_true, Y_pred):
    """
    计算二分类模型的精确率 (Precision)。

    参数：
    - Y_true (list)：真实标签。
    - Y_pred (list)：预测标签。

    返回值：
    - precision (float)：精确率分数。
    """
    true_positives = get_true_positives(Y_true, Y_pred)
    false_positives = get_false_positives(Y_true, Y_pred)
    precision = true_positives/(true_positives + false_positives)
    return precision

```

```python
print(f"Precision of the standard Naive Bayes model: {get_precision(Y_test, Y_pred_naive_bayes):.4f}")
print(f"Precision of the log Naive Bayes model: {get_precision(Y_test, Y_pred_log_naive_bayes):.4f}")

```

原始模型的精确率仅有 59.57%。换句话说，在它标记为垃圾邮件的 100 封邮件里，有大约 40 封其实是重要的正常邮件——这种漏检虽少但误杀严重的系统完全无法在生产环境中上线使用。

而在改用对数似然进行数值稳定性优化后，模型的精确率跃升至惊人的 98.42%！ 此时在模型判定的每 100 封垃圾邮件中，仅有约 2 封属于误判，系统的可用性与稳健度达到了真正可工程化落地的标准。

祝贺你！你不仅圆满完成了本次作业的全部实现任务，还深入领会了附录中关于数值下溢与分类评估指标的实战优化！