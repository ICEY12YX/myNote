___
# 朴素贝叶斯 (Naive Bayes)

欢迎来到“机器学习与数据科学中的概率与统计 (Probability and Statistics for Machine Learning and Data Science) ”课程的第一个作业，这也是“机器学习与数据科学数学基础 (Math for Machine Learning and Data Science) ”专项课程的最后一门课程！

在本次作业中，你将像在课程视频中学习的那样，亲手实现朴素贝叶斯 (Naive Bayes) 算法来解决垃圾邮件检测问题。第 1 至 3 节为该问题提供了实用的背景知识。在第 4 节中，你将编写具体的函数来实现该算法。第 5 节则包含一些不计分但非常有趣的拓展内容。

___

___

## 1 - 简介

朴素贝叶斯算法是机器学习 (Machine Learning) 和数据科学 (Data Science) 领域的奠基之作，它巧妙地利用贝叶斯定理 (Bayes' Theorem) 来推断某个数据样本究竟归属于哪一个具体类别。该算法做出了一个看似天真的“朴素假设” (naive assumption) ：即各个特征之间彼此相互独立。在现实世界的数据中，这个假设几乎必然是不成立的；但正是这一假设，让算法的实现变得极其简单轻巧，而且正如你即将看到的，它所取得的效果往往令人惊艳。需要说明的是，朴素贝叶斯是一种监督学习 (Supervised Learning) 算法，这意味着它必须依赖已经标注好的数据才能正常运转。在我们即将展开的示例中，这就要求在训练算法之前，必须先收集好一批已经被明确标记为“垃圾邮件” (spam) 或“正常邮件” (ham) 的样本。

### 运用朴素贝叶斯进行垃圾邮件检测

本项任务专注于一个经典的二分类 (Binary Classification) 问题：区分垃圾邮件 (spam) 与非垃圾邮件（在英文日常语境中常被称为“ham”，即正常邮件）。在本次任务中，我们将垃圾邮件标记为 $1$，正常邮件标记为 $0$。

对于一封给定的邮件，我们重点关注的条件概率记为：

$$P(\text{spam} \mid \text{email})$$

这个概率值越大，意味着该邮件属于垃圾邮件的可能性越高。在课堂讲授中提到的贝叶斯定理，正是通过以下方式应用于这一计算的：

$$P(\text{spam} \mid \text{email}) = \frac{P(\text{email} \mid \text{spam}) \cdot P(\text{spam})}{P(\text{email})}$$

各项的具体含义如下：

* $P(\text{spam})$：随机抽取一封邮件为垃圾邮件的先验概率，在数值上等同于数据集中垃圾邮件所占的比例。


* $P(\text{email} \mid \text{spam})$：在已知邮件是垃圾邮件的前提下，出现当前特定这封邮件内容的似然概率。


* $P(\text{email})$：当前这封邮件内容在所有邮件中出现的总边缘概率。



在这种方法中，我们可以巧妙地走一个“捷径”：直接忽略分母上的 $P(\text{email})$ 项。因为我们的最终目的并不是精确求解具体的概率数值，而是比较“这封邮件是垃圾邮件的概率”与“它是正常邮件的概率”谁大谁小。下面列出 $P(\text{spam} \mid \text{email})$ 和 $P(\text{ham} \mid \text{email})$ 的完整表达式：

$$P(\text{spam} \mid \text{email}) = \frac{P(\text{email} \mid \text{spam}) \cdot P(\text{spam})}{P(\text{email})}$$

$$P(\text{ham} \mid \text{email}) = \frac{P(\text{email} \mid \text{ham}) \cdot P(\text{ham})}{P(\text{email})}$$

由于对于任何已存在的邮件都有 $P(\text{email}) > 0$，并且它在两个公式的分母中完全相同，因此比较两者大小时，只需评估分子的大小即可，完全可以把分母抛在一边。

## 2 - 必要的依赖导入

接下来的代码块将导入本作业所需的全部依赖库和核心函数，并载入单元测试工具，在你的编写过程中实时给出反馈。

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

### 3.1 加载并探索数据集

下面的代码块会将数据集加载到内存中。你将使用 Pandas 库将其读取为核心数据结构 DataFrame。不过如果你对 Pandas 还不算熟悉，**完全不必担心**。此处加载它只是为了直观展示数据结构，在后续的实际运算中，你将使用熟悉的 NumPy 数组进行操作。

```python
dataframe_emails = pd.read_csv('emails.csv')
dataframe_emails.head()

```

让我们初步探索一下数据集：

```python
print(f"Number of emails: {len(dataframe_emails)}")
print(f"Proportion of spam emails: {dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")
print(f"Proportion of ham emails: {1-dataframe_emails.spam.sum()/len(dataframe_emails):.4f}")

```

可以看到，这个数据集是**不平衡** (unbalanced) 的。其中正常邮件的数量是垃圾邮件的两倍以上！在任何数据分析项目中，这类背景信息都至关重要，因为它可能会影响包括朴素贝叶斯在内的诸多机器学习算法的实际运行效果。

### 3.2 预处理数据集

该数据表包含两列。名为 `text` 的列存储邮件的具体文本内容，名为 `spam` 的列则用数值指示该邮件是否为垃圾邮件。请记住：$1$ 代表垃圾邮件，而 $0$ 代表正常邮件 (ham) 。接下来的函数将完成两项关键的预处理步骤：

* 原数据集中每封邮件开头均带有 `Subject:` 前缀。该函数会剔除每封邮件最前方的这部分字符。


* 对数据集进行随机打乱 (shuffle) 。原始数据中所有垃圾邮件都集中在最前面，其后才是正常邮件。只有先打乱数据，才能在后续合理地将数据划分为训练集和测试集。



如果你对该函数中的某些 Python 写法稍有陌生也无需介意，这里展示它旨在提醒大家：在正式展开模型分析前，探索并预处理数据往往是必经的关键步骤。

```python
def preprocess_emails(df):
    """
    预处理来自 DataFrame 的邮件数据。

    参数：
    - df (pandas.DataFrame): 包含邮件数据的输入 DataFrame，带有 'text' 和 'spam' 列。

    返回值：
    - tuple: 包含两个元素的元组：
        1. X (numpy.array): 剔除 "Subject:" 前缀后的邮件内容数组。
        2. Y (numpy.array): 指示每封邮件是否为垃圾邮件 (1) 或正常邮件 (0) 的标签数组。

    该函数会打乱输入的 DataFrame 以避免训练集/测试集划分中的偏差。
    随后提取邮件文本和垃圾邮件标签，并去除每封邮件的 "Subject:" 前缀。
    """
    # 打乱数据集
    df = df.sample(frac = 1, ignore_index = True, random_state = 42)
    # 去除构成每封邮件前 9 个字符的 "Subject:" 字符串，并转换为 numpy 数组
    X = df.text.apply(lambda x: x[9:]).to_numpy()
    # 将标签转换为 numpy 数组
    Y = df.spam.to_numpy()
    return X, Y

```

```python
X, Y = preprocess_emails(dataframe_emails)

```

让我们打印前 $5$ 封邮件看看：

```python
print(X[:5])

```

以及前 $5$ 个标签：

```python
print(Y[:5])

```

请注意，NumPy 数组 `X` 是由字符串构成的数组，其中每个元素代表一封邮件，且 `X` 中的索引位置与 `Y` 中的索引位置严格对应，指明了该邮件是否属于垃圾邮件。**你可以试着修改 `email_index` 的数值**，查看不同邮件的文本内容及其分类标签。牢记：0 代表正常邮件，1 代表垃圾邮件。

```python
email_index = 30
print(f"Email index {email_index}: {X[email_index]}\n\n")
print(f"Class: {Y[email_index]}")

```

### 3.3 预处理文本

这部分内容虽然在先前的课堂视频中未作深入探讨，且不包含计分任务，但它对于处理文本数据极为关键，掌握它必将为你在机器学习和数据科学道路上的探索提供极大助益！

在人类语言中，通常存在一些本身并不承载太多核心语义信息的词汇，例如介词、代词等。这类词汇被称为**停用词** (stopwords) 。因为它们在几乎所有文本中都高频出现，往往无法为我们的分类任务提供有价值的区分信息。因此，文本预处理的核心思路就是剔除所有的停用词和标点符号，最终只留下一个精炼且更具代表性的词汇集合。下面的函数所做的正是这件事。

另一个重要步骤是邮件的**分词** (tokenization) 。所谓分词，就是将一整封邮件拆解成一系列独立的**词元** (tokens) ，本质上就是提取出其中的各个单词。最终，针对每一封邮件，处理结果都会被转化为一个 NumPy 数组，其中整齐排列着剔除了停用词和标点符号后的所有单词。

```python
def preprocess_text(X):
    """
    通过剔除停用词和标点符号来预处理文本数据集合。

    参数：
    - X (str 或 array-like): 待处理的输入文本数据。如果传入单个字符串，
      会被转换为单元素 numpy 数组。

    返回值：
    - numpy.array: 预处理后的文本数据数组，其中每个元素代表一个剔除了停用词和标点符号的文档。

    说明：
    - 该函数使用自然语言处理工具包 (nltk) 库进行分词和停用词过滤。
    - 如果输入是单个字符串，它会被转换为单元素 numpy 数组。
    """
    # 构建包含停用词和标点符号的集合
    stop = set(stopwords.words('english') + list(string.punctuation))

    # 处理传入单封邮件而非邮件数组的情况
    if isinstance(X, str):
        X = np.array([X])

    # 结果将存储在列表中
    X_preprocessed = []

    for i, email in enumerate(X):
        email = np.array([i.lower() for i in word_tokenize(email) if i.lower() not in stop]).astype(X.dtype)
        X_preprocessed.append(email)
        
    if len(X) == 1:
        return X_preprocessed[0]
    return X_preprocessed

        
        

```

```python
# 该函数运行可能需要几秒钟，通常不到 1 分钟。
X_treated = preprocess_text(X)

```

预处理完成后，每封邮件的文本都被转化为了一个去除了所有停用词的 NumPy 词汇数组。下面的例子展示了一个随机抽取的邮件样本 (此处索引为 989) 在预处理前后的具体对比。你可以随意尝试不同的索引值，观察不同邮件的处理效果。这个清洗完毕的单词序列，才是后续算法真正读取和计算的输入。

```python
email_index = 989
print(f"Email before preprocessing: {X[email_index]}")
print(f"Email after preprocessing: {X_treated[email_index]}")

```

### 3.4 划分训练集与测试集

现在我们将数据集划分为训练集与测试集。我们按照 80/20 的经典比例进行划分，即 80% 的数据用于模型训练，剩余 20% 的数据用于评估测试。

```python
TRAIN_SIZE = int(0.80*len(X_treated)) # 80% 的样本用于训练。

X_train = X_treated[:TRAIN_SIZE]
Y_train = Y[:TRAIN_SIZE]
X_test = X_treated[TRAIN_SIZE:]
Y_test = Y[TRAIN_SIZE:]

```

回想一下，整个数据集中大约有 24% 的邮件属于垃圾邮件。检查训练集和测试集中这一比例是否大致持平非常重要，否则训练出的模型可能会产生意外的偏差。

```python
print(f"Proportion of spam in train dataset: {sum(Y_train == 1)/len(Y_train):.4f}")
print(f"Proportion of spam in test dataset: {sum(Y_test == 1)/len(Y_test):.4f}")

```

两者虽然并非分毫不差，但数值已经非常接近，完全满足实验要求！

## 4 - 实现朴素贝叶斯算法

再次重申我们的核心目标：对比 $P(\text{spam} \mid \text{email})$ 与 $P(\text{ham} \mid \text{email})$ 的相对大小，判断哪一个值更高。为了完成这一对比，我们只需分别计算 $P(\text{spam}) \cdot P(\text{email} \mid \text{spam})$ 和 $P(\text{ham}) \cdot P(\text{email} \mid \text{ham})$ 即可。

### 4.1 计算 $P(\text{email} \mid \text{spam})$ 与 $P(\text{email} \mid \text{ham})$

两者的计算逻辑完全一致，我们先以垃圾邮件的情况为例展开分析。

每一封邮件本质上都是由一连串单词构成的列表。我们的目标是计算：在已知这封邮件是垃圾邮件的前提下，观察到这组词汇序列出现的概率有多大。实现该目标的方法是运用概率的乘法规则。将一封邮件表示为单词集合 $\text{email} = \{\text{word}_1, \text{word}_2, \ldots, \text{word}_n \}$，计算式展开为：

$$P(\text{email} \mid \text{spam}) = P(\text{word}_1 \mid \text{spam}) \cdot P(\text{word}_2 \mid \text{spam}) \cdots P(\text{word}_n \mid \text{spam})$$

这正是我们引入“朴素假设” (naive assumption) 的关键所在，也是“朴素贝叶斯”之名的由来！我们大胆假设：每个单词在邮件中出现的概率，都与其他单词的出现完全独立。不用怀疑，这个假设在现实中肯定是站不住脚的。包含“聚会 (party) ”一词的邮件，显然更有可能出现“邀请 (invitation) ”；而出现“大奖 (prize) ”的邮件，也很可能伴随着“祝贺 (congratulations) ”。然而，正是通过做出这种在统计上并不完全符合真实的独立性假设，我们才得以直接应用连乘规则。我们不必再去费心估算词与词之间极为复杂的联合条件概率网，只需简单假定各词独立，并将一系列相对简单的单词条件概率相乘即可。虽然朴素贝叶斯建立在对数据的粗糙假设之上，但正如你接下来会看到的，它往往能产生令人刮目相看的出色效果！

那么，在已知邮件为垃圾邮件的条件下，某个特定单词 $\text{word}_1$ 出现的概率到底该如何实际计算呢？公式如下：

$$P(\text{word}_1 \mid \text{spam}) = \frac{\text{\# 包含 } \text{word}_1 \text{ 的垃圾邮件数量}}{\text{\# 垃圾邮件总数}}$$

其中符号 # 代表元素个数，即分子表示包含该单词的垃圾邮件总篇数。

这其实是一个非常直观的计数计算：统计出包含 $\text{word}_1$ 的垃圾邮件篇数，再除以垃圾邮件的总篇数即可。遍历数据集中出现的每一个单词并重复此过程，你就做好了评估任意邮件在垃圾或正常类别下出现概率的准备。**因此，你的第一个任务就是创建一个名为 `word_frequency` 的字典，用来记录数据集中每个单词分别在正常邮件和垃圾邮件中出现的频次**。

#### 4.1.1 处理连乘中的 0 概率

如果遇到某个只在垃圾邮件中出现、或从未在垃圾邮件中出现过的生词，可能会导致 $P(\text{word} \mid \text{spam}) = 0$（或者在正常邮件中出现类似情况），进而让整个乘积瞬间归零。这种情况极为不利，因为仅仅一个罕见词就会抹杀掉整封邮件累积的全部概率信息。为了消除这一隐患，你将**在统计每个单词的垃圾/正常频次时，默认从 1 开始计数**。通过人为假定每个单词都至少在垃圾邮件和正常邮件中各出现过一次（即拉普拉斯平滑思想），便彻底杜绝了计算中出现 $0$ 的可能。

### 4.2 计算 $P(\text{spam})$ 与 $P(\text{ham})$

在使用贝叶斯定理时，我们还需要结合整批邮件中正常邮件与垃圾邮件各自的宏观出现概率（先验概率）。这项计算非常简单，只需取数据集中垃圾邮件与正常邮件各自所占的比例即可：

$$P(\text{spam}) = \frac{\text{\# 垃圾邮件数量}}{\text{\# 邮件总数}}$$

$$P(\text{ham}) = \frac{\text{\# 正常邮件数量}}{\text{\# 邮件总数}}$$

### 4.3 组合模型

要评估一封邮件究竟更有可能是垃圾邮件还是正常邮件，只需将先前计算得到的各项相乘，然后对比哪一方的数值更大即可：

* $P(\text{spam}) \cdot P(\text{email} \mid \text{spam})$

* $P(\text{ham}) \cdot P(\text{email} \mid \text{ham})$


### 练习 1

你的任务是实现一个函数，用于生成一个词频字典，记录数据集中各个词汇分别在垃圾邮件 (1) 和正常邮件 (0) 中出现的频次。

```python
def get_word_frequency(X,Y):
    """
    计算分类为垃圾邮件 (1) 或非垃圾邮件 (0) 的邮件集合中每个单词的频次。

    参数：
    - X (numpy.array): 邮件数组，其中每封邮件表示为一个单词列表。
    - Y (numpy.array): 与 X 中每封邮件对应的标签数组。1 表示垃圾邮件，0 表示正常邮件。

    返回值：
    - word_dict (dict): 一个字典，键为邮件中出现的唯一词汇，值为包含每个词在垃圾邮件 (1) 
      和正常邮件 (0) 中出现频次的字典。
    """
    # 创建一个空字典
    word_dict = {}

    ### START CODE HERE ###

    num_emails = None

    # 遍历每封处理后的邮件及其标签
    for i in range(num_emails):
        # 获取第 i 封邮件
        email = X[i] 
        # 获取第 i 个标签。指示该邮件是否为垃圾邮件。1 = None
        # 变量名 cls 是 class 的缩写，因为 class 在 Python 中是保留关键字。
        cls = Y[i] 
        # 为了避免同一封邮件中重复计算同一个单词，通过将邮件转为集合 (set) 来去重
        email = set(email) 
        # 遍历邮件中的每个唯一单词
        for word in email:
            # 如果该单词尚不在字典中，手动添加它。记住：每个单词在垃圾和正常邮件中的初始计数均从 1 开始
            if word not in word_dict.keys():
                word_dict[word] = {"spam": None, "ham": None}
            # 当 cls == 0 时在键 ham 下增加一次出现频次；当 cls == 1 时在 spam 下增加一次
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

##### **预期输出** (键的输出顺序可能有所不同，关键在于每个单词对应的数值统计)

```Python
{'going': {'spam': 2, 'ham': 1}, 'river': {'spam': 2, 'ham': 3}, 'like': {'spam': 2, 'ham': 1}, 'deep': {'spam': 1, 'ham': 2}, 'love': {'spam': 1, 'ham': 2}, 'hate': {'spam': 1, 'ham': 2}}

```

接下来的代码块将测试你的函数。别担心，这部分尚不计入最终评分，仅用于确保你的函数运行无误。若未通过单元测试，控制台会输出提示信息，方便你在进入下一项练习前进行调试修改。

```python
w1_unittest.test_get_word_frequency(get_word_frequency)

```

```python
# 这将使用训练集构建 word_frequency 字典。
word_frequency = get_word_frequency(X_train,Y_train)

```

你还需要一个类别频次字典，用来记录训练集中正常邮件 (0) 和垃圾邮件 (1) 的总数。下面的代码行会自动为你创建它。

```python
# 要统计垃圾邮件和正常邮件的数量，只需对训练数据集中的 1 和 0 分别求和即可，因为约定 spam = 1 且 ham = 0。
class_frequency = {'ham': sum(Y_train == 0), 'spam': sum(Y_train == 1)}

```

```python
print(class_frequency)

```

若要获取训练集中垃圾邮件的占比，只需执行：

```python
# 核心思路是计算 (垃圾邮件数量) / (邮件总数)。
# 由于邮件非黑即白（要么是垃圾邮件，要么是正常邮件），所以邮件总数 = (正常邮件数量) + (垃圾邮件数量)。
proportion_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam'])
print(f"The proportion of spam emails in training is: {proportion_spam:.4f}")

```

可以看到，该结果与此前在数据探索单元中得到的比例完全吻合！

### 练习 2

在接下来的练习中，你将实现计算 $P(\text{word} \mid \text{spam})$ 与 $P(\text{word} \mid \text{ham})$ 的函数。由于两种邮件类别的计算逻辑一致，你可以统一编写一个计算 $P(\text{word} \mid \text{class})$ 的通用函数，其中的类别 `class` 既可以是垃圾邮件 ($1$) 也可以是正常邮件 ($0$) 。

回顾其计算公式：

$$P(\text{word}_i \mid \text{class}) = \frac{\text{\# 该类别 (垃圾或正常)  中包含 } \text{word}_i \text{ 的邮件数量}}{\text{\# 该给定类别 (垃圾或正常)  的邮件总数}}$$

注意：当前我们暂且不需要担心单词是否在字典中存在的问题，这一逻辑将在后续函数中统一处理。

```python
def prob_word_given_class(word, cls, word_frequency, class_frequency):
    """
    计算给定单词在特定类别中出现的条件概率。

    参数：
    - word (str): 计算概率的目标单词。
    - cls (str): 目标类别，可以是 'spam' 或 'ham'。
    - word_frequency (dict): 包含词频统计的字典。
    - class_frequency (dict): 包含类别频次统计的字典。

    返回值：
    - float: 给定单词在指定类别中出现的条件概率。
    """
    ### START CODE HERE ###
    
    # 获取该单词在指定类别下出现的次数 (类别存储在变量中)
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

##### **预期输出** (最后几位小数位可能略有差异)

```Python
P(lottery | spam) = 0.00807899461400359
P(lottery | ham) = 0.0002883506343713956
P(schedule | spam) = 0.008976660682226212
P(schedule | ham) = 0.10294117647058823

```

接下来的代码块将测试你的函数。别担心，这部分尚不计分，仅用于验证函数正确性。若测试失败会显示反馈提示。

```python
w1_unittest.test_prob_word_given_class(prob_word_given_class, word_frequency, class_frequency)

```

### 练习 3

接下来，你将实现计算 $P(\text{email} \mid \text{class})$ 的函数，其中类别可以是垃圾邮件 (1) 或正常邮件 (0) 。你将使用经典的*朴素假设*：

$$P(\text{email} \mid \text{class}) = P(\text{word}_1 \mid \text{class}) \cdot P(\text{word}_2 \mid \text{class}) \cdots P(\text{word}_n \mid \text{class})$$

其核心逻辑是遍历邮件中的每个单词，并在每一步迭代中，将当前累积概率乘以该单词的条件概率 $P(\text{word} \mid \text{class})$。

提示：在 Python 中更新连乘值时，可以使用简洁的 `value *= update` 语法，它与 `value = value * update` 完全等价。

```python
def prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    根据处理后的邮件内容，计算邮件属于特定类别 (如 spam 或 ham) 的概率。

    参数：
    - treated_email (list): 邮件中经过预处理的单词列表。
    - cls (str): 邮件的类别标签，可以是 'spam' 或 'ham'。
    - word_frequency (dict): 包含词频统计的字典。
    - class_frequency (dict): 包含类别频次统计的字典。

    返回值：
    - float: 给定邮件属于指定类别的概率。
    """

    # prob 初始设为 1，因为在每次迭代中它都会通过乘以当前的 P(word | class) 进行更新
    prob = 1

    ### START CODE HERE ###

    for word in None:
        # 仅对存在于词频字典中的单词执行计算
        if word in word_frequency.keys(): 
            # 将 prob 乘以 P(word | class) 来更新概率。切记传入 word_frequency 和 class_frequency 参数！
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

##### **预期输出** (最后几位小数位可能略有差异)

```Python
Email: Click here to win a lottery ticket and claim your prize!
Email after preprocessing: ['click' 'win' 'lottery' 'ticket' 'claim' 'prize']
P(email | spam) = 5.3884806600117164e-11
P(email | ham) = 1.2428344868918976e-15

```

接下来的代码块将测试你的函数，以确保代码正确无误：

```python
w1_unittest.test_prob_email_given_class(prob_email_given_class, word_frequency, class_frequency)

```

### 练习 4

在本练习中，你将执行以下两组关键计算，从而评估一封邮件分别属于垃圾邮件和正常邮件的可能性大小：

* $P(\text{spam}) \cdot P(\text{email} \mid \text{spam})$

* $P(\text{ham}) \cdot P(\text{email} \mid \text{ham})$


算法最终会将两者中数值更高的一方作为该邮件的预测类别。请注意，下面的函数中包含一个专门的参数选项，允许函数直接返回这两个似然值，而非仅仅输出最终分类结果。

**注**：你会注意到函数的最终预测输出是一个整数类别代码。虽然直接返回字符串形式的 "spam" 或 "ham" 也很直观，但让模型直接输出数值代码更有利于后续的指标评估与数学运算。

```python
def naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood = False):    
    """
    用于垃圾邮件检测的朴素贝叶斯分类器。

    该函数基于朴素贝叶斯算法计算邮件为垃圾邮件 (1) 或正常邮件 (0) 的概率。
    它使用处理后邮件在 spam 和 ham 下的条件概率，以及两类的先验概率。
    最终决策通过比较计算出的概率值做出。

    参数：
    - treated_email (list): 输入邮件预处理后的表示。
    - word_frequency (dict): 包含词频统计的字典。
    - class_frequency (dict): 包含类别频次统计的字典。
        - return_likelihood (bool): 若为 True，则返回 spam 和 ham 的似然值。

    返回值：
    若 return_likelihood = False:
        - int: 若邮件分类为垃圾邮件返回 1，若分类为正常邮件返回 0。
    若 return_likelihood = True:
        - tuple: 格式为 (spam_likelihood, ham_likelihood) 的元组。
    """

    ### START CODE HERE ###
    
    # 使用上面刚刚定义的函数计算 P(email | spam)。别忘了传入 word_frequency 和 class_frequency 参数！
    prob_email_given_spam = None

    # 使用上面刚刚定义的函数计算 P(email | ham)。别忘了传入 word_frequency 和 class_frequency 参数！
    prob_email_given_ham = None

    # 使用 class_frequency 字典并利用公式 #spam emails / #total emails 计算 P(spam)
    p_spam = None

    # 使用 class_frequency 字典并利用公式 #ham emails / #total emails 计算 P(ham)
    p_ham = None

    # 计算乘积 P(spam) * P(email | spam)，我们称之为 spam_likelihood
    spam_likelihood = None * None

    # 计算乘积 P(ham) * P(email | ham)，我们称之为 ham_likelihood
    ham_likelihood = None * None


    ### END CODE HERE ###
    
    # 若传入 return_likelihood = True，则返回目标元组
    if return_likelihood == True:
        return (spam_likelihood, ham_likelihood)
    
    # 比较两个数值并选择对应较大数值的类别
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

接下来的代码块将测试你的函数，以确保代码能够正常工作：

```python
w1_unittest.test_naive_bayes(naive_bayes, word_frequency, class_frequency)

```

### 4.4 模型表现

本小节不包含计分练习，因为其探讨的主题已经超出了核心讲义的理论范围。但我们非常建议你仔细研读并理解其中的逻辑，因为评估和衡量模型表现是构建任何机器学习模型时的必修功课。

在这一节中，你将全面检验刚刚搭建的模型在实际数据上的表现。回顾之前的步骤，我们在 80% 的数据上训练了模型，并预留了 20% 的独立测试数据对其进行验证。此时最自然的问题便是：该模型在全新测试集上的分类准确率到底如何？为了回答这一问题，最经典的评估指标就是准确率 (Accuracy) 。它衡量的是模型正确预测样本数占总样本数的比例。

要计算准确率，你需要统计：

* 模型成功识别为垃圾邮件的真实垃圾邮件数量（称为**真正例** (True Positives) ）


* 模型成功识别为正常邮件的真实正常邮件数量（称为**真反例** (True Negatives) ）



最后，将真正例与真反例的总和除以总观测样本数，便得到了准确率。若模型达到完美预测，准确率将为 1（即 100%）。下面的代码块实现了这一指标的计算过程：

```python
def get_true_positives(Y_true, Y_pred):
    """
    计算二分类任务中的真正例 (True Positives) 数量。

    参数：
    - Y_true (list): 每个样本的真实标签列表 (0 或 1)。
    - Y_pred (list): 每个样本的预测标签列表 (0 或 1)。

    返回值：
    - int: 真实标签和预测标签均为 1 的真正例数量。
    """
    # Y_true 和 Y_pred 的长度必须一致。
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_positives = 0
    # 遍历列表中的每个元素
    for i in range(n):
        # 获取当前邮件的真实标签
        true_label_i = Y_true[i]
        # 获取当前邮件的模型预测标签
        predicted_label_i = Y_pred[i]
        # 仅当真实标签为 1 且预测标签为 1 时计数器加 1 (真正例)
        if true_label_i == 1 and predicted_label_i == 1:
            true_positives += 1
    return true_positives
        
def get_true_negatives(Y_true, Y_pred):
    """
    计算二分类任务中的真反例 (True Negatives) 数量。

    参数：
    - Y_true (list): 每个样本的真实标签列表 (0 或 1)。
    - Y_pred (list): 每个样本的预测标签列表 (0 或 1)。

    返回值：
    - int: 真实标签和预测标签均为 0 的真反例数量。
    """
    
    # Y_true 和 Y_pred 的长度必须一致。
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)
    true_negatives = 0
    # 遍历列表中的每个元素
    for i in range(n):
        # 获取当前邮件的真实标签
        true_label_i = Y_true[i]
        # 获取当前邮件的模型预测标签
        predicted_label_i = Y_pred[i]
        # 仅当真实标签为 0 且预测标签为 0 时计数器加 1 (真反例)
        if true_label_i == 0 and predicted_label_i == 0:
            true_negatives += 1
    return true_negatives
        

```

```python
# 让我们获取测试集上的预测结果：

# 创建空列表存储预测值
Y_pred = []


# 遍历测试集中的每封邮件
for email in X_test:
    # 执行预测
    prediction = naive_bayes(email, word_frequency, class_frequency)
    # 添加到列表中 
    Y_pred.append(prediction)

# 检查 Y_pred 与 Y_test (真实标签) 的长度是否匹配：
print(f"Y_test and Y_pred matches in length? Answer: {len(Y_pred) == len(Y_test)}")

```

```python
# 获取真正例数量：
true_positives = get_true_positives(Y_test, Y_pred)

# 获取真反例数量：
true_negatives = get_true_negatives(Y_test, Y_pred)

print(f"The number of true positives is: {true_positives}\nThe number of true negatives is: {true_negatives}")

# 通过将真反例与真正例相加并除以数据集总元素数来计算准确率。
# 由于 Y_pred 和 Y_test 长度相同，使用哪一个都可以。
accuracy = (true_positives + true_negatives)/len(Y_test)

print(f"Accuracy is: {accuracy:.4f}")

```

干得漂亮！你已经成功构建了一个扎实的朴素贝叶斯模型，尽管它仅仅基于“邮件中的每个词都是彼此孤立”这一朴素假设，却依然在测试集上取得了高达 84.82% 的优异准确率！现在，你可以在下面的代码块中输入自己编写的邮件，亲自感受模型的预测能力。

```python
email = "Please meet me in 2 hours in the main building. I have an important task for you."
# email = "You win a lottery prize! Congratulations! Click here to claim it"

# 预处理邮件
treated_email = preprocess_text(email)
# 获取预测结果，为了更清晰地打印输出，若输出为 1 则显示为 "spam"，否则显示为 "ham"。
prediction = "spam" if naive_bayes(treated_email, word_frequency, class_frequency) == 1 else "ham"
print(f"The email is: {email}\nThe model predicts it as {prediction}.")

```

## 5 - 附录 (非评分部分)

以下各节不属于作业计分范围，但展示了基于上述工作的一些极具深度的工程延伸。如果你希望现在直接提交作业获取评分，可以先行提交；如果你渴望进一步探究底层奥秘，不妨继续阅读以下内容。

### 5.1 朴素贝叶斯模型中隐藏的问题

在当前模型中，潜藏着一个严重制约其性能发挥的隐蔽缺陷。让我们通过一个具体的测试用例，手动执行一次朴素贝叶斯计算来揭示这一问题。

```python
example_index = 4798
example_email = X[example_index]
treated_email = preprocess_text(example_email)
print(f"The email is:\n\t{example_email}\n\nAfter preprocessing:\n\t:{treated_email}")

```

现在我们调用 `return_likelihood = True`，分别计算这封邮件的 $P(\text{spam}) \cdot P(\text{email} \mid \text{spam})$ 和 $P(\text{ham}) \cdot P(\text{email} \mid \text{ham})$：

```python
spam_likelihood, ham_likelihood = naive_bayes(treated_email, word_frequency = word_frequency, class_frequency = class_frequency, return_likelihood = True)
print(f"spam_likelihood: {spam_likelihood}\nham_likelihood: {ham_likelihood}")

```

奇怪的事情发生了：垃圾邮件和正常邮件的似然值居然**全部变成了 0**！这怎么可能呢？顺便一提，根据我们预设的判断规则，当 $\text{spam\_likelihood} \geq \text{ham\_likelihood}$ 时模型将判定为 1（垃圾邮件），因此由于两者皆为 0，这封邮件被直接武断地归为了垃圾邮件。让我们对比一下真实标签与预测标签：

```python
print(f"The example email is labeled as: {Y[example_index]}")
print(f"Naive bayes model classifies it as: {naive_bayes(treated_email, word_frequency, class_frequency)}")

```

果然，这封原本正常的邮件被错误地丢进了垃圾箱！更反常的是，两个似然值竟然同时为 0。究竟是什么原因造成了这种现象？答案其实隐藏在底层的数学与计算机存储机制中！

回顾朴素贝叶斯的核心公式：

$$P(\text{email} \mid \text{spam}) = P(\text{word}_1 \mid \text{spam}) \cdot P(\text{word}_2 \mid \text{spam}) \cdots P(\text{word}_n \mid \text{spam})$$

这是邮件中**每一个**单词条件概率的连乘积。

```python
print(f"The example email has: {len(treated_email)} words in the product.")

```

原来，我们正在测试的这封邮件在预处理后足足包含 2657 个单词！让我们看看其中前 3 个词在正常邮件下的条件概率 $P(\text{word} \mid \text{ham})$：

```python
for i in range(3):
    word = treated_email[i]
    p_word_given_ham = prob_word_given_class(word, cls = 'ham', word_frequency = word_frequency, class_frequency = class_frequency)
    print(f"Word: {word}. P({word} | ham) = {p_word_given_ham}")

```

因为它们都是概率值，所以必然是介于 $0$ 到 $1$ 之间的小数。这意味着我们实际上是在将 2657 个处于 0 到 1 之间的小数连续相乘！哪怕在最理想的情况下，每个词的概率都维持在 $10^{-1}$ 的数量级（类似于上例中的第一个词），整个连乘的结果也将跌落至 $10^{-2657}$ 的微小量级——这是任何现代计算机在标准浮点数精度下都无法直接表达的**极小数值**。让我们查看一下 Python 浮点数 (float) 的存储极限：

```python
import sys

print(sys.float_info)

```

显而易见，Python 能够表示的最小正浮点数量级大约在 $10^{-308}$，它比 $10^{-2657}$ 要庞大得多。因此，在连乘进行到一定阶段后，Python 只能无可奈何地将其截断为 0，导致所有的概率信息彻底丢失。换句话说，在当前的算法实现下，一旦邮件长度超出某个临界阈值，所有的邮件都会被无一例外地错误判定为垃圾邮件！这一问题根植于朴素贝叶斯固有的庞大连乘机制中，我们必须设法予以解决。

#### 5.1.1 数值下溢问题 (The Underflow Problem)

你所遭遇的这一瓶颈在计算机科学中被称为**数值下溢问题** (underflow problem) ，它特指因数值过于微小而超出了计算机浮点精度的表示极限。在本例中，罪魁祸首正是朴素贝叶斯中过长的连乘链条。幸运的是，数学工具为我们提供了一个极其优雅的解决方案。

回想一下，在朴素贝叶斯中，我们其实根本不需要求出精确的后验概率绝对数值，算法的核心本质始终只是**比较相对大小**。这正是我们之前可以心安理得地扔掉分母的原因：

$$P(\text{spam} \mid \text{email}) = \frac{P(\text{spam}) \cdot P(\text{email} \mid \text{spam})}{P(\text{email})}$$

$$P(\text{ham} \mid \text{email}) = \frac{P(\text{ham}) \cdot P(\text{email} \mid \text{ham})}{P(\text{email}) }$$

既然目标仅仅是找出两者的较大者，且两者共享一个严格大于零的相同分母，那么决定胜负的唯有分子。换句话说，这两个分子连乘项的具体数值究竟是多小并不重要：

$$P(\text{spam}) \cdot P(\text{email} \mid \text{spam})$$

$$P(\text{ham}) \cdot P(\text{email} \mid \text{ham})$$

只要我们能准确判断出它们谁比谁大，分类任务就能够顺利完成。

此时，如果我们能找到一个数学函数，在作用于这些数值后依然能够**严格保持其原有的大小顺序**（单调性），那么比较函数变换后的输出结果，就与比较原数值具有完全相同的决策效力。

任何一个**严格单调递增函数**都天生具备这一优良特性：它能够严格保持极大值点的位置不变。因此，解决问题的核心就在于：寻找一个单调递增函数，把朴素贝叶斯中令人头疼的连乘积化繁为简。你能想到这样一个函数吗？没错，就是**对数函数** ($\log$)！众所周知，对数运算拥有化**乘积**为**求和**的神奇魔力！加之 $\log$ 是严格单调递增的，它完美保全了相对大小关系。因此，我们可以直接转而对比以下两个对数量：

$$\log \left(P(\text{spam}) \cdot P(\text{email} \mid \text{spam}) \right)$$

$$\log \left(P (\text{ham}) \cdot P(\text{email} \mid \text{ham}) \right)$$

只需从中选出数值较大者即可。将类别统称为 $\text{class}$（spam 或 ham），根据对数性质可展开为：

$$\log \left(P(\text{class}) \cdot P(\text{email} \mid \text{class}) \right) = \log \left(P(\text{class}) \right) + \log \left( P(\text{email} \mid \text{class}) \right)$$

并且：

$$\log \left( P(\text{email} \mid \text{class}) \right) = \log  \left(P(\text{word}_1 \mid \text{class}) \cdot P(\text{word}_2 \mid \text{class}) \cdots P(\text{word}_n \mid \text{class}) \right) = \log  \left(P(\text{word}_1 \mid \text{class}) \right) + \log \left(P(\text{word}_2 \mid \text{class})\right) + \cdots + \log \left( P(\text{word}_n \mid \text{class}) \right)$$

通过这一巧妙转化，我们把原本极其危险的大规模浮点数连乘，转变成了一个在数值计算上高度稳定的大规模**对数求和**！现在，我们将运用这一数学利器来重构我们的核心函数。我们需要调整以下两个函数：

* `prob_email_given_class`：将单词概率的乘积替换为对数概率的求和。


* `naive_bayes`：将乘积 $P(\text{class}) \cdot P(\text{email} \mid \text{class})$ 替换为相应的对数相加 $\log P(\text{class}) + \log P(\text{email} \mid \text{class})$。



这两个升级后的新函数将被命名为 `log_prob_email_given_class` 和 `log_naive_bayes`。

```python
def log_prob_email_given_class(treated_email, cls, word_frequency, class_frequency):
    """
    根据处理后的邮件内容，计算邮件属于特定类别 (如 spam 或 ham) 的对数概率。

    参数：
    - treated_email (list): 邮件中经过预处理的单词列表。
    - cls (str): 类别标签 ('spam' 或 'ham')。
    

    返回值：
    - float: 给定邮件属于指定类别的对数概率。
    """

    # prob 初始设为 0，因为在每次迭代中它都会通过累加当前的 log(P(word | class)) 进行更新
    prob = 0

    for word in treated_email: 
        # 仅对存在于词频字典中的单词执行计算
        if word in word_frequency.keys(): 
            # 将 prob 累加 log(P(word | class)) 来更新对数概率
            prob += np.log(prob_word_given_class(word, cls,word_frequency, class_frequency))

    return prob

```

```python
# 考虑只有 1 个单词的邮件，因此计算退化为求 P(word | class) 或 log(P(word | class))。
one_word_email = ['schedule']
word = one_word_email[0]
prob_spam = prob_email_given_class(one_word_email, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency)
log_prob_spam = log_prob_email_given_class(one_word_email, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency)
print(f"For word {word}:\n\tP({word} | spam) = {prob_spam}\n\tlog(P({word} | spam)) = {log_prob_spam}")

```

请注意，$\text{log}$ 成功地将一个原本无限逼近 0 的极其微小的正数，转化为了一个数值规模适度、便于计算机精准处理的负数。更重要的是，整个算法现在是在安全地执行加法，而不是在危险的边缘执行连乘。

接下来的代码块实现了 `log_naive_bayes` 函数。

```python
def log_naive_bayes(treated_email, word_frequency, class_frequency, return_likelihood = False):    
    """
    用于垃圾邮件检测的朴素贝叶斯分类器，通过比较对数概率而非原始概率进行判别。

    该函数基于朴素贝叶斯算法计算邮件为垃圾邮件 (1) 或正常邮件 (0) 的对数概率。
    它使用处理后邮件在 spam 和 ham 下的条件对数概率，以及两类的先验概率。
    最终决策通过比较计算出的对数概率值做出。

    参数：
    - treated_email (list): 输入邮件预处理后的表示。
    - return_likelihood (bool): 若为 True，则返回 spam 和 ham 的 log_likelihood。

    返回值：
    - int: 若邮件分类为垃圾邮件返回 1，若分类为正常邮件返回 0。
    """
    
    # 使用新的对数函数计算 P(email | spam)
    log_prob_email_given_spam = log_prob_email_given_class(treated_email, cls = 'spam',word_frequency = word_frequency, class_frequency = class_frequency) 

    # 使用上面刚刚定义的函数计算 P(email | ham)
    log_prob_email_given_ham = log_prob_email_given_class(treated_email, cls = 'ham',word_frequency = word_frequency, class_frequency = class_frequency) 

    # 使用 class_frequency 字典并利用公式 #spam emails / #total emails 计算 P(spam)
    p_spam = class_frequency['spam']/(class_frequency['ham'] + class_frequency['spam']) 

    # 使用 class_frequency 字典并利用公式 #ham emails / #total emails 计算 P(ham)
    p_ham = class_frequency['ham']/(class_frequency['ham'] + class_frequency['spam']) 

    # 计算量 log(P(spam)) + log(P(email | spam))，我们称之为 log_spam_likelihood
    log_spam_likelihood = np.log(p_spam) + log_prob_email_given_spam 

    # 计算量 log(P(ham)) + log(P(email | ham))，我们称之为 log_ham_likelihood
    log_ham_likelihood = np.log(p_ham) + log_prob_email_given_ham 

    # 若传入 return_likelihood = True，则返回目标元组
    if return_likelihood == True:
        return (log_spam_likelihood, log_ham_likelihood)
    
    # 比较两个数值并选择对应较大数值的类别。
    # 由于对数是单调递增函数，数值较大的一方依旧保持该性质。
    if log_spam_likelihood >= log_ham_likelihood:
        return 1
    else:
        return 0

```

重新审视本节开头的例子，我们将重新计算 `log_spam_likelihood` 与 `log_ham_likelihood`：

```python
log_spam_likelihood, log_ham_likelihood = log_naive_bayes(treated_email,word_frequency = word_frequency, class_frequency = class_frequency,return_likelihood = True)
print(f"log_spam_likelihood: {log_spam_likelihood}\nlog_ham_likelihood: {log_ham_likelihood}")

```

太棒了！现在我们得到了两个清晰、截然不同且非零的有效数值！可以看到，`log_ham_likelihood` 明显大于 `log_spam_likelihood`，因此 `log_naive_bayes` 准确无误地将这封邮件归类为了正常邮件：

```python
print(f"The example email is labeled as: {Y[example_index]}")
print(f"Log Naive bayes model classifies it as: {log_naive_bayes(treated_email,word_frequency = word_frequency, class_frequency = class_frequency)}")

```

依托这一改进后的算法，让我们重新评估测试集上的整体模型准确率：

```python
# 让我们获取测试集上的预测结果：

# 创建空列表存储预测值
Y_pred = []


# 遍历测试集中的每封邮件
for email in X_test:
    # 执行预测
    prediction = log_naive_bayes(email,word_frequency = word_frequency, class_frequency = class_frequency)
    # 添加到列表中 
    Y_pred.append(prediction)

# 获取真正例数量：
true_positives = get_true_positives(Y_test, Y_pred)

# 获取真反例数量：
true_negatives = get_true_negatives(Y_test, Y_pred)

print(f"The number of true positives is: {true_positives}\nThe number of true negatives is: {true_negatives}")

# 通过将真反例与真正例相加并除以数据集总元素数来计算准确率。
# 由于 Y_pred 和 Y_test 长度相同，使用哪一个都可以。
accuracy = (true_positives + true_negatives)/len(Y_test)

print(f"The accuracy is: {accuracy:.4f}")

```

这是一个**堪称惊艳**的飞跃！在未经任何特征工程干预的情况下，模型在测试集上的准确率从 84.82% 一跃攀升至 **99.21%**，整体准确率提升了近 17 个百分点！我们根本没有触碰底层数据集的一分一毫，这一突破完全来源于对底层**数学逻辑**的深刻理解与巧妙重构。这正是数学在工程落地中的非凡力量！

### 5.2 提升模型表现：朴素贝叶斯的工程实践

#### 5.2.1 业务背景引入

在本小节中，你将综合运用上述定义的两代朴素贝叶斯模型（标准版与对数稳定版），来解决一个现实业务场景中的具体诉求：

假设你需要为一款具体的电子邮件客户端开发一套高效的垃圾邮件过滤模块。我们一直使用的这个数据集就是该软件收集到的真实邮件库。在实际业务中，你必须构建一套既能强力拦截垃圾邮件，**又必须极力避免将正常邮件误杀进垃圾箱**的防御机制——因为一旦误杀，用户很可能会错失极其重要的工作或私人信件。相比之下，偶尔漏过一两封无关痛痒的垃圾邮件进入收件箱，反而是用户容忍度高得多的次要问题。

#### 5.2.2 准确率的局限性

站在真实业务的角度审视，我们迄今为止构建的模型到底表现如何？单纯依靠先前定义的准确率指标，在面对垃圾邮件检测这一特定场景时存在着显著的局限性。我们在 Notebook 开头就已经观察到，整个数据集中垃圾邮件的占比仅为 23.88%。这意味着，如果你制定一条毫无技术含量的荒谬规则——**把所有邮件无脑全部放行进收件箱**，那么这条摆烂规则的“准确率”也能躺赢达到 76.12%！

为了真正衡量算法的商业可用性，我们需要深入追踪以下两个核心问题：

* 算法究竟成功逮住了多少真正的垃圾邮件？这些被称为**真正例** (True Positives) 。


* 算法究竟把多少**正常邮件****错误地当成了垃圾邮件**？这些被称为**假正例** (False Positives) 。**这才是我们必须拿着放大镜严密监控的关键指标**。



第一个问题直接引出了评估领域的另一个基石指标——召回率 (Recall) 。为了回答这个问题，我们必须统计数据集中客观存在的所有垃圾邮件总数，并看模型能够准确从中挑出多少（即真正例）。召回率的定义如下：

$$\text{recall} = \frac{\text{真正例 (被成功识别为垃圾邮件的垃圾邮件数量)}}{\text{所有客观存在的垃圾邮件总数}}$$

另一种等价的数学表达是：任何一封真实的垃圾邮件，要么被模型成功识别（真正例），要么被模型不幸漏判为正常邮件（假反例，False Negatives），因此：

$$\text{recall} =\frac{\text{真正例}}{\text{真正例} + \text{假反例}}$$

现在我们编写计算召回率的函数：

```python
def get_recall(Y_true, Y_pred):
    """
    计算二分类任务的召回率 (Recall)。

    参数：
    - Y_true (array-like): 真实标签。
    - Y_pred (array-like): 预测标签。

    返回值：
    - recall (float): 召回率得分，即真正例与实际正例总数的比值。
    """
    # 获取垃圾邮件的总数。由于数据中垃圾邮件记为 1，只需对数组 Y 中所有值求和即可。
    total_number_spams = Y_test.sum()
    # 获取真正例
    true_positives = get_true_positives(Y_true, Y_pred)
    
    # 计算召回率
    recall = true_positives/total_number_spams
    return recall

```

```python
# 使用朴素贝叶斯模型 (标准版与对数版) 对测试集中的每封邮件进行分类
Y_pred_naive_bayes = []
Y_pred_log_naive_bayes = []

for email in X_test:
 prediction = naive_bayes(email,word_frequency = word_frequency, class_frequency = class_frequency)
 log_prediction = log_naive_bayes(email,word_frequency = word_frequency, class_frequency = class_frequency)
 Y_pred_naive_bayes.append(prediction)
 Y_pred_log_naive_bayes.append(log_prediction)

# 计算两个模型的召回率
recall_naive_bayes = get_recall(Y_test, Y_pred_naive_bayes)
recall_log_naive_bayes = get_recall(Y_test, Y_pred_log_naive_bayes)

```

```python
print(f"The proportion of spam emails the standard Naive Bayes model can correctly classify as spam (recall) is: {recall_naive_bayes:.4f}")
print(f"The proportion of spam emails the log Naive Bayes model can correctly classify as spam (recall) is: {recall_log_naive_bayes:.4f}")

```

测试表明，两个版本的模型在**捕捉垃圾邮件**方面的能力都极度敏锐，能够精准捕获超过 98% 的垃圾邮件！这个指标印证了模型的敏感度 (Sensitivity) ，展现了它在搜寻垃圾邮件时的天罗地网能力。

但关键的考验在于第二个问题，它对应着另一个极其关键的质量指标——精确率 (Precision) 。

要回答这一问题，我们必须把目光锁定在所有被模型扣上“垃圾邮件”帽子的邮件池中：在这个池子里，到底有多少是**名副其实**的真垃圾邮件？这是一个极为致命的指标，因为正如我们所见，一个把所有邮件都统统打上垃圾邮件标签的模型，固然能够斩获 100% 的垃圾邮件召回率，但它实际上完全是个灾难！在日常业务中，我们必须千方百计避免将普通正常信件扔进垃圾桶，防止用户痛失重要信息。

这一关切与**假正例** (False Positives) 紧密相连。换句话说，我们现在需要仔细清点算法到底把多少无辜的正常邮件错误归入了垃圾箱。下面的代码块实现了统计假正例数量的函数：

```python
def get_false_positives(Y_true, Y_pred):
    """
    计算二分类中的假正例 (False Positives) 实例数量。

    参数：
    - Y_true (list): 每个样本的真实标签列表 (0 或 1)。
    - Y_pred (list): 每个样本的预测标签列表 (0 或 1)。

    返回值：
    - int: 真实标签为 0 且预测标签为 1 的假正例数量。
    """
    
    # Y_true 和 Y_pred 的长度必须一致。
    if len(Y_true) != len(Y_pred):
        return "Number of true labels and predict labels must match!"
    n = len(Y_true)

    false_positives = 0
    # 遍历列表中的每个元素
    for i in range(n):
        # 获取当前邮件的真实标签
        true_label_i = Y_true[i]
        # 获取当前邮件的模型预测标签
        predicted_label_i = Y_pred[i]
        # 仅当真实标签为 0 且预测标签为 1 时计数器加 1 (假正例)
        if true_label_i == 0 and predicted_label_i == 1:
            false_positives += 1
    return false_positives

```

```python
# 统计被误标为垃圾邮件的正常邮件数量 (假正例)。使用上面定义的 get_false_positives 函数
 
false_positives_naive_bayes = get_false_positives(Y_test, Y_pred_naive_bayes)
false_positives_log_naive_bayes = get_false_positives(Y_test, Y_pred_log_naive_bayes)

```

```python
print(f"Number of false positives in the standard Naive Bayes model: {false_positives_naive_bayes}")
print(f"Number of false positives in the log Naive Bayes model: {false_positives_log_naive_bayes}")

```

这真是一个质的飞跃！标准版朴素贝叶斯将多达 169 封正常的无辜邮件误判为了垃圾邮件，而改进后的对数版模型仅误判了 4 封！为了将这一对比量化为更具普适意义的指标，我们计算以下比率：

* 在所有被模型预测为垃圾邮件的集合中，真实垃圾邮件（真正例）所占的比例。请注意，被预测为垃圾邮件的集合包含两部分：**被正确揪出的垃圾邮件**（真正例）与**被无辜冤枉的正常邮件**（假正例）。



这一关键指标被称为**精确率** (Precision) ，其严谨定义如下：

$$\text{precision} = \frac{\text{真正例}}{\text{真正例} + \text{假正例}}$$

该指标直接反映了模型输出结果的**可信度与相关性**。正如前文所述，一个闭着眼睛把所有邮件都判定为垃圾邮件的模型固然能找全所有的垃圾邮件，但由于它把所有正常邮件全部判错，其预测结果在实际生产中毫无参考价值。接下来我们实现精确率计算函数：

```python
def get_precision(Y_true, Y_pred):
    """
    计算精确率 (Precision)，这是分类模型的一项评估指标，
    通过计算真正例与 (真正例 + 假正例) 之和的比值获得。

    参数：
    - Y_true (list): 真实标签。
    - Y_pred (list): 预测标签。

    返回值：
    - precision (float): 精确率得分。
    """
    # 获取真正例和假正例
    true_positives = get_true_positives(Y_true, Y_pred)
    false_positives = get_false_positives(Y_true, Y_pred)
    precision = true_positives/(true_positives + false_positives)
    return precision

```

```python
print(f"Precision of the standard Naive Bayes model: {get_precision(Y_test, Y_pred_naive_bayes):.4f}")
print(f"Precision of the log Naive Bayes model: {get_precision(Y_test, Y_pred_log_naive_bayes):.4f}")

```

数据对比令人震撼：第一版标准模型的精确率仅仅只有 59.57%。换句话说，在该模型挑出的每 100 封所谓“垃圾邮件”中，实际上只有大约 60 封是真的垃圾邮件，剩下的 40 封全是蒙受不白之冤的正常重要邮件！这表明旧模型虽然看似“灵敏”，但在实际生产环境中极不可靠。

而经过对数数值稳定性改造后的升级模型，精确率直接飙升至 **98.42%**！这意味着在它判定为垃圾邮件的 100 封信件中，大约仅有 2 封可能是误判，其余全部精准命中。输出结果的可靠度实现了质的蜕变。

祝贺你！你不仅完整攻克了整套作业的核心代码实现，还彻底吃透了附录中关于数值稳定性与工业级评估指标的高阶拓展内容！