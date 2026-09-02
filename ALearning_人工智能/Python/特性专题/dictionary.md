___
#dictionary字典 

# 其他方法
字典还有几个出场率极高的方法，我把它们分为**安全获取**、**三大视野**和**进阶修改**三类来给你梳理：

### 1. 安全拿数据：`get()`

你现在是用 `a_dic["apple"]` 来拿数据，这有一个潜在危险：==如果键不存在，程序会直接报错崩溃==（抛出 KeyError）。

为了安全地拿数据，Python 提供了 `get()` 方法：
* **用法**：`字典.get(键)`
* **好处**：如果键在字典里，它就把值给你；如果键不在，它**不会报错**，而是默默返回一个 `None`（或者你指定的备用值）。

```python
a_dic = {"apple": 1, "pear": 2}

# 1. 正常获取
print(a_dic.get("apple"))  # 输出: 1

# 2. 键不存在时，不会报错，返回 None
print(a_dic.get("orange")) # 输出: None

# 3. 你还可以设置一个“兜底方案”：如果找不到，就返回你给的默认值
print(a_dic.get("orange", "没找到这个水果")) # 输出: 没找到这个水果

```

### 2. 字典的三大视野：`keys()`, `values()`, `items()`

这三个方法常用于提取字典里的信息，并且**它们是 `for` 循环的最佳搭档**。

* **`keys()`**：把字典里所有的键（标签）打包拿出来。
* **`values()`**：把字典里所有的**值**打包拿出来。
* **`items()`**：把字典里所有的**键值对**（标签和值组合在一起）拿出来。<mark style="background: #FF5582A6;">结果是一个包含元组的列表,每个   元组    就是一个键值对</mark>

```python
a_dic = {"apple": 1, "pear": 2, "banana": 80}

# 拿所有的键
print(a_dic.keys())    # 输出: dict_keys(['apple', 'pear', 'banana'])

# 拿所有的值
print(a_dic.values())  # 输出: dict_values([1, 2, 80])

# 拿所有的键值对
print(a_dic.items())   # 输出: dict_items([('apple', 1), ('pear', 2), ('banana', 80)])

```

⭐️⭐️⭐️**实战运用（结合你之前学的解包）：**
当你用 `for` 循环搭配 `items()` 时，可以直接用两个变量把键和值“解包”出来，这是 Python 里遍历字典最优雅的写法！

```python
for key, value in a_dic.items():
    print(f"水果 {key} 的数量是 {value}")

```

### 3. 进阶修改：`update()` 和 `pop()`

你之前学过用 `a_dic["banana"] = 80` 来新增键值对。如果你想**一次性塞入好几个键值对**，或者把两个字典合并，就可以用 `update()`。

* **`update()`**：批量更新。==如果有相同的键，就用新的值覆盖==；如果没有，就新增。

```python
a_dic = {"apple": 1}
new_info = {"apple": 100, "orange": 5}

# 把 new_info 里的东西全部合并到 a_dic 里
a_dic.update(new_info)
print(a_dic)  # 输出: {'apple': 100, 'orange': 5}

```

* **`pop()`**：你之前学过列表的 `pop()` 是拔萝卜。字典的 `pop()` 也是一样，通过**键**把那个键值对拔出来，并且把**值**交给你。==它比 `del` 更灵活==。

```python
a_dic = {"apple": 1, "pear": 2}

# 把 pear 拔出来，并把它对应的值 2 存入变量
pear_value = a_dic.pop("pear")

print(pear_value)  # 输出: 2
print(a_dic)       # 输出: {'apple': 1}

```
___
# 字典的推导式

字典推导式（Dictionary Comprehension）的逻辑和列表推导式几乎一模一样。
唯一的区别在于：**外面要用大括号 `{}`，而且每次必须要同时生成一个“键”和一个“值”，中间用冒号 `:` 隔开。**
==(生成一个字典)==
### 核心公式

`{ 【键】: 【值】 for 【变量】 in 【数据源】 if 【过滤条件】 }`

---

### 场景一：极简生成（比如从列表生成字典）

假设你有一个包含水果名称的列表，现在你想直接生成一个字典：把水果名字当成**键**，把名字的长度（字母个数）当成**值**。

```python
fruits = ["apple", "pear", "banana"]

# 解读：{把水果名做键 : 把它的长度做值  for  每次拿出一个水果  in  水果列表}
fruit_len_dic = {fruit: len(fruit) for fruit in fruits}

print(fruit_len_dic)  
# 输出: {'apple': 5, 'pear': 4, 'banana': 6}

```

### ⭐️场景二：经典操作之“字典键值对反转”

有时候，我们想把==字典里的“标签”和“数据”对调一下==（键变成值，值变成键）。这时候，搭配你上一节刚刚学过的 `.items()`，字典推导式就能大显身手。

```python
a_dic = {"apple": 1, "pear": 2, "banana": 80}

# 1. k, v 分别代表原来的键(key)和值(value)
# 2. 我们在前面把位置反过来写成 v: k
reverse_dic = {v: k for k, v in a_dic.items()}

print(reverse_dic)
# 输出: {1: 'apple', 2: 'pear', 80: 'banana'}

```

### 场景三：边过滤边生成

我们把难度升级一点。还是刚才那个字典，现在我不仅想反转它，而且我**只想要原来字典里“值大于1”的那些水果**。

套用公式，只要在最后面加上 `if` 就可以了：

```python
a_dic = {"apple": 1, "pear": 2, "banana": 80}

# 严格按照公式排兵布阵：
# {【新键:新值】 for 【解包出来的k, v】 in 【字典所有的键值对】 if 【过滤条件】}
filtered_reverse_dic = {v: k for k, v in a_dic.items() if v > 1}

print(filtered_reverse_dic)
# 原来的 "apple": 1 被过滤掉了
# 输出: {2: 'pear', 80: 'banana'}

```
___
# 处理方式补充

### 1. 判断某个“键”在不在字典里：`in`

这是最高频的操作之一。在从字典里拿数据或者修改数据之前，我们经常需要先判断一下这个标签存不存在。直接用 `in` 关键字就可以，它不仅简单，而且在字典里查找的速度极快。

```python
scores = {"Alice": 95, "Bob": 88}

# 检查字典里有没有 "Alice" 这个键
if "Alice" in scores:
    print("Alice 参加了考试！")

# 检查字典里有没有 "Charlie"
if "Charlie" not in scores:
    print("Charlie 缺考了。")

```

### 2. 两个字典的“优雅合并”：`|` 运算符 (Python 3.9+)

你之前学过用 `update()` 可以把一个字典合并到另一个字典里，但这会修改原来的字典。如果你想保留原字典，直接生成一个**合并后的新字典**，在现代 Python 中，你可以直接用竖线 `|`（管道符）来进行合并。

```python
dict_a = {"apple": 1, "pear": 2}
dict_b = {"banana": 3, "apple": 100} # 注意这里也有 apple

# 用 | 直接合并，生成一个全新的字典
# 遇到冲突的键（比如 apple），⭐️右边的 dict_b 会覆盖左边的 dict_a
dict_c = dict_a | dict_b

print(dict_c)  # 输出: {'apple': 100, 'pear': 2, 'banana': 3}

```

### 3. 数据统计（经典套路）：`get(key, 0) + 1`

假设你有一堆水果，你想用字典统计每种水果出现了几次。如果不熟练，可能会写很长的 `if-else` 来判断。但如果利用 `get()` 方法自带的“备用值”功能，一行核心代码就能搞定：

```python
fruits = ["apple", "banana", "apple", "pear", "banana", "apple"]
count_dict = {}

for fruit in fruits:
    # 核心套路：
    # 如果找不到这个水果，⭐️默认它是 0 个，然后 +1
    # 如果找到了，就拿出原来的数量，然后 +1
    count_dict[fruit] = count_dict.get(fruit, 0) + 1

print(count_dict) 
# 输出: {'apple': 3, 'banana': 2, 'pear': 1}

```
这个不理解,先放着
### 4. 字典排序：`sorted()`

虽然字典主要是靠“键”来找东西的，但有时候我们就是想把字典里的数据排个序，比如做个排行榜。Python 提供了一个内置的 `sorted()` 函数。

**情况 A：按“键”（标签）排序**

```python
scores = {"Tom": 88, "Alice": 95, "Bob": 70}

# 默认情况下，sorted 会直接把字典的 ⭐️“键”⭐️ 按字母表排好序，放进一个列表里
sorted_keys = sorted(scores)
print(sorted_keys)  # 输出: ['Alice', 'Bob', 'Tom']

```

**情况 B：按“值”（分数）排序（进阶操作）**
如果要按分数高低排序，我们需要用到 `items()` 把键值对拿出来，并且借用一个小工具 `lambda`（匿名函数）来告诉 Python：“请盯着分数（也就是位置 1 的元素）来排”。

```python
scores = {"Tom": 88, "Alice": 95, "Bob": 70}

# x 代表每个键值对，x[1] 就是分数
# reverse=True 表示从大到小降序排列
# 这里sorted一共填了3个参数
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

# 排序后的结果会变成一个包含元组的列表
print(sorted_scores) 
# 输出: [('Alice', 95), ('Tom', 88), ('Bob', 70)]

```
这里sorted一共填了3个参数,意思是: 对 scores.items() 这个列表进行排序,排序的依据是每个元组的第二个元素(即分数),并且是降序排列

⭐️每个参数的用法是:
 `sorted(iterable, key=None, reverse=False)`
     `iterable`: 要排序的可迭代对象
     `key`: 一个函数,用来从每个元素中提取用于排序的关键字
     `reverse`: 是否反向排序,True 表示降序,False 表示升序

### 总结
* **判断存在**：用 `in`。
* **无损合并**：用 `|`。
* **频次统计**：用 `字典[键] = 字典.get(键, 0) + 1`。
* **想要排序**：用 `sorted()` 配合 `items()`。