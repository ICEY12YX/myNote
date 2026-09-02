___
 这段代码带你解锁了 Python 中处理“唯一性”和“数学集合”的终极武器：**集合 (Set)**。

它的底层依赖哈希表（Hash Table），核心特点就是**天生去重**和**绝对无序**。在未来的人工智能自然语言处理（NLP）中，我们经常用它来提取文章的“不重复词汇表”。

#### 模块 1：集合的创建与天生去重

```python
char_list = ["a", "b", "c", "d", "d"]
# 用set会把重复的元素去掉
# 输出的set里面没有顺序所以是乱序的
print(set(char_list))
# 类型就是set(不是字典dict)
print(type(set(char_list)))

sentence_a = "Welcome Back to This Tutorial"
# 句子的话,会认为每一个字母是一个元素
# 大小写是不一样的元素
print(set(sentence_a))

# print(set(sentence_a, char_list))
# 报错TypeError: set expected at most 1 argument, got 2

```

* **解释**：
    `set()` 是一个强转函数，能够把列表或字符串变成==**集合**==。一旦变成集合，里面所有的重复元素（比如多余的 `"d"`）都会被瞬间抹除。
* **疑问解答 (报错原因)**：
    `TypeError: set expected at most 1 argument`（最多只需 1 个参数，你却给了 2 个）。`set()` 就像一个榨汁机，==每次只能往里面放一个完整的容器==（比如一整个列表，或者一整个字符串），你不能同时把两个容器当成两个独立的参数硬塞进去。

#### 模块 2：增删元素与“不可哈希”陷阱

```python
char_list = ["a", "b", "c", "d", "d", "d", "e", "f"]
unique_list = set(char_list)
unique_list.add("a")
# unique_list.add(['a','b']) 不能一次性添加一个list
print(unique_list)

unique_list.remove("a")
# print(unique_list.remove("a")) 打印出来None
# unique_list.remove("y")  # 会报错,因为找不到这个元素
# 为了避免这种情况,可以使用discard方法,找不到元素也不会报错
unique_list.discard("y")
unique_list.discard("b")
print(unique_list)

unique_list.clear()   
print(unique_list)

```

* **疑问解答 (为什么不能 add 一个 list?)**：
    如果你取消注释 `unique_list.add(['a','b'])`，Python 会报一个非常经典的错误：`TypeError: unhashable type: 'list'`（不可哈希类型）。
    * **底层原理解密**：因为 `set` 为了保证极速查找，底层是用“哈希值”来排座位的。这意味着==塞进 `set` 里的东西必须是**永远不变的 (Immutable)**==。列表（List）是可变的（随时可以增删改），它的==哈希值不固定==，所以系统绝对不允许你把列表塞进集合里。
    * 🌙 这个不好理解,先记住吧

* **`remove` vs `discard` (优雅排雷)**：
    * `remove` 是个暴脾气，找不到元素直接报错死机。
    * `discard` 情绪非常稳定，找不到就算了，程序继续安稳往下跑。在实际工程中，我们==强烈推荐多用 `discard`==。

#### 模块 3：集合的数学运算 (交并差)

```python
char_list = ["a", "b", "c", "d", "d", "d", "e", "f"]
unique_list = set(char_list)

set1 = unique_list
set2 = {"a", "d", "i"}
print(set1.difference(set2))  # 差集
print(set1.intersection(set2)) # 交集

```

* **解释**：
    这完全就是初中数学里的集合运算。
* `difference` (差集)：在 `set1` 里，但不在 `set2` 里的元素。
* `intersection` (交集)：两个集合里都有的公共元素。

>[!NOTE]  **🌟 现代 AI 推荐写法 (符号魔法)**
> 每次敲 `.difference()` 或 `.intersection()` 太长了。Python 为集合提供了一套极其优雅的符号运算：
> * **交集 (Intersection):** `set1 & set2`
> * **并集 (Union):** `set1 | set2`
> * **差集 (Difference):** `set1 - set2`
> 这些符号不仅写起来清爽，而且执行速度极快！
> 
> 

---
#set集合
[[set]]