___
### 关于 `self` 的三大核心规则

Python 类的核心规则——`self`。

#### 1. 占位必须在第一位

在类里面定义普通方法时，**第一个参数必须留给对象本身**。不管你后面需要几个参数，第一个位置永远是被预定的。这个参数我们习惯性地命名为 `self`。
#### 2. Python 负责传 `self`，你只负责传剩下的

你在调用方法时，**千万不要手动给 `self` 传值**。Python 每次都会自动把调用这个方法的对象传给 `self`，你只需要传入后面的业务参数即可。

⭐️以上两点就是说, 
调用类方法时,python会自动把对象当做第一个参数传进类方法
这个对象 就会被当做self (就是赋值的过程)
<mark style="background: #FF5582A6;">不管类方法你怎么定义的输入参数,第一个输入参数永远是python自动传来的self</mark>
#### 3. `self` 就是对象自己

`self` 的作用是让方法知道“现在是谁在调用我”。如果你的类里有属性（比如 `self.name`），你需要通过 `self` 才能在方法内部拿到或者修改这个具体对象的数据。
___

```python
class Calculator:
    def add(a, b):
        result = a + b
        return result


calculator = Calculator()
print(calculator.add(1, 2))
# 报错TypeError: add() takes 2 positional arguments but 3 were given
```
这段代码报错，是因为**Python在调用类的方法时，会在暗中自动传递一个参数**。

当你在代码里写下 `calculator.add(1, 2)` 时，Python 实际上在后台做了一个转换，它真正执行的是：

```python
# Python 秘密转换后的样子：
Calculator.add(calculator, 1, 2)

```

你看，Python 把 `calculator` 这个对象本身，当作**第一个参数**硬塞给了 `add` 方法。

这就导致了你的报错：

1. 你的 `add(a, b)` 只准备接收 **2** 个参数。
2. 但 Python 实际上塞进去了 **3** 个参数（`calculator` 对象, 数字 `1`, 数字 `2`）。
所以报错提示：`add() takes 2 positional arguments but 3 were given`（add只需要2个参数，但给出了3个）。
___

### 正确的代码写法

为了让代码跑通，你只需要在定义 `add` 方法时，在最前面加上 `self` 占位：
命名为self只是约定俗成的习惯, 当然也可以命名为其他东西

```python
class Calculator:
    # 规则1：第一个参数加上 self
    def add(self, a, b):
        result = a + b
        return result

calculator = Calculator()

# 规则2：调用时忽略 self，只传 a 和 b
print(calculator.add(1, 2))  # 输出: 3

```

这样，`self` 接收了 `calculator` 对象，`a` 接收了 `1`，`b` 接收了 `2`，皆大欢喜。

____
[[静态方法 和 类方法]]