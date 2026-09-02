___
#### 模块 1：类的定义、`self` 与属性赋值

```python
class Calculator:
    name = "Good Calculator"  # 类属性, 类属性可以通过类名访问,也可以通过对象访问
    price = 1000  # 类属性

    def add(self, a, b):  # 实例方法, 实例方法必须有self参数, self参数表示对象本身
        print("the name is", self.name)  # 访问类属性
        result = a + b
        return result

    def minus(self, a, b):
        result = a - b
        return result

    def multiply(self, a, b):
        result = a * b
        return result

    def divide(self, a, b):
        if b == 0:
            print("除数不能为0")
            return None
        result = a / b
        return result

# 先创建这个类
calculator = Calculator()  # 创建对象, 调用类的构造方法
# 构造方法是一个特殊的方法, 用于创建对象, 构造方法的名称是__init__, 但是这里没有定义构造方法, 所以使用默认的构造方法

calculator.name = "Bad Calculator"  # 修改类属性, 通过对象修改类属性, 但是这里是修改了对象(就是实例)的属性, 并没有修改类的属性
print(calculator.name)  # 访问类属性, 通过对象访问类属性
print(calculator.price)  # 访问类属性, 通过对象访问类属性
print(calculator.add(1, 2))  # 调用实例方法, 通过对象调用实例方法

```

* **解释**：
    **⭐没有 `new` 关键字**：
        在 Java 或 C++ 中，实例化对象必须写 `new Calculator()`。在 Python 中，直接当成函数调用 `Calculator()` 即可生成对象，非常干脆。
    **神奇的 `self`**：
        方法定义里的 `self` 相当于 Java/C++ 里的 **`this` 指针**。
        Python **强制要求**你在定义类方法时，**第一个参数必须写上 `self`**。
        ⭐但在**调用时**（如 `calculator.add(1, 2)`），你**不需要传** `self`，Python 会自动把 `calculator` 这个对象本身作为第一个参数偷偷传进去。

看这段代码
```python
class Calculator:
    def add(a, b):
        result = a + b
        return result


calculator = Calculator()
print(calculator.add(1, 2))
# 报错TypeError: add() takes 2 positional arguments but 3 were given
```
#self
[[关于self]]


* **属性屏蔽陷阱（划重点）**：
    当你执行 `calculator.name = "Bad Calculator"` 时，你并没有修改类公用的那个 `name`，而是在这个特定的实例（对象）上，**新建了一个同名的实例属性**。
    这和我们在函数里讲的“局部变量屏蔽全局变量”的底层逻辑是一模一样的。实例属性优先级高于类属性，所以打印出来是 `Bad Calculator`。
从下面的代码也可以理解
```python
class Calculator:
    name = "Good Calculator"


calculator = Calculator()
calculator1 = Calculator()
print(calculator.name)  # 输出Good Calculator
print(calculator1.name)  # 输出Good Calculator

Calculator.name = "Bad Calculator"
calculator.name = "sss Calculator"
print(calculator.name)  # 输出sss Calculator
print(calculator1.name)  # 输出Bad Calculator
```
#类属性 
[[类属性]]

#### 模块 2：构造方法 (`__init__`) 与传参报错

```python
class Calculator:
    name = "Good Calculator"  # 类属性, 类属性可以通过类名访问,也可以通过对象访问
    price = 1000  # 类属性

    def __init__(self, name, price, fan=0, yuan=9):  # 构造方法, 用于创建对象
        self.name = name  # 实例属性, 实例属性必须通过self访问
        self.price = price
        self.fan = fan
        self.yuan = yuan

    def add(self, a, b):  # 实例方法, 实例方法必须有self参数, self参数表示对象本身
        print("the name is", self.name)  # 访问类属性
        result = a + b
        return result

# c = Calculator("Bad Calculator")
# 报错 TypeError: __init__() missing 3 required positional arguments: 'price', 'fan', and 'yuan'
c = Calculator("Bad Calculator", 2000, 10, 20)  # 创建对象, 调用类的构造方法
print(c.yuan)

c = Calculator("Bad Calculator", 2000, 10)
print(c.yuan)  # 输出9, 因为构造方法中定义了默认值

```

* **解释（构造函数）**：
    在 Java 里，构造函数的名字必须和类名一模一样（比如 `public Calculator()`）。
    ⭐️但在 Python 中，构造函数有着固定的名字：**`__init__`**（前后各两个下划线，这类方法在 Python 中被称为魔术方法/Dunder methods）。当你执行 `Calculator(...)` 创建对象时，系统会自动调用这个方法进行初始化。


* **避坑指南（深入分析 TypeError）**：
* 为什么 `c = Calculator("Bad Calculator")` 会报错？
* 因为你的 `__init__` 规定了除了 `self` 自动传入外，还必须接收 `name`、`price`  这两个个没有默认值的必填参数。
<mark style="background: #FF5582A6;">上面写的类属性赋值, 是不算赋了默认值的</mark>

> [!NOTE] **🌟 现代 AI 推荐写法：类型提示 (Type Hint)**
> 现代 Python 非常鼓励在类和方法中加入类型提示，这对于有着强类型背景的你来说会非常亲切。
> **现代写法：** `def __init__(self, name: str, price: int, fan: int, yuan: int = 9) -> None:`
> 这不会改变运行结果，但能在你的 VS Code 或 PyCharm 里提供极其强大的代码自动补全功能！
> 
> 
