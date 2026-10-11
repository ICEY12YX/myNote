___
在 Python 里，如果你想在创造一个对象的那一瞬间，就给它配置好属于它自己的属性，你需要用到一个特殊的方法：**`__init__`**。

`__init__` 的全称是 initialization（初始化）。你可以把它理解为对象的“出生设置”**或**“出厂配置”**。每当你用类创建一个新对象时，Python 就会**自动且立刻执行这个类里面的 `__init__` 方法。

### 代码演示

让我们继续用 `Calculator` 来举例。假设我们希望每个计算器在被创建时，就能拥有自己专属的“品牌名”和“颜色”：

```python
class Calculator:
    # 这是出厂设置方法，注意 init 前后各有两个下划线
    def __init__(self, brand, color):
        # 把外部传进来的 brand 和 color 参数，绑定到自己（self）身上
        self.brand = brand
        self.color = color

    def show_info(self):
        # 任何普通方法都可以通过 self 拿到自己的专属属性
        print(f"我是一个 {self.color} 的 {self.brand} 计算器")

# 创建对象的一瞬间，直接在括号里按照 __init__ 的要求传入参数
calc1 = Calculator("卡西欧", "黑色")
calc2 = Calculator("得力", "白色")

# 验证一下
calc1.show_info()  # 输出: 我是一个 黑色 的 卡西欧 计算器
calc2.show_info()  # 输出: 我是一个 白色 的 得力 计算器

```

### `__init__` 的三个核心细节

1. **名字绝对不能错**：
    `init` 的前后**各有两个下划线**（`__init__`）。在 Python 里，带有这种双下划线的方法被称为<mark style="background: #FFF3A3A6;">“魔术方法”</mark>。它的魔力在于：Python 会在对象诞生的那一刻帮你自动触发它，你永远**不需要手动去写** `calc1.__init__()`。
2. **第一位依然是 `self`**：
    刚刚出生、还在流水线上的那个新对象，会被 Python 自动传给 `self`。你只需要在括号里填上后面需要的参数（比如 `brand` 和 `color`）。

只要你在 `__init__` 里通过 `self.xxx` 贴好了属性，以后在这个类的任何其他普通方法里，都可以随时用 `self.xxx` 把它们调出来用。

[[继承]]