___
除了定义函数的基本用法
还要注意**作用域与执行顺序**。

---
#### 模块 1：函数的定义与参数传递

```python
def functionA():  # 同样注意tab缩进, 缩进的是函数里面内容,没缩进的就不是
    print("functionA is running")
    a = 1 + 2
    print(a)
functionA()  # 调用函数, 执行函数里面的内容

# ---------------------------------------------------
def functionB(x, y):  # 函数可以有参数, 也可以没有参数
    z = x + y
    print("the z is", z)  # 这个输出可以注意到,is后面自己带了空格
    print("the z is " + str(z))  # 注意两种print写法
    
functionB(1, 2)  # 顺序传参
functionB(x=4, y=7)  # 关键字传参
# functionB(a=1, b=2)  # 参数名不对, 会报错

```

* **解释**：
    使用 `def` (define) 关键字定义函数。
    ⭐ 对比你的 C/Java 经验，Python 定义函数时**不需要声明返回类型，参数也不需要声明类型**，极其奔放。**就是输入输出参数都不需要显示定义类型**
* **参数命名绑定 (强推)**：
    `functionB(x=4, y=7)` 叫 #关键字传参。
[[顺序or关键字传参]]

#### 模块 2：默认参数的“排队规则”

```python
# def functionC(price, color="red", discount=0.8, text): 
    # 报错：默认参数后面不能再有非默认参数

def functionC(price, color="red", discount=0.8):
    print("the price is", price)
    print("the color is", color)
    print("the discount is", discount)

functionC(100)  # 只传必填项，其余用默认
functionC(100, color="blue")  # 覆盖默认项

```

* **语法铁律**：
    如果某些参数有默认值（如 `color="red"`），它们**必须乖乖排在 所有没有默认值的参数 后面**。否则，当你只传一个参数时，Python 根本不知道你是想传给前面的必填项，还是想覆盖后面的默认项。
    比如`def functionC (color="red", price):... `  此时调用的时候写`functionC(100)`, 100是给prize还是覆盖color, 就不清楚了

#### 模块 3：变量作用域与“ referenced before assignment” 报错解密

```python
APPLE = 88  # 全局变量
def functionD(c):
    a = 10
    b = APPLE
    return a + b + c

c = 9
print(functionD(c))
```
#函数编译顺序
[[函数编译顺序]]


```python
APPLE = 88  # 全局变量
def functionD():
    a = 10
    b = APPLE
    
    # APPLE = 99 
    # 疑问：为什么写这句会报错 local variable 'APPLE' referenced before assignment?
    # 意思是APPLE 被引用了 (在被赋值之前)
    return a + 10

```

#全局变量
[[函数编译顺序]]

(以下是简单总结, 具体看上面)
* **⭐解密 Python 机制 (划重点)**：
    Python 的设计逻辑是：**只要你在函数内部对某个变量进行了“赋值操作”（写了等号），Python 就会在函数刚开始扫描时，强行把这个变量判定为“本函数的局部变量”**。
* **报错原因**：
    当你写了 `APPLE = 99`，`APPLE` 就变成了局部变量。当代码执行到前面的 `b = APPLE` 时，系统发现“局部变量 APPLE 还没被赋值呢，你怎么就先用起来了？”，于是直接报错。

```python
APPLE = 88

def functionD():
    global APPLE   # 第一步：举牌子声明
    a = 10
    b = APPLE      # 此时读取的是全局的 88
    APPLE = 99     # 此时修改的也是全局的 APPLE
    return a + 10
    
print(functionD())

```
* **正确解法**：
    如果你要在函数里修改全局变量，必须先写一句 `global APPLE` 声明一下。


#### 模块 4：执行顺序与 Jupyter 特有“幽灵”

正确做法
```python
p = 99  # 全局变量

def functionD():
    global p
    p = 100  # global关键字可以在函数内部修改全局变量的值
    print("the p is", p)
    return p + 999

print(functionD())  # 调用函数, 并打印返回值
print(p)  # 函数内部的全局变量可以在函数外部使用

```

报错做法
```python
# p = 99  # 全局变量

def functionD():
    global p
    p = 100  # global关键字可以在函数内部修改全局变量的值
    print("the p is", p)
    return p + 999

print(p);
# 老师直接打印会报错，因为函数没调用。
# 疑问：为什么你这里却可以输出？
```
尝试下来发现
    py里会报错, jypter里不会(但是输出99)

按道理来说, (`p = 99`这句没有的情况下) 
如果不调用 `functionD()`，变量 `p` 根本不存在，正常情况下 `print(p)` 必定报错。
py报错`NameError: name 'p' is not defined`

* **为什么你的没报错？(Jupyter 幽灵陷阱)**：
    因为你使用的是交互式环境（Jupyter / IDE 终端）。你肯定在**之前的某次运行中**，不小心调用过 `functionD()`。Jupyter 的内核把 `p = 100` 永远记在了后台内存里。即便你现在把调用函数的代码删了，只要你不重启内核，`p` 就一直活着。
确实! 重启以后报错了
    ![[Pasted image 20260831225249.png]]
* **避坑指南**：
    以后如果遇到代码逻辑明明不对却能跑通，点击上方菜单栏的 `Kernel -> Restart`（重启内核），清除记忆再跑一次，幽灵 Bug 就会原形毕露。

重启在这里
![[Pasted image 20260901045924.png|548]]