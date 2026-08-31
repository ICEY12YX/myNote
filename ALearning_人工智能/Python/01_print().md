___
输出、字符串与类型转换
___
#### 模块 1：最基础的打印输出

```python
print(1)
```

* **解释**：`print()` 就是打印，相当于你在前端用的 `console.log()` 
    这行代码的作用是在屏幕上直接打印出数字 1。
* **注意**：在 Python 中，纯数字打印不需要加引号，它属于整数类型（`int`）。

#### 模块 2：字符串与引号的碰撞

```python
# print(we're going to do something)
    # SyntaxError: EOL while scanning string literal
# print(we)
    # NameError: name 'we' is not defined
print("we're going to do something")
print('we')

# print('we're going to do something')
    # SyntaxError: invalid syntax
print("we're going to do something")
print('we\'re going to do something')
```

* **解释**：在 Python 里，表示一段文本（字符串 `str`）**必须**用引号包裹起来。
    与 C++ 和 Java 不同（它们用单引号表示单个字符，双引号表示字符串），**在 Python 中，单引号 `''` 和双引号 `""` 的作用是完全等价的**。

* **错误示范 1 避坑**：`print(we're going...)` 报错是因为你没加引号。
    Python 试图去理解 `we` 是不是一个变量名，结果后面的语法完全读不懂，直接报语法错误（SyntaxError）。
* **错误示范 2 避坑**：`print(we)` 报错**也**是因为你没加引号。
    Python 试图去理解 `we` 是不是一个变量名，结果后面的语法完全读不懂，直接报语法错误（SyntaxError）。
* **错误示范 3 避坑**：`print('we're...')` 报错是因为**引号冲突**。
    Python 看到第一个 `'` 认为字符串开始，看到 `we` 后面的 `'` 就认为字符串结束了，那后面的 `re going...` 它就完全不知道是什么鬼东西了。

* **正确解法**：
1. **内外岔开**：如果文本里有单引号，外面就用双引号包裹（如 `"we're"`），反之亦然。这是最常用的做法。
2. **使用转义符**：用反斜杠 `\` 告诉 Python：“后面的这个单引号只是个普通字符，不是用来结束字符串的”（如 `\'`）。

#### 模块 3：字符串的拼接与“强类型”限制

```python
print('fan'+'yuan')

# print('fan'+11)
    # TypeError: can only concatenate str (not "int") to str
print('fan'+'11')
print('fan'+str(11))
```

* **解释**：在 Python 里，加号 `+` 不仅能做数学题，还能把两个字符串“粘”在一起。`'fan'+'yuan'` 就会输出 `fanyuan`。

* **错误示范避坑**：`print('fan'+11)` 报类型错误（TypeError）。
    这里你需要警惕！在 JavaScript 中，字符串加数字会自动变成字符串（前端思维：`'fan'+11` 会自动变成 `'fan11'`）；但 **Python 是强类型语言**（这极像 C），它绝对不允许你把文本（`str`）和数字（`int`）直接相加。
    但是记区别也很难记, 只能现阶段熟悉什么语言, 隐隐约约记住吧 (( ¬_¬))

* **正确解法**：
1. 手动加引号，把它变成文本：`'11'`。
2. **强制类型转换**：使用 `str(11)` 强行把数字 11 转换成字符串形式。


> [!NOTE] **🌟 时代眼泪与现代 AI 推荐写法 (划重点)**
> 用 `+` 和 `str()` 来拼接变量是 2017 年以前老版本的无奈之举，写起来很啰嗦。在现代 Python（3.6 版本之后）以及目前的 AI 代码中，我们几乎 100% 使用 **f-string (格式化字符串)**。
> **现代写法：** `print(f'fan{11}')`
> **用法：** 在字符串的**引号前面**加一个小写字母 `f`，然后把所有**非文本**的变量或数字用大括号 `{}` 括起来，Python 会自动帮你完美转换并拼接！

#### 模块 4：基础计算与数值类型转换

```python
print(1+2)
print('1+2')

print(int('1')+2)

# print(int('1.2')+2)
    # ValueError: invalid literal for int() with base 10: '1.2'
print(float('1.2')+2)
```

* **解释**：
    * `print(1+2)`：纯数字，执行数学加法，输出 3。
    * `print('1+2')`：加了引号，它就是一段毫无感情的文本，原样输出 `1+2`。

* **类型转换**：
    * `int()`：把字符串转成**整数**（没有小数点的数）。所以 `int('1')+2` 变成了数学计算 `1+2`，输出 3。

* **错误示范避坑**：`int('1.2')` 报值错误（ValueError）。
    因为 `int` 是整数的意思，你硬塞给它一个带小数点的文本 `'1.2'`，它不知道该怎么把小数点抠掉，直接罢工。
* **正确解法**：在 Python 中，带小数点的数字叫**浮点数**（`float`）。所以遇到带小数点的文本，必须用 `float('1.2')` 来转换，最后输出 `3.2`。


