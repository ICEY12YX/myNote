___
#### 模块 1：基础导入与“起别名” (极度高频)

```python
# time属于标准库,不需要额外安装
import time
print(time.localtime())

import time as t
print(t.localtime())

```

* **解释**：
     `import time`：把整个 `time` 工具箱搬过来。你想用里面的工具，必须带上前缀（比如 `time.localtime()`），告诉系统这个工具是哪来的。
     `import time as t`：给工具箱**起个别名**。这在 AI 开发中极其常用！程序员为了偷懒，通常会把名字很长的库缩写。

* ✨**AI 业界规范前瞻**：未来你肯定会遇到这些全行业统一的缩写，千万不要自己瞎起名：
    * `import numpy as np`
    * `import pandas as pd`
    * `import torch.nn as nn`

#### 模块 2：精准导入与“星号”陷阱

```python
from time import time, localtime
print(localtime())
print(time())

from time import *
print(localtime())
print(time())
# 其他函数也可以直接用...

```

* **解释 (`from ... import ...`)**：
    我不想要整个工具箱，我只想把里面的 `localtime` 这一件工具拿出来。好处是，调用时直接写 `localtime()` 即可，不需要再加前缀。
* **避坑指南 (`from ... import *`)**：
    * `*` 代表“把里面所有的东西全倒出来”。
    * **❗️ 现代 AI 编码规范强烈抵制这种写法！** 我们前面刚讲过“变量名屏蔽（Shadowing）”的危险。如果你用 `*` 把几百个函数全倒进你的代码里，==一旦里面某个函数名恰好和你自己定义的变量名重名，就会发生可怕的覆盖==，导致极其难以排查的 Bug。请务必老老实实写清楚导入了什么。


#### 模块 3：自定义模块与 Jupyter 的“顽固记忆”

```python
import m1

m1.functionA("fanfanfan")
print(m1.functionB("yuanyuanyuan"))
# jypter里面要是m1里写了funB,再回来调用会报错
# 此时好像要刷新一下内核(右上角"重启")

```

* **解释**：
    你不仅可以导入官方库，只要在同一个文件夹下，你自己写的 `m1.py` 也能作为一个工具箱被直接 `import m1` 导入。
* **疑问解答 (为什么改了代码会报错？)**：
    * 正如我们之前聊过的 Jupyter “幽灵记忆”陷阱：为了追求极速，==Jupyter **只会导入一个模块一次**==。
    * 当你第一次 `import m1` 时，它把 `m1` 放进了内存。如果你这时去修改了 `m1.py`（比如加了个 `funB`），==Jupyter 根本不会再去重新读一遍硬盘，它依然使用的是刚才缓存在内存里的旧版本==，所以它找不到 `funB` 从而报错。
* **解决办法**：你的直觉非常准，点击右上角重启内核（Restart Kernel）清除记忆，是最简单粗暴且有效的方法。

#### 模块 4：解密神秘的 `__pycache__` 文件夹

```python
# 疑问: 会多一个_pycache_文件夹是啥

```

* **深入底层机制解答**：
    * 还记得我们之前讨论过 Python 在执行前会先进行“扫描（Parsing）”吗？
    * ⭐️ Python 虽然是解释型语言，但它并不傻。当你 `import m1` 时，Python 觉得“既然你要把别人当工具箱，我干脆把你==编译成效率更高的机器字节码（Bytecode）存起来==，下次再导入时就快多了”。
    * 于是，Python 会在后台偷偷生成一个 `.pyc` 文件，并把它藏在这个名叫 `__pycache__` 的文件夹里。
* **对比你的 C/Java 经验**：这个文件夹里的东西，就完全等同于 C 语言编译产生的 `.obj` 文件, 你平时完全不用管它，就算直接删掉，下次运行 Python 时也会自动重新生成。
