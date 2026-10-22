# 选修实验：Python 与 Jupyter Notebooks 简明指南
欢迎来到第一个选修实验！
这些选修实验旨在为你：
- 提供背景信息——就像你正在阅读的这份交互式笔记本 (notebook) 一样
- 通过亲手实践的案例，巩固课堂上讲授的理论内容
- 提前展示计分实验中会用到的核心代码例程 (routines) 的实际运行示例

## 实验目标
在本次实验中，你将：
- 初步认识 Jupyter 笔记本 (Jupyter notebooks) 的基本概念
- 跟着导览快速熟悉 Jupyter notebooks 的界面
- 弄清楚 Markdown 单元格 (markdown cells) 与代码单元格 (code cells) 之间的区别
- 动手练习一些基础的 Python 编程操作


想要快速上手 Jupyter notebooks，最省力的办法就是点击上方“帮助” (Help) 菜单，开启一段内置的界面导览 (tour) ：

&lt;figure&gt;
    &lt;center&gt; &lt;img src=&quot;./images/C1W1L1_Tour.PNG&quot;  alt=&#39;missing&#39; width=&quot;400&quot;  &gt;&lt;center/&gt;
&lt;figure/&gt;

在本课程中，我们主要会用到 Jupyter notebooks 的两种单元格。第一种就像你现在看到的这段文字一样，专门用来编写说明文档，被称为 `Markdown 单元格` (Markdown Cells) 。它的名字源于单元格内部所使用的轻量级排版标记语言 Markdown。在课程中，你不需要自己动手创建 markdown 单元格。不过，熟悉一下下方图片中展示的 `单元格下拉菜单` (cell pulldown) 会很有帮助。有时候，某个单元格可能会不小心切换到错误的模式，这时你就可以通过这个下拉菜单将它恢复到正常状态：

&lt;figure&gt;
   &lt;img src=&quot;./images/C1W1L1_Markdown.PNG&quot;  alt=&#39;missing&#39; width=&quot;400&quot;  &gt;
&lt;figure/&gt;

另一种则是用来编写程序的 `代码单元格` (code cell) ，你后续所有的代码都会写在这里：


```python
#This is  a 'Code' Cell
print("This is  code cell")
```

    This is  code cell


## Python 基础操作
你可以直接在代码单元格里编写代码。
想要运行写好的代码，只需先选中对应的单元格，然后任选以下两种方式之一操作即可：
- 按住 Shift 键 (shift-key) 不放，同时按下回车键 ('enter' 或 'return') 
- 直接点击顶部工具栏中的“运行” ('run') 箭头按钮
&lt;figure&gt;
    &lt;img src=&quot;./images/C1W1L1_Run.PNG&quot;  width=&quot;400&quot;  &gt;
&lt;figure/&gt;

 

### 打印语句 (Print statement) 
在课程中，打印语句通常会采用 Python 简洁直观的格式化字符串 (f-string) 风格。  
不妨在下方的单元格里动手写一行属于你自己的 print 语句吧。  
建议把上面提到的两种运行单元格的方法都亲自试一试。


```python
# print statements
variable = "right in the strings!"
print(f"f strings allow you to embed variables {variable}")
```

    f strings allow you to embed variables right in the strings!


# 恭喜！
现在，你已经掌握了在 Jupyter Notebook 中自如探索的基本操作。
