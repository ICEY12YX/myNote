___
- [[#选修实验：使用计算图 (Computation Graph) 进行反向传播 (Back propagation)|选修实验：使用计算图 (Computation Graph) 进行反向传播 (Back propagation)]]
	- [[#选修实验：使用计算图 (Computation Graph) 进行反向传播 (Back propagation)#计算图|计算图]]
		- [[#计算图#前向传播 (Forward Propagation)|前向传播 (Forward Propagation)]]
		- [[#计算图#反向传播 (Backprop)|反向传播 (Backprop)]]
		- [[#计算图#$\frac{\partial J}{\partial a}$|$\frac{\partial J}{\partial a}$]]
			- [[#$\frac{\partial J}{\partial a}$#算术方法|算术方法]]
			- [[#$\frac{\partial J}{\partial a}$#符号计算方法 (Symbolically)|符号计算方法 (Symbolically)]]
		- [[#计算图#$\frac{\partial J}{\partial w}$|$\frac{\partial J}{\partial w}$]]
			- [[#$\frac{\partial J}{\partial w}$#算术方法|算术方法]]
	- [[#选修实验：使用计算图 (Computation Graph) 进行反向传播 (Back propagation)#简单神经网络的计算图|简单神经网络的计算图]]
		- [[#简单神经网络的计算图#前向传播|前向传播]]
		- [[#简单神经网络的计算图#反向传播 (Backprop)|反向传播 (Backprop)]]
		- [[#简单神经网络的计算图#$\frac{\partial J}{\partial d}$|$\frac{\partial J}{\partial d}$]]
			- [[#$\frac{\partial J}{\partial d}$#算术方法|算术方法]]
			- [[#$\frac{\partial J}{\partial d}$#符号计算方法|符号计算方法]]
		- [[#简单神经网络的计算图#$\frac{\partial J}{\partial a}$|$\frac{\partial J}{\partial a}$]]
			- [[#$\frac{\partial J}{\partial a}$#算术方法|算术方法]]
			- [[#$\frac{\partial J}{\partial a}$#符号计算方法|符号计算方法]]
		- [[#简单神经网络的计算图#$\frac{\partial J}{\partial c}$，$\frac{\partial J}{\partial b}$|$\frac{\partial J}{\partial c}$，$\frac{\partial J}{\partial b}$]]
		- [[#简单神经网络的计算图#$\frac{\partial J}{\partial w}$|$\frac{\partial J}{\partial w}$]]
	- [[#选修实验：使用计算图 (Computation Graph) 进行反向传播 (Back propagation)#恭喜！|恭喜！]]



___
# 选修实验：使用计算图 (Computation Graph) 进行反向传播 (Back propagation)

完成本次实验将帮助你深入理解大多数机器学习 (Machine Learning) 框架都在使用的一项核心算法。梯度下降 (Gradient Descent) 需要计算代价 (Cost) 关于网络中每一个参数 (Parameter) 的导数 (Derivative) 。而神经网络 (Neural Networks) 往往拥有数百万甚至数十亿个参数。**反向传播**算法正是用来高效计算这些导数的，而**计算图**则被用来简化这一运算过程。下面让我们一探究竟。

```python
from sympy import *
import numpy as np
import re
%matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import ipywidgets as widgets
from lab_utils_backprop import *
```

## 计算图
计算图通过将复杂的导数计算拆解成若干个较小的步骤，从而大大简化了计算过程。让我们看看它是如何运作的。

我们来计算一个稍微复杂一点的表达式的导数：$J = (2+3w)^2$。我们希望求出 $J$ 关于 $w$ 的导数，即 $\frac{\partial J}{\partial w}$。

```python
plt.close("all")
plt_network(config_nw0, "./images/C2_W2_BP_network0.PNG")
```
(这个图懒得存了, 反正就是一个让你填空的互动程序)
![[c2_w2_backprop-1794091086286.webp|446]]


从上图中可以看到，我们将该表达式拆解成了两个可以独立处理的节点 (Nodes) 。如果你通过课堂学习已经对这一过程有了很好的理解，可以直接在上面的图表中填写各个方框。你需要先从左往右依次填写蓝色方框，然后再从右往左依次填写绿色方框。
如果你填入的数值正确，它们会显示为绿色或蓝色；如果数值有误，则会显示为红色。请注意，这个交互式图形界面并不是特别稳定，如果你在操作界面时遇到问题，只需重新运行上面的代码单元格即可重启。

如果你对具体步骤还不太确定，下面我们将一步步推导这个例子。

(后面没啥好看的,就是课上教你的手推反向传播)
### 前向传播 (Forward Propagation)   
首先，让我们沿着前向路径计算出各个节点的数值。

>关于本节的一点提示：这里使用了全局变量，并在计算推进的过程中重复使用它们。如果你不按顺序运行单元格，可能会得到奇怪的结果。万一出现这种情况，只需回到这里并按顺序重新运行它们即可。


```python
w = 3
a = 2+3*w
J = a**2
print(f"a = {a}, J = {J}")
```

现在你可以把这些数值填入上图的蓝色方框中了。

### 反向传播 (Backprop)
&lt;img align=&quot;left&quot; src=&quot;./images/C2_W2_BP_network0_j.PNG&quot;     style=&quot; width:100px; padding: 10px 20px; &quot; &gt; 反向传播是我们用来计算导数的算法。正如课堂上所讲，反向传播从最右侧开始，一路向左推进。我们要考虑的第一个节点是 $J = a^2 $，而第一步就是求出 $\frac{\partial J}{\partial a}$。


### $\frac{\partial J}{\partial a}$ 
#### 算术方法
通过观察当 $a$ 发生微小变化时 $J$ 会如何变化，来求出 $\frac{\partial J}{\partial a}$。这一点在关于导数的选修实验中有详细介绍。


```python
a_epsilon = a + 0.001       # a epsilon
J_epsilon = a_epsilon**2    # J_epsilon
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")
```

$\frac{\partial J}{\partial a}$ 的结果是 22，也就是 $2\times a$。我们的算术计算结果之所以不完全等于 $2 \times a$，是因为我们选取的 epsilon 值还不是无穷小。
#### 符号计算方法 (Symbolically)
现在，让我们像在导数选修实验中那样，使用 SymPy 以符号计算的方式来求取导数。我们会在变量名称前加上前缀 's'，以表示这是一个*符号 (symbolic) *变量。


```python
sw,sJ,sa = symbols('w,J,a')
sJ = sa**2
sJ
```


```python
sJ.subs([(sa,a)])
```


```python
dJ_da = diff(sJ, sa)
dJ_da
```

因此，$\frac{\partial J}{\partial a} = 2a$。当 $a=11$ 时，$\frac{\partial J}{\partial a} = 22$。这与我们上面的算术计算结果完全一致。
如果你还没有填写的话，现在可以回到上图中把 $\frac{\partial J}{\partial a}$ 的数值填进去了。

### $\frac{\partial J}{\partial w}$ 
&lt;img align=&quot;left&quot; src=&quot;./images/C2_W2_BP_network0_a.PNG&quot;     style=&quot; width:100px; padding: 10px 20px; &quot; &gt;  继续从右向左推进，下一个我们要计算的数值是 $\frac{\partial J}{\partial w}$。为此，我们需要先计算出 $\frac{\partial a}{\partial w}$，它描述了当输入 $w$ 发生微小变化时，当前节点的输出 $a$ 会随之发生怎样的变化。

#### 算术方法
通过观察当 $w$ 发生微小变化时 $a$ 会如何变化，来求出 $\frac{\partial a}{\partial w}$。


```python
w_epsilon = w + 0.001       # a  plus a small value, epsilon
a_epsilon = 2 + 3*w_epsilon
k = (a_epsilon - a)/0.001   # difference divided by epsilon
print(f"a = {a}, a_epsilon = {a_epsilon}, da_dw ~= k = {k} ")
```

通过算术方法计算得出，$\frac{\partial a}{\partial w} \approx 3$。下面让我们再用 SymPy 来试一试。


```python
sa = 2 + 3*sw
sa
```


```python
da_dw = diff(sa,sw)
da_dw
```

>接下来是最核心、也最有趣的部分：
> - 我们知道，$w$ 的微小变化会导致 $a$ 产生 3 倍的变化量。
> - 我们又知道，$a$ 的微小变化会导致 $J$ 产生 $2\times a$ 倍的变化量 (在本例中 a=11) 。  
 把这两点结合起来，
> - 我们就能知道，$w$ 的微小变化会导致 $J$ 产生 $3 \times 2\times a$ 倍的变化量。
> 
> 这种层层传递的连锁变化，在数学上就叫做*链式法则 (Chain Rule) *。它可以写成如下形式：
 $$\frac{\partial J}{\partial w} = \frac{\partial a}{\partial w} \frac{\partial J}{\partial a} $$
 
如果这一点你还没有完全想明白，非常值得花点时间仔细琢磨一下，因为这是本次实验最关键的核心收获。
 
 下面让我们实际计算一下：
 


```python
dJ_dw = da_dw * dJ_da
dJ_dw
```

在本例中 $a$ 等于 11，因此 $\frac{\partial J}{\partial w} = 66$。我们可以用算术方法来验证一下：


```python
w_epsilon = w + 0.001
a_epsilon = 2 + 3*w_epsilon
J_epsilon = a_epsilon**2
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

太棒了！如果你还没有填写的话，现在可以把 $\frac{\partial a}{\partial w}$ 和 $\frac{\partial J}{\partial w}$ 的数值填入上面的图表中了。

**换个视角来看**  
我们还可以用下面这种方式来直观地理解这种层层放大的连锁变化：  
&lt;img align=&quot;center&quot; src=&quot;./images/C2_W2_BP_network0_diff.PNG&quot;  style=&quot; width:500px; padding: 10px 20px; &quot; &gt;  
$w$ 的一个微小变化先乘以 $\frac{\partial a}{\partial w}$，产生了一个放大为 3 倍的变化；随后，这个变大了的变化量又乘以 $\frac{\partial J}{\partial a}$，最终产生了一个放大为 $3 \times 22 = 66$ 倍的变化。

## 简单神经网络的计算图
下面是课堂上所讲的那个神经网络的计算图，只是换了一组不同的数值。请试着把各个方框中的数值填写完整。请注意，交互式图形界面并不是特别稳定，如果你在操作时遇到问题，只需重新运行下面的单元格即可重启。


```python
plt.close("all")
plt_network(config_nw1, "./images/C2_W2_BP_network1.PNG")
```
![[c2_w2_backprop-1794091220025.webp]]

下面，我们将详细推导填满上述计算图所需的全部计算过程。我们依然从前向路径开始。

### 前向传播
前向路径中的计算正是你最近在神经网络课程中学到的那些基础运算。你可以将下面计算出的数值与你在上图中填写的数值进行核对。


```python
# Inputs and parameters
x = 2
w = -2
b = 8
y = 1
# calculate per step values   
c = w * x
a = c + b
d = a - y
J = d**2/2
print(f"J={J}, d={d}, a={a}, c={c}")
```

### 反向传播 (Backprop)
&lt;img align=&quot;left&quot; src=&quot;./images/C2_W2_BP_network1_jdsq.PNG&quot;     style=&quot; width:100px; padding: 10px 20px; &quot; &gt; 正如课堂上所讲，反向传播从最右侧开始，逐步向左推进。我们要考虑的第一个节点是 $J = \frac{1}{2}d^2 $，而第一步就是求出 $\frac{\partial J}{\partial d}$。


### $\frac{\partial J}{\partial d}$ 

#### 算术方法
通过观察当 $d$ 发生微小变化时 $J$ 会如何变化，来求出 $\frac{\partial J}{\partial d}$。


```python
d_epsilon = d + 0.001
J_epsilon = d_epsilon**2/2
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dd ~= k = {k} ")
```

$\frac{\partial J}{\partial d}$ 的结果是 3，正好等于 $d$ 的数值。我们的算术结果之所以不完全等于 $d$，是因为我们的 epsilon 值还不是无穷小。
#### 符号计算方法
现在，让我们像在导数选修实验中那样，使用 SymPy 以符号计算的方式来求取导数。我们会在变量名称前加上前缀 's'，以表示这是一个*符号*变量。


```python
sx,sw,sb,sy,sJ = symbols('x,w,b,y,J')
sa, sc, sd = symbols('a,c,d')
sJ = sd**2/2
sJ
```


```python
sJ.subs([(sd,d)])
```


```python
dJ_dd = diff(sJ, sd)
dJ_dd
```

因此，$\frac{\partial J}{\partial d}$ = d。当 $d=3$ 时，$\frac{\partial J}{\partial d}$ = 3。这与我们上面的算术计算结果完全吻合。
如果你还没有填写的话，现在可以回到上图中把 $\frac{\partial J}{\partial d}$ 的数值填进去了。

### $\frac{\partial J}{\partial a}$ 
&lt;img align=&quot;left&quot; src=&quot;./images/C2_W2_BP_network1_d.PNG&quot;     style=&quot; width:100px; padding: 10px 20px; &quot; &gt;  继续从右向左推进，下一个我们要计算的数值是 $\frac{\partial J}{\partial a}$。为此，我们需要先计算出 $\frac{\partial d}{\partial a}$，它描述了当输入 $a$ 发生微小变化时，当前节点的输出会如何变化。 (请注意，我们并不关心当 $y$ 变化时输出会如何变化，因为 $y$ 并不是模型的可训练参数。) 

#### 算术方法
通过观察当 $a$ 发生微小变化时 $d$ 会如何变化，来求出 $\frac{\partial d}{\partial a}$。


```python
a_epsilon = a + 0.001         # a  plus a small value
d_epsilon = a_epsilon - y
k = (d_epsilon - d)/0.001   # difference divided by epsilon
print(f"d = {d}, d_epsilon = {d_epsilon}, dd_da ~= k = {k} ")
```

通过算术方法计算得出，$\frac{\partial d}{\partial a} \approx 1$。下面让我们再用 SymPy 来试一试。
#### 符号计算方法


```python
sd = sa - sy
sd
```


```python
dd_da = diff(sd,sa)
dd_da
```

与算术计算的结果一样，符号计算得出的 $\frac{\partial d}{\partial a}$ 也等于 1。  
>接下来依然是那个最关键的步骤，让我们在这个例子中再温习一遍：
> - 我们知道，$a$ 的微小变化会导致 $d$ 产生 1 倍的变化量。
> - 我们又知道，$d$ 的微小变化会导致 $J$ 产生 $d$ 倍的变化量 (在本例中 d=3) 。  
 把这两点结合起来，
> - 我们就能知道，$a$ 的微小变化会导致 $J$ 产生 $1\times d$ 倍的变化量。
> 
>这同样是*链式法则*，可以写成如下形式：
 $$\frac{\partial J}{\partial a} = \frac{\partial d}{\partial a} \frac{\partial J}{\partial d} $$
 
 让我们实际计算一下：
 


```python
dJ_da = dd_da * dJ_dd
dJ_da
```

在本例中 $d$ 为 3，因此 $\frac{\partial J}{\partial a} = 3$。我们可以用算术方法来验证一下：


```python
a_epsilon = a + 0.001
d_epsilon = a_epsilon - y
J_epsilon = d_epsilon**2/2
k = (J_epsilon - J)/0.001   
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")
```

完全吻合！如果你还没有填写的话，现在可以把 $\frac{\partial d}{\partial a}$ 和 $\frac{\partial J}{\partial a}$ 的数值填入上面的图表中了。

> **反向传播的基本步骤**   
>既然你已经亲手推导了好几个节点，现在我们可以总结出反向传播的基本套路了：\
> 从右向左依次处理每一个节点：
>- 计算当前节点的局部导数 (Local Derivative(s)) 
>- 利用链式法则，将其与右侧节点传过来的代价导数相乘合并。  

所谓的“局部导数”，就是当前节点的输出关于其所有输入或参数的导数。

让我们继续完成剩下的部分。既然你已经熟悉了这套方法，接下来的叙述就会更加简明扼要。

### $\frac{\partial J}{\partial c}$，$\frac{\partial J}{\partial b}$
&lt;img align=&quot;left&quot; src=&quot;./images/C2_W2_BP_network1_a.PNG&quot;     style=&quot; width:100px; padding: 10px 20px; &quot; &gt;下一个节点有两个我们感兴趣的导数。我们需要计算 $\frac{\partial J}{\partial c}$，以便继续向左传播；同时我们还需要计算 $\frac{\partial J}{\partial b}$。求出代价关于参数 $w$ 和 $b$ 的导数，正是反向传播的最终目的。我们将先求出局部导数 $\frac{\partial a}{\partial c}$ 和 $\frac{\partial a}{\partial b}$，然后再将它们与从右侧传过来的导数 $\frac{\partial J}{\partial a}$ 相乘合并。


```python
# calculate the local derivatives da_dc, da_db
sa = sc + sb
sa
```


```python
da_dc = diff(sa,sc)
da_db = diff(sa,sb)
print(da_dc, da_db)
```


```python
dJ_dc = da_dc * dJ_da
dJ_db = da_db * dJ_da
print(f"dJ_dc = {dJ_dc},  dJ_db = {dJ_db}")
```

而在我们的例子中，d = 3。

###  $\frac{\partial J}{\partial w}$
&lt;img align=&quot;left&quot; src=&quot;./images/C2_W2_BP_network1_c.PNG&quot;     style=&quot; width:100px; padding: 10px 20px; &quot; &gt; 本例中的最后一个节点负责计算 `c`。在这里，我们关心的是 J 关于参数 w 的变化率。由于我们不需要向输入特征 $x$ 继续反向传播，因此我们对 $\frac{\partial J}{\partial x}$ 并不感兴趣。让我们先从计算局部导数 $\frac{\partial c}{\partial w}$ 开始。


```python
# calculate the local derivative
sc = sw * sx
sc
```


```python
dc_dw = diff(sc,sw)
dc_dw
```

这个导数比刚才那个要稍微有意思一点，它的大小取决于输入 $x$ 的取值。在我们的例子中，这个值是 2。

将其与 $\frac{\partial J}{\partial c}$ 相乘合并，即可求出 $\frac{\partial J}{\partial w}$。


```python
dJ_dw = dc_dw * dJ_dc
dJ_dw
```


```python
print(f"dJ_dw = {dJ_dw.subs([(sd,d),(sx,x)])}")
```

由于 $d=3$，因此在我们的例子中 $\frac{\partial J}{\partial w} = 6$。  
让我们用算术方法来测试验证一下：


```python
J_epsilon = ((w+0.001)*x+b - y)**2/2
k = (J_epsilon - J)/0.001  
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

结果完全一致！太棒了。现在你可以把 $\frac{\partial J}{\partial w}$ 填入上面的图表中，我们的推导分析也就圆满完成了。

## 恭喜！
你已经利用计算图完整推导了一个反向传播的实例。只要遵循同样逐节点推进的方法，你就可以将这一原理应用到规模大得多的复杂网络中。
