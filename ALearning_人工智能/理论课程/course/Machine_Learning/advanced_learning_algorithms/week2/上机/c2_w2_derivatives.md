# 选修实验 - 导数 (Derivatives)
本次实验将帮助你对导数建立起更加直观的理解。它将向你展示一种通过算术方法计算导数的简单方式，同时还会介绍一个非常实用的 Python 库，让你能够以符号计算 (Symbolic Computation) 的方式求取导数。


```python
from sympy import symbols, diff
```

## 导数的直观定义

导数的正式数学定义往往包含极限 (Limits) 以及“趋近于零”的变量，看起来难免让人望而生畏。但其实它的核心思想非常简单。

函数的导数，描述的是当某个输入变量发生微小变化时，该函数的输出会随之发生怎样的变化。

让我们以代价函数 (Cost Function) $J(w)$ 为例。代价 $J$ 是输出，而 $w$ 是输入变量。  
让我们给这个“微小变化”起个名字，叫做 *epsilon* 或 $\epsilon$。之所以使用这些希腊字母，是因为数学界习惯用 *epsilon* ($\epsilon$) 或 *delta* ($\Delta$) 来代表一个非常小的数值。你可以把它想象成 0.001 或者其他某个很小的数字。  

$$
\begin{equation}
\text{if } w \uparrow \epsilon \text{ causes }J(w) \uparrow \text{by }k \times \epsilon \text{ then}  \\
\frac{\partial J(w)}{\partial w} = k \tag{1}
\end{equation}
$$

这个式子的意思很简单：如果你把函数 $J(w)$ 的输入稍微改变一点点，而输出随之改变了那个微小量的 $k$ 倍，那么 $J(w)$ 的导数就等于 $k$。

让我们亲自试一试。下面我们来看看函数 $J(w) = w^2$ 在 $w=3$ 且 $\epsilon = 0.001$ 处的导数。


```python
J = (3)**2
J_epsilon = (3 + 0.001)**2
k = (J_epsilon - J)/0.001    # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k:0.6f} ")
```

我们将输入值稍微增加了一点点 (0.001) ，使得输出从 9 变成了 9.006001，增加的幅度大约是输入增加量的 6 倍。对照上面的公式 (1) ，这意味着 $k=6$，因此 $\frac{\partial J(w)}{\partial w} \approx 6$。如果你学过微积分 (Calculus) ，你一定知道它的符号表达式是 $\frac{\partial J(w)}{\partial w} = 2 w$。当 $w=3$ 时，结果就是 6。我们上面的算术计算结果之所以不完全等于 6，是因为要想绝对精确，$\epsilon$ 必须无穷小，也就是小到极致。正因如此，我们使用的是约等于号 $\approx$ 或 ~=，而不是等号 =。让我们看看如果把 $\epsilon$ 变得更小会发生什么。


```python
J = (3)**2
J_epsilon = (3 + 0.000000001)**2
k = (J_epsilon - J)/0.000000001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

随着我们不断缩小 $\epsilon$ 的数值，计算结果越来越接近精确的 6。你可以随时尝试把这个值改得更小一些。

## 求取符号导数 (Symbolic Derivatives)
在反向传播 (Backprop) 算法中，了解简单函数在任意输入值下的导数是非常有用的。换句话说，我们希望知道的是“符号”导数，而不是具体的“算术”导数。符号导数的一个例子就是上面 $J(w) = w^2$ 的导数表达式 $\frac{\partial J(w)}{\partial w} = 2 w$。有了符号导数，你就可以直接求出在任意输入值 $w$ 下的导数值。  

如果你上过微积分课，你一定对数学家们为了根据函数表达式求导而总结出的众多求导法则 (Differentiation Rules) 并不陌生。好消息是，如今这一过程已经通过符号微分 (Symbolic Differentiation) 程序实现了自动化。在 Python 中，SymPy 库就是一个典型的代表。下面让我们来看看如何使用它。

### $J = w^2$
首先定义 Python 变量及其对应的符号名称。


```python
J, w = symbols('J, w')
```

定义并打印该数学表达式。请注意，SymPy 会生成一段 LaTeX 字符串，从而渲染出非常美观易读的数学公式。


```python
J=w**2
J
```

使用 SymPy 的 `diff` 函数对表达式 $J$ 关于 $w$ 求导。可以看到，结果与我们前面的例子完全吻合。


```python
dJ_dw = diff(J,w)
dJ_dw
```

通过将具体的数值“代入 (substituting) ”符号变量，我们可以计算出导数在几个特定点上的数值。在第一个例子中，我们将 $w$ 替换为 $2$。


```python
dJ_dw.subs([(w,2)])    # derivative at the point w = 2
```


```python
dJ_dw.subs([(w,3)])    # derivative at the point w = 3
```


```python
dJ_dw.subs([(w,-3)])    # derivative at the point w = -3
```

## $J = 2w$


```python
w, J = symbols('w, J')
```


```python
J = 2 * w
J
```


```python
dJ_dw = diff(J,w)
dJ_dw
```


```python
dJ_dw.subs([(w,-3)])    # derivative at the point w = -3
```

将其与算术计算的结果进行比较：


```python
J = 2*3
J_epsilon = 2*(3 + 0.001)
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

对于函数 $J=2w$，很容易看出，无论 $w$ 的起始值是多少，$w$ 的任何变化都会导致输出 $J$ 发生 2 倍的变化。我们的符号计算与算术计算结果都证实了这一点。

## $J = w^3$


```python
J, w = symbols('J, w')
```


```python
J=w**3
J
```


```python
dJ_dw = diff(J,w)
dJ_dw
```


```python
dJ_dw.subs([(w,2)])   # derivative at the point w=2
```

将其与算术计算的结果进行比较：


```python
J = (2)**3
J_epsilon = (2+0.001)**3
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

## $J = \frac{1}{w}$


```python
J, w = symbols('J, w')
```


```python
J= 1/w
J
```


```python
dJ_dw = diff(J,w)
dJ_dw
```


```python
dJ_dw.subs([(w,2)])
```

将其与算术计算的结果进行比较：


```python
J = 1/2
J_epsilon = 1/(2+0.001)
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

## $J = \frac{1}{w^2}$


```python
J, w = symbols('J, w')
```

如果你有时间，请试着对函数 $J = \frac{1}{w^2}$ 重复上述步骤，并求出它在 w=4 处的导数值。


```python

```


```python

```


```python

```

将其与算术计算的结果进行比较：


```python
J = 1/4**2
J_epsilon = 1/(4+0.001)**2
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

&lt;details&gt;
  &lt;summary&gt;&lt;font size=&quot;3&quot; color=&quot;darkgreen&quot;&gt;&lt;b&gt;点击查看提示&lt;/b&gt;&lt;/font&gt;&lt;/summary&gt;
    
```python 
J= 1/w**2
dJ_dw = diff(J,w)
dJ_dw.subs([(w,4)])
```
  

&lt;/details&gt;

    


## 恭喜！
通过运行上面的这些例子，你已经明白了导数描述的就是当函数的输入发生微小变化时，其输出随之产生的变化率。同时，你也学会了如何在 Python 中使用 *SymPy* 来求取函数的符号导数。
