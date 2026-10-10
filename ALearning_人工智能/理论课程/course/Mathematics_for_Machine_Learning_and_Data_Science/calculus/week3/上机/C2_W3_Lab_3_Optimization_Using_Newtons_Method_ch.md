___
# 利用牛顿法进行优化 (Optimization Using Newton's Method)



在本实验中，你将通过代码实现牛顿法 (Newton's method)，对一元和二元函数进行优化求解。同时，你还将把它与梯度下降法 (gradient descent) 进行对比，亲身体会这两种算法各自的优缺点。

# 目录

* 1 - 一元函数优化


* 2 - 二元函数优化



## 依赖库

运行以下代码单元格，加载本实验所需的函数库。

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

```

## 1 - 一元函数优化

我们将使用牛顿法来优化一元函数 $f\left(x\right)$。为了寻找到导数等于零的极值点，我们需要从某个初始点 $x_0$ 出发，计算该点处的一阶导数与二阶导数 ( $f'(x_0)$ 与 $f''(x_0)$ )，然后依据下面的递推公式迈向下一个点：

$$x_1 = x_0 - \frac{f'(x_0)}{f''(x_0)},\tag{1}$$

接着按此规律迭代循环。迭代次数 $n$ 通常也是一个可调的超参数。

现在，让我们利用牛顿法来优化函数 $f\left(x\right)=e^x - \log(x)$ ( 定义域为 $x>0$ )。在代码实现中，我们先定义目标函数 $f\left(x\right)=e^x - \log(x)$ 及其一阶导数 $f'(x)=e^x - \frac{1}{x}$ 和二阶导数 $f''(x)=e^x + \frac{1}{x^2}$：

```python
def f_example_1(x):
    return np.exp(x) - np.log(x)

def dfdx_example_1(x):
    return np.exp(x) - 1/x

def d2fdx2_example_1(x):
    return np.exp(x) + 1/(x**2)

x_0 = 1.6
print(f"f({x_0}) = {f_example_1(x_0)}")
print(f"f'({x_0}) = {dfdx_example_1(x_0)}")
print(f"f''({x_0}) = {d2fdx2_example_1(x_0)}")

```

绘制函数图像，直观查看它的全局最小值 (global minimum)：

```python
def plot_f(x_range, y_range, f, ox_position):
    x = np.linspace(*x_range, 100)
    fig, ax = plt.subplots(1,1,figsize=(8,4))

    ax.set_ylim(*y_range)
    ax.set_xlim(*x_range)
    ax.set_ylabel('$f\,(x)$')
    ax.set_xlabel('$x$')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position(('data', ox_position))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.autoscale(enable=False)
    
    pf = ax.plot(x, f(x), 'k')
    
    return fig, ax

plot_f([0.001, 2.5], [-0.3, 13], f_example_1, 0.0)

```

实现上述的牛顿法：

```python
def newtons_method(dfdx, d2fdx2, x, num_iterations=100):
    for iteration in range(num_iterations):
        x = x - dfdx(x) / d2fdx2(x)
        print(x)
    return x

```

在这个实现中，除了需要一阶和二阶导数外，还有两个输入参数：迭代步数 `num_iterations` 和初始点 `x`。为了优化函数，我们设定好参数并调用该算法：

```python
num_iterations_example_1 = 25; x_initial = 1.6
newtons_example_1 = newtons_method(dfdx_example_1, d2fdx2_example_1, x_initial, num_iterations_example_1)
print("Newton's method result: x_min =", newtons_example_1)

```

可以看到，从初始点 $x_0 = 1.6$ 出发，牛顿法仅经过 $6$ 次迭代就实现了收敛。在实际工程中，当每一步的 $x$ 变化不再明显 ( 或者一阶导数已极度接近于零 ) 时，就可以提前跳出循环。

如果从同一个初始点出发，改用梯度下降法 (gradient descent) 会发生什么？

```python
def gradient_descent(dfdx, x, learning_rate=0.1, num_iterations=100):
    for iteration in range(num_iterations):
        x = x - learning_rate * dfdx(x)
        print(x)
    return x

num_iterations = 25; learning_rate = 0.1; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.2; x_initial = 1.6
gd_example_1 = gradient_descent(dfdx_example_1, x_initial, learning_rate, num_iterations)
print("Gradient descent result: x_min =", gd_example_1) 

```

梯度下降法多了一个控制步长的参数——学习率 (learning rate) `learning_rate`。若在本例中将其设为 `0.1`，算法大约需要 $15$ 次迭代才会逐渐收敛 ( 目标精度达到小数点后 4-5 位 )。即使将学习率提升到 `0.2`，收敛也需要约 $12$ 次迭代，速度依然明显落后于牛顿法。

相比牛顿法，这就是梯度下降法的劣势所在：需要额外调节超参数，而且收敛速度相对较慢。但它也有显著的优势——每一步迭代都无需计算二阶导数；而在更复杂的工程场景下，求二阶导数的计算开销是极其高昂的。因此，梯度下降法单步迭代的计算成本远比牛顿法轻量得多。

这就是数值优化 (numerical optimization) 的真实世界：收敛速度与最终结果高度依赖于初始参数的选择。世上没有“完美”的算法，每种方案都是利弊权衡的结果。

## 2 - 二元函数优化

当扩展到二元函数时，牛顿法所需要的计算量将进一步攀升。从初始坐标点 $(x_0, y_0)$ 出发，更新到下一个点的递推公式变为：

$$\begin{bmatrix}x_1 \\ y_1\end{bmatrix} = \begin{bmatrix}x_0 \\ y_0\end{bmatrix} -  H^{-1}\left(x_0, y_0\right)\nabla f\left(x_0, y_0\right),\tag{2}$$

式中，$H^{-1}\left(x_0, y_0\right)$ 表示函数在点 $(x_0, y_0)$ 处海森矩阵 (Hessian matrix) 的逆矩阵，而 $\nabla f\left(x_0, y_0\right)$ 则为该点处的梯度向量 (gradient)。

让我们用代码实现这个过程。参照讲解视频，定义测试函数 $f(x, y)$ 及其对应的梯度与海森矩阵：

\begin{align}
f\left(x, y\right) &= x^4 + 0.8 y^4 + 4x^2 + 2y^2 - xy - 0.2x^2y,\
\nabla f\left(x, y\right) &= \begin{bmatrix}4x^3 + 8x - y - 0.4xy \ 3.2y^3 + 4y - x - 0.2x^2\end{bmatrix}, \
H\left(x, y\right) &= \begin{bmatrix}12x^2 + 8 - 0.4y && -1 - 0.4x \ -1 - 0.4x && 9.6y^2 + 4\end{bmatrix}.
\end{align}

```python
def f_example_2(x, y):
    return x**4 + 0.8*y**4 + 4*x**2 + 2*y**2 - x*y -0.2*x**2*y

def grad_f_example_2(x, y):
    return np.array([[4*x**3 + 8*x - y - 0.4*x*y],
                     [3.2*y**3 +4*y - x - 0.2*x**2]])

def hessian_f_example_2(x, y):
    hessian_f = np.array([[12*x**2 + 8 - 0.4*y, -1 - 0.4*x],
                         [-1 - 0.4*x, 9.6*y**2 + 4]])
    return hessian_f

x_0, y_0 = 4, 4
print(f"f{x_0, y_0} = {f_example_2(x_0, y_0)}")
print(f"grad f{x_0, y_0} = \n{grad_f_example_2(x_0, y_0)}")
print(f"H{x_0, y_0} = \n{hessian_f_example_2(x_0, y_0)}")

```

运行以下代码单元格，将函数的曲面及等高线可视化：

```python
def plot_f_cont_and_surf(f):
    
    fig = plt.figure( figsize=(10,5))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.set_facecolor('#ffffff')
    gs = GridSpec(1, 2, figure=fig)
    axc = fig.add_subplot(gs[0, 0])
    axs = fig.add_subplot(gs[0, 1],  projection='3d')
    
    x_range = [-4, 5]
    y_range = [-4, 5]
    z_range = [0, 1200]
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X,Y = np.meshgrid(x,y)
    
    cont = axc.contour(X, Y, f(X, Y), cmap='terrain', levels=18, linewidths=2, alpha=0.7)
    axc.set_xlabel('$x$')
    axc.set_ylabel('$y$')
    axc.set_xlim(*x_range)
    axc.set_ylim(*y_range)
    axc.set_aspect("equal")
    axc.autoscale(enable=False)
    
    surf = axs.plot_surface(X,Y, f(X,Y), cmap='terrain', 
                    antialiased=True,cstride=1,rstride=1, alpha=0.69)
    axs.set_xlabel('$x$')
    axs.set_ylabel('$y$')
    axs.set_zlabel('$f$')
    axs.set_xlim(*x_range)
    axs.set_ylim(*y_range)
    axs.set_zlim(*z_range)
    axs.view_init(elev=20, azim=-100)
    axs.autoscale(enable=False)
    
    return fig, axc, axs

plot_f_cont_and_surf(f_example_2)

```

公式 $(2)$ 所描述的二元牛顿法由以下函数实现：

```python
def newtons_method_2(f, grad_f, hessian_f, x_y, num_iterations=100):
    for iteration in range(num_iterations):
        x_y = x_y - np.matmul(np.linalg.inv(hessian_f(x_y[0,0], x_y[1,0])), grad_f(x_y[0,0], x_y[1,0]))
        print(x_y.T)
    return x_y

```

现在运行以下代码来寻找函数的极小值点：

```python
num_iterations_example_2 = 25; x_y_initial = np.array([[4], [4]])
newtons_example_2 = newtons_method_2(f_example_2, grad_f_example_2, hessian_f_example_2, 
                                     x_y_initial, num_iterations=num_iterations_example_2)
print("Newton's method result: x_min, y_min =", newtons_example_2.T)

```

在本例中，以 $(4, 4)$ 作为初始点，牛顿法大约经过 $9$ 次迭代即可收敛。现在让我们来看看梯度下降法的表现：

```python
def gradient_descent_2(grad_f, x_y, learning_rate=0.1, num_iterations=100):
    for iteration in range(num_iterations):
        x_y = x_y - learning_rate * grad_f(x_y[0,0], x_y[1,0])
        print(x_y.T)
    return x_y

num_iterations_2 = 300; learning_rate_2 = 0.02; x_y_initial = np.array([[4], [4]])
# num_iterations_2 = 300; learning_rate_2 = 0.03; x_y_initial = np.array([[4], [4]])
gd_example_2 = gradient_descent_2(grad_f_example_2, x_y_initial, learning_rate_2, num_iterations_2)
print("Gradient descent result: x_min, y_min =", gd_example_2) 

```

显而易见，梯度下降法的收敛速度远慢于牛顿法。如果尝试调大步长提高学习率，甚至可能导致算法震荡发散而完全失效。这再次体现了梯度下降法相比牛顿法的局限。然而必须注意的是，牛顿法需要求解海森矩阵的逆；当面对成千上万个参数的复杂模型时，求逆运算带来的算力开销将是极其巨大的。

祝贺你顺利完成本实验！

```python


```