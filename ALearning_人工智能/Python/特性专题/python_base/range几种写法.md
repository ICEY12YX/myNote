___
在 Python 中，`range()` 主要有 **3 种写法**：

## 1. `range(stop)` —— 只给一个参数

从 `0` 开始，到 `stop-1` 结束，步长为 `1`。

```python
for i in range(5):
    print(i)   # 0 1 2 3 4
```

## 2. `range(start, stop)` —— 给两个参数

从 `start` 开始，到 `stop-1` 结束，步长为 `1`。

```python
for i in range(1, 10):
    print(i)   # 1 2 3 4 5 6 7 8 9
```

## 3. `range(start, stop, step)` —— 给三个参数

从 `start` 开始，到 `stop-1` 结束，步长为 `step`（可为负数）。

```python
for i in range(0, 10, 2):
    print(i)   # 0 2 4 6 8

for i in range(10, 0, -1):
    print(i)   # 10 9 8 7 6 5 4 3 2 1
```

## 总结表格

| 写法 | 含义 |
|------|------|
| `range(stop)` | 0 → stop-1，步长 1 |
| `range(start, stop)` | start → stop-1，步长 1 |
| `range(start, stop, step)` | start → stop-1，步长 step |

## 补充说明

- `range()` 返回的是一个**惰性可迭代对象**（不是列表），要用 `list(range(...))` 才能看到全部内容。
- 参数必须是**整数**，不支持浮点数（需要小数步长可以用 `numpy.arange`）。
- `step` 为 `0` 会报错 `ValueError`。
- 结束值 `stop` **永远取不到**，这是最常见的坑。

所以你写的 `range(1, 10)` 就是第 2 种写法，表示 1 到 9。