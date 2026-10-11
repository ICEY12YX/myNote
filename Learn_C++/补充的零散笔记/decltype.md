___
![[Pasted image 20250630164141.webp]]
![[Pasted image 20250630164203.webp]]
![[Pasted image 20250630164220.webp]]
![[Pasted image 20250630164301.webp]]
下面的先看看
![[Pasted image 20250630164325.webp]]
![[Pasted image 20250630164334.webp]]

```cpp
template <typename T, typename U>
auto add(T a, U b) -> decltype(a + b) { // 尾置返回类型
    return a + b;
}
```
![[Pasted image 20250630164511.webp]]
![[Pasted image 20250630164537.webp]]
![[Pasted image 20250630164559.webp]]

然后上面`auto add(T a, U b) -> decltype(a + b)` 中的auto是尾置返回类型的固定写法
![[Pasted image 20250630164741.webp]]