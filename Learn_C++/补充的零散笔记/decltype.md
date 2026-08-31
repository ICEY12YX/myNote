___
![[Pasted image 20250630164141.png]]
![[Pasted image 20250630164203.png]]
![[Pasted image 20250630164220.png]]
![[Pasted image 20250630164301.png]]
下面的先看看
![[Pasted image 20250630164325.png]]
![[Pasted image 20250630164334.png]]

```cpp
template <typename T, typename U>
auto add(T a, U b) -> decltype(a + b) { // 尾置返回类型
    return a + b;
}
```
![[Pasted image 20250630164511.png]]
![[Pasted image 20250630164537.png]]
![[Pasted image 20250630164559.png]]

然后上面`auto add(T a, U b) -> decltype(a + b)` 中的auto是尾置返回类型的固定写法
![[Pasted image 20250630164741.png]]