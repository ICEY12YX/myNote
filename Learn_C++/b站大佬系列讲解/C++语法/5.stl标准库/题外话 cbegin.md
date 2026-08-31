___
```cpp
std::vector<int> ivec{1,2,3,4};

std::vector<int> ivec2(ivec.cbegin(), ivec.cend());
for (int elem : ivec2) {
	std::cout << elem << std::endl;
}
```
cbegin和begin的区别
    ![[Pasted image 20250617165739.png]]
    ![[Pasted image 20250617165758.png]]
    ![[Pasted image 20250617165810.png]]
    ![[Pasted image 20250617165827.png]]

但是insert的时候也可以用
```cpp
std::vector<int> ivec(3, 5);
ivec.insert(ivec.cbegin(), 100);  //当然填begin()也可以
```
![[Pasted image 20250617165922.png]]
(不大看的懂,总之就是能用,可能insert第一个参数,只表示位置,是不是只读insert根本不care)