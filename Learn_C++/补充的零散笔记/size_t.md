___
像这里的std::size_t是啥
```cpp
#include <iostream>

template <typename T, std::size_t N>
class FixedArray {
public:
    T data[N];

    //因为我们自己写了一个数组类
    //为了让这个数组类可以通过arr[0]这样取值,要重载[]运算符
    T& operator[](std::size_t index) {
        return data[index];
    }

    void print() const {
        for(std::size_t i = 0; i < N; ++i)
            std::cout << data[i] << " ";
        std::cout << std::endl;
    }
};

int main() {
    FixedArray<int, 5> arr;
    for(int i = 0; i < 5; ++i)
        arr[i] = i * 10;
    
    arr.print(); // 输出：0 10 20 30 40 
    return 0;
}
```

具体解释
    ![[Pasted image 20250628151820.png]]
    ![[Pasted image 20250628151851.png]]
    ![[Pasted image 20250628151905.png]]
    ![[Pasted image 20250628152037.png]]
    ![[Pasted image 20250628152052.png]]