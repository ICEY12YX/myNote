___
# 视频解码知识
H.264,YUV也可以都当做一种文件格式
![[Pasted image 20250528170947.png]]
___
# 环境搭建
![[Pasted image 20250528172444.png]]

先创建一个新项目(这是还没编译运行的项目)
拷贝文件之前
F:\University\Audio-and-video\coding\Project1  (应该还有个x64文件夹)
![[Pasted image 20250528173027.png]]
F:\University\Audio-and-video\coding\Project1\Project1 (应该也还有个x64文件夹)
![[Pasted image 20250528173040.png]]

后面我根据cherno的做法,把应该拷贝过来的文件整理了一下
先F:\University\Audio-and-video\coding\Project1\Project1创建src,把cpp文件放这里

F:\University\Audio-and-video\coding\Project1  新建Dependencies文件夹,在里面创建ffmpeg文件夹

把include和lib放这里
F:\University\Audio-and-video\coding\Project1\Dependencies\ffmpeg
![[Pasted image 20250528173748.png]]

然后dll文件复制到
F:\University\Audio-and-video\coding\Project1\x64\Debug
![[Pasted image 20250528174458.png]]

然后进行配置
![[Pasted image 20250528175154.png]]
![[Pasted image 20250528175330.png]]

把lib文件中的文件名通通复制过来
F:\University\Audio-and-video\coding\Project1\Dependencies\ffmpeg\lib
![[Pasted image 20250528175708.png]]![[Pasted image 20250528175730.png]]

dll文件的话是动态库不需要配置,上一步复制进来就算完成了

## 报错记录
首先代码是没问题的
```cpp
#define __STDC_CONSTANT_MACROS  //注意开头是两个横线
extern "C"
{
#include "libavcodec/avcodec.h"
}
#include <iostream>

int main() {
	std::cout << "Hello" << std::endl;
	printf("%s", avcodec_configuration());
	std::cin.get();

}
```

bug1: 报错了error C2065: “PTRDIFF_MAX”: 未声明的标识符
**FFmpeg 的头文件目录中包含旧版本或冲突的标准库文件**（如 `stdint.h`、`inttypes.h`），这些文件与 Visual Studio 自带的 C/C++ 标准库冲突，导致编译器优先使用了错误版本的头文件，从而引发 `error C2065` 未定义标识符错误。
要把include里面三个头文件删掉
![[Pasted image 20250528194745.png]]

bug2:error LNK2019: 无法解析的外部符号 avcodec_configuration，函数 main 中引用了该符号
1>F:\University\Audio-and-video\coding\Project2\Dependencies\ffmpeg\lib\avcodec.lib : warning LNK4272: 库计算机类型“x86”与目标计算机类型“x64”冲突

改x86
![[Pasted image 20250528194934.png]]
改完以后,会多一个debug文件夹,很无语,为了统一你至少也要先x86再debug啊,不知道VS的设计人员怎么设计的
注意此时要把dll文件复制到这个Debug文件夹中
![[Pasted image 20250528193204.png]]