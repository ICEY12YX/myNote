___
# 初始引入头文件
```cpp
#define __STDC_CONSTANT_MACROS

extern "C"
{
#include "libavcodec/avcodec.h"
}
```

![[Pasted image 20250529163323.png]]
![[Pasted image 20250529163348.png]]
___

# 基本介绍
## 函数

![[Pasted image 20250529161249.png]]
概括来说
![[Pasted image 20250529163938.png]]

详细说
![[Pasted image 20250529164042.png]]
![[Pasted image 20250529164115.png]]
![[Pasted image 20250529164141.png]]
![[Pasted image 20250529165504.png]]![[Pasted image 20250529165541.png]]
![[Pasted image 20250529165632.png]]

![[Pasted image 20250529165733.png]]![[Pasted image 20250529165651.png]]
![[Pasted image 20250529165756.png]]

![[Pasted image 20250529170649.png]]
avformat_open_input()
![[Pasted image 20250529170716.png]]

avformat_find_stream_info()
![[Pasted image 20250529170733.png]]

## 结构体
嵌套关系
(可以以后把这些类型当做一个类型看待)
![[Pasted image 20250529171021.png]]

上下文是指一些运行时参数,如下面的AVCodecContext就存储解码器/编码器的运行时参数
![[Pasted image 20250529171308.png]]
___
# 数据结构具体介绍

(下面1,2,3就对应第几个是啥意思)
1.因为是嵌套关系,所以iformat是嵌套的AVInputFormat类型
2.就是有几个流视频/音频/字幕,是个int
![[Pasted image 20250529171402.png]]


![[Pasted image 20250529171632.png]]
name 和 long_name的区别
![[Pasted image 20250529200622.png]]

时基 指 先理解为计量时间的单位,后面用来算某一帧该在什么时间播
![[Pasted image 20250529200547.png]]

![[Pasted image 20250529200823.png]]
![[Pasted image 20250529200836.png]]
主要用的是pts, bts的话是因为帧的解码顺序是和具体时间顺序有所不同的
pts数是100,200 它最终会乘以前面的时基,得到某一帧到底在几分几秒播
![[Pasted image 20250529201959.png]]

![[Pasted image 20250529202209.png]]