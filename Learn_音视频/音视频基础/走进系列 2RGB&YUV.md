https://blog.csdn.net/u011686167/article/details/126217050?ops_request_misc=%257B%2522request%255Fid%2522%253A%25227a1302b586418f1508f9c567c9b03a5f%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=7a1302b586418f1508f9c567c9b03a5f&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-7-126217050-null-null.nonecase&utm_term=%E8%B5%B0%E8%BF%9B&spm=1018.2226.3001.4450
___
# RGB
取值范围[0, 255]
## RGBA8888
每个通道占8位，即一个字节
![[Pasted image 20250524201910.png]]
___
## RGB565
R占5位，G占6位，B占5位
(但是排列是B在前)
![[Pasted image 20250524202006.png]]
___
## 图像的像素阵列
P代表一个像素pixel
![[Pasted image 20250524202142.png]]

<mark style="background: #ABF7F7A6;">为了内存对齐，会使用stride来填充</mark>                                                                                                                                                                                                                                 

### 为什么要内存对齐
![[Pasted image 20250524205143.png]]
![[Pasted image 20250526170133.png]]
![[Pasted image 20250526170147.png]]
___
## stride
Stride 是图像在内存中<mark style="background: #FFF3A3A6;">每行</mark>像素实际占用的字节数，它可能比图像的**实际宽度更大**（为了内存对齐而填充多余字节）

![[Pasted image 20250524204628.png]]
![[Pasted image 20250524204759.png]]
___
# YUV
Y代表Luma亮度，U代表Chroma色度，V代表Contrast对比度

<mark style="background: #FFF3A3A6;">YUV整体占比是3/2，RGB整体占比是6/2，YUV所占存储空间比RGB少了3/2。因此，默认采用YUV作为视频压缩存储格式</mark>
(3/2=(1+1+4)/4像素)

## 完全,水平,垂直采样
![[Pasted image 20250526171119.png]]

![[Pasted image 20250526172720.png]]
![[Pasted image 20250526173059.png]]
![[Pasted image 20250526173111.png]]
___
## YUV420p
(Y1,Y2,Y9,Y10,共享U1,V1)
![[Pasted image 20250526174418.png]]
___
## YUV420sp
UV分量交错存储
![[Pasted image 20250526174842.png]]
___
## NV21
(和上面有点区别,但是差不多)
![[Pasted image 20250526174918.png]]
___
# 一些代码
## RGB转YUV
```cpp
// *yuv *rgb就当做数组,这里是原数据存在rgb数组里
void rgb_to_yuv(int8_t *yuv, int *rgb, int width, int height) {
    int rgbIndex = 0;   //在下面以这样rgb[rgbIndex]使用,代表现在取到哪了
    int yIndex = 0;
    int uIndex = width * height;  //u的index从width * height开始(因为是先存Y)
    int vIndex = width * height * 5 / 4; //key2:看下面解释
    int R, G, B;
    float Y, U, V;
 
    // 遍历图像，获取所有像素点
    // i,j代表当前取到每行每列的第几个像素
    //(和像素点对应,所以一行该取width次,一列该取height次)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // key1: 从像素点获取R、G、B分量(为什么这样获取,下面有解释)
            R = (rgb[rgbIndex] & 0xFF0000) >> 16;
            G = (rgb[rgbIndex] & 0xFF00) >> 8;
            B = (rgb[rgbIndex] & 0xFF);
            
            // 使用公式把RGB转成YUV
            Y = 0.299 * R + 0.587 * G + 0.114 * B;
            U = -0.147 * R - 0.289 * G + 0.436 * B;
            V = 0.615 * R - 0.515 * G - 0.100 * B;
            
            // YUV分量赋值给yuv数组
            yuv[yIndex++] = (int8_t)Y;
            //这里是4个Y共享一个UV
            //i,j又代表当前取到每行每列的第几个像素了
            //所以i % 2 == 0 && j % 2 == 0代表当前处在2X2方格左上角那个位置
            //这个时候才需要存一个U一个V
            if (i % 2 == 0 && j % 2 == 0) {
                yuv[uIndex++] = (int8_t) U;
                yuv[vIndex++] = (int8_t) V;
            }
            rgbIndex++;
        }
        
    } 
}
```

**key1**
![[Pasted image 20250524185456.png]]

**key2**
![[Pasted image 20250526191819.png]]![[Pasted image 20250526191912.png]]
![[Pasted image 20250526192004.png]]
就是5/4
___
## NV21转换为YUV420p
```cpp
//len为像素个数
static void nv21_to_yuv420p(int8_t *dst, int8_t *src, int len) {
    memcpy(dst, src, len); // y
    for (int i = 0; i < len / 4; ++i) {
	    //NV21是交错存储，4个Y共享一组UV，而且是VUVU这样排列。
	    //所以，我们需要把偶数的V分量、奇数的U分量读出来
        *(dst + len + i) = *(src + len + i * 2 + 1);  // u
        *(dst + len * 5 / 4 + i) = *(src + len + i * 2); // v
    }
}
```
___
## YUV旋转
![[Pasted image 20250526193430.png]]
![[Pasted image 20250526193501.png]]

```cpp
static void yuv420p_rotate90(int8_t *dst, const int8_t *src, int width, int height) {
    int n = 0;
    int offset = width * height;
    int half_width = width / 2;
    int half_height = height / 2;
    // y
    for (int j = 0; j < width; j++) {
        for (int i = height - 1; i >= 0; i--) {
            dst[n++] = src[width * i + j];  //由此看出,这里源数据src是一维数组
        }
    }
    // u
    for (int j = 0; j < half_width; i++) {
        for (int i = height - 1; i >= 0; i--) {
            dst[n++] = src[offset + (i * half_width + j)];
        }
    }
    // v
    for (int j = 0; j < half_width; i++) {
        for (int i = height - 1; i >= 0; i--) {
            dst[n++] = src[offset + offset / 4 + (i * half_width + j)];
        }
    }
}
```
![[Pasted image 20250527161702.png]]
![[Pasted image 20250527161714.png]]



Y分量
下面应该说错了,原图第一列 变 变第一行
![[Pasted image 20250526194001.png]]

U分量,V分量同理,offset是偏移值

转180度就是对原图完全逆序输出
```cpp
static void yuv420p_rotate180(int8_t *dst, const int8_t *src, int width, int height) {
    int n = 0;
    int half_width = width / 2;
    int half_height = height / 2;
    // y
    for (int i = height - 1; i >= 0; i--) {
        for (int j = width; j > 0; j--) {
            dst[n++] = src[width * i + j - 1];
        }
    }
    // u
    int offset = width * height;
    for (int i = half_height - 1; i >= 0; i--) {
        for (int j = half_width; j > 0; j--) {
            dst[n++] = src[offset + half_width * i + j - 1];
        }
    }
    // v
    offset += half_width * half_height;
    for (int i = half_height - 1; i >= 0; i--) {
        for (int j = half_width; j > 0; j--) {
            dst[n++] = src[offset + half_width * i + j - 1];
        }
    }
}
```

```cpp
static void yuv420p_rotate270(int8_t *dst, const int8_t *src, int width, int height) {
	int n = 0;
	
    for (int j = 0; j < width; j++) {
        for (int i = 1; i <= height; i++) {
            dst[n++] = src[i * width - j - 1];
        }
    }

	//*src_u只是指针写法,不用管,和上面一样的
    auto *src_u = const_cast<int8_t *>(src + width * height);
    for (int j = 0; j < width / 2; j++) {
        for (int i = 1; i <= height / 2; i++) {
            dst[n++] = *(src_u + i * width / 2 - j - 1);
        }
    }
 
    auto *src_v = const_cast<int8_t *>(src + width * height * 5 / 4);
    for (int j = 0; j < width / 2; j++) {
        for (int i = 1; i <= height / 2; i++) {
            dst[n++] = *(src_v + i * width / 2 - j - 1);
        }
    }
}
```

```cpp
static void yuv420p_rotate(int8_t *dst, int8_t *src, int width, int height, int degree) {
    switch(degree) {
        case 0:
            memcpy(dst, src, width * height * 3 / 2);
            break;
        case 90:
            yuv420p_rotate90(dst, src, width, height);
            break;
        case 180:
            yuv420p_rotate180(dst, src, width, height);
            break;
        case 270:
            yuv420p_rotate270(dst, src, width, height);
            break;
        default:
            break;
    }
```