https://blog.csdn.net/weixin_72357342/article/details/132104889?ops_request_misc=%257B%2522request%255Fid%2522%253A%252277f2e16bb8796f0ca810e32b7bf3c94c%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=77f2e16bb8796f0ca810e32b7bf3c94c&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-132104889-null-null.142^v102^pc_search_result_base2&utm_term=memcpy&spm=1018.2226.3001.4187
___
**从源头指向的内存块拷贝固定字节数的数据到目标指向的内存块.**
```cpp
void * memcpy ( void * destination, const void * source, size_t num );
```