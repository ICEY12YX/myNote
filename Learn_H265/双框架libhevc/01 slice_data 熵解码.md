___
`cabac_init_idc` 0,1,2
cabac_init_flag:  (slice header)   仅 P/B slice 有；I slice 无此字段（隐含为 0）
这里也可以看出  在slice header里面
![[Pasted image 20251119191220.webp]]

解释
    ![[Pasted image 20251119190953.webp]]
    **I slice 固定用 idc=0**
    **P slice 有两种选择**
    ![[Pasted image 20251119191033.webp]]

```cpp
slice_qp = CLIP3(slice_qp, 0, 51);
```
解释
    ![[Pasted image 20251119193835.webp]]
    ![[Pasted image 20251119194338.webp]]
    后面if这段代码相关知识
    ![[Pasted image 20251119194633.webp]]

cabac重置初始化条件
    初始化条件
    ![[Pasted image 20251119195828.webp]]

wavefront
    ![[Pasted image 20251119200222.webp]]
    ![[Pasted image 20251119200241.webp]]

cab_ctxt_t  和  gau1_ihevc_cab_ctxts
    ![[Pasted image 20251119201023.webp]]
    ![[Pasted image 20251119201748.webp]]
    ![[Pasted image 20251119202002.webp]]
    ![[Pasted image 20251119202043.webp]]
    ![[Pasted image 20251119202113.webp]]
    (这个表写死了,也可能是py算出来的)
    gau1_ihevc_cab_ctxts这个表的值就代表au1_ctxt_models[CTX_TABLE_SIZE]初始取多少

为什么`&gau1_ihevc_cab_ctxts[cabac_init_idc][slice_qp][0]` 最后一维初始定为0,但是我又不知道现在解的是哪个语法元素,为什么就定ctxIdx=0?
    ![[Pasted image 20251119203139.webp]]

## ihevcd_cabac_init
解释
    ![[Pasted image 20251119203706.webp]]

![[Pasted image 20251119204915.webp]]
解释
    ![[Pasted image 20251119204929.webp]]
    ![[Pasted image 20251119205016.webp]]