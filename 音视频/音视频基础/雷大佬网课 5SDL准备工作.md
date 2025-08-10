___
# bug
## bug1
报错error C4716: “SDL_main”: 必须返回一个值
![[Pasted image 20250530163744.png]]
所以明确写上return 0就好了(一般我们用惯了VS给我们默认加上return 0 )
## bug2
报错warning LNK4098: 默认库“msvcrt.lib”与其他库的使用冲突；请使用 /NODEFAULTLIB:library
![[Pasted image 20250530170158.png]]

改两处
![[Pasted image 20250530165924.png]]
![[Pasted image 20250530165951.png]]

## bug3
报错LINK : warning LNK4217: 符号“_exit”(在“ libucrt.lib(exit.obj)”中定义)已由“SDL2main.lib(SDL_windows_main.obj)”(函数“_main”中)导入
![[Pasted image 20250530170234.png]]
![[Pasted image 20250530170308.png]]