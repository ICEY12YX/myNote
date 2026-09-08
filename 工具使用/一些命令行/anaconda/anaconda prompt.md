___
```shell
# 查看所有创建的虚拟环境列表
conda env list

# 先激活环境
conda activate pytorch
# 然后查看下载的包列表
conda list
#或者不激活环境，直接指定环境名来查看
conda list -n pytorch
```

___
pip和conda都可以安装包
但是不大一样, 一般常用pip

两者不要混用, 
pip安装的包,conda感知不到 (upgrade的时候感知不到)
所以还是都用pip比较好
___
```shell
# 然后再决定是否需要升级
conda update numpy    # 用 conda 升级（推荐）
# 或
pip install --upgrade numpy    # 用 pip 升级

#安装(可以几个库一起存)
pip install numpy pandas matplotlib
```