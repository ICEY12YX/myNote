# 0.文件夹进出

基本切换目录
```shell
cd 文件夹路径

cd C:\Users\YourName\Documents  # 进入绝对路径**
cd Projects                    # 进入当前目录下的子文件夹
```

返回上一级目录
```shell
cd ..

cd ..      # 返回上一级**
cd ..\..   # 返回上两级
```

直接切换到磁盘根目录
```shell
cd \        # 进入当前磁盘的根目录（如 C:\）
```

切换磁盘驱动器
```shell
盘符:

D:          # 切换到 D 盘（无需 `cd`）
cd D:\Data  # 再进入 D 盘的文件夹
```

快速进入当前用户的个人文件夹
```shell
cd %USERPROFILE%
```
(这会直接进入当前用户的个人目录（如 `C:\Users\YourName`）。)

查看当前目录路径
```shell
chdir 或 cd
```

![[Pasted image 20250519215434.png]]
___

# 1.初始化
参数:
![[Pasted image 20250519215836.png]]


```shell
git config –-global user.name “ICEYYX”
git config –-global user.email 1353479050@qq.com
```


```shell
git config –-global credential.helper store
```
用于将 Git 的认证信息（用户名和密码）保存在本地，避免每次推送（git push）或拉取（git pull）代码时重复输入账号密码

![[Pasted image 20250519220835.png]]

![[Pasted image 20250519220819.png]]


```shell
git config –-global –-list
```
用于 列出当前用户的所有全局 Git 配置。这些配置会影响你电脑上所有的 Git 仓库（除非某个仓库单独覆盖了这些设置）。

![[Pasted image 20250519220756.png]]
![[Pasted image 20250519220620.png]]
___
# 2.创建仓库
## 本地新建仓库

```shell
git init     #在当前目录下建仓库
git init my-repo #在当前目录下建一个my-repo文件夹,并在那里建仓库
```
![[Pasted image 20250519221331.png]]

## 查看.git文件夹

补充:
	`ls` 命令用于 **列出当前目录下的文件和子目录**，但它的具体用法和可用性取决于你使用的环境：

![[Pasted image 20250519221450.png]]
![[Pasted image 20250519221515.png]]

结果:
![[Pasted image 20250519222623.png]]

有.git文件夹就代表当前这个目录是个本地git仓库, 删掉这个.git文件夹就是删掉这个本地仓库(不是删源码,只是删掉git对当前目录的管理)

## 远程clone过来仓库
(clone过来也是在本地建仓库)
```shell
git clone https://github.com/ICEY12YX/git-demo.git #跟上clone地址
```
___
# 3.工作区域和文件状态
## 工作区域

![[Pasted image 20250520134424.png]]
![[Pasted image 20250520134258.png]]

## 文件状态

![[Pasted image 20250520134546.png]]
![[Pasted image 20250520140735.png]]
上面应该是对的,就是命令有点错位了

![[Pasted image 20250520134530.png]]
上面这幅图只展示了2个状态,有点怪,先不管吧
上面这些命令,后面会讲
___
# 4.添加和提交文件

```shell
git status  #查看当前仓库状态

git ls-files #显示 已被 Git 跟踪的所有文件,就是add过的文件,不一定在暂存区
git ls-files -c         # 显示已缓存（暂存）的文件
git ls-files -o         # 显示未跟踪（未暂存）的文件
git ls-files -d         # 显示已删除的文件

git ls-remote  #查看远程仓库文件列表

git ls-tree -r HEAD --name-only #当前分支的最新提交中的所有文件(暂时记住,可以用来看本地仓库有哪些文件)
```

项目结构:
![[Pasted image 20250520145247.png]]
结果:
![[Pasted image 20250520145408.png]]
master指主分支, 版本不同也可以叫main
我们还没有add过仓库,所以文件都处于未跟踪的状态(所以上面显示untracked)

```shell
git add .      #添加当前目录所有的文件修改到暂存区
git add 文件名  #文件加入暂存区(不是正式提交到本地仓库)
git add *.js   #添加所有 `.js` 文件的修改到暂存区
```

结果:
下面最后,指有修改了但是还没被commit提交的文件
还说了,可以用"git rm --cached 文件名"命令取消文件的暂存
![[Pasted image 20250520150319.png]]

```shell
git commit -m "提交描述"  #将 `git add` 添加到暂存区的所有变更提交到本地仓库

git commit -a -m "提交描述" 
#用-a,就是同时干了add和commit两件事
#`-a` 或 `--all`：自动暂存所有已跟踪文件（tracked files）的修改（新增的未跟踪文件仍需手动 `git add`）
#新增的未跟踪文件就是指一次都没add过的文件
#已跟踪文件:已经被 Git 管理的文件（即之前执行过 `git add` 或文件已被提交过）,修改这些文件后，Git 会检测到变更
#新创建文件:新创建的文件，Git 之前从未见过,从未执行过 `git add`

git commit --amend -m "提交描述"  # 替换上一次提交
```
只将暂存区文件,添加到本地仓库,在工作区没有add过的文件,不会commit!!!

结构:
![[Pasted image 20250520151143.png]]
结果:
![[Pasted image 20250520151125.png]]
![[Pasted image 20250520153236.png]]

不写-m "..." 就会转到vim可视化窗口,让我们填写本次提交的描述
这个窗口不是很好操作,建议不要进来
![[Pasted image 20250520152305.png]]

```shell
git log  #按时间倒序列出所有提交记录（最新提交在最上面）
git log --oneline  #查看简洁的提交记录
#可以配合参数,筛选这输出 略
```
结果:
![[Pasted image 20250520153419.png]]
这个commit ==f6bcafe49726e7ec0119e72dd0a6a9f4bdce8349==  叫提交哈希

![[Pasted image 20250520153832.png]]
___

# 5.回退版本
## HEAD指针

**`HEAD` 是一个特殊的指针**，它代表 **当前所在的分支或提交**

![[Pasted image 20250520155611.png]]
![[Pasted image 20250520155749.png]]
第一个父提交,就是指上一个提交

## cat命令

`cat`（全称 **concatenate**）是一个用于 **查看文件内容** 或 **合并文件** 的基础命令。
![[Pasted image 20250520161048.png]]

## reset回退

```shell
git reset commit-hash
#commit-hash指提交哈希
#--soft 回退到提交commit-hash,修改保留(源码不动),暂存保留(比如现在回退到版本2,后面我add了file3,并commit版本3,那现在回退以后暂存区会有file3,相当于把commit版本3的动作取消了,add还在)
#默认的--mixed 回退到提交commit-hash，修改保留(源码不动),但未暂存(比如现在回退到版本2,后面我add了file3,并commit版本3,那现在回退以后暂存区什么也没有)
#--hard 回退到提交commit-hash,修改和暂存区就都回退到提交commit-hash时的状态,这个提交后面可能写了新的代码,然后提交了,这些都不会被保留
```

![[Pasted image 20250520154151.png]]

用处:
回退后:
--mixed 需要在重新commit之前,把一些文件add进入暂存区
--soft 不需要
--hard 真的要放弃本地的所有修改,所以要谨慎使用
但是误操作--hard, 这也是可以回退的

```shell
git reflog
```
是 Git 中的 **"安全网"**，它记录了所有 **本地仓库中 HEAD 和分支指针的移动历史**（包括被删除的提交）。当你误操作（如 `reset --hard`、删除了分支或提交）时，可以通过它找回丢失的内容。

测试:
![[Pasted image 20250520162345.png]]
___
# 6.查看差异

可以查看工作区/暂存区/本地仓库之间的差异
可以查看不同commit版本之间的差异
可以查看不同分支之间的差异

```shell
git diff
#什么都不加,默认比较工作区与暂存区,
#就是写完代码后，检查改了哪些内容，再决定是否 `git add`

git diff HEAD #比较工作区和本地仓库的差异

git diff --staged  #比较暂存区和本地仓库的差异
# 或
git diff --cached

git diff commit-hash1 commit-hash2  #对比两个版本提交
#示例:
git diff abc123 def456   # 对比两次提交的差异
git diff HEAD~ HEAD     # 对比当前与前一次交的差异
git diff HEAD^ HEAD     # 对比当前与前一次提交的差异
git diff HEAD~2 HEAD     # 对比当前与前前次提交的差异

git diff branch1 branch2  #对比两个分支

git diff 文件名 #单查看某个文件的差异
git diff -- 文件路径 #单查看某个文件的差异
#示例:
git diff HEAD~ HEAD -- src/app.js #注意"--"前面后面都有空格
```

输出:
![[Pasted image 20250520163329.png]]
测试:
![[Pasted image 20250520163149.png]]
___
# 7.删除文件
## 方法一 直接删+add

先把文件拖回收站里
删之前
![[Pasted image 20250520165603.png]]
删之后
上面没commit那段不管
下面提示我们要把删掉的文件重新add进暂存区, 大概因为此时暂存区还有删掉的文件的数据,所以我们需要同步一下
![[Pasted image 20250520165658.png]]
然后我们用add .或add 文件名都可以
![[Pasted image 20250520165922.png]]

## 方法二 git rm

方法一太麻烦了
就可用git rm

```shell
git rm 文件名  #同时删除工作区和暂存区的某个文件（相当于 `删除文件` + `git add` 的复合操作）
#要是这个文件commit过了,仍需要手动commit同步本地仓库

git rm --cached 文件名 #仅从 Git 暂存区移除文件（不再跟踪），但工作区源码

git rm -r 目录名      # 删除目录及内容（物理删除）
git rm -r --cached 目录名  # 仅停止跟踪目录（保留本地文件）
```

测试:
![[Pasted image 20250520171220.png]]
___
# 8.gitignore文件

我们写好项目以后,不是所有文件都需要放到版本库中的
像下面这些就不需要
![[Pasted image 20250520171411.png]]

所以我们可以用.gitignore文件,列出我们想忽略的文件规则,然后git会自动帮我们忽略
![[Pasted image 20250520171347.png]]
但是如果,某个文件已经add了,被git跟踪了,再去写.gitignore文件让git忽略的话,那个文件的修改还是会被监测到,所以不能这样
___
# 9.SSH配置和克隆仓库

## 配置SSH

github ssh文档
https://docs.github.com/zh/authentication

.ssh位置
/c/Users/叶馨/.ssh/id_rsa

命令: ssh-keygen
密码我直接敲了回车
![[Pasted image 20250520192554.png]]

.pub的是公匙  没有.pub的是私匙
![[Pasted image 20250520192752.png]]


![[Pasted image 20250520194305.png]]
![[Pasted image 20250520194323.png]]

## 连接远程库github

先在github里面创建一个remote-repo
然后使用ssh来 git clone
![[Pasted image 20250520195132.png]]
结果:
因为我们上面配置了ssh所以成功了
![[Pasted image 20250520195248.png]]

## 本地同步至远程

上面clone了远程仓库
在本地多写一个文件,要怎么再同步到远程?

```shell
git push
git pull  #拉到本地仓库

git push <远程名称> <本地分支>:<远程分支>
git pull <远程名称> <远程分支>:<本地分支>
#示例(两个用法差不多,下面仅展示一个):
#将本地 `main` 分支的提交推送到远程仓库（`origin`）的同名分支 `origin/main`
#远程仓库叫main
#推送的部分,本地与远程同名
git push origin main:main #（可省略重复的 main）
git push origin main  #上一句省略版

git push origin dev:dev  #将本地 `dev` 分支的提交推送到远程仓库的 `dev` 分支

#push独有
git push -u origin main  #首次推送时建立跟踪关系（后续可简写 `git push`）
git push -f origin main  
#如果远程仓库有本地仓库没有的文件,直接把本地push到远程不行的,就是想把远程变的和本地一模一样,不管远程那里有什么,就用-f
```

![[Pasted image 20250520202314.png]]
___
# 10.本都仓库关联远程仓库

## 命令

`git remote` 是 Git 中用于 **管理远程仓库链接** 的命令，它可以查看、添加、修改或删除本地仓库与远程仓库的关联。
```shell
git remote -v  #列出所有远程仓库的别名+地址(SSH的)
```

![[Pasted image 20250520203423.png]]
前面的origin指远程仓库的别名,可以看到一般默认都叫远程仓库origin,这个名字和远程仓库叫什么名字无关

```shell
git remote add <远程名称> <仓库地址>  #本地仓库关联一个新的远程仓库
#示例
git remote add origin git@github.com:ICEY12YX/first-repo.git
#上面关联一个后还能关联别的远程库
git remote add upstream git@github.com:......  #另一个远程库的别名比如叫upstream

git remote remove <远程名称>  #删除远程仓库关联
#示例
git remote remove origin

git remote rename <旧名称> <新名称>  #重命名远程仓库
#示例
git remote rename origin upstream
```
## 具体操作

先创建一个远程仓库first-repo,再本地创建一个my-repo
把my-repo关联上first-repo
![[Pasted image 20250520204247.png]]
注意写了上面这些,只是关联了仓库,本地仓库中的内容还没有被push进远程
所以还得写下面这些
![[Pasted image 20250520205708.png]]
注意,可能版本原因,本地创建的项目主分支默认叫master,github上主分支默认叫main
所以这样写会报错
![[Pasted image 20250520205831.png]]

还可以比如远程仓库多了一些东西,然后我要把远程同步至本地,就用git pull
![[Pasted image 20250520211204.png]]
___
# 11.分支
## 快速创建文件

echo命令核心功能是 **将输入的字符串或变量值显示在屏幕上**，同时常用于脚本编写、文件操作和 Git 相关操作中。

```shell
echo "Hello World"  #输出字符串

#快速创建文件
echo "# My Project" > README.md  # 创建 README 文件并写入内容
```

## GUI工具 
https://git-scm.com/downloads/guis
俺现在下了gitKraken(在D盘,找到文件夹打开就好了)

## gitKraken辅助认识分支

```shell
#查看分支
git branch     #列出所有本地分支(其中前面打星号*的是现在所处的分支)
git branch -a  #列出所有分支（包括远程分支）
git branch -v  #查看分支的最后提交信息

#创建分支
git branch <分支名>   #只是创建,不会自动切换到新创建的分支,可用git branch验证

#删除分支
git branch -d <分支名>
git branch -D <分支名>  #强制删除未合并的分支
git push origin --delete <远程分支名>   #删除远程分支

#分支管理
git branch -m <旧分支名> <新分支名>  #重命名分支
git checkout <分支名>        #切换到已有分支
git checkout -b <新分支名>   #创建并切换到新分支

#但是checkout还有恢复文件的作用,为了防止歧义,建议用switch命令来切换分支
git switch <分支名>     #切换到已有分支
git switch -c <分支名>  #创建并切换到新分支
```

测试:
先用gitKraken图形化界面创建分支,会自动切换到该分支
![[Pasted image 20250521153825.png]]
用git branch <分支名>创建分支,不会切换到此分支
![[Pasted image 20250521153239.png]]

现在创建了两个分支
main里除去第一次系统自动的提交,一共三次提交,并分别创建了了3个txt
dev在上面操作完以后创建,一共两次提交,并分别创建了了2个txt
![[Pasted image 20250521154530.png]]

分别在两个分支中用ls列出文件
可以看到main里看不见dev分支中的文件
而dev分支看的见main分支文件
这是因为操作的时机导致的, dev分支上的修改还没有合并到main分支中
![[Pasted image 20250521154757.png]]

又在main创建了两个文件
可以从图形化界面发现,这里分叉了
这个图是从下往上看的
![[Pasted image 20250521155243.png]]

```shell
#合并分支
git switch main       # 切换到接受合并的分支（如 main）
git merge dev # 将 dev 分支合并到 main
#一定是把merge后面的分支合并到当前切换到的分支

#merge以后git会自动为我们commit一次,然后会弹出vim框,输入commit信息
#所以也可以当场就写好commit信息
git merge -m "合并登录功能" feature-login
```

测试:
![[Pasted image 20250521160347.png]]
merge以后的图
![[Pasted image 20250521160330.png]]
merge以后,dev分支还是存在的,除非此时手动删除
可用git branch -d <分支名>删除已经合并过的分支,没有合并过的分支要用-D
![[Pasted image 20250521160612.png]]

```shell
#查看分支图
git log --graph --oneline --decorate --all
```
![[Pasted image 20250521160532.png]]
___
# 12.解决合并冲突

比如两个分支,分别改了同一个文件的同一行代码, 就会产生冲突,merge的时候会报错
这个时候调用git status会显示报错信息
这个时候需要手动解决冲突
就是手动修改文件到底要写成什么样

也可以终止这次合并
```shell
git merge --abort  # 回到合并前的状态
```
___
# 13.rebase和回退

`git rebase` 核心功能是将当前分支的提交“移动”到另一个基准点（通常是目标分支的最新提交），从而生成一个**线性的、更整洁的提交历史**。
总结:将当前分支的提交(最新提交)“嫁接”到目标分支的提交(最新提交)上

比如下图左边
此时"当前分支"为dev,目标分支为main
(所以可以理解为 和merge思路反一反)
![[Pasted image 20250521165142.png]]

这里有个专有名词叫变基
![[Pasted image 20250521170737.png]]
___
