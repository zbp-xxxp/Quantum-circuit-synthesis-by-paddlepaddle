# 飞桨常规赛：量子电路合成4月第5名方案

本方案基于[飞桨常规赛：量子电路合成3月第3名方案](https://aistudio.baidu.com/aistudio/projectdetail/1620929)，我在其基础上从**超参数的角度**进行优化 将成绩从2.1748分（我在3月份的最终得分）/18.0889（红白黑大佬在3月份的最终得分）提高到19.4944分（我在4月份的最终得分）。

# 一、赛题分析与重点难点剖析

比赛包含六道题目，其中前四题为基础题 (满分分别为 1 分，2 分，3分，10 分)，第五题为进阶题 (满分为 25 分)，第六题为挑战题 (满分为 60 分)。六道题又简单到复杂，但**本质上都是构建指定量子的量子电路求解最优门的$\theta$值**

> 下面是红白黑大佬在量子电路合成3月第3名方案上讲解的思路。
   
## 赛题解答核心

利用量桨的优势，结合paddle的反向传播优化机制快速搭建量子电路优化。
    
## 模型构建思路

1. 设置量子数目，根据量子数目构建量子电路模块--量子数目决定量子电路的输入端个数
2. 【2-3可交换顺序】设置theta参数形状，从而确定电路过程中y门需要优化的参数$\theta$
3. 上一步之前可以先配置量子电路网络后再确定其中theta的参数形状
4. 构建优化网络部分，将theta的形状传入网络中，从而生成可优化参数--通过paddle的优势进行优化【这一步之前需要设计优化损失，自定义——但要根据反向优化的最小值求解为基础设计损失函数，避免优化错误】
5. 创建优化器
6. 迭代优化，记录theta参数与loss曲线
7. 根据问题公式评估得分
8. 保存训练后所需的theta值
            
## 损失函数构建思路

利用paddle的最小化优化方法，与优化参数矩阵与目标矩阵的最大相似结合:

> 创建目标函数: loss = 1 - 相似矩阵求解的值  or  loss = - 相似矩阵求解的值


## 问题求解说明

1. 问题1：无直接的数据集比对，对给定量子电路，所以只需对应搭建y门电路即可
2. 问题2，3：对给定的电路进行theta优化，然后与直接的数据集中的数据进行比对，得到得分
3. 问题4、5：均参考paddle-quantum的内置若纠错网络结构进行构建电路，实现简单网络求解
   
说明：
- 在问题4中，对弱纠缠网络进行展开成y门进行逐一搭建
- 在问题5中，对弱纠缠网络的组件网络进行展开——其源码可参考quantum的real_block_layer

以上网络求解不超过30秒

4. 问题6：简单用单一的弱纠缠网络与强纠缠网络搭建求解电路暂未能解决问题，所以需要未来做一点其它的尝试进行求解（当前未完成）

# 二、优化思路介绍

我的主要优化思路是从超参数下手。

1.降低学习率，提高训练epoch数，从而使计算结果的精度提高

2.尝试使用不同的优化器以找到得分最高的$\theta$值

3.从数学的角度解析量子电路求解最优门的$\theta$值

# 三、具体方案分享

## 降低学习率，提高迭代次数

量子电路合成3月第3名方案的学习率及迭代次数为：
- lr = 0.32 # 学习率
- iters = 100 # 迭代次数

我将学习率降低的同时提高迭代次数：
- lr = 0.001 # 学习率
- iters = 1000 # 迭代次数

## 尝试使用不同的优化器

量子电路合成3月第3名方案使用的是Adam优化器：
- opts = optimizer.Adam(learning_rate=lr, parameters=model.parameters())

飞桨提供的优化器API多达10个，可尝试更换不同的优化器：
- adadelta = paddle.optimizer.Adadelta(learning_rate=0.0003, epsilon=1.0e-6, rho=0.95, parameters=model.parameters())
- adagrad = paddle.optimizer.Adagrad(learning_rate=0.1, parameters=model.parameters())
- lamb = paddle.optimizer.Lamb(learning_rate=0.002, parameters=model.parameters(), lamb_weight_decay=0.01)
- ... ...

## 从数学的角度解析量子电路求解最优门的θ值

以第一题单量子比特门近似为例。

### 问题描述
寻找合适的参数 $\theta$，使用 $R_y(\theta)$旋转门来近似单量子比特门 U
$$
U:=\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -1 \\ 1 & 1\end{bmatrix}
$$

![](https://img-blog.csdnimg.cn/2021030615550947.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3picF8xMjEzOA==,size_16,color_FFFFFF,t_70)

### 输入数据
无

### 输出数据

 Question_1_Answer.txt 文件。该文件描述答案的量子电路结构，数据格式需满足 “提交内容说明” 要求，即文件内容格式必须为


> R 0 $\theta$

其中 $\theta$ 是选手给出的 float 类型实数。

### 评分机制
算分程序根据选手提交的量子电路结构数据解析出 $\theta$ 值，计算量子门保真度函数F
$$
F(U,Ry(θ))=∣Tr⁡(U×RyT(θ))∣/2
$$

然后将 $F$ 作为最终分数（精确到小数点后四位）。

举例说明： 选手提交的 Question_1_Answer.txt 文件内容为

> R 0 3.1416

算分程序解析出 $θ=3.1416$，因为


$$
\begin{aligned}
F\left(U, R_{y}(3.1416)\right) &=\frac{1}{2}\left|\operatorname{Tr}\left(\frac{1}{\sqrt{2}}\left[\begin{array}{cc}
1 & -1 \\
1 & 1
\end{array}\right] \times\left[\begin{array}{cc}
\cos \left(\frac{3.1416}{2}\right) & -\sin \left(\frac{3.1416}{2}\right) \\
\sin \left(\frac{3.1416}{2}\right) & \cos \left(\frac{3.1416}{2}\right)
\end{array}\right]^{T}\right)\right| \\
&=\frac{1}{2}\left|\operatorname{Tr}\left(\frac{1}{\sqrt{2}}\left[\begin{array}{cc}
1 & -1 \\
1 & 1
\end{array}\right] \times\left[\begin{array}{cc}
\cos \left(\frac{3.1416}{2}\right) & \sin \left(\frac{3.1416}{2}\right) \\
-\sin \left(\frac{3.1416}{2}\right) & \cos \left(\frac{3.1416}{2}\right)
\end{array}\right]\right)\right| \\
& \approx 0.7071
\end{aligned}
$$

所以他的分数为 0.7071。

### 解题思路

这题只有1分，属于这个比赛的Hello World吧，其实就是数学计算。

我们看评分机制，要求计算量子门保真度函数F，F算得多少就是多少分，这题满分1分，所以其实就是求$F(U,Ry(θ))=1$时，$θ$的取值。

根据量子门保真度函数F，我们可以知道
$$
∣Tr⁡(U×RyT(θ))∣/2 = 1
$$

把2乘到等式的右边：
$$
∣Tr⁡(U×RyT(θ))∣ = 2
$$

接下来先计算Tr() 里的结果，简单提一下，Tr(A) 表示矩阵 A 的迹 (Trace)，运算规则为取 n×n 矩阵 A 的主对角线所有元素之和。

题目已经给了：
$$
U=\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -1 \\ 1 & 1\end{bmatrix}
$$

$$
R_{y}(\theta)=\left[\begin{array}{ll}
\cos \frac{\theta}{2} & -\sin \frac{\theta}{2} \\
\sin \frac{\theta}{2} & \cos \frac{\theta}{2}
\end{array}\right]
$$

因此：
$$
R_{y}^{T}(\theta)=\left[\begin{array}{ll}
\cos \frac{\theta}{2} & -\sin \frac{\theta}{2} \\
\sin \frac{\theta}{2} & \cos \frac{\theta}{2}
\end{array}\right]^{T}
=\left[\begin{array}{ll}
\cos \frac{\theta}{2} & \sin \frac{\theta}{2} \\
-\sin \frac{\theta}{2} & \cos \frac{\theta}{2}
\end{array}\right]
$$

带入下面这个式子：
$$
U×RyT(θ)=\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -1 \\ 1 & 1\end{bmatrix} × \left[\begin{array}{ll}
\cos \frac{\theta}{2} & \sin \frac{\theta}{2} \\
-\sin \frac{\theta}{2} & \cos \frac{\theta}{2}
\end{array}\right]
$$

化简一下：
$$
U×RyT(θ)={\sqrt{2}} × (\cos \frac{\theta}{2} + \sin \frac{\theta}{2})
$$

该矩阵的迹是其本身，因此有：
$$
∣Tr⁡(U×RyT(θ))∣ ={\sqrt{2}} × (\cos \frac{\theta}{2} + \sin \frac{\theta}{2}) = 2
$$

解下述方程即可：
$$
{\sqrt{2}} × (\cos \frac{\theta}{2} + \sin \frac{\theta}{2}) = 2
$$

$$
 (\cos \frac{\theta}{2} + \sin \frac{\theta}{2}) = {\sqrt{2}}
$$

两边平方：
$$
1 + 2 \sin \frac{\theta}{2} \cos \frac{\theta}{2}  = 2
$$

化简一下：
$$
\sin \frac{\theta}{2} \cos \frac{\theta}{2}  = 0.5
$$

不难算出：
$$
\theta = \frac{Π}{2} = \frac{3.14}{2} = 1.57
$$

所以这一题的答案是1.57，即参数 $\theta$=1.57时，能使$R_y(\theta)$旋转门来近似单量子比特门 U

# 四、具体代码实现

本方案代码主要参考量子电路合成3月第3名方案代码，具体解析可参考[飞桨常规赛：量子电路合成 3月第3名分享与指导](https://aistudio.baidu.com/aistudio/projectdetail/1780841)


```python
# 下载必要的依赖
!python -m pip install -r work/requirements.txt
```


```python
# 解压数据当前目录
!unzip -oq /home/aistudio/data/data71784/飞桨常规赛：量子电路合成.zip

# 创建提交文件的文件根目录
import os
os.mkdir('Anwser')
```


```python
# 问题1解答-保存结果-vdl显示
!python work/quest1.py
```


```python
# 问题2解答-保存结果-vdl显示
!python work/quest2.py
```


```python
# 问题3解答-保存结果-vdl显示
!python work/quest3.py
```


```python
# 问题4解答-保存结果-vdl显示
!python work/quest4.py
```


```python
# 问题5解答-保存结果-vdl显示
!python work/quest5.py
```


```python
# 问题6未能解答-保存结果-vdl显示
!python work/quest6.py
```


```python
# 保存赛题提交文件
!zip -r Answer.zip Anwser
```

      adding: Anwser/ (stored 0%)
      adding: Anwser/Question_3_Answer.txt (deflated 30%)
      adding: Anwser/Question_4_Answer.txt (deflated 49%)
      adding: Anwser/Question_6_Answer.txt (stored 0%)
      adding: Anwser/Question_5_Answer.txt (deflated 54%)
      adding: Anwser/Question_1_Answer.txt (stored 0%)
      adding: Anwser/Question_2_Answer.txt (deflated 25%)


# 五、总结与升华

量子力学的奠基人波尔曾说：

> “如果你第一次学量子力学认为自己懂了，那说明你还没懂。”

我刚开始接触量子计算时，看到它的数学原理，我不禁想到，这不就是线性代数吗？随着深入学习，我越来越感觉到量子计算的奇妙之处，也感觉到自己可能并没有真正领会量子计算的“奥义”，就像量子力学的奠基人波尔说的那样，如果你第一次学量子力学认为自己懂了，那说明你还没懂。

但不管怎么说，量子计算给我们带来了无限的遐想空间，相信在不久的将来，量子计算会给我们的生活带来新的变化。


# 个人简介

> 北京联合大学 机器人学院 自动化专业 2018级 本科生 郑博培

> 中国科学院自动化研究所复杂系统与智能科学实验室实习生

> 百度飞桨开发者技术专家 PPDE

> 百度飞桨官方帮帮团、答疑团成员

> 深圳柴火创客空间 认证会员

> 百度大脑 智能对话训练师

> 阿里云人工智能、DevOps助理工程师

我在AI Studio上获得至尊等级，点亮9个徽章，来互关呀！！！<br>
[https://aistudio.baidu.com/aistudio/personalcenter/thirdview/147378](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/147378)

![](https://ai-studio-static-online.cdn.bcebos.com/182d11007d3a47248ce40081a4ac5d53fe80e30dc9c443338020bfa8648fe141)
