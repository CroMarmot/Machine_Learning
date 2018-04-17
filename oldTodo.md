 # [网易Stanford机器学习](https://open.163.com/movie/2008/1/M/C/M6SGF6VB4_M6SGHFBMC.html)

cs 229

[线上资源](http://cs229.stanford.edu)

## 以下链接是github搜到的

[Github资源](https://github.com/Kivy-CN/Stanford-CS-229-CN)

[课件？](https://github.com/econti/cs229)

[练习](https://github.com/zyxue/stanford-cs229)

[笔记和练习](https://github.com/HuangCongQing/MachineLearning_Ng)

还有qa的邮箱 具体见视频

- [x] 1 机器学习的动机与应用：

先修课(计算机的基础知识(数据结构) 数学知识(高数 线代)等等...),机器学习应用的一些具体方向,会需要用Octave / Matlab

连续 回归 监督学习问题

离散 分类 监督学习问题

无监督学习：分类，图形聚类，声音分离

ICA(独立成分分析算法) `[W,s,v]=svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');`

强化学习，回馈函数,多次 中等不是最差 的行为不停学习

- [x] 2 监督应用学习:梯度下降：

线性回归，梯度下降，正则方程组

m = 训练样本数目

x = 输入变量/属性

y = 输出变量/目标变量

(x,y) = 训练样本

第i个训练样本 (x_i,y_i)

[训练集合]-> 学习算法->产生 假设函数h

h(新的 输入)->产生估计的结果

h(x) = sum({theta_i} * {x_i}), i=1~n,线性关系假设函数

希望 通过调整 theta使sum((h(x_i) - y_i)^2)尽量小

初始theta+梯度下降(theta = theta - (learning rate 学习速度)*偏导theta(J(theta))) ->达到一个局部最小值，其中J(theta)是上面一行的sum/2【然后老师推了一下 单X的 变化情况 XD 就是个求偏导的过程没啥好记录的，从而有了线性回归方程的表达式

如果m很大(数据量很大)，则使用随机梯度下降/增量梯度下降算法

    repeat
        j=1 to m
            all i
                theta_i = theta_i - (learning rate)*偏导(J(theta))

每加一个样本 改动一次所有theta，过程中可能还有与梯度方向偏离较大的，最后 会接近局部最小，但没有梯度下降那么接近，但是快

▽J = [偏导数J/theta_0 , 偏导数J/theta_1 , ... , 偏导数J/theta_n ,]' 梯度

theta = theta - (learning rate) * ▽J

然后 写了一个 输入参数是向量的 情况 把 ▽J 展开的样子

```
矩阵的迹 tr(AB) = tr(BA) , tr(ABC) = tr CAB = tr BCA

▽(trAB,A) = B^T

▽(tr A B A^T C,A) = CAB + C^T AB^T
```

X * THETA - Y = [h(x_i)-y_i, ,.... ]'

1/2 * [X * THETA - Y ]^T * [X * THETA - Y] = 1/2 * sum (1->m) (h(x_i)-y_i)^2 = J(theta)

▽(J,THETA)
= 1/2 ▽((X THETA - Y)^T (X THETA - Y)) 
= 1/2 ▽(THETA^T X^T X THETA - THETA^T X^T Y - Y^T X THETA + Y^T Y) 
= 1/2 ▽ tr(THETA^T X^T X THETA - THETA^T X^T Y - Y^T X THETA) 因为关于ＴＨＥＴＡ求偏导所以倒Ｙ就为０ ,以及　tr实数＝实数
= 1/2 (▽ tr(THETA THETA^T X^T X ) - ▽ tr(Y^T X THETA) - ▽ tr(Y^T X THETA))
= 1/2 ((X^T X THETA I + X^T X THETA I)  - (X^T Y) - (X^T Y)) 通过上面的公式转化
= Ｘ^T X THETA - X^T Y 
希望
= 0

Ｘ^T X THETA ＝ X^T Y 由此　被叫做 Normal equations 也就有THETA = (X^T X)^-1 X^T Y

然后说 如果没有逆 ，可以通过伪逆 来达到最小值

- [x] 3 欠拟合与过拟合的概念

局部加权拟合 ， 如果想 对具体值x，用函数h估计出值

->要让 sum{w_i * (y_i - theta X_i)^2} 其中w_i=e^(-(x_i-x)^2/2),也就是近处权很大

w_i 也可以选取其它权值函数

【如果有兴趣海量数据下的这个算法的应用 可以去看KD tree

证明 最小二乘法是最好(?)的

假设  y_i=theta^T X_i + 误差_i

假设（各种理由 这样假设合理 且比较好） 误差_i ~ N(0 , o^2), 误差_i之间IID(独立同分布) ，P(误差_i) = 高斯/正太分布函数 

迷之加法 拆成条件P(y_i | x_i; theta) ~ N(theta^T X_i , o^2)

L(theta)
=P(Y|X;THETA)
=MULTI  (P(y_i | x_i;theta)) = 按概率公式展开

所以L都出来的 ，然后 真的就用了极大似然估计 XD...... 哦 所以我还是不懂 第二讲中 的J的1/2是从哪里来的了

---

第一个分类算法

二元分析？ 所以 只能 单侧0 另一侧1 混杂的就不能了？

h(X)_theta = g(theta^T x) = 1/(1+e^(-theta^T * x)), 符号函数 逻辑函数

P(y=1 | x ; theta) = h(x)_theta

P(y=0 | x ; theta) = 1 - h(x)_theta

L(theta)=P(y|x;theta)=MULTI (h(x_i)_theta)^y_i (1-h(x_i)_theta)^(1-y_i)

最大化(梯度下降是减 这里是加 以及这里的h 和前面的h函数也不同了) 

theta:=theta+2▽(Ln(L(theta)),theta)

▽(Ln(L(theta)),theta)= sum((y_i-h(x_i))x_i)

感知算法， 用g(z) {1 if z>=0 |, 0 otherwise} 说是 theta的推出结论和上面的梯度上升一样

- [x] 4 牛顿方法

theta_{i+1} = theta_i - f(theta_i)/f'(theta_i),也就是牛顿切线法，这里的theta是要找的零点 而f对应前面讲的也就是在极大似然过程中的l函数

比logistics 回归效果更好，但是f需要满足 某些 更复杂的条件

收敛速度快(比如和梯度上升比较) 二次收敛

theta = theta - H^-1 ▽ (l,theta),其中H 是矩阵 H_{i,j} = l 偏导 theta_i, theta_j

---

广义线性模型 GLM

P(y;theta) = b(y) e^(theta^转置 T(y) - a(theta)) ,,, 其中不同的a b  T, T是一个什么充分统计量 也是概率分布

theta- natural parameter 分布的自然参数

伯努利和高斯分布N(mu,o^2)都是这个的特例

伯努利

```
P(y;theta) 
= theta^y (1-theta)^(1-y)
=e^(y log(theta) + (1-y) log(1-theta))
=提取y 对应 广义线性模型

广义中n = log(theta/(1-theta)) -> theta =1/(1+e^-广义中n)
T(y)=y
b(y)=1
```

f 高斯分布函数= 提取拆解成广义线性模型的样子 然后对应 a b T

老实说 关于 变成高斯分布 过程都在讲义上

泊松分布通常 计数过程建模，然后说了一堆都能转化成指数分布族

---

如何生成 广义线性模型

假设
```
(1) y|x;theta ~ ExpFamily( alta)
(2)Given X ,want h(X) = E[T(y)|X]
(3) alta = theta^T X = 实数
```

上面说了广义线性模型可以表示伯努利以及高斯 (也就是这两个是广义线性模型的特殊例子)，下面这部分 就是在证明 伯努利对应的函数正好是 1/(1+e^(theta^T X)), 以及高斯对应的函数正好是最小二乘模型 (老师说这部分省略了)

伯努利分布下，y只取0/1, T(y)=y;

h(x)=E[y|x; theta] = P(y=1|x;theta)=1/(1+e^-alta)=1/(1+e^(theta^T x))

正则相应函数 g(alta)=E(y;alta)=1/(1+e^-alta)
正则关联函数 g^-1

[高斯的 过程省略了]

多项式分布(属于广义线性模型模型)，输出的y的取值范围是 {1,.....,k} 

图形举例也就是 把原数据分为多个类

参数 phi_1,phi_2,....phi_k

p(y=i)=phi_i

phi_k = 1- (phi_1+phi_2+...phi_{k-1})因为有冗余，所以 真实只定义k-1个

T(y) = 列向量，其中第y个为1，其它为0

指示函数 1{True}=1, 1{Flase}=0

so , T(y)_i = 1{y=i}

```
P(y)
= MULTI { phi_i ^T(y)_i }
=b(y)e^(alta^T  T(y) - a(alta))

alta = [log(phi_1/phi_k), ... , log(phi_{k-1}/phi_k) ]'
a(alta) = -log(phi_k) b(y)=1

phi_i = e^{n_i} / (1+ sum(e^n_j)) ,,j=1~n-1
```

h(x) = E [T(Y)| X ; theta] = E[] 展开Y 为 1{y=i}的列向量 = [phi_1,phi_2,,,,phi_3]' 每一个phi_i 在用上面的式子表示

softmax 回归 (被认为是logistics 回归的推广，因为 logistics回归是分为两类，而这个是分为k类)

依然是训练样本 去做极大似然

L(theta) = MULTI( p(y_i | x_i; theta) )= ...

所以这一课讲得是 几分钟的牛顿法在寻找theta的应用 + 广义线性回归(验证 前面讲过的方法，再用softmax进行举例 推演)，整体这几节课的思路 大框是 (模型假设+分布假设+极大似然)，然后他的假设选取不是随便假设的，按照大数的常见的结论进行的。

- [x] 5 买了个ipad pro，终于可以相对舒服的记录公式了，然后这里就记一记录进度。
- [x] 6
- [x] 7
对偶问题 https://wenku.baidu.com/view/a3276747168884868762d678.html
KKT http://blog.csdn.net/johnnyconstantine/article/details/46335763
- [x] 8
- [x] 9
- [x] 10
- [x] 11
- [x] 12
- [x] 13
- [x] 14
- [x] 15
- [x] 16
- [x] 17
- [x] 18
- [x] 19
- [x] 20

- [ ] 感受随记 & TODO

不能舒服的打希腊字母 好难受,将就看

证明 2讲 中  直接给出的公式


哇 这个课要是我大一大二 都绝对看不懂，需要一堆 高数/线代 基础，至少到第三章就是这样了

也见识了一些常见的假设方法，以前以为只在题目里出现，结果他每次要证一堆东西，就会先假设再假设，假设中有一些常见的。

老师建议把 证明过程盖住 自己再证明一遍

从某些程度上想感谢字幕，虽然说很明显有些字母没听懂，但是老师说得很明白了，也写在黑板上了，但字母硬是写的 `没听清`，而且标题 真的是这样取标题的吗？？？这样看来 字幕 本身并不想学，真的是在做贡献 thx

之后整理一下整个的内容更细节的目录，现在的做字母和标题和视频内容细节还是有差别。 之后再把这个转换成git，

从 第9-11讲开始比较轻松，这个和李航 的《统计学习方法》的 顺序不太一样，把范的放在后面

容我说一句 里面的学生有的时候提的问题真的是，比如X讲有学生提的问题A，而我觉得，这个问题A本身提出来是OK的，但是明明在X-3讲的时候就应该弄懂了，不然我都不知道 这三讲他是怎么学的，看来外国也有学生上课划水啊，不过别人划水以后还敢提问，还没同学怼他也是服气。

之后看要不要把前面几讲重看一遍整理到ipad上 XD，现在每一讲结束画一下脑图XD，同时把前面的几讲也进行整理

TODO然后目前的额外想法是 写个相对线性的总结，目前想到的结构是，大分类，具体算法名称，能够解决的理论问题，算法的思想理解，【具体公式？】，实例举例

目前课程视屏学完了，现在接下来做的是

- [ ] 整理之前的手稿，建立4-11讲的脑图，建立知识点依赖文档
- [ ] 做一个框架类整理文章 放在github & 公众号上
