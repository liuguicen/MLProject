\torch.range(start=1, end=6)的结果是会包含end的
而torch.arange(start=1, end=6)的结果并不包含end
两者创建的tensor的类型也不一样,前者float32，后者int64

tensor.view()
1、共享内存，改变tensor展现出来的形状，但是不能改变元素数量，这就是取名为view的意思，改变形状外观,但不改变元素数量，要注意因为共享内存，改其中一个的数据值，另外一个的数据值也会跟着改变。

注意按照前面说的，view不是改变自己外观，而是返回改变外观的tensor

如下例所示
tensor.view(可变长参数表示各个维度重构后的大小)
维度大小为-1 表示该维度自动计算
tensor.view(-1) 比如，变成一维, 长度自动计算
tensor.view(2, 3, -1, 4, 5 ) 第三个维度自动计算

tensor.resize()
相对于view，改变形状，并且还可以改变总元素数量，增加减少都行

permute()
permute译为重新排列，高维转置操作，从代数上理解就是加入就是通过permute(..j...)，将第j个维度放到第i个维度上，
那么下标访问的时候，新的tentor访问第i个维度xx下标，就相当于访问原来第j个维度的xx下标，
举例子a.permute(1,2,0)，那么新的a[3,4,5]=原来的a[4,5,3], 即第0维度与第1个维度交换，现在第0个维度下标3，就相当于访问原来的第1维下标3
注意这个 从几何上不好理解的
区别
和transpose的区别， transpose只能用于2维，意思转置，但两者本质上有共同性
和View区别，permute和view都有改变尺寸的作用，但view是只改变尺寸，不考虑下标的重新排列关系，它是把整个数据拉平，然后重新改变尺寸，它能变成各种尺寸，但permute不行，只能重新排列尺寸
应用上对于一个图像batch，其形状为[batch, channel, height, width]，我们可以使用tensor.permute(0,3,2,1)得到形状为[batch, width, height, channel]的tensor.

torch.repeat(维度，可变长参数)
这个函数没理解到，理解起来很复杂，记下例子

重复通道
torch.tensor(
[[1,1],
 [2,2]]).repeat(1,1,3)).permute(1, 2, 0))
tensor([[[1, 1],
         [2, 2]],

        [[1, 1],
         [2, 2]],

        [[1, 1],
         [2, 2]]])
         
下标操作相关

torch的tensor切片的时候还能增加维度，通过切片操作在增加的维度上面传入None的方式
二维的a这样写变成3维
a[:, :, None]
还可以在下标中括号运算里面传入列表[a,b,c]，表示取第0维的a,b,c号元素，组成列表，还可以是多维的！秀！


torch.meshgrid(x, y)
平面上画网格的函数，就是传上x轴上的点，y轴上的点，产生len(x) * len(y)个点
返回值是两个二维矩阵的形式，分别表示每个点的x和y坐标
这个函数还真没啥意思吧，自己写就行了，不然还得去理解api

save和load
save的可以是任意对象，它只是特别的对张量做了处理，其它和python的一张
load的时候可以指定map_location参数
        默认情况下，参数先读取为cpu形式，再转移到保存时的形式，比如gpu，
         map_location参数指定要转移到哪种形式，它可以是device 的形式，还可以是函数的形式，怎么用未知
         
## squeeze() 
torch.squeeze(input, dim=None, out=None) → Tensor
官方文档：https://pytorch.org/docs/master/generated/torch.squeeze.html
参数说明：
input (Tensor)：输入的张量
dim (int, optional) ：可选参数，如果不指定，该方法会把所有值为 1 的维度移除，如果指定，该方法则指移除指定的那个维度
out (Tensor, optional) ：可选，指定输出的张量.

## torch.cat(list, dim)
cat是concatnate的意思：拼接 
在给定维度上对输入的张量序列seq 进行连接操作
对于不同的参数形式，要风情拼接的是哪个，通过边长参数输入，拼接参数，通过list输入，拼接list内部的数据，不是list
参数
inputs : 待连接的张量序列，可以是任意相同Tensor类型的python 序列
dim : 选择的扩维, 必须在0到len(inputs[0])之间，沿着此维连接张量序列。
2. 重点
输入数据必须是序列，序列中数据是任意相同的shape的同类型tensor

## module及其子类以及模型 层之间的关系 网址
https://zhuanlan.zhihu.com/p/282863934   
sequential 放入的大概也分为两类，就是列表和字典map  
https://zhuanlan.zhihu.com/p/340453841
nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
放入字典

# 分布式并行：
使用上只需要调用model = nn.DataParallel(model),然后就像一个单卡的模型一样使用即可，绝大部分情况下对用户来说并行化是透明的，逻辑交给框架内部来完成，还是很厉害的   
要注意的几个点是使用分布式的时候学习率要调整（损失函数等等要根据情况，比如批数量增大等，看是否调整）

基本原理，首先要明白非并行情况下前向传播和反向传播的流程，画个计算图可以知道，批量梯度下降前向传播中每个样本都会用到参数w...，所以这个参数有多条(m+条)路径链接到最后的输出中，反向传播的时候需要累加偏导数

基于这个，分布式就是类似的，把若干个样本对应的路径合成一组，然后累加梯度，最后将所有的组的梯度再累加，就和原来一样了，简单的想就是计算图上把若干条路径归到一组

具体实现流程就是，用一个管理者GPU，把当前参数模型传递给不同的GPU，然后将样本分成多个组，交给不同的GPU，每个GPU进行前向传播，然后反向传播求出偏导，最后管理GPU整合所以GPU的偏导数
注意这里使用了装饰器模式，也是这个模式的优点，透明的，用户可以像使用原来的接口一样使用

然后重写了forward函数，在里面进行样本分配和参数整合
 scatter 函数，负责将 tensor 分成大概相等的块并将他们分给不同的 GPU。对其他的数据类型，则是复制给不同的 GPU 。
 
 # 模型导出
 详见模型导出文件
 
 nn.xxx 和 nn.function.xx的区别
 functional中的是函数化的，不保存参数，需要手动传入。
 
# 不同维度、尺寸张量相乘：
 broadcast
点积是broadcast的。broadcast是torch的一个概念，简单理解就是在一定的规则下允许高维Tensor和低维Tensor之间的运算。broadcast的概念稍显复杂，在此不做展开，可以参考官方文档关于broadcast的介绍.  
这里举一个点积broadcast的例子。在例子中，a是二维Tensor，b是三维Tensor， 则c[i,*,*] = a * b[i, *, *] 可以简单理解为a维度不够，那么直接在不够的维度上a乘以b的所有，对于a中尺寸不够，但是为1的，直接去掉这个维度，再按照维度不够相乘

size 计算元素数量
x.numel()

# conv2d参数 卷积参数 的解释
group 分组卷积，详解自己笔记或者相关文档
dilation 空洞卷积或者叫扩张卷积 比如设置为2，就是让卷积核去卷2倍长宽的区域，那么现在卷积核的一个点从四个点中选一个来卷

# relu相关
relu6 mobilenet发明
leackrelu 参数为负时不为0，低斜率下降
参数，inplace 表示tensor原地计算，不产生新的tensor 节省内存

# 创建，类型处理 
2.类型转换
方法一：简单后缀转换

tensor.int()
tensor.float()
tensor.double()
方法二：使用torch.type()函数

tensor.type(torch.FloatTensor)
方法三：使用type_as（tensor）将tensor转换为指定tensor的类型