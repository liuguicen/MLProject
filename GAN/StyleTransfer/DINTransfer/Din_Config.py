import torch
import torch.nn as nn

# 注 文中存疑的地方，全都写的这个配置文件里面了，通过该配置确认这些地方
############################                         存疑的点                    ########################################

# 是否添加规范层， 这个存疑，有的用的，有的没用如adain
# normal = nn.BatchNorm2d
normal = nn.InstanceNorm2d

# 文中提到，默认din 过滤器size设置为1，为了减少计算消耗，
# 但是附录里面卷积核大小是3

# 然后它的附录研究了动态卷积不同尺寸的效果，看上去设置为1效果好
# 有疑问的是这个尺寸到底是不是生成weight和bias的卷积的卷积核
dinLayer_filterSize = 1

# 第二个深度可分离卷积的逐点卷积层的 kernel_size，正常来说应该是1，但附录材料中给出的是3
pw2_kernal_size = 3

# 关于DIN层的VGG参数，生成风格特征VGG的层数
# 论文中没有说明，采用和Adain一样的层数，有的模型不是这样的，有的是用的vgg16 有的用的vgg19
todo1 = 19

# DIN层的结构
# ![](.DIN-MobileNet替代vgg的-没代码-笔记_images/3d78676c.png)
# 动态卷积层那个圆圈叉叉符号 应该就是 就是 乘上 然后 加
# 从几个地方可以佐证：
# 6-1、大佬文章
# https://blog.csdn.net/kevinoop/article/details/115099530?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522161773091516780274130898%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=161773091516780274130898&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-2-115099530.pc_search_result_before_js&utm_term=Dynamic+Instance+Normalization+for+Arbitrary+Style+Transfer
# ![](.DIN-MobileNet替代vgg的-没代码-笔记_images/6f18b859.png)
# ![](.DIN-MobileNet替代vgg的-没代码-笔记_images/702e431d.png)
# 6-2、这个很像adain 学到的就像是用来反规范化的参数，加和乘的
# 6-3、结构图中给出的符号应该就是加乘的意思，如果是其它的，不应该是这个符号
# 6-4、这个技术的原文Dynamic Filter Networks的样例代码应该就是直接加的
# ![](.DIN-MobileNet替代vgg的-没代码-笔记_images/10e4db58.png)
# 6-5、1*1 的深度卷积，加上偏置，= x * w + b，相乘相加效果一样，所以把这个层说成卷积xx也是可以的
todo2 = 1

# batch_size 没有说明，参照adain
# batch_size = 16
batch_size = 2

# 激活层 没有说明 网上说用relu有问题，用leakyrelu才行，mobilenet用的relu6
active_layer = nn.LeakyReLU(inplace=True)
# active_layer = nn.ReLU6(inplace=True)
# weight和bias用那种池化，文中没有说明, 网上说最大池化保留纹理信息，应该是这个
# 因为输入的大小不固定，输出是固定的，所以采用自适应池化AdaptiveXxxPool，pytorch包装了，自己换算也可以的
weight_bias_pool_layer = torch.nn.functional.adaptive_max_pool2d


# 根据卷积的输出size，反向计算输入size, 文中没有说明几倍，这里让其经过layer3之后是outputSize的两倍，先乘以2，再算出卷积前的大小
# 卷积尺寸公式：size_out = (size_in + 2 * pad - k) / s + 1
def get_adapool1_output_size(output_size):
    return (output_size * 2 - 1) * 2 + 1


# 将输出变到-1-1，网上说的用这个好， 但是

# metaStyleTransfer-master里面没有这一层 然后反normal之后传入vgg计算loss
# adain没有这层，直接将输出放到VGG里面，但是这个应该做的呀
# LinearStyleTransfer 同adain
useTanh = True

############################                         存疑的点                    ########################################


lam = 10

learning_rate_enco_deco = 0.0001
learning_rate_din_layer = 0.001

epoch = 20

train_content_dir = 'content'
train_style_dir = 'style'
test_content_dir = 'content'
test_style_dir = 'style'

checkpoint_interval = 10


test_res_dir = 'test_res'
check_point = 'check_point'