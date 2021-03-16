import cv2 as cv
import numpy as np

"""
hed边缘检测
特点：
1.内部会把图像缩放到500*500，似乎只有这样效果才最好，尺寸相同了，耗时都是约1s左右
2.因为利用了深度学习方法，hed检测的边缘有较好的语义含义，而传统的方法如canny等，它的无语义边缘比较多
3.实际观察到的，hed检测的边缘是粗而平滑的，就像粗笔手绘的，canny的反之，很细，然后不平滑锯齿抖动很多
4.论文中还提到传统方法断线，hed不会
canny 只需要几到几十毫秒 但是它效果不够好 
"""


# ! [CropLayenr]
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        # self.ystart = (inputShape[2] - targetShape[2]) / 2
        # self.xstart = (inputShape[3] - targetShape[3]) / 2

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)

        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]


# ! [CropLayer]
# ! [Register]
cv.dnn_registerLayer('Crop', CropLayer)  # 裁剪图片的？
# ! [Register]

# 模型配置和预训练权重
net = cv.dnn.readNet('D:\MLProject\ml_base\edge\deploy.prototxt',
                     'D:\MLProject\ml_base\edge\hed_pretrained_bsds.caffemodel')


def detectEdge(img):
    '''
    :return: 注意返回的是浮点格式的图像
    '''
    if isinstance(img, np.ndarray):
        if len(img.shape) == 3 and img.shape[2] > 3:  # 变成RGB格式
            img = img[:, :, 0:3]

    # size表示调整图像宽高到这个值，貌似500*500时效果才好, 训练就是这个大小？？
    inp = cv.dnn.blobFromImage(img, scalefactor=1.0, size=(500, 500),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)

    out = net.forward()
    out = out[0, 0]
    out = cv.resize(out, (img.shape[1], img.shape[0]))
    return out
