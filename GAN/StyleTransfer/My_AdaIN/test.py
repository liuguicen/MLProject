import argparse
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
import AdaConfig

import ImageUtil
import MlUtil
from CommonModels.CommonModels import MyVgg
from MlUtil import Timer, Const
from mobile.model_export import exportModule
# from model import Model
from mobileBaseModel import MobileBasedModel
import onnx

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

savePath = r'D:\AndroidProject\ML_Android\StyleTransferDemo\app\src\main\assets'


def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


def denorm(tensor):
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def t_estResIm(out):
    out = out.to('cpu', torch.uint8).numpy()
    im = Image.fromarray(out)
    im.save('test.jpg')
    ImageUtil.imShow(im)


def save_img(args, c, out, s):
    if out == None: return
    if args.output_name is None:
        c_name = os.path.splitext(os.path.basename(args.content))[0]
        s_name = os.path.splitext(os.path.basename(args.style))[0]
        args.output_name = f'{c_name}_{s_name}'
    torchvision.utils.save_image(out, f'{args.output_name}.jpg', nrow=1)
    o = Image.open(f'{args.output_name}.jpg')
    demo = Image.new('RGB', (c.width * 2, c.height))
    o = o.resize(c.size)
    s = s.resize((i // 4 for i in c.size))
    demo.paste(c, (0, 0))
    demo.paste(o, (c.width, 0))
    demo.paste(s, (c.width, c.height - s.height))
    demo.save(f'{args.output_name}_style_transfer_demo.jpg', quality=95)
    o.paste(s, (0, o.height - s.height))
    o.save(f'{args.output_name}_with_style_image.jpg', quality=95)
    print(f'result saved into files starting with {args.output_name}')


def convertForAndroid(out):
    '''
    转变成Android中易于处理的格式 注意其中通道维度在最后，因为这个在pytorch里面可以加速 变成图片时改过来
    '''
    out = out.squeeze_().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).int()
    return out
    # testResIm()

    # out.type(torch.uint8) jit不支持这个函数
    #
    # out.type(bytes)
    # out = out.int()
    # out = out.byte() # 转换成byte，Android不支持

    # w, h = out.size()[0], out.size()[1]
    # for i in range(w):
    #     for j in range(h):  # 透明通道不能赋值，赋值python直接溢出报错，有毒阿，这你管个啥
    #         out[i][j][0] = out[i][j][0].item() << 16 | out[i][j][1].item() << 8 | out[i][j][2].item()
    # result = OrderedDict()
    # result['wh'] = torch.tensor([out.size()[1], out.size()[0]], dtype=torch.int32)  # 这里是先高后宽
    # result['im'] = out
    # return result


if AdaConfig.isExportModel:
    device = 'cpu'
else:
    device = MlUtil.gpu


class AdainDecoder(nn.Module):
    def __init__(self, decoder):
        nn.Module.__init__(self)
        self.decoder = decoder

    def forward(self, content_features, style_features, alpha):
        # adain
        t = adain(content_features, style_features)

        t = alpha * t + (1 - alpha) * content_features
        # Timer.print_and_record('adain耗时 = ')

        # 解码
        out = self.decoder(t)
        # Timer.print_and_record('decoder耗时= ')

        out = denorm(out)
        #
        out = convertForAndroid(out)
        # Timer.print_and_record('格式转换耗时 = ')
        # Timer.print_and_record('总耗时 = ', recordKey=Const.total)
        return out


def encoder2Onnx(torch_model, path):
    torch_model.to("cpu")
    torch_model.eval()
    input = torch.rand(1, 3, 1024, 1024)
    output = torch.rand(1, 512, 128, 128)
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {
        "input": {2: "width", 3: "height"}
    }
    torch.onnx.export(torch_model,  # model being run
                      args=input,  # model input (or a tuple for multiple inputs)
                      f=path,
                      verbose=True,
                      example_outputs=output,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=11
                      )

    # 下面验证导出结果是否正确
    # Load the ONNX model
    model = onnx.load(path)
    # Check that the IR is well formed
    onnx.checker.check_model(model)
    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)
    print('导出oxx模型并验证完成， path = \n', path)


def decoder2Onnx(torch_model, path):
    torch_model.to("cpu")
    torch_model.eval()
    input = (torch.rand(1, 512, 128, 128), torch.rand(1, 512, 128, 128), torch.tensor(1, dtype=float))
    # Define attributes for ONNX export
    input_names = ["content", "style", "alpha"]
    output_names = ["output"]
    dynamic_axes = {
        "content": {2: "width", 3: "height"},
        "style": {2: "width", 3: "height"},
        "output": {2: "width", 3: "height"}
    }
    torch.onnx.export(torch_model,  # model being run
                      args=input,  # model input (or a tuple for multiple inputs)
                      f=path,
                      verbose=True,
                      example_outputs=torch.rand(1, 1000, 1000, 3),
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes,
                      opset_version=11
                      )

    # 下面验证导出结果是否正确
    # Load the ONNX model
    model = onnx.load(path)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)
    print('导出oxx模型并验证完成， path = \n', path)

def onnx_to_tf(onnx_path, tf_path):
    """
    Converts ONNX model to TF 2.X saved file
    :param onnx_path: ONNX model path to load
    :param tf_path: TF path to save
    """
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)  # Checks signature
    tf_rep = prepare(onnx_model)  # Prepare TF representation
    tf_rep.export_graph(tf_path)  # Export the model



from model import Model

from CommonModels.CommonModels import MyVgg_Inference


def main():
    # set model
    # model = MobileBasedModel()
    encoder = MyVgg_Inference()
    encoder.to(device)

    model = Model()
    if AdaConfig.model_state_path is not None:
        model.load_state_dict(torch.load(AdaConfig.oringalModelState, map_location=lambda storage, loc: storage))
    model = model.to(device)
    adainDecoder = AdainDecoder(model.decoder)

    if AdaConfig.isExportModel:
        encoder2Onnx(encoder, 'adain_encoder.onnx')
        decoder2Onnx(adainDecoder, 'adain_decoder.onnx')


    c = Image.open(AdaConfig.content)
    s = Image.open(AdaConfig.style)
    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)

    with torch.no_grad():
        # 编码
        out = None
        Timer.record()
        Timer.record(Const.total)

        content_features = encoder(c_tensor)
        style_features = encoder(s_tensor)

        Timer.print_and_record('编码')

        out = adainDecoder(content_features, style_features, torch.tensor(1, dtype=float))
        MlUtil.showModelOutImage(out)


if __name__ == '__main__':
    main()
