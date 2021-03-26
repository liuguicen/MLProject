import argparse
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

import imgBase
from MUtil import Timer, Const
from model import Model

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])

savePath = r'C:\Users\liugu\Downloads\android-demo-app-master\android-demo-app-master\HelloWorldApp\app\src\main\assets'


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
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def testResIm(out):
    out = out.to('cpu', torch.uint8).numpy()
    im = Image.fromarray(out)
    im.save('test.jpg')
    imgBase.imShow(im)


def convertForAndroid(out):  # 转变成Android Studio易于处理的格式 注意通道维度在最后 这个可以加速 变成图片时改过来
    out = out.squeeze_().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).int()

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
    result = OrderedDict()
    result['wh'] = torch.tensor([out.size()[1], out.size()[0]], dtype=torch.int32)  # 这里是先高后宽
    result['im'] = out
    return result


class AdainDecoder(nn.Module):
    def __init__(self, decoder):
        nn.Module.__init__(self)
        self.decoder = decoder

    def forward(self, content_features, style_features, alpha: float):
        # adain
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features
        # Timer.print_and_record('adain耗时 = ')

        # 解码
        out = self.decoder(t)
        # Timer.print_and_record('decoder耗时= ')

        out = denorm(out)
        out = convertForAndroid(out)
        # Timer.print_and_record('格式转换耗时 = ')
        # Timer.print_and_record('总耗时 = ', recordKey=Const.total)
        return out


class MyVgg(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        vgg = torchvision.models.vgg19(pretrained=True).features[:21]
        vgg.to('cpu')
        vgg.cpu()
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input):
        h1 = self.slice1(input)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h4


from torch.utils.mobile_optimizer import optimize_for_mobile


def exportModule(module, path):
    scripted_module = torch.jit.script(module)
    if os.path.exists(path):
        os.remove(path)
    # pytorch 提供的移动优化 实测似乎没什么用
    torchscript_model_optimized = optimize_for_mobile(scripted_module)
    torch.jit.save(torchscript_model_optimized, path)
    print('导出', path, '成功')


def exportOnnx(torch_model, input, path, dynamic_axes=None):
    torch_model.eval()
    torch.onnx.export(torch_model,  # model being run
                      input,  # model input (or a tuple for multiple inputs)
                      path, verbose=True,
                      dynamic_axes=dynamic_axes
                      )
    import onnx

    # Load the ONNX model
    model = onnx.load(path)

    # Check that the IR is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    onnx.helper.printable_graph(model.graph)
    print('导出oxx模型并验证完成， path = \n', path)


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


def parseArgs():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--content', '-c', type=str, default=None,
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', type=str, default=None,
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--alpha', '-a', type=float, default=1,
                        help='alpha control the fusion degree in Adain')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                        help='save directory for result and loss')
    args = parser.parse_args()
    return args


def main():
    args = parseArgs()
    alpha = args.alpha

    device = 'cpu'

    # set model
    model = Model()
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
    model = model.to(device)

    myVgg = MyVgg()
    adainDecoder = AdainDecoder(model.decoder)

    c = Image.open(args.content)
    s = Image.open(args.style)
    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)

    with torch.no_grad():
        # 编码
        out = None
        Timer.record()
        Timer.record(Const.total)

        content_features = myVgg(c_tensor)
        style_features = myVgg(s_tensor)

        Timer.print_and_record('编码')

        out = adainDecoder(content_features, style_features, alpha)

        exportModule(myVgg, os.path.join(savePath, 'vgg_encoder.pt'))
        # exportOnnx(myVgg, c_tensor, os.path.join(savePath, 'vgg_encoder.onnx'),
        #            dynamic_axes={'input_1': {1: 'width',
        #                                      2: 'height'},
        #
        #                          'input_2': {1: 'width',
        #                                      2: 'height'},
        #
        #                          'output': {1: 'width',
        #                                     2: 'height'}
        #                          })

        exportModule(adainDecoder, os.path.join(savePath, 'adain_decoder.pt'))
        # content_features = model.vgg_encoder(c_tensor, output_last_feature=True)
        # style_features = model.vgg_encoder(s_tensor, output_last_feature=True)
        # # adain
        # t = adain(content_features, style_features)
        # t = alpha * t + (1 - alpha) * content_features
        #
        # # 解码
        # out = model.decoder(t)
        # out = denorm(out, 'cpu')
        # save_img(args, c, out, s)


if __name__ == '__main__':
    main()
