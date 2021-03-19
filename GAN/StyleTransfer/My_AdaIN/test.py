import argparse
import os
import time

import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from model import Model
from model import adain


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
    return parser.parse_args()


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

normal = transforms.Compose([transforms.ToTensor(),
                             normalize])
savePath = r'C:\Users\liugu\Downloads\android-demo-app-master\android-demo-app-master\HelloWorldApp\app\src\main\assets'


def denorm(tensorim):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).cpu()
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).cpu()
    res = torch.clamp(tensorim * std + mean, 0, 1)
    return res


def useNewOrder(model, c_tensor, s_tensor, alpha):
    with torch.no_grad():
        content_features = model.vgg_encoder(c_tensor)
        style_features = model.vgg_encoder(s_tensor)
        # 99%的时间是花在上面这两步了，这里的这些东西都可以预处理，从而提高速度
        # 把feature直接获取出来
        start_time = time.time()
        print('vgg time', time.time() - start_time)
        start_time = time.time()
        t = adain(content_features, style_features)  # 执行adain
        t = alpha * t + (1 - alpha) * content_features  # 混合
        print('adain time', time.time() - start_time)
        start_time = time.time()
        out = model.decoder(t)
        print('generate time', time.time() - start_time)
        return out


def save_res(args, out, c, s):
    if args.output_name is None:
        c_name = os.path.splitext(os.path.basename(args.content))[0]
        s_name = os.path.splitext(os.path.basename(args.style))[0]
        args.output_name = f'{c_name}_{s_name}'

    save_image(out, f'{args.output_name}.jpg', nrow=1)
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


def exportModule(module, path):
    scripted_module = torch.jit.script(module)
    if os.path.exists(path):
        os.remove(path)

    torch.jit.save(scripted_module, path)


def exportFunction(function, *param, name=''):
    sm = torch.jit.trace(function, *param)
    torch.jit.save(sm, name + '.pt')


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

    def forward(self, input):
        h1 = self.slice1(input)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h4


def main():
    args = parseArgs()

    # set device on GPU if available, else CPU
    device = 'cpu'

    myVgg = MyVgg()
    myVgg.eval()
    exportModule(myVgg, os.path.join(savePath, 'vgg.pt'))

    # set model
    model = Model(myVgg)
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
    model = model.to(device)

    c = Image.open(args.content)
    s = Image.open(args.style)
    c_tensor = normal(c).unsqueeze(0).to(device)
    s_tensor = normal(s).unsqueeze(0).to(device)

    out = useNewOrder(model, c_tensor, s_tensor, args.alpha)

    model.vgg_encoder.to('cpu')
    exportModule(model.vgg_encoder, 'vgg_encoder')

    exportFunction(adain, (c_tensor, s_tensor), name='adain')
    model.decoder.to('cpu')
    exportModule(model.decoder, 'decoder')

    save_res(args, out, c, s)


if __name__ == '__main__':
    main()
