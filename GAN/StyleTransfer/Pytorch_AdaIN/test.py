import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

import AdaConfig
import FileUtil
from model import Model

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def main():
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
    parser.add_argument('--model_state_path', type=str,
                        default=r'E:\重要_dataset_model\预训练模型\pure_pyroch_adain_style.pth',
                        help='save directory for result and loss')

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and AdaConfig.gpu >= 0:
        device = torch.device(f'cuda:{AdaConfig.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model = Model()
    if AdaConfig.model_state_path is not None:
        model.load_state_dict(torch.load(AdaConfig.model_state_path, map_location=lambda storage, loc: storage))
    model = model.to(device)

    c = Image.open(AdaConfig.content)
    s = Image.open(AdaConfig.style)
    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(c_tensor, s_tensor, AdaConfig.alpha)

    out = denorm(out, device)

    if AdaConfig.output_name is None:
        c_name = os.path.splitext(os.path.basename(AdaConfig.content))[0]
        s_name = os.path.splitext(os.path.basename(AdaConfig.style))[0]
        AdaConfig.output_name = f'{c_name}_{s_name}'

    save_image(out, f'{AdaConfig.output_name}.jpg', nrow=1)
    o = Image.open(f'{AdaConfig.output_name}.jpg')

    demo = Image.new('RGB', (c.width * 2, c.height))
    o = o.resize(c.size)
    s = s.resize((i // 4 for i in c.size))

    demo.paste(c, (0, 0))
    demo.paste(o, (c.width, 0))
    demo.paste(s, (c.width, c.height - s.height))
    demo.save(f'{AdaConfig.output_name}_style_transfer_demo.jpg', quality=95)

    o.paste(s,  (0, o.height - s.height))
    o.save(f'{AdaConfig.output_name}_with_style_image.jpg', quality=95)

    print(f'result saved into files starting with {AdaConfig.output_name}')

    # test_multi_size_0(args, device, model)


def t1est_multi_size_0(args, device, model):
    c = Image.open(AdaConfig.content)
    for root, dir, file in os.walk(r'D:\MLProject\GAN\StyleTransfer\Pytorch_AdaIN\style'):
        for f in file:
            for ratio in [0.125, 0.25, 0.5, 1, 2]:
                stylePath = os.path.join(root, f)
                s = Image.open(stylePath)  # type:Image
                nc = c.resize((int(c.size[0] * ratio), int(c.size[1] * ratio)))
                AdaConfig.style = stylePath
                t1est_multi_size(args, nc, device, model, s, ratio)


def t1est_multi_size(args, c, device, model, s, ratio):
    c_tensor = trans(c).unsqueeze(0).to(device)
    s_tensor = trans(s).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(c_tensor, s_tensor, AdaConfig.alpha)
    out = denorm(out, device)
    c_name = os.path.splitext(os.path.basename(AdaConfig.content))[0]
    s_name = os.path.splitext(os.path.basename(AdaConfig.style))[0]
    AdaConfig.output_name = f'{c_name}_{s_name}'
    save_image(out, f'{AdaConfig.output_name}_{c.size}_{s.size}.jpg', nrow=1)
    # o = Image.open(f'{AdaConfig.output_name}.jpg')
    # demo = Image.new('RGB', (c.width * 2, c.height))
    # o = o.resize(c.size)
    # s = s.resize((i // 4 for i in c.size))
    # demo.paste(c, (0, 0))
    # demo.paste(o, (c.width, 0))
    # demo.paste(s, (c.width, c.height - s.height))
    # demo.save(f'{AdaConfig.output_name}_style_transfer_demo.jpg', quality=95)
    # o.paste(s, (0, o.height - s.height))
    # o.save(f'{AdaConfig.output_name}_with_style_image.jpg', quality=95)
    # print(f'result saved into files starting with {AdaConfig.output_name}')


if __name__ == '__main__':
    main()
