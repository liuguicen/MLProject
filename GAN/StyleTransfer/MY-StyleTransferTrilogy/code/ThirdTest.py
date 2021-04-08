# %% md

# 导入必要的库

# %%

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

import torch.optim as optim

import random
import shutil
from glob import glob
from tqdm import tqdm

from models import *
from matplotlib import pyplot as plt
from Thirdutils import imshow, read_image, mean_std, tensor_normalizer


def rmrf(path):
    try:
        shutil.rmtree(path)
    except:
        pass


for f in glob('runs/*/.AppleDouble'):
    rmrf(f)

rmrf('runs/metanet')
rmrf('runs/transform_net')

# 搭建模型

vgg16 = models.vgg16(pretrained=True)
vgg16 = VGG(vgg16.features[:23]).to(device).eval()

base = 32
transform_net = TransformNet(base).to(device)
transform_net.get_param_dict()


class MetaNet(nn.Module):
    def __init__(self, param_dict):
        super(MetaNet, self).__init__()
        self.param_num = len(param_dict)
        self.hidden = nn.Linear(1920, 128 * self.param_num)
        self.fc_dict = {}
        for i, (name, params) in enumerate(param_dict.items()):
            self.fc_dict[name] = i
            setattr(self, 'fc{}'.format(i + 1), nn.Linear(128, params))

    # ONNX 要求输出 tensor 或者 list，不能是 dict
    def forward(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i + 1))
            filters[name] = fc(hidden[:, i * 128:(i + 1) * 128])
        return list(filters.values())

    def forward2(self, mean_std_features):
        hidden = F.relu(self.hidden(mean_std_features))
        filters = {}
        for name, i in self.fc_dict.items():
            fc = getattr(self, 'fc{}'.format(i + 1))
            filters[name] = fc(hidden[:, i * 128:(i + 1) * 128])
        return filters


metanet = MetaNet(transform_net.get_param_dict()).to(device)

# 输出计算图到 tensorboard

mean_std_features = torch.rand(4, 1920).to(device)

rands = torch.rand(4, 3, 256, 256).to(device)


# 测试速度
def testSpeed():
    metanet.load_state_dict(torch.load('models/metanet_base32_style50_tv1e-06_tagnohvd.pth'))
    transform_net.load_state_dict(torch.load('models/metanet_base32_style50_tv1e-06_tagnohvd_transform_net.pth'))
    X = torch.rand((1, 3, 256, 256)).to(device)

    import time

    for i in range(1000):
        features = vgg16(X)
        mean_std_features = mean_std(features)
        weights = metanet.forward2(mean_std_features)
        transform_net.set_weights(weights)
        print('任务1 第 ', i, '个 time =', time.time())
        del features, mean_std_features, weights

    for i in range(1000):
        transform_net(X)

    for i in range(1000):
        features = vgg16(X)
        mean_std_features = mean_std(features)
        weights = metanet.forward2(mean_std_features)
        transform_net.set_weights(weights)
        transform_net(X)
        del features, mean_std_features, weights


# 可视化


width = 256

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(width, scale=(256 / 480, 1), ratio=(1, 1)),
    transforms.ToTensor(),
    tensor_normalizer
])

content_dataset = torchvision.datasets.ImageFolder(r'C:\Users\liugu\Desktop\test', transform=data_transform)


# %%
def trainModel():
    style_dataset = torchvision.datasets.ImageFolder('/home/ypw/WikiArt/', transform=data_transform)

    # %%
    style_weight = 50
    content_weight = 1
    tv_weight = 1e-6
    batch_size = 8

    trainable_params = {}
    trainable_param_shapes = {}
    for model in [vgg16, transform_net, metanet]:
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param
                trainable_param_shapes[name] = param.shape

    optimizer = optim.Adam(trainable_params.values(), 1e-3)
    content_data_loader = torch.utils.data.DataLoader(content_dataset, batch_size=batch_size, shuffle=True)
    style_image = read_image('../images/test.jpg', target_width=256).to(device)
    style_features = vgg16(style_image)
    style_mean_std = mean_std(style_features)

    # metanet.load_state_dict(torch.load('models/metanet_base32_style50_tv1e-06_tagnohvd.pth'))
    # transform_net.load_state_dict(torch.load('models/metanet_base32_style50_tv1e-06_tagnohvd_transform_net.pth'))

    n_batch = 20
    with tqdm(enumerate(content_data_loader), total=n_batch) as pbar:
        for batch, (content_images, _) in pbar:
            x = content_images.cpu().numpy()
            if (x.min(-1).min(-1) == x.max(-1).max(-1)).any():
                continue

            optimizer.zero_grad()

            # 使用风格图像生成风格模型
            weights = metanet.forward2(mean_std(style_features))
            transform_net.set_weights(weights, 0)

            # 使用风格模型预测风格迁移图像
            content_images = content_images.to(device)
            transformed_images = transform_net(content_images)

            # 使用 vgg16 计算特征
            content_features = vgg16(content_images)
            transformed_features = vgg16(transformed_images)
            transformed_mean_std = mean_std(transformed_features)

            # content loss
            content_loss = content_weight * F.mse_loss(transformed_features[2], content_features[2])

            # style loss
            style_loss = style_weight * F.mse_loss(transformed_mean_std,
                                                   style_mean_std.expand_as(transformed_mean_std))

            # total variation loss
            y = transformed_images
            tv_loss = tv_weight * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                   torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            # 求和
            loss = content_loss + style_loss + tv_loss

            loss.backward()
            optimizer.step()

            if batch > n_batch:
                break

    content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)
    # while content_images.min() < -2:
    #     print('.', end=' ')
    #     content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)
    transformed_images = transform_net(content_images)

    transformed_images_vis = torch.cat([x for x in transformed_images], dim=-1)
    content_images_vis = torch.cat([x for x in content_images], dim=-1)

    plt.figure(figsize=(20, 12))
    plt.subplot(3, 1, 1)
    imshow(style_image)
    plt.subplot(3, 1, 2)
    imshow(content_images_vis)
    plt.subplot(3, 1, 3)
    imshow(transformed_images_vis)


# %%
import time


def test():
    style_image = read_image(r'D:\MLProject\GAN\StyleTransfer\s.jpg', target_width=256).to(device)
    start_time = time.time()
    features = vgg16(style_image)
    mean_std_features = mean_std(features)
    weights = metanet.forward2(mean_std_features)
    transform_net.set_weights(weights)

    content_images = torch.stack([random.choice(content_dataset)[0] for i in range(1)]).to(device)
    # while content_images.min() < -2:
    #     print('.', end=' ')
    #     content_images = torch.stack([random.choice(content_dataset)[0] for i in range(4)]).to(device)
    start_time = time.time()
    transformed_images = transform_net(content_images)
    print('耗时', time.time() - start_time)

    transformed_images_vis = torch.cat([x for x in transformed_images], dim=-1)
    content_images_vis = torch.cat([x for x in content_images], dim=-1)
    plt.figure(figsize=(20, 12))
    plt.subplot(3, 1, 1)
    imshow(style_image)
    plt.subplot(3, 1, 2)
    imshow(content_images_vis)
    plt.subplot(3, 1, 3)
    imshow(transformed_images_vis)


if __name__ == '__main__':
    test()