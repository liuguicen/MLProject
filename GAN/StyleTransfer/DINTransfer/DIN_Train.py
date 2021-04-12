import os

import torch.nn.functional as F
import torchvision

import Din_Config
import MLUtil
import fileUtil


def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def calc_content_loss(out_features, t):
    return F.mse_loss(out_features, t)


def calc_style_loss(content_middle_features, style_middle_features):
    loss = 0
    for c, s in zip(content_middle_features, style_middle_features):
        c_mean, c_std = calc_mean_std(c)
        s_mean, s_std = calc_mean_std(s)
        loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
    return loss


'''
loss 计算的几个函数都来自于adain
'''


def compute_loss(vgg, out, content, style):
    styleFeature_multiLayer = vgg(style, True)

    contentFeature_h4 = vgg(content)

    outFeature_multiLayer = vgg(out, True)
    outFeature_h4 = outFeature_multiLayer[3]

    loss_c = calc_content_loss(outFeature_h4, contentFeature_h4)

    loss_s = calc_style_loss(outFeature_multiLayer, styleFeature_multiLayer)
    loss = loss_c + Din_Config.lam * loss_s
    return loss


import torch
from DIN_Data import PreprocessDataset
from torch.utils.data import DataLoader
from Din_Model import DINModel
from tqdm import tqdm


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


import ml_base.fileUtil


def train():
    print('train start！\n')
    print(f'# Minibatch-size: {Din_Config.batch_size}')
    print(f'# epoch: {Din_Config.epoch}')
    print('')

    # 读取上次的运行配置
    record = fileUtil.readRunLog(Din_Config.runRecordPath)
    Din_Config.runRecord = record if record is not None else Din_Config.runRecord

    device = 'cuda:0'
    # 数据集和数据加载器
    train_dataset = PreprocessDataset(Din_Config.train_content_dir, Din_Config.train_style_dir)
    test_dataset = PreprocessDataset(Din_Config.test_content_dir, Din_Config.test_style_dir)
    iters = len(train_dataset)
    print(f'Length of train image pairs: {iters}')

    train_loader = DataLoader(train_dataset, batch_size=Din_Config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=Din_Config.batch_size, shuffle=False)
    test_iter = iter(test_loader)

    test_content, test_style = next(test_iter)
    # 模型和优化器
    dinModel = DINModel().to(device)  # type:DINModel

    optEncoder = torch.optim.Adam(dinModel.encoder.parameters(), lr=Din_Config.learning_rate_enco_deco)
    optDecoder = torch.optim.Adam(dinModel.decoder.parameters(), lr=Din_Config.learning_rate_enco_deco)

    optDin1 = torch.optim.Adam(dinModel.dinLayer1.parameters(), lr=Din_Config.learning_rate_din_layer)
    optDin2 = torch.optim.Adam(dinModel.dinLayer2.parameters(), lr=Din_Config.learning_rate_din_layer)

    loss_list = []
    batch_number = 0
    for e in range(1, Din_Config.epoch + 1):
        for i, (content, style) in tqdm(enumerate(train_loader, 1)):
            batch_number += 1
            optEncoder.zero_grad()
            optDecoder.zero_grad()
            optDin1.zero_grad()

            content = content.to(device)
            style = style.to(device)
            out = dinModel(content, style)
            loss = compute_loss(dinModel.vgg, out, content, style)
            loss_list.append(loss.item())
            loss.backward()

            optEncoder.step()
            optDecoder.step()
            optDin1.step()
            optDin2.step()
            if batch_number % Din_Config.checkpoint_interval == 0:
                content = test_content.to(device)
                style = test_style.to(device)
                with torch.no_grad():
                    if Din_Config.debugMode:
                        for model in dinModel.getMySubModel():
                            MLUtil.registerMiddleFeaturePrinter(model)
                    out = dinModel(content, style)

                content = denorm(content, device)
                style = denorm(style, device)
                res = torch.cat([content, style, out], dim=0)
                res = res.to('cpu')
                if not os.path.exists(Din_Config.test_res_dir):
                    fileUtil.mkdir(Din_Config.test_res_dir)
                if not os.path.exists(Din_Config.check_point):
                    fileUtil.mkdir(Din_Config.check_point)

                torchvision.utils.save_image(res, f'{Din_Config.test_res_dir}/{e}_epoch_{i}_iteration.png',
                                             nrow=Din_Config.batch_size)
                torch.save(dinModel.state_dict(), f'{Din_Config.check_point}/{e}_epoch.pth')


if __name__ == '__main__':
    train()
