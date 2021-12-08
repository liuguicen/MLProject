import warnings

import torchvision

import AdaConfig
import BaseRunRecord
import FileUtil

import os
# import matplotlib as mpl
# mpl.use('Agg')

import matplotlib.pyplot as plt
import MlUtil
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import PreprocessDataset, denorm
from model import Model
from mobileBaseModel import MobileBasedModel
from mobileBaseModel import RC

# set device on GPU if available, else CPU
if torch.cuda.is_available() and AdaConfig.gpu >= 0:
    device = torch.device(f'cuda:{AdaConfig.gpu}')
    print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
else:
    device = 'cpu'


def saveCheckPoint(model, test_content, test_style, loss_list, epoch, iter):
    content = test_content.to(device)
    style = test_style.to(device)
    with torch.no_grad():
        t, out = model.generate(content, style)
    content = denorm(content, device)
    style = denorm(style, device)
    out = denorm(out, device)
    res = torch.cat([content, style, out], dim=0)
    res = res.to('cpu')

    # 把要保存的变量型数据 赋值给记录的对象
    record = AdaConfig.record
    record.check_point_epoch = epoch
    record.check_point_iter = iter
    record.check_point_path = f'{record.model_state_dir}/{epoch}_epoch_{iter}_iteration.pth'

    # 保存数据
    torch.save(model.state_dict(), record.check_point_path)
    img_name = f'{epoch}_epoch_{iter}_iteration.png'
    torchvision.utils.save_image(res, f'{record.tes_res_dir}/{img_name}',
                                 nrow=AdaConfig.batch_size)
    img_name = f'{epoch}_epoch_{iter}_iteration_m.png'
    MlUtil.saveMiddleFeature(t, 10, img_name, f'{record.tes_res_dir}/{img_name}')
    # MLUtil.printAllMiddleFeature(model, content, style, type=torchvision.models.mobilenetv2.InvertedResidual)
    # MLUtil.printAllMiddleFeature(model, content, style, type=RC)
    # plt绘制并保存loss
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'{record.loss_dir}/train_loss.png')
    with open(f'{record.loss_dir}/loss_log.txt', 'w') as f:
        for l in loss_list:
            f.write(f'{l}\n')
    print(f'Loss saved in {record.loss_dir}')

    # pkl保存对象
    record.saveRunRecord('runRecord.pkl')


def main():
    # create directory to save
    if not os.path.exists(AdaConfig.check_point_dir):
        os.mkdir(AdaConfig.check_point_dir)

    print(f'# Minibatch-size: {AdaConfig.batch_size}')
    print(f'# epoch: {AdaConfig.epoch}')
    print('')

    # prepare dataset and dataLoader
    train_dataset = PreprocessDataset(AdaConfig.train_content_dir, AdaConfig.train_style_dir)
    test_dataset = PreprocessDataset(AdaConfig.test_content_dir, AdaConfig.test_style_dir)
    iters = len(train_dataset)
    print(f'Length of train image pairs: {iters}')

    train_loader = DataLoader(train_dataset, batch_size=AdaConfig.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=AdaConfig.batch_size, shuffle=False)
    test_iter = iter(test_loader)
    test_content, test_style = next(test_iter)

    # set model and optimizer
    if AdaConfig.use_mobile_based:
        model = MobileBasedModel().to(device)
    else:
        model = Model().to(device)
    record = AdaConfig.record
    if os.path.exists(record.check_point_path) and AdaConfig.use_check_point_state:
        model.load_state_dict(torch.load(record.check_point_path))
    optimizer = Adam(model.parameters(), lr=AdaConfig.learning_rate)

    # start training
    loss_list = []
    for e in range(record.check_point_epoch, AdaConfig.epoch + 1):
        print(f'Start {e} epoch')
        for i, (content, style) in tqdm(enumerate(train_loader, 0)):
            content = content.to(device)
            style = style.to(device)
            loss = model(content, style)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'[{e}/total {AdaConfig.epoch} epoch],[{i} /'
                  f'total {round(iters / AdaConfig.batch_size)} iteration]: {loss.item()}')

            if i % AdaConfig.check_point_interval == AdaConfig.check_point_interval - 1:
                saveCheckPoint(model, test_content, test_style, loss_list, e, i)


if __name__ == '__main__':
    main()
