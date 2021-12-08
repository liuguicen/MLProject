import argparse

import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models.yolo import Model
from utils import google_utils
from utils.datasets import *
from utils.utils import *

#  设置混精度训练，需要安装英伟达的apex，默认为True,笔者没用到就设置为False
mixed_precision = False
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

# 超参数
hyp = {'optimizer': 'SGD',  # 优化器['adam', 'SGD', None] if none, default is SGD
       'lr0': 0.01,  # 学习率initial learning rate (SGD=1E-2, Adam=1E-3)
       'momentum': 0.937,  # 学习率动量SGD momentum/Adam beta1
       'weight_decay': 5e-4,  # 权重衰减系数optimizer weight decay
       'giou': 0.05,  # giou损失的系数giou loss gain
       'cls': 0.58,  # 分类损失的系数cls loss gain
       'cls_pw': 1.0,  # 分类BCELoss中正样本的权重cls BCELoss positive_weight
       'obj': 1.0,  # 有无物体损失的系数obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # 有无物体BCELoss中正样本的权重obj BCELoss positive_weight
       'iou_t': 0.20,  # 标签与anchors的iou阈值iou training threshold
       'anchor_t': 4.0,  # 标签的长h宽w/anchor的长h_a宽w_a阈值, 即h/h_a, w/w_a都要在(1/4, 4)之间anchor-multiple threshold
       'fl_gamma': 0.0,  # focal loss gamma, 设为0则表示不使用focal loss(efficientDet default is gamma=1.5)
       # 下面是一些数据增强的系数, 包括颜色空间和图片空间
       'hsv_h': 0.014,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.68,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 0.0,  # image rotation (+/- deg)
       'translate': 0.0,  # image translation (+/- fraction)
       'scale': 0.5,  # image scale (+/- gain)
       'shear': 0.0}  # image shear (+/- deg)


def train(hyp):
    print(f'Hyperparameters {hyp}')
    # 获取记录训练日志的路径
    """
    训练日志包括：权重、tensorboard文件、超参数hyp、设置的训练参数opt(也就是epochs,batch_size等),result.txt
    result.txt包括: 占GPU内存、训练集的GIOU loss, objectness loss, classification loss, 总loss, 
    targets的数量, 输入图片分辨率, 准确率TP/(TP+FP),召回率TP/P ; 
    测试集的mAP50, mAP@0.5:0.95, GIOU loss, objectness loss, classification loss.
    还会保存batch<3的ground truth
    """
    log_dir = tb_writer.log_dir  # run directory
    # 设置保存权重的路径
    wdir = str(Path(log_dir) / 'weights') + os.sep  # weights directory

    os.makedirs(wdir, exist_ok=True)
    last = wdir + 'last.pt'
    best = wdir + 'best.pt'
    # 设置保存results的路径
    results_file = log_dir + os.sep + 'results.txt'

    # Save run settings
    # 保存hyp和opt
    with open(Path(log_dir) / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(Path(log_dir) / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # 设置轮次、批次、权重
    epochs = opt.epochs  # 300
    batch_size = opt.batch_size  # 64
    weights = opt.weights  # initial training weights

    # Configure
    # 设置随机种子
    init_seeds(1)
    # 加载数据配置信息
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    # 获取训练集、测试集图片路径
    train_path = data_dict['train']
    test_path = data_dict['val']
    # 获取类别数量和类别名字
    # 如果设置了opt.single_cls则为一类
    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Remove previous results
    # 移除之前的图片结果
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):
        os.remove(f)

    # Create model
    # 创建模型
    model = Model(opt.cfg, nc=nc).to(device)

    # Image sizes
    # 获取模型总步长和模型输入图片分辨率
    gs = int(max(model.stride))  # grid size (max stride)
    # 检查输入图片分辨率确保能够整除总步长gs
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Optimizer
    """
    nbs为模拟的batch_size; 
    就比如默认的话上面设置的opt.batch_size为16,这个nbs就为64，
    也就是模型梯度累积了64/16=4(accumulate)次之后
    再更新一次模型，变相的扩大了batch_size
    """
    nbs = 32  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置权重衰减系数
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    # 将模型分成三组(weight、bn, bias, 其他所有参数)优化
    for k, v in model.named_parameters():
        if v.requires_grad:
            if '.bias' in k:
                pg2.append(v)  # biases
            elif '.weight' in k and '.bn' not in k:
                pg1.append(v)  # apply weight decay
            else:
                pg0.append(v)  # all else

    # 选用优化器，并设置pg0组的优化方式
    if hyp['optimizer'] == 'adam':  # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    # 设置weight、bn的优化方式
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # 设置biases的优化方式
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # 设置学习率衰减，这里为余弦退火方式进行衰减
    # 就是根据以下公式lf与epoch进行衰减
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.9 + 0.1  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # plot_lr_scheduler(optimizer, scheduler, epochs, save_dir=log_dir)

    # Load Model
    # 加载模型，从google云盘中自动下载模型
    # 但通常会下载失败，建议提前下载下来放进weights目录
    google_utils.attempt_download(weights)
    # 初始化开始训练的epoch和最好的结果
    # best_fitness是以[0.0, 0.0, 0.1, 0.9]为系数并乘以[精确度, 召回率, mAP@0.5, mAP@0.5:0.95]再求和所得
    # 根据best_fitness来保存best.pt
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt'):  # pytorch format
        # 加载检查点
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        # load model
        # 加载模型
        try:
            ckpt['model'] = {k: v for k, v in ckpt['model'].float().state_dict().items()
                             if model.state_dict()[k].shape == v.shape}  # to FP32, filter
            model.load_state_dict(ckpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
                "Please delete or update %s and try again, or use --weights '' to train from scratch." \
                % (opt.weights, opt.cfg, opt.weights, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        # 加载优化器与best_fitness
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # load results
        # 加载训练结果result.txt
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # epochs
        # 加载训练的轮次
        start_epoch = ckpt['epoch'] + 1
        """
        如果新设置epochs小于加载的epoch，
        则视新设置的epochs为需要再训练的轮次数而不再是总的轮次数
        """
        if epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt

    # Mixed precision training https://github.com/NVIDIA/apex
    # 如果设置混精度训练，初始化混精度训练
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Distributed training
    # 如果不在cpu上计算且gpu数量大于1且pytorch允许分布式，则设置分布式训练
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # distributed backend
                                init_method='tcp://127.0.0.1:9999',  # init method
                                world_size=1,  # number of nodes
                                rank=0)  # node rank
        model = torch.nn.parallel.DistributedDataParallel(model)
        # pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html

    # Trainloader
    # 创建训练集dataloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect)

    """
    获取标签中最大的类别值，并于类别数作比较
    如果小于类别数则表示有问题
    """
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Correct your labels or your model.' % (mlc, nc, opt.cfg)

    # Testloader
    # 创建测试集dataloader
    testloader = create_dataloader(test_path, imgsz_test, batch_size, gs, opt,
                                   hyp=hyp, augment=False, cache=opt.cache_images, rect=True)[0]

    # Model parameters
    # 根据自己数据集的类别数设置分类损失的系数
    hyp['cls'] *= nc / 80.  # scale coco-tuned hyp['cls'] to current dataset
    # 设置类别数，超参数
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    """
    设置giou的值在objectness loss中做标签的系数, 使用代码如下
    tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * giou.detach().clamp(0).type(tobj.dtype)
    这里model.gr=1，也就是说完全使用标签框与预测框的giou值来作为该预测框的objectness标签
    """
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    # 根据labels初始化图片采样权重
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    # 获取类别的名字
    model.names = names

    # Class frequency
    # 将所有样本的标签拼接到一起shape为(total, 5)，统计后做可视化
    labels = np.concatenate(dataset.labels, 0)
    # 获得所有样本的类别
    c = torch.tensor(labels[:, 0])  # classes
    # cf = torch.bincount(c.long(), minlength=nc) + 1.
    # model._initialize_biases(cf.to(device))
    # 根据上面的统计对所有样本的类别，中心点xy位置，长宽wh做可视化
    plot_labels(labels, save_dir=log_dir)
    # 添加类别的直方图到tensorboard中
    if tb_writer:
        tb_writer.add_histogram('classes', c, 0)

    # Check anchors
    """
    计算默认锚点anchor与数据集标签框的长宽比值
    标签的长h宽w与anchor的长h_a宽w_a的比值, 即h/h_a, w/w_a都要在(1/hyp['anchor_t'], hyp['anchor_t'])是可以接受的
    如果标签框满足上面条件的数量小于总数的99%，则根据k-mean算法聚类新的锚点anchor
    """
    if not opt.noautoanchor:
        check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)

    # Exponential moving average
    # 为模型创建EMA指数滑动平均
    ema = torch_utils.ModelEMA(model, updates=start_epoch * nb / accumulate)
    print(ema.updates)

    # Start training
    t0 = time.time()
    # 获取热身训练的迭代次数
    nw = max(3 * nb, 1e3)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # 初始化mAP和results
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    """
    设置学习率衰减所进行到的轮次，
    目的是打断训练后，--resume接着训练也能正常的衔接之前的训练进行学习率衰减
    """
    scheduler.last_epoch = start_epoch - 1  # do not move
    """
    打印训练和测试输入图片分辨率
    加载图片时调用的cpu进程数
    从哪个epoch开始训练
    """
    print('Image sizes %g train, %g test' % (imgsz, imgsz_test))
    print('Using %g dataloader workers' % dataloader.num_workers)
    print('Starting training for %g epochs...' % epochs)
    # torch.autograd.set_detect_anomaly(True)
    # 训练
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        # if epoch == 250:
        #     exit()
        model.train()

        # Update image weights (optional)
        """
        如果设置进行图片采样策略，
        则根据前面初始化的图片采样权重model.class_weights以及maps配合每张图片包含的类别数
        通过random.choices生成图片索引indices从而进行采样
        """
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(4, device=device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        # tqdm 创建进度条，方便训练时 信息的展示
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            # 计算迭代的次数iteration
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0

            # Warmup
            """
            热身训练(前nw次迭代)
            在前nw次迭代中，根据以下方式选取accumulate和学习率
            """
            if ni <= nw:
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    """
                    bias的学习率从0.1下降到基准学习率lr*lf(epoch)，
                    其他的参数学习率从0增加到lr*lf(epoch).
                    lf为上面设置的余弦退火的衰减函数
                    """
                    x['lr'] = np.interp(ni, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    # 动量momentum也从0.9慢慢变到hyp['momentum'](default=0.937)
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [0.9, hyp['momentum']])

            # Multi-scale
            # 设置多尺度训练，从imgsz * 0.5, imgsz * 1.5 + gs随机选取尺寸
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)

            # Loss
            # 计算损失，包括分类损失，objectness损失，框的回归损失
            # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
            loss, loss_items = compute_loss(pred, targets.to(device), model)
            # 检查loss是否无穷大(可能时梯度爆炸，或者计算损失梯度时存在log(score)->log(0)->无穷大)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            # 如果设置混精度训练，混合精度反向传播
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            # 模型反向传播accumulate次之后再根据累积的梯度更新一次参数
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            # 打印显存，进行的轮次，损失，target的数量和图片的size等信息
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.4g' * 6) % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
            # 进度条显示以上信息
            pbar.set_description(s)

            # Plot
            # 将前三次迭代batch的标签框在图片上画出来并保存
            if ni < 3:
                f = str(Path(log_dir) / ('train_batch%g.jpg' % ni))  # filename
                result = plot_images(images=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer and result is not None:
                    tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        # 进行学习率衰减
        scheduler.step()

        # mAP
        # 更新EMA的属性
        ema.update_attr(model)
        # 判断该epoch是否为最后一轮
        final_epoch = epoch + 1 == epochs
        # 对测试集进行测试，计算mAP等指标
        # 测试时使用的是EMA模型
        if not opt.notest or final_epoch:  # Calculate mAP
            results, maps, times = test.test(opt.data,
                                             batch_size=batch_size,
                                             imgsz=imgsz_test,
                                             save_json=final_epoch and opt.data.endswith(os.sep + 'coco.yaml'),
                                             model=ema.ema,
                                             single_cls=opt.single_cls,
                                             dataloader=testloader,
                                             save_dir=log_dir)

        # Write
        # 将指标写入result.txt
        with open(results_file, 'a') as f:
            f.write(s + '%10.4g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        # 如果设置opt.bucket, 上传results.txt到谷歌云盘
        if len(opt.name) and opt.bucket:
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Tensorboard
        # 添加指标，损失等信息到tensorboard显示
        if tb_writer:
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        # 更新best_fitness
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        """
        保存模型，还保存了epoch，results，optimizer等信息，
        optimizer将不会在最后一轮完成后保存
        model保存的是EMA的模型
        """
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if save:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': ema.ema,
                        'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(ckpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(ckpt, best)
            del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    # Strip optimizers
    """
    模型训练完后，strip_optimizer函数将optimizer从ckpt中去除；
    并且对模型进行model.half(), 将Float32的模型->Float16，
    可以减少模型大小，提高inference速度
    """
    n = ('_' if len(opt.name) and not opt.name.isnumeric() else '') + opt.name
    fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
    for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
        if os.path.exists(f1):
            os.rename(f1, f2)  # rename
            ispt = f2.endswith('.pt')  # is *.pt
            strip_optimizer(f2) if ispt else None  # strip optimizer
            # 上传结果到谷歌云盘
            os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    # Finish
    # 可视化results.txt文件
    if not opt.evolve:
        plot_results(save_dir=log_dir)  # save as results.png
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    # 释放显存
    dist.destroy_process_group() if device.type != 'cpu' and torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    """
    opt参数解析：
    cfg:模型配置文件，网络结构
    data:数据集配置文件，数据集路径，类名等
    hyp:超参数文件
    epochs:训练总轮次
    batch-size:批次大小
    img-size:输入图片分辨率大小
    rect:是否采用矩形训练，默认False
    resume:接着打断训练上次的结果接着训练
    nosave:不保存模型，默认False
    notest:不进行test，默认False
    noautoanchor:不自动调整anchor，默认False
    evolve:是否进行超参数进化，默认False
    bucket:谷歌云盘bucket，一般不会用到
    cache-images:是否提前缓存图片到内存，以加快训练速度，默认False
    weights:加载的权重文件
    name:数据集名字，如果设置：results.txt to results_name.txt，默认无
    device:训练的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)
    multi-scale:是否进行多尺度训练，默认False
    single-cls:数据集是否只有一个类别，默认False
    adam:是否使用adam优化器
    sync-bn:是否使用跨卡同步BN,在DDP模式使用
    local_rank:gpu编号
    logdir:存放日志的目录
    workers:dataloader的最大worker数量
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    opt = parser.parse_args()

    # Set DDP variables
    """
    设置DDP模式的参数
    world_size:表示全局进程个数
    global_rank:进程编号
    """
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)
    if opt.global_rank in [-1, 0]:
        # 检查你的代码版本是否为最新的(不适用于windows系统)
        check_git_status()

    # Resume
    # 是否resume
    if opt.resume:  # resume an interrupted run
        # 如果resume是str,则表示传入的是模型的路径地址
        # get_latest_run()函数获取runs文件夹中最近的last.pt
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        log_dir = Path(ckpt).parent.parent  # runs/exp0
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        # opt参数也全部替换
        with open(log_dir / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))  # replace
        # opt.cfg设置为'' 对应着train函数里面的操作(加载权重时是否加载权重里的anchor)
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        logger.info('Resuming training from %s' % ckpt)

    else:
        # 获取超参数列表
        opt.hyp = opt.hyp or ('data/hyp.finetune.yaml' if opt.weights else 'data/hyp.scratch.yaml')
        # 检查配置文件信息
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        # 扩展image_size为[image_size, image_size]一个是训练size，一个是测试size
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
        # 根据opt.logdir生成目录
        log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)  # runs/exp1

    # 选择设备
    device = select_device(opt.device, batch_size=opt.batch_size)

    # DDP mode
    # DDP 模式
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        # 根据gpu编号选择设备
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        # 初始化进程组
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        # 将总批次按照进程数分配给各个gpu
        opt.batch_size = opt.total_batch_size // opt.world_size

    # 打印opt参数信息
    logger.info(opt)
    # 加载超参数列表
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    # Train
    # 如果不进行超参数进化，则直接调用train()函数，开始训练
    if not opt.evolve:
        tb_writer = None
        if opt.global_rank in [-1, 0]:
            # 创建tensorboard
            logger.info('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opt.logdir)
            tb_writer = SummaryWriter(log_dir=log_dir)  # runs/exp0

        train(hyp, opt, device, tb_writer)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        # 超参数进化列表,括号里分别为(突变规模, 最小值,最大值)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.1, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'giou': (1, 0.02, 0.2),  # GIoU loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                # 'anchors': (1, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mixup': (1, 0.0, 1.0)}  # image mixup (probability)

        assert opt.local_rank == -1, 'DDP mode not implemented for --evolve'
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        yaml_file = Path('runs/evolve/hyp_evolved.yaml')  # save best result here
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        # 默认进化100次
        """
        这里的进化算法是：根据之前训练时的hyp来确定一个base hyp再进行突变；
        如何根据？通过之前每次进化得到的results来确定之前每个hyp的权重
        有了每个hyp和每个hyp的权重之后有两种进化方式；
        1.根据每个hyp的权重随机选择一个之前的hyp作为base hyp，random.choices(range(n), weights=w)
        2.根据每个hyp的权重对之前所有的hyp进行融合获得一个base hyp，(x * w.reshape(n, 1)).sum(0) / w.sum()
        evolve.txt会记录每次进化之后的results+hyp
        每次进化时，hyp会根据之前的results进行从大到小的排序；
        再根据fitness函数计算之前每次进化得到的hyp的权重
        再确定哪一种进化方式，从而进行进化
        """
        for _ in range(100):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                # 选择进化方式
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                # 加载evolve.txt
                x = np.loadtxt('evolve.txt', ndmin=2)
                # 选取至多前5次进化的结果
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                # 根据results计算hyp的权重
                w = fitness(x) - fitness(x).min()  # weights
                # 根据不同进化方式获得base hyp
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                # 超参数进化
                mp, s = 0.9, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                # 获取突变初始值
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                # 设置突变
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                # 将突变添加到base hyp上
                # [i+7]是因为x中前七个数字为results的指标(P, R, mAP, F1, test_losses=(GIoU, obj, cls))，之后才是超参数hyp
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            # 修剪hyp在规定范围里
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            # 训练
            results = train(hyp.copy())

            # Write mutation results
            """
            写入results和对应的hyp到evolve.txt
            evolve.txt文件每一行为一次进化的结果
            一行中前七个数字为(P, R, mAP, F1, test_losses=(GIoU, obj, cls))，之后为hyp
            保存hyp到yaml文件
            """
            print_mutation(hyp.copy(), results, yaml_file, opt.bucket)

        # Plot results
        plot_evolution(yaml_file)
        print('Hyperparameter evolution complete. Best results saved as: %s\nCommand to train a new model with these '
              'hyperparameters: $ python train_comment.py --hyp %s' % (yaml_file, yaml_file))

