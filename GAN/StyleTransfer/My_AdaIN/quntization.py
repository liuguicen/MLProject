# 量化模型
import torch
import torch.nn as nn
import os


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

# Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
def fuse_model(model):
    for m in model.modules():
        if type(m) == ConvBNReLU:
            torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
        if type(m) == InvertedResidual:
            for idx in range(len(m.conv)):
                if type(m.conv[idx]) == nn.Conv2d:
                    torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


def moreQuantization(per_channel_quantized_model):
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
    per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    print(per_channel_quantized_model.qconfig)

    torch.quantization.prepare(per_channel_quantized_model, inplace=True)
    # evaluate(per_channel_quantized_model, criterion, data_loader, num_calibration_batches)
    torch.quantization.convert(per_channel_quantized_model, inplace=True)
    # top1, top5 = evaluate(per_channel_quantized_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    # print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))
    # torch.jit.save(torch.jit.script(per_channel_quantized_model), path)
    return per_channel_quantized_model

def quantization(myModel: nn.Module):
    num_calibration_batches = 32

    myModel.eval()

    # Fuse Conv, bn and relu
    myModel.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    myModel.qconfig = torch.quantization.default_qconfig
    print(myModel.qconfig)
    torch.quantization.prepare(myModel, inplace=True)

    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

    # Calibrate with the training set
    evaluate(myModel, criterion, data_loader, neval_batches=num_calibration_batches)
    print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.quantization.convert(myModel, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',
          myModel.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(myModel)

    top1, top5 = evaluate(myModel, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f' % (num_eval_batches * eval_batch_size, top1.avg))

