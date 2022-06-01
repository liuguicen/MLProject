import os

import torch
from torch.utils.mobile_optimizer import optimize_for_mobile


def exportModule(model, path):
    model.cpu()
    model.eval()
    # model = quntization.quantization2(model)
    scripted_module = torch.jit.script(model)
    if os.path.exists(path):
        os.remove(path)
    # pytorch 提供的移动优化 实测似乎没什么用
    torchscript_model_optimized = optimize_for_mobile(scripted_module)
    torch.jit.save(torchscript_model_optimized, path)
    print('导出', path, '成功')