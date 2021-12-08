import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

input_channel = 512
output_channel = 3


class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""

    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # self.activated = activated

    def forward(self, x):
        # x = self.pad(x)
        x = F.pad(x, (1, 1, 1, 1), 'reflect')
        x = self.conv(x)

        x = F.interpolate(x, scale_factor=2)

        # return F.relu(x)
        return x


def exportOnnx(torch_model, path):
    torch_model.eval()
    input = (torch.rand(1, input_channel, 100, 100))
    output = (torch.rand(1, output_channel, 100, 100))
    # Define attributes for ONNX export
    input_names = ["content"]
    output_names = ["output"]
    dynamic_axes = {
        "content": {2: "width", 3: "height"},
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


import MlUtil

if __name__ == '__main__':
    rc = RC(input_channel, output_channel, 3, 1)
    out = rc(torch.randn(1, 512, 100, 100))
    MlUtil.showModelOutImage(out)
    exportOnnx(rc, 'TestModel.onnx')
