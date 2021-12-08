import numpy as np
import torch

def write_pytorch_data(output_path, data, data_name_list):
    """
    Save the data of Pytorch needed to align TNN model.

    The input and output names of pytorch model and onnx model may not match,
    you can use Netron to visualize the onnx model to determine the data_name_list.

    The following example converts ResNet50 to onnx model and saves input and output:
    >>> from torchvision.models.resnet import resnet50
    >>> model = resnet50(pretrained=False).eval()
    >>> input_data = torch.randn(1, 3, 224, 224)
    >>> input_names, output_names = ["input"], ["output"]
    >>> torch.onnx.export(model, input_data, "ResNet50.onnx", input_names=input_names, output_names=output_names)
    >>> with torch.no_grad():
    ...     output_data = model(input_data)
    ...
    >>> write_pytorch_data("input.txt", input_data, input_names)
    >>> write_pytorch_data("output.txt", output_data, output_names)

    :param output_path: Path to save data.
    :param data: The input or output data of Pytorch model.
    :param data_name_list: The name of input or output data. You can get it after visualization through Netron.
    :return:
    """

    if type(data) is not list and type(data) is not tuple:
        data = [data, ]
    assert len(data) == len(data_name_list), "The number of data and data_name_list are not equal!"
    with open(output_path, "w") as f:
        f.write("{}\n" .format(len(data)))
        for name, data in zip(data_name_list, data):
            data = data.numpy()
            shape = data.shape
            description = "{} {} ".format(name, len(shape))
            for dim in shape:
                description += "{} ".format(dim)
            data_type = 0 if data.dtype == np.float32 else 3
            fmt = "%0.6f" if data_type == 0 else "%i"
            description += "{}".format(data_type)
            f.write(description + "\n")
            np.savetxt(f, data.reshape(-1), fmt=fmt)


def write_tensorflow_data(output_path, data, data_name_list, data_usage=1):
    """
    Save the data of TensoFlow needed to align TNN model.

    :param output_path: Path to save data. "You should use input.txt or output.txt to name input or output data"
    :param data: The input or output data of TensorFlow model.
    :param data_name_list: The name of input or output data. You can get it after visualization through Netron.
    :param data_usage: Specify the data usage. If the data is input data, data_usage=0;
                       if the data if outptu data, data_usage=1.
    :return:
    """
    def convert_nhwc(data):
        assert len(data.shape) <= 4
        if len(data.shape) == 2:
            return data
        orders = (0, 2, 1) if len(data.shape) == 3 else (0, 2, 3, 1)
        return data.transpose(orders)

    if type(data) is not list and type(data) is not tuple:
        data = [data, ]
    assert len(data) == len(data_name_list), "The number of data and data_name_list are not equal!"
    with open(output_path, "w") as f:
        f.write("{}\n" .format(len(data)))
        for name, data in zip(data_name_list, data):
            data = convert_nhwc(data) if data_usage == 0 else data
            shape = data.shape
            description = "{} {} ".format(name, len(shape))
            for dim in shape:
                description += "{} ".format(dim)
            data_type = 0 if data.dtype == np.float32 else 3
            fmt = "%0.6f" if data_type == 0 else "%i"
            description += "{}".format(data_type)
            f.write(description + "\n")
            np.savetxt(f, data.reshape(-1), fmt=fmt)


