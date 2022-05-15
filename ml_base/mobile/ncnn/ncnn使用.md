ubuntu安装ncnn 参考
https://blog.csdn.net/litt1e/article/details/116000118

# 克隆ncnn 
git clone ncnn
# 安装依赖包
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev
问题点
其中，protobuf冲突，因为多个位置安装，比如系统,conda，删除conda的
然后使用git官网推荐的安装方法，安装完成之后，加上一个
protoc -h
查看是否安装成功，之后应该就可以了

# 编译构建ncnn
cd ncnn
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON ..
make -j$(nproc)
# 测试ncnn是否构建成功
cd ../examples
../build/examples/squeezenet ../images/256-ncnn.png
终端打印出信息就说明成功了
532 = 0.165951
920 = 0.094098
716 = 0.062193

# 查看onnx相关是否构建成功
需要onnx目录下有
cmake_install.cmake
Makefile
onnx.pb.cc
onnx.pb.h
onnx2ncnn
这些文件

# 获取onnx
## 简化onnx
首先，安装onnx-smiplifier
pip install onnx-simplifier
然后简化onnx模型
python3 -m onnxsim name.onnx name-sim.onnx


# 进入构建好的目录
cd ncnn/build/tools/onnx
# 转换
/D/tools/ncnn/ncnn/build/tools/onnx/onnx2ncnn my_mobileface-sim.onnx my_mobileface.param my_mobileface.bin

# 优化
**优化应该是很重要的！！测试一个picodet模型大小减半！！！ 参数文件287行优化到了187行，**
现在在 ncnn/build/tools/onnx 目录下
../ncnnoptimize bytetrack_s.param bytetrack_s.bin bytetrack_s_op.param bytetrack_s_op.bin 65536


# ncnn量化
## 准备好ncnn优化后的模型 官网最后的值是0
./ncnnoptimize mobilenet.param mobilenet.bin mobilenet-opt.param mobilenet-opt.bin 0

## 创建校准表文件
下面命令是在/D/tools/ncnn/ncnn/build/tools/quantize目录下执行的，其它目录似乎会有问题
验证集图片文件夹为images，放到该目录下，然后
find images/ -type f > imagelist.txt

生成校准表
./ncnn2table yolov4-tiny-opt.param yolov4-tiny-opt.bin imagelist.txt yolov4-tiny.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl

./ncnn2table nanodet.param          nanodet.bin        imagelist.txt nanodet.table     mean=[104,116,124] norm=[0.017,0.017,0.017] shape=[416,416,3] pixel=BGR thread=8 method=kl


# 量化

./ncnn2int8 nanodet.param  nanodet.bin  nanodet-416-1.5x-int8.param nanodet-416-1.5x-int8.bin    nanodet.table
 
# 推导式开启相关选项
    this->Net->opt.use_vulkan_compute = false; //hasGPU && useGPU;  // gpu
    this->Net->opt.use_fp16_arithmetic = true;
    this->Net->opt.use_fp16_packed = true;
    this->Net->opt.use_fp16_storage = true;


# 提取中间结果
ncnn::Extractor ex = MyNet.create_extractor();
ex.set_light_mode(true);
ex.input("data", in);
ncnn::Mat prob1;
ncnn::Mat conv4;
ex.extract("prob1", prob1);
ex.extract("conv4", conv4);

您的意思是如果只是提取中间某层的结果，前向计算就会执行到当前层？
NEU-Gou 我这边的实验结果是这样的
nihui  是的，只会计算需要算的部分

@NEU-Gou @nihui 确实如此，多谢多谢+1
还得请教一下：同一个Extractor，输入图像大小不变，我想循环处理N张不同的图像，有没有简单的方法？多谢^_^
如下面的这种形式，第二次调extract会直接给出第一次extract的结果，而不会真正forward一次：
ncnn::Extractor t_ex = T_Net.create_extractor();
for (int i = 0; i < N; i++)
{
t_ex.extract(...);
}

for (int i = 0; i < N; i++)
{
ncnn::Extractor t_ex = T_Net.create_extractor();
t_ex.extract(...);
}

# 原理篇
https://zhuanlan.zhihu.com/p/307543402