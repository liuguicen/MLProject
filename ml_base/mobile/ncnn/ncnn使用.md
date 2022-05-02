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
python3 -m onnxsim my_mobileface.onnx my_mobileface-sim.onnx


# 进入构建好的目录
cd ncnn/build/tools/onnx
# 转换
./onnx2ncnn my_mobileface-sim.onnx my_mobileface.param my_mobileface.bin

# 优化
**优化应该是很重要的！！测试一个picodet模型大小减半！！！ 参数文件287行优化到了187行，**
现在在 ncnn/build/tools/onnx 目录下
../ncnnoptimize bytetrack_s.param bytetrack_s.bin bytetrack_s_op.param bytetrack_s_op.bin 65536

