# 编写新的文件是一键导入常用的库，不用重复写代码
# 然后直接使用相关类或者文件即可

# 用法 from ml_base.common_lib_import import *
from os import path
import FileUtil
import random
import image_util

if __name__ == '__main__':
    FileUtil
    image_util
    random.random()
    path.join('a','b')