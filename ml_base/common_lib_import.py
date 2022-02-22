# 编写新的文件是一键导入常用的库，不用重复写代码
# 方法 from ml_base.common_lib_import import *
# 然后直接使用相关类或者文件即可
from os import path
import FileUtil

if __name__ == '__main__':
    path.join('a','b')