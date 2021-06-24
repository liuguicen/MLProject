from ml_base.BaseRunRecord import BaseRunRecord
import pickle
import os

class RunRecord(BaseRunRecord):
    def __init__(self, checkPointPath='check_point'):
        BaseRunRecord.__init__(self, checkPointPath)
        self.check_point_path = r''
        self.check_point_epoch = 0
        self.check_point_iter = 0
        # super().createDir()  # 创建相应的中间结果需要的目录