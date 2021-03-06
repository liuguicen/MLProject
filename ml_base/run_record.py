import os
import pickle

import FileUtil

default_dir = 'check_point'
default_record_path = os.path.join(default_dir, 'RunRecord.pkl')


class RunRecord:
    def __init__(self, check_point_dir=default_dir):
        self.common_iter_count = 0
        '''
        通用的循环迭代次数
        '''
        self.check_point_path = ''
        self.check_point_epoch = 0
        self.check_point_iter = 0
        self.check_point_dir = check_point_dir
        self.loss_dir = f'{check_point_dir}/loss'
        self.model_state_dir = f'{check_point_dir}/model_state'
        self.tes_res_dir = f'{check_point_dir}/test_res'
        self.createDir()

    def reset(self):
        self.check_point_path = ''
        self.check_point_epoch = 0
        self.check_point_iter = 0

    def createDir(self):
        '''
        以dir结尾的属性名将创建对应的文件夹
        '''
        name_value = self.__dict__
        for name, value in name_value.items():
            if name.endswith('dir'):
                FileUtil.mkdir(value)

    def saveRunRecord(self, epoch, iter, path=default_record_path):
        '''
        保存运行记录，直接用下面的代码就行，写在这里防止忘记
        '''
        self.check_point_epoch = epoch
        self.check_point_iter = iter
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def readFromDisk(path=default_record_path):
        if os.path.exists(path):
            with open(path, "rb") as file:
                record = pickle.load(file)
                record.createDir()
                return record
        else:
            print('没有运行记录')


def saveRunRecord(record, path=default_record_path):
    '''
    保存运行记录，直接用下面的代码就行，写在这里防止忘记
    '''
    with open(path, "wb") as f:
        pickle.dump(record, f)


class TestSubClass(RunRecord):
    def __init__(self):
        RunRecord.__init__(self)
        super().createDir()


if __name__ == "__main__":
    print('abc'.endswith('c'))
