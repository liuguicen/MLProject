import FileUtil


class BaseRunRecord:
    def __init__(self, check_point_dir = 'check_point'):
        self.check_point_path = ''
        self.check_point_epoch = 0
        self.check_point_iter = 0
        self.check_point_dir = check_point_dir
        self.loss_dir = f'{check_point_dir}/loss'
        self.model_state_dir = f'{check_point_dir}/model_state'
        self.tes_res_dir = f'{check_point_dir}/test_res'

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


class TestSubClass(BaseRunRecord):
    def __init__(self):
        BaseRunRecord.__init__(self)
        super().createDir()


if __name__ == "__main__":
    print('abc'.endswith('c'))
