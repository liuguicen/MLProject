import time


class Const:
    '''
    常用字符串等，避免写错
    '''
    total = 'total'
    default = 'defalt'


class Timer:
    '''
    recordkey 用于需要多个记录时
    '''
    startT = {Const.default: time.time()}

    @classmethod
    def record(cls, recordkey=Const.default):
        cls.startT[recordkey] = time.time()

    @classmethod
    def print_and_record(cls, msg='', recordKey=Const.default):
        cls.print(msg, recordKey)
        cls.record(recordKey)

    @classmethod
    def print(cls, msg='', recordKey=Const.default):
        print(msg, time.time() - cls.startT[recordKey])


if __name__ == '__main__':
    Timer.print_and_record()
