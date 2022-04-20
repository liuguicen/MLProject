# 不只是python相关工具，还有使用说明
# 使用说明如果写在文档里面，每次查询相当不方便，写到代码里面查询更方便
class dictUtil:
    @staticmethod
    def contains(self):
        '''
        使用__contain__
        '''
        pass


class stringUtil:

    @staticmethod
    def concatS(*objList, splicer = ""):
        '''
        python没有拼接字符串的语法
        '''
        s = ''
        for obj in objList:
            s += str(obj) + str(splicer)
        return s


if __name__ == "__main__":
    print(stringUtil.concatS(1, "2,", 3, 4, " "))
