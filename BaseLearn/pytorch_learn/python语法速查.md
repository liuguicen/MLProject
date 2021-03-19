python里面列表乘以一个数，不是数乘，而是扩大列表尺寸？？？这又是什么反常语法
[1] * 2 = [2, 2]
反常语法

1、变量后面注释 `#type: Image.Image`
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201108111402524.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2huX2xnYw==,size_16,color_FFFFFF,t_70#pic_center)

2、使用isinstance指定
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201115041028583.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2huX2xnYw==,size_16,color_FFFFFF,t_70#pic_center)
  
python的max，min,等类似函数   
里面可以直接传入多个相同类型的数，注意类型一样，不一样可能转换，转化不了就报错  
然后后面还可以加一个明明参数key=xxx, 就是函数或者lambda表达式，表示对比较列表中的元素先做某种处理之后在比较，注意最后返回的还是原始  
这种方式是有利于提示时空效率的  
max()

## yield语法
使用了yield的函数不再是普通的函数，执行函数后会从yield位置返回，python会记录函数的状态，包括方法内部的临时变量，当再次执行函数时，代码从 yield b 的下一条语句继续执行，直到再次遇到yield，
进一步的这样的函数可以被看成一种生成器，也可以看成迭代器，可以调用next() 方法，不用自己写，可以放到迭代语句中
   
例子   
```
def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        # print b
        a, b = b, a + b
        n = n + 1
```
要以此生成多个斐波那契数时，使用yield, 可以更方便的使用。
更深层的说，这就是一种有状态方法，不同于我们一般的无状态方法，yield对应的方法会记录运行过程中的一些状态


也可以手动调用 fab(5) 的 next() 方法（因为 fab(5) 是一个 generator 对象，该对象具有 next() 方法），这样我们就可以更清楚地看到 fab 的执行流程：  

清单 6. 执行流程  

参数解析器的使用：  
parser = argparse.ArgumentParser()  
这个玩意真不咋地，好几个坑的地方  
1、定义的参数里面的减号-可以自动替换成下划线_  我用得着你多此一举，都最后结果别人找不到谁对应谁，find操作也不好找  
2、参数不用输全就可以起作用，这多此一举吧，说不定别人根本就不想设置这个参数呢？？？  
3、在pycharm 里面，同样的输入，调试和运行解析的到参数经验不一样，找不到原因？？？？  

# 注意python的 and or   
这和其它语言的逻辑运算不同，逻辑运算返回真/假  
这个是返回的对象  
or返回遇到的第一个非空对象，包括字符串非空  
andand 在布尔上下文中从左到右演算表达式的值，如果布尔上下文中的所有值都为真，那么 and 返回最后一个值。  
如果布尔上下文中的某个值为假，则 and 返回第一个假值  

# zip()
函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表  
zip( iterable, ...) 注意里面的迭代器可以是多个，不是只能2个  
如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表  
另外在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换  

# 导入包 导入模块
同一个项目目录下的，文件夹上右键，make source as
或者打开项目结构 子目录选中 右键 source
项目目录不同 把它们的共同父目录作为项目根目录，重新打开，或者在项目结构那里，先删除content root，再加上共同根目录即可  
![](.python语法速查_images/698c93dc.png)  

any 判断迭代器是否包含至少一个元素为true
all 判断迭代器是否所有元素都为true


动态导入模块，
lib = importlib.import_module(模块x相对路径)
然后可以使用x内部的所有类，包括x中的import语句里面的哪些模块类
lib.C  #取出class C
lib.C.c  #取出class C 中的c 方法
或者lib.__dict__.items() 方法模块内部所有的类名字和类对象形成的dict对

捕获异常
```python

try:
    ...
 except Exception as e: # Exception就是异常根父类
    ... 
    raise e   
```
文件相关
创建父目录
 os.makedirs
os.walk 遍历目录树，包含自己
如果想要只处理一级目录下的，那么只处理它的第一个返回值就行了

获取父目录
os.path.dirname('...')
os.path.split(path)[0]
获取路径中的文件名
os.path.split(path)[1] # 带后缀
os.path.splitext(带后缀的文件名)[0] # 不带后缀 
获取后缀
os.path.splitext(path)[1] # 带有一个.

# 图片相关
# PIL
for PIL import Image
打开图片文件,支持PNG，透明度在第4通道
Image.open(path)

与numpy互相转换
numpy.array(im)
Image.fromarray(img.astype('uint8')).convert('RGB')
显示图片
 plt.figure("dog")
 plt.imshow(src)
 plt.show()

    
保存图片
im.save(path)