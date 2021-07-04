def process(image, para):
    s = 0
    for i in range(1000000):
        s += i
    return image, para


if __name__ == '__main__':
    print('test finish')
    pro = process
    print(pro(1, 2))
