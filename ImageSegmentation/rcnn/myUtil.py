def comparePara(a, b):
    print('\n\nmy resnet para')
    aPara = a.named_parameters()
    bpara = b.named_parameters()

    print('lib res my res compare')
    for i, data in enumerate(zip(aPara, bpara)):
        print('\n第', i, '个参数')
        para, myPara = data[0], data[1]
        print('name = ', para[0], '  type = ', type(para[1]), 'size = ', para[1].size())
        print('name = ', myPara[0], '  type = ', type(myPara[1]), 'size = ', myPara[1].size())
