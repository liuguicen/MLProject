

def print_para(net):
    print(net.state_dict())
    print('网络结构：\n', net)
    print('参数结构:')
    paras_list = []
    for k, v in net.named_parameters():
        print(k, v.size())
        print('paras is', v)
        tolist = v.data.view(-1).numpy().tolist()
        print('para to list is ', tolist)
        paras_list.extend(tolist)
    print(-8.7923e-02)
    print(paras_list)