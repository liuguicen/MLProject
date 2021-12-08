import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class rec_model(nn.Module):

    ### user_max_dict/movie_max_dict：用户/电影字典，即用户/电影的一些属性的最大值，决定我们的模型的embedding表的宽度。
    def __init__(self, user_max_dict, movie_max_dict, convParams, embed_dim=32, fc_size=200):
        '''

        Args:
            user_max_dict: the max value of each user attribute. {'uid': xx, 'gender': xx, 'age':xx, 'job':xx}
            user_embeds: size of embedding_layers.
            movie_max_dict: {'mid':xx, 'mtype':18, 'mword':15}
            fc_sizes: fully connect layer sizes. normally 2
        '''

        super(rec_model, self).__init__()

        # --------------------------------- user channel ----------------------------------------------------------------
        # user embeddings
        # 词嵌入技术，大意是把单词映射成向量，一一对应的关系，如‘deep’映射成【12,23.,34.,45.,56.]等等
        # 然后似乎还能通过学习得到词之间的关系，具有相似意义的词具有相似的表示，比如教师和上课，教室关系近等
        # 这种方法在自然语言处理中具有重要地位，词嵌入是自然语言处理的重要突破之一。
        # 之所以希望把每个单词都变成一个向量，目的还是为了方便计算目的还是为了方便计算，
        # 比如“猫”，“狗”，“爱情”三个词。对于我们人而言，我们可以知道“猫”和“狗”表示的都是动物，
        # 而“爱情”是表示的一种情感，但是对于机器而言，这三个词都是用0,1表示成二进制的字符串而已，无法对其进行计算。
        # 而通过词嵌入这种方式将单词转变为词向量，机器便可对单词进行计算，通过计算不同词向量之间夹角余弦值cosine而得出单词之间的相似性。
        # 此外，词嵌入还可以做类比，比如：v(“国王”)－v(“男人”)＋v(“女人”)≈v(“女王”)，v(“中国”)＋v(“首都”)≈v(“北京”)，
        # 当然还可以进行算法推理。有了这些运算，机器也可以像人一样“理解”词汇的意思了。
        # 当您在自然语言处理项目中使用词嵌入时，您可以选择自主学习词嵌入，当然这需要大量的数百万或数十亿文本数据，以确保有用的嵌入被学习。
        # 您也可以选择采用开源的预先训练好的词嵌入模型，研究人员通常会免费提供预先训练的词嵌入，例如word2vec和GloVe词嵌入都可以免费下载。
        # 常见的几种完成方式
        # Embedding Layer
        # Word2Vec和Doc2Vec
        # GloVe
        # 词嵌入在自然语义理解领域内所有任务中都担任着最基础、最核心的功能，包括文本分类、文本摘要、信息检索、自动对话等，
        # 通过词嵌入得到好的词向量作为模型的初始参数，可以帮助几乎所有的各类 NLP 任务取得更好的效果。
        # 你在电脑上存储的单词的ascii码，但是它仅仅代表单词怎么拼写,词嵌入则把词映射到数字空间里面，用数字的方式表达词
        # 处理自然语言输入时，输入的是字，词，句，段，篇，中文的字也 ≈ 英文的词，词可以看成是自然语言输入的一个基本的特征
        # 那么这个就和cv的一个像素一样，然后对词进行处理，如同图像的卷积一样，只是处理的方式不一样，这两者也该都可以算成一个特征处理层
        # 两者可进行一定的类比，像素的量有数亿，组合是二维，或者说3维，数量无穷，词的数量数百万千万，组合算是二维，数量也是无穷
        self.embedding_uid = nn.Embedding(user_max_dict['uid'], embed_dim)
        self.embedding_gender = nn.Embedding(user_max_dict['gender'], embed_dim // 2)
        self.embedding_age = nn.Embedding(user_max_dict['age'], embed_dim // 2)
        self.embedding_job = nn.Embedding(user_max_dict['job'], embed_dim // 2)

        # user embedding to fc: the first dense layer
        self.fc_uid = nn.Linear(embed_dim, embed_dim)
        self.fc_gender = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_age = nn.Linear(embed_dim // 2, embed_dim)
        self.fc_job = nn.Linear(embed_dim // 2, embed_dim)

        # concat embeddings to fc: the second dense layer
        self.fc_user_combine = nn.Linear(4 * embed_dim, fc_size)

        # --------------------------------- movie channel -----------------------------------------------------------------
        # movie embeddings
        self.embedding_mid = nn.Embedding(movie_max_dict['mid'], embed_dim)  # normally 32
        # nn.EmbeddingBag：在构建词袋模型（bag - of - words model）时，在Sum或Mean之后执行Embedding是一种常见做法。对于不同长度的序列，计算词袋嵌入涉及到掩码。我们提供一个单独的nn.EmbeddingBag，它能够更高效、快捷地计算词袋嵌入，尤其是不同长度的序列。
        self.embedding_mtype_sum = nn.EmbeddingBag(movie_max_dict['mtype'], embed_dim, mode='sum')

        self.fc_mid = nn.Linear(embed_dim, embed_dim)
        self.fc_mtype = nn.Linear(embed_dim, embed_dim)

        # movie embedding to fc
        self.fc_mid_mtype = nn.Linear(embed_dim * 2, fc_size)

        # text convolutional part
        # wordlist to embedding matrix B x L x D  L=15 15 words
        self.embedding_mwords = nn.Embedding(movie_max_dict['mword'], embed_dim)

        # input word vector matrix is B x 15 x 32
        # load text_CNN params
        kernel_sizes = convParams['kernel_sizes']
        # 8 kernel, stride=1,padding=0, kernel_sizes=[2x32, 3x32, 4x32, 5x32]
        # sequential 快速搭建神经网络，module里面可以添加一些自己的东西
        self.Convs_text = [nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(k, embed_dim)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(15 - k + 1, 1), stride=(1, 1))
        ).to(device) for k in kernel_sizes]

        # movie channel concat
        self.fc_movie_combine = nn.Linear(embed_dim * 2 + 8 * len(kernel_sizes), fc_size)  # tanh

        # BatchNorm layer
        # BN是归一化函数，
        # 这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
        self.BN = nn.BatchNorm2d(1)

    def forward(self, user_input, movie_input):
        # pack train_data
        uid = user_input['uid']
        gender = user_input['gender']
        age = user_input['age']
        job = user_input['job']

        mid = movie_input['mid']
        mtype = movie_input['mtype']
        mtext = movie_input['mtext']
        if torch.cuda.is_available():
            uid, gender, age, job, mid, mtype, mtext = \
                uid.to(device), gender.to(device), age.to(device), job.to(device), mid.to(device), mtype.to(
                    device), mtext.to(device)
        # user channel
        feature_uid = self.BN(F.relu(self.fc_uid(self.embedding_uid(uid))))
        feature_gender = self.BN(F.relu(self.fc_gender(self.embedding_gender(gender))))
        feature_age = self.BN(F.relu(self.fc_age(self.embedding_age(age))))
        feature_job = self.BN(F.relu(self.fc_job(self.embedding_job(job))))

        # feature_user B x 1 x 200
        feature_user = F.tanh(self.fc_user_combine(
            torch.cat([feature_uid, feature_gender, feature_age, feature_job], 3)
        )).view(-1, 1, 200)

        # 电影通道
        feature_mid = self.BN(F.relu(self.fc_mid(self.embedding_mid(mid))))
        feature_mtype = self.BN(F.relu(self.fc_mtype(self.embedding_mtype_sum(mtype)).view(-1, 1, 1, 32)))

        # feature_mid_mtype = torch.cat([feature_mid, feature_mtype], 2)

        # 文本卷积网络
        feature_img = self.embedding_mwords(mtext)  # to matrix B x 15 x 32
        flattern_tensors = []
        for conv in self.Convs_text:
            flattern_tensors.append(
                conv(feature_img.view(-1, 1, 15, 32)).view(-1, 1, 8))  # each tensor: B x 8 x1 x 1 to B x 8
        # Dropout是为了防止过拟合采取的一种方式，具有简单性并取得良好的结果：
        # 就是在原网络中丢掉部分神经元，其输出设为0
        # 以概率p舍弃神经元并让其它神经元以概率q=1-p保留。每个神经元被关闭的概率是相同的。
        feature_flattern_dropout = F.dropout(torch.cat(flattern_tensors, 2), p=0.5)  # to B x 32

        # feature_movie B x 1 x 200
        feature_movie = F.tanh(self.fc_movie_combine(
            torch.cat([feature_mid.view(-1, 1, 32), feature_mtype.view(-1, 1, 32), feature_flattern_dropout], 2)
        ))
        # 256 * 1 * 200  的张量转换成了 256 * 1的张量
        output = torch.sum(feature_user * feature_movie, 2)  # B x rank
        # 我们的目的就是要训练出用户特征和电影特征，在实现推荐功能时使用。得到这两个特征以后，就可以选择任意的方式来拟合评分了。我使用了两种方式，一个是上图中画出的将两个特征做向量乘法，
        # 将结果与真实评分做回归，采用MSE优化损失。因为本质上这是一个回归问题，另一种方式是，将两个特征作为输入，再次传入全连接层，输出一个值，将输出值回归到真实评分，采用MSE优化损失。
        # 实际上第二个方式的MSE loss在0.8附近，第一个方式在1附近，5次迭代的结果。
        return output, feature_user, feature_movie
