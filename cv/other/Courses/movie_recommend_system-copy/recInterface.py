# Recommendation Interface

import pickle as pkl

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MovieRankDataset
from douban_dataset import DoubanMovieRankDataset


# 推荐系统训练完毕后，根据模型的中间输出结果作为电影和用户的特征向量，这个推荐接口根据这些向量的空间关系提供一些定向推荐结果
# 模型训练结束后，我们可以得到电影的特征和用户的特征（可以看网络图中最后一层连接前两个通道的输出即为用户/电影特征，我们在训练结束后将其返回并保存起来）。
def saveMovieAndUserFeature(model, datasets):
    '''
    Save Movie and User feature into HD

    '''

    batch_size = 256

    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=False, num_workers=4)

    # format: {id(int) : feature(numpy array)}
    user_feature_dict = {}
    movie_feature_dict = {}
    movies = {}
    users = {}
    with torch.no_grad():
        for i_batch, sample_batch in enumerate(dataloader):
            user_inputs = sample_batch['user_inputs']
            movie_inputs = sample_batch['movie_inputs']

            # B x 1 x 200 = 256 x 1 x 200
            _, feature_user, feature_movie = model(user_inputs, movie_inputs)

            # B x 1 x 200 = 256 x 1 x 200
            feature_user = feature_user.cpu().numpy()
            feature_movie = feature_movie.cpu().numpy()

            for i in range(user_inputs['uid'].shape[0]):
                uid = user_inputs['uid'][i]  # uid
                gender = user_inputs['gender'][i]
                age = user_inputs['age'][i]
                job = user_inputs['job'][i]

                mid = movie_inputs['mid'][i]  # mid
                mtype = movie_inputs['mtype'][i]
                mtext = movie_inputs['mtext'][i]

                if uid.item() not in users.keys():
                    users[uid.item()] = {'uid': uid, 'gender': gender, 'age': age, 'job': job}
                if mid.item() not in movies.keys():
                    movies[mid.item()] = {'mid': mid, 'mtype': mtype, 'mtext': mtext}

                if uid not in user_feature_dict.keys():
                    user_feature_dict[uid] = feature_user[i]
                if mid not in movie_feature_dict.keys():
                    movie_feature_dict[mid] = feature_movie[i]

            print('Solved: {} samples'.format((i_batch + 1) * batch_size))
    feature_data = {'feature_user': user_feature_dict, 'feature_movie': movie_feature_dict}
    dict_user_movie = {'user': users, 'movie': movies}
    pkl.dump(feature_data, open('Params/feature_data.pkl', 'wb'))
    pkl.dump(dict_user_movie, open('Params/user_movie_dict.pkl', 'wb'))


def getKNNitem(itemID, itemName='movie', K=1):
    '''
    Use KNN at feature data to get K neighbors

    Args:
        itemID: target item's id
        itemName: 'movie' or 'user'
        K: K-neighbors

    return:
        a list of item ids of which close to itemID

逻辑很简单：

根据itemName提取保存在本地的相应的用户/电影特征集合, 是所有的特征
根据itemID获取目标用户的特征
求其特征与其他所有用户/电影的cosine相似度
排序后返回前k个用户/电影即可

    '''

    assert K >= 1, 'Expect K bigger than 0 but get K<1'

    # get cosine similarity between vec1 and vec2
    def getCosineSimilarity(vec1, vec2):
        cosine_sim = float(vec1.dot(vec2.T).item()) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return cosine_sim

    feature_data = pkl.load(open('Params/feature_data.pkl', 'rb'))

    feature_items = feature_data['feature_' + itemName]

    assert itemID in feature_items.keys(), 'Expect item ID exists in dataset, but get None.'
    feature_current = feature_items[itemID]

    id_sim = [(item_id, getCosineSimilarity(feature_current, vec2)) for item_id, vec2 in feature_items.items()]
    id_sim = sorted(id_sim, key=lambda x: x[1], reverse=True)

    return [id_sim[i][0] for i in range(K + 1)][1:]


def getUserMostLike(uid):
    '''
    Get user(uid) mostly like movie
    feature_user * feature_movie

    Args:
        model: net model
        uid: target user's id

    return:
        the biggest rank movie id

    getUserMostLike(uid): 获取用户id为uid的用户最喜欢的电影

过程也很容易理解：

依次对uid对应的用户特征和所有电影特征做一个点积操作
该点击操作视为用户对电影的评分，对这些评分做一个sort操作
返回评分最高的即可。
    '''

    # user_movie_ids = pkl.load(open('Params/user_movie_dict.pkl','rb'))
    #
    # assert uid in user_movie_ids['user'], 'Expect user whose id is uid exists, but get None'
    #
    # movie_dict = user_movie_ids['movie']
    # user_dict = user_movie_ids['user']
    # user_dict[uid]['uid']=user_dict[uid]['uid'].view(1,1,1)
    # user_dict[uid]['gender'] = user_dict[uid]['gender'].view(1,1, 1)
    # user_dict[uid]['age'] = user_dict[uid]['age'].view(1,1, 1)
    # user_dict[uid]['job'] = user_dict[uid]['uid'].view(1,1, 1)
    # mid_rank={}
    #
    # # Step 1. Go through net to get user_movie score
    # with torch.no_grad():
    #     for mid in movie_dict.keys():
    #         movie_dict[mid]['mid']=movie_dict[mid]['mid'].view(1,1,1)
    #         movie_dict[mid]['mtype']=movie_dict[mid]['mtype'].view(1,-1)
    #         movie_dict[mid]['mtext']=movie_dict[mid]['mtext'].view(1,-1)
    #         movie_inputs = movie_dict[mid]
    #         user_inputs = user_dict[uid]
    #
    #         rank, _, _ = model(user_inputs,movie_inputs)
    #
    #         if mid not in mid_rank.keys():
    #             mid_rank[mid]=rank.item()

    # 模型训练后得到的用户和电影特征向量
    feature_data = pkl.load(open('Params/feature_data.pkl', 'rb'))
    user_movie_ids = pkl.load(open('Params/user_movie_dict.pkl', 'rb'))
    assert uid in user_movie_ids['user'], \
        'Expect user whose id is uid exists, but get None'
    feature_user = feature_data['feature_user'][uid]

    movie_dict = user_movie_ids['movie']
    mid_rank = {}
    for mid in movie_dict.keys():
        feature_movie = feature_data['feature_movie'][mid]
        rank = np.dot(feature_user, feature_movie.T)
        if mid not in mid_rank.keys():
            mid_rank[mid] = rank.item()

    mid_rank = [(mid, rank) for mid, rank in mid_rank.items()]
    mids = [mid[0] for mid in sorted(mid_rank, key=lambda x: x[1], reverse=True)]

    return mids[0]
