from torch.utils.data import Dataset
import pickle as pkl
import torch
from pandas import DataFrame

from datas_preprocessing import vectorize_feature, process_score_data


class DoubanMovieRankDataset(Dataset):
    def __init__(self):
        self.movies_dict = vectorize_feature.get_movies_data()
        self.score_list, self.user_id_dict, self.movie_id_dict = process_score_data.get_score_data()

    def __len__(self):
        return len(self.score_list)

    def __getitem__(self, idx):
        # user data
        # score_list的行结构
        # user id,origin user id,movie id,origin movie id,score
        uid = int(self.score_list[idx][0])
        print('输入用户', uid)
        gender = 0
        age = 1
        job = 0

        # movie data
        # 缩减后的movie_id
        mid = self.score_list[idx][2]
        # 变成原始的movie_id, 用原始的movie id 得到movie的信息
        oringin_mid = self.movie_id_dict[mid]
        movie_data = self.movies_dict[oringin_mid]
        # 先用到dict 中，再转型
        mid = int(mid)
        mtext = movie_data[1]
        mtype = movie_data[2]

        # torch.FloatTensor()向量组装成张量
        score = float(self.score_list[idx][4])
        rank = torch.FloatTensor([score])

        # 注意下面数据转换成了2维的
        user_inputs = {
            'uid': torch.LongTensor([uid]).view(1, -1),
            'gender': torch.LongTensor([gender]).view(1, -1),
            'age': torch.LongTensor([age]).view(1, -1),
            'job': torch.LongTensor([job]).view(1, -1)
        }

        movie_inputs = {
            'mid': torch.LongTensor([mid]).view(1, -1),
            'mtype': torch.LongTensor(mtype),
            'mtext': torch.LongTensor(mtext)
        }

        sample = {
            'user_inputs': user_inputs,
            'movie_inputs': movie_inputs,
            'target': rank
        }
        return sample
