from model import rec_model
from dataset import MovieRankDataset
from douban_dataset import DoubanMovieRankDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
from datas_preprocessing import file_util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# --------------- hyper-parameters------------------
user_max_dict = {
    'uid': 6041,  # 6040 users
    'gender': 2,
    'age': 7,
    'job': 21
}

movie_max_dict = {
    'mid': 3953,  # 3952 movies
    'mtype': 18,
    'mword': 5215  # 5215 words
}

convParams = {
    'kernel_sizes': [2, 3, 4, 5]
}


def train(model, datasets, num_epochs=2, lr=0.0001):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataloader = DataLoader(datasets, batch_size=256, shuffle=True)

    losses = []
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        loss_all = 0
        for i_batch, sample_batch in enumerate(dataloader):

            user_inputs = sample_batch['user_inputs']
            movie_inputs = sample_batch['movie_inputs']
            target = sample_batch['target'].to(device)

            model.zero_grad()

            tag_rank, _, _ = model(user_inputs, movie_inputs)

            loss = loss_function(tag_rank, target)
            if i_batch % 20 == 0:
                writer.add_scalar('epoch%d data/loss' % epoch, loss, i_batch * 20)
                print(loss)
            # if i_batch > 20:
            #     break
            loss_all += loss
            loss.backward()
            optimizer.step()
        print('Epoch {}:\t loss:{}'.format(epoch, loss_all))
    writer.export_scalars_to_json("./test.json")
    writer.close()


def get_user_like(user_id):
    print('\n\n用户', datasets.user_id_dict[str(user_id)], '最喜欢的电影为')
    like = getUserMostLike(user_id, feature_data, dict_user_movie)
    print(movie_dict[datasets.movie_id_dict[str(like)]])


if __name__ == '__main__':
    model = rec_model(user_max_dict=user_max_dict, movie_max_dict=movie_max_dict, convParams=convParams)
    model = model.to(device)

    # train model

    # datasets = MovieRankDataset(pkl_file='data.p')
    datasets = DoubanMovieRankDataset()

    train(model=model, datasets=datasets, num_epochs=5)
    torch.save(model.state_dict(), 'Params/model_params.pkl')

    # get user and movie feature
    # model.load_state_dict(torch.load('Params/model_params.pkl'))
    from recInterface import saveMovieAndUserFeature

    feature_data, dict_user_movie = saveMovieAndUserFeature(model=model, datasets=datasets)

    # test recsys
    from recInterface import getKNNitem, getUserMostLike

    movie_id, user_id = 3, 20
    origin_movie_id = datasets.movie_id_dict[str(movie_id)]
    movie_dict = file_util.read_movie_data()
    print('查询电影', movie_dict[origin_movie_id], '的临近电影')
    knn_item = getKNNitem(itemID=movie_id, K=10, feature_data=feature_data)
    knn_list = []
    for item in knn_item:
        origin_id = datasets.movie_id_dict[str(item)]
        print(movie_dict[origin_id])

    for i in range(10):
        get_user_like(user_id + i)
