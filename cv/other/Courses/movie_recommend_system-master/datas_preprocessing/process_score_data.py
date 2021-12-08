import csv
import numpy as np

score_data_path = 'D:\MLProject\Courses\movie_recommend_system-master\datas\scores_data_copy1.csv'
score_with_id_data_path = 'D:\MLProject\Courses\movie_recommend_system-master\datas\scores_with_id_data.csv'

IS_TEST = False


# 读取原始评分数据
def read_original_score_data():
    origin_score_data_list = []
    with open(score_data_path, 'r', newline='') as movie_data_file:
        csv_reader = csv.reader(movie_data_file)
        for i, row in enumerate(csv_reader, 0):
            if i >= 1:
                # print(row)
                origin_score_data_list.append(row)
    return origin_score_data_list


# 缩减ID
def shrink_index(list, col_id):
    # 通过这个影射获取缩减后的id对应的原来的ID
    id_dict = {}
    a = np.array(list)
    # 按第i列的大小排序
    sorted_by_mid = a[np.lexsort(a.T[col_id, None])]
    new_list = []
    shrunk_id = 0
    last_origin_id = sorted_by_mid[0][col_id]
    for one in sorted_by_mid:
        # 原来的id会有重复，当原来的id改变时，缩减的id+1
        if one[col_id] != last_origin_id:
            shrunk_id += 1
            last_origin_id = one[col_id]

        new_list.append(np.insert(one, col_id, shrunk_id).tolist())

    return new_list


def create_movies_data():
    # python2可以用file替代open
    with open(score_with_id_data_path, 'w', newline='') as movie_data_file:
        writer = csv.writer(movie_data_file)
        # 先写columns_name
        writer.writerow(['user id', 'origin user id', "movie id", 'origin movie id', 'score'])


def write_score_with_id_data(score_data_list):
    # 写入多行用writerows
    with open(score_with_id_data_path, 'a+', newline='') as movie_data_file:
        writer = csv.writer(movie_data_file)
        writer.writerows(score_data_list)


# 从原始评分数据收缩ID，并且将数据存放到文件中
def shrink_id_in_score_2_file():
    score_data_list = read_original_score_data()
    score_data_list = shrink_index(score_data_list, 0)
    score_data_list = shrink_index(score_data_list, 2)
    if IS_TEST:
        for one in score_data_list:
            print(one)
    create_movies_data()
    write_score_with_id_data(score_data_list)
    print('压缩id完成')


# 从文件中读取缩放id后的评分数据
# 同时返回了缩减后的ID与原始ID的映射
# 读取的表行的结构为
# user id,origin user id,movie id,origin movie id,score
def get_score_data():
    score_data_list = []
    # 缩减后的ID到原始ID的映射
    user_id_dict = {}
    movie_id_dict = {}
    with open(score_with_id_data_path, 'r', newline='') as score_with_id_data_file:
        csv_reader = csv.reader(score_with_id_data_file)
        for i, row in enumerate(csv_reader, 0):
            if i > 0:
                if int(row[0]) > 6000:
                    continue
                # print(row)
                score_data_list.append(row)
                user_id_dict[row[0]] = row[1]
                movie_id_dict[row[2]] = row[3]
    return score_data_list, user_id_dict, movie_id_dict


# 处理用户评分数据，主要是缩减ID，得到缩减后的ID与原始ID的映射dict
if __name__ == '__main__':
    # score_data_list, user_id_dict, movie_id_dict = read_score_data_by_file()
    # print(score_data_list)
    # print(user_id_dict)
    # print(movie_id_dict)
    IS_TEST = True
    shrink_id_in_score_2_file()
