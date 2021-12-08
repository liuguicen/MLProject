import csv
import re

MOVIE_TYPE_LEN = 18
MOVIE_TYPE_SUPP = 16
MOVIE_NAME_LEN = 15
MOVIE_NAME_SUPP = 3422
IS_TEST = False
# type_dict = {
#     '剧情': 0,
#     '喜剧': 1,
#     '动作': 2,
#     '爱情': 3,
#     '科幻': 4,
#     '动画': 5,
#     '悬疑': 6,
#     '惊悚': 7,
#     '恐怖': 8,
#     '犯罪': 9,
#     '同性': 10,
#     '音乐': 11,
#     '歌舞': 12,
#     '传记': 13,
#     '历史': 14,
#     '战争': 15,
#     '西部': 16,
#     '奇幻': 17,
#     '冒险': 18,
#     '灾难': 19,
#     '武侠': 20,
#     '情色': 21,
#     '纪录片': 22,
#     '运动':23
# }
# 将相关数据向量化，包括电影名，电影类型等
movie_data_path = 'D:\MLProject\Courses\movie_recommend_system-master\datas\movies_data_copy.csv'


def split_movie_name(name):
    return re.split('[ :\']+', name)


# 读取电影数据，并且将电影名、类型向量化
def split_movie_type(stype):
    split = re.split('[ ]+', stype)
    return [one.strip() for one in split if len(one.strip()) >= 1]


# 将列表中某一列非向量的数据向量化，如单词
# 将单词列表转换成向量列表，向量值从0 ~ n-1与单词一一对应
def vectorize_column(m_list, col, row_len, supp_number):
    m_dict = {}
    for row in m_list:
        vec = []
        # 利用dict缩减id大小
        for one in row[col]:
            if not (one in m_dict):
                cur_len = len(m_dict)
                m_dict[one] = cur_len
                vec.append(cur_len)
            else:
                vec.append(m_dict[one])
        # 补足长度，让向量长度一致
        while len(vec) < row_len:
            vec.append(supp_number)
        row[col] = vec


# 主函数，从原始电影数据读取电影数据，返回向量化后的电影数据
def red_and_vec_movie_feature():
    movie_data_list = []
    with open(movie_data_path, 'r') as movie_data_file:
        csv_reader = csv.reader(movie_data_file)
        for i, row in enumerate(csv_reader, 0):
            if i > 1 and i % 2 == 0:
                if IS_TEST:
                    print(row)
                row[0] = [row[0]]
                row[1] = split_movie_name(row[1])
                row[2] = split_movie_type(row[2])
                if IS_TEST:
                    print('after split: \n', row)
                movie_data_list.append(row)
        if IS_TEST:
            print('向量化之前\n', movie_data_list)
        # 向量化电影名
        vectorize_column(movie_data_list, 1, MOVIE_NAME_LEN, MOVIE_NAME_SUPP)
        # 向量化电影类型
        vectorize_column(movie_data_list, 2, MOVIE_NAME_LEN, MOVIE_NAME_SUPP)
        if IS_TEST:
            print('向量化之后\n', movie_data_list)
    return movie_data_list


# 从原始电影数据读取电影数据，变成向量化后的电影数据
# 返回一个dict，key为id，便于用id获取电影信息
def get_movies_data():
    movie_data_list = red_and_vec_movie_feature()
    movies_dict = {}
    for movie in movie_data_list:
        movies_dict[movie[0][0]] = movie
    return movies_dict


if __name__ == '__main__':
    IS_TEST = True
    print(get_movies_data())
