import csv

worm_log_path = 'D:\MLProject\worm\worm_log.txt'
movies_data_path = 'D:\MLProject\worm\datas\movies_data.csv'
scores_data_path = 'D:\MLProject\worm\datas\scores_data.csv'
movies_data_path_copy = 'D:\MLProject\worm\datas\movies_data_copy.csv'


def write_log(movie_page, movie_item):
    print('写入爬取位置记录', movie_page, movie_item)
    f = open(worm_log_path, 'w')
    f.writelines([str(movie_page) + " " + str(movie_item)])


def read_log():
    global worm_log_path
    with open(worm_log_path, "r") as f:
        # 为a+模式时，因为为追加模式，指针已经移到文尾，读出来的是一个空字符串。
        line = f.readline()  # 也是一次性读全部，但每一行作为一个子句存入一个列表
        split = line.split(' ')
        return int(split[0]), int(split[1])


def create_scores_data():
    # python2可以用file替代open
    with open(scores_data_path, 'w', newline='') as movie_data_file:
        writer = csv.writer(movie_data_file)
        # 先写columns_name
        writer.writerow(["user_id", "movie_id", "score"])


def create_movies_data():
    # python2可以用file替代open
    with open(movies_data_path, 'w', newline='') as movie_data_file:
        writer = csv.writer(movie_data_file)
        # 先写columns_name
        writer.writerow(["id", "name", "type"])


def add_movie_data(movie_data_list):
    # 写入多行用writerows
    with open(movies_data_path, 'a+', newline='') as movie_data_file:
        writer = csv.writer(movie_data_file)
        writer.writerows(movie_data_list)


def add_score_data(score_data_list):
    # 写入多行用writerows
    with open(scores_data_path, 'a+', newline='') as movie_data_file:
        writer = csv.writer(movie_data_file)
        writer.writerows(score_data_list)


# 读取电影信息，返回一个dict，key为id，便于用id获取电影信息
def read_movie_data():
    movies_data_dict = {}
    with open(movies_data_path_copy, 'r', newline='') as movie_data_file:
        csv_reader = csv.reader(movie_data_file)
        for i, row in enumerate(csv_reader, 0):
            if i >= 1:
                movies_data_dict[row[0]] = row
    return movies_data_dict


def read_user_data():
    pass


if __name__ == '__main__':
    print(read_movie_data())
    print(read_log())
    add_movie_data([[1, 2, 2]])
