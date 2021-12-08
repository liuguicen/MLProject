import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer


'''一、读取文件'''
# csv文件路径，放在当前py文件统一路径(存放路径自己选择)
# 测试集和训练集的存放路径
train_file_path = 'train.xlsx'
test_file_path = 'test.xlsx'
# pandas库读取训练集csv文件（训练集1460行，测试集1459行，评分的时候要一致，所以训练集要删掉一行）
train_data = pd.read_excel(train_file_path)
column_headers = list(train_data.columns.values)[1:10]+list(train_data.columns.values)[11:]
# pandas库读取测试集csv文件
test_data = pd.read_excel(test_file_path)

'''二、确认预测特征变量和选择要训练的特征'''
# 确定要预测的特征变量（标签）
train_y = train_data.preloss.apply(lambda x:round(x,1))
# print(train_y)
# 要训练的特征列表,LotArea:占地面积;OverallQual:整体的材料和成品质量;YearBuilt:最初施工日期;TotRmsAbvGrd:房间的总数(不包含浴室)
predictor_cols = column_headers
# 要训练的真正的数据
train_X = train_data[predictor_cols]
# 删除有缺失的行(这里选取的特征列表都没有缺失)
train_X = train_X.dropna(axis=0)


# XGBoost训练过程
model = xgb.XGBRegressor(max_depth=20, learning_rate=0.6, n_estimators=300, objective='reg:gamma')
model.fit(train_X, train_y)

# 对测试集进行预测
test_X = test_data[predictor_cols]
test_y = test_data.preloss
ans = model.predict(test_X)
plt.figure(figsize=(6, 4))

plt.plot(range(len(ans)), ans,'b', label='pre_loss')
plt.plot(range(len(ans)), test_y, 'r', label='true_loss')
# plt.plot(self.x_seed, self.func(self.x_seed), 'bo', label='seed')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('XgBOOST.png', dpi=500)
plt.show()
plt.close()

# ans_len = len(ans)
# id_list = np.arange(10441, 17441)
# data_arr = []
# for row in range(0, ans_len):
#     data_arr.append([int(id_list[row]), ans[row]])
# np_data = np.array(data_arr)

# 写入文件
# pd_data = pd.DataFrame(np_data, columns=['id', 'y'])
# # print(pd_data)
# pd_data.to_csv('submit.csv', index=None)

my_submission = pd.DataFrame({'Id': test_data.id, 'preloss': ans})
# you could use any filename. We choose submission here
my_submission.to_csv('submit.csv', index=False)

# 显示重要特征
importance = model.get_booster().get_score()
imp_dic = dict(importance)
label_list = sorted(imp_dic.items(), key=lambda item:item[1], reverse=True)
label_list = label_list[:30]
label = [lab[0] for lab in label_list]
print(label_list)
# label = list(imp_dic.keys())
with open("bootout.csv","w",encoding='gbk') as f:
    for item in label:
        f.write(item+',')
        f.write(str(imp_dic[item]))
        f.write('\n')
    f.close()
# train_X = train_data[predictor_cols]
# train_X = np.array(train_X)
# list = train_X[133,:].array
# pre = model.predict(list)
# print(pre)
# print("---"*10)
# print(imp_dic.keys())
# print(imp_dic.values())
# print(imp_dic)
# print(importance)
# plot_importance(model)
# plt.show()


