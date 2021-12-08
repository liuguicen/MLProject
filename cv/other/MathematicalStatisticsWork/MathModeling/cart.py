from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import numpy as np
import csv

'''一、读取文件'''
# csv文件路径，放在当前py文件统一路径(存放路径自己选择)
# 测试集和训练集的存放路径
train_file_path = 'train.xlsx'
test_file_path = 'test.xlsx'
# pandas库读取训练集csv文件（训练集1460行，测试集1459行，评分的时候要一致，所以训练集要删掉一行）
train_data = pd.read_excel(train_file_path)
f = csv.reader(open('bootout.csv','r',encoding='gbk'))
column_headers =[i[0] for i in f]
# pandas库读取测试集csv文件
test_data = pd.read_excel(test_file_path)

'''二、确认预测特征变量和选择要训练的特征'''
# 确定要预测的特征变量（标签）
train_y = train_data.preloss
# print(train_y)
# 要训练的特征列表,LotArea:占地面积;OverallQual:整体的材料和成品质量;YearBuilt:最初施工日期;TotRmsAbvGrd:房间的总数(不包含浴室)
predictor_cols = column_headers
# 要训练的真正的数据
train_X = train_data[predictor_cols]
# 删除有缺失的行(这里选取的特征列表都没有缺失)
train_X = train_X.dropna(axis=0)
#print(train_X)
#print(train_y)


'''三、创建模型和训练'''
# 创建随机森林模型
my_model = RandomForestRegressor(n_estimators=1000)
# 把要训练的数据丢进去，进行模型训练
my_model.fit(train_X,train_y)
print(train_y)
print("=="*20)

'''四、用测试集预测房价'''
test_X = test_data[predictor_cols]
test_y = test_data.preloss.apply(lambda x:round(x,1))
print(np.isnan(test_X).any())
predicted_prices = my_model.predict(test_X).round(1)
print(predicted_prices)

plt.plot(range(len(predicted_prices)), predicted_prices,'b', label='pre_loss')
plt.plot(range(len(predicted_prices)), test_y, 'r', label='true_loss')
# plt.plot(self.x_seed, self.func(self.x_seed), 'bo', label='seed')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('随机深林预测RON损失.png', dpi=500)
plt.show()
plt.close()

# print("="*20)
# prelist = [0.32,21.8,2.7,640]
# prelist = np.array(prelist).reshape(1,-1)
# pre = my_model.predict(prelist)
# print(pre)
# print("*"*10)

'''五、使用(RMSE)均方对数误差是做评价指标'''
# print("*"*20)
# print(predicted_prices.shape)
# print(train_y.shape)
print(metrics.mean_squared_log_error(predicted_prices, test_y))


'''六、把预测的值按照格式保存为csv文件'''
my_submission = pd.DataFrame({'Id': test_data.id, 'preloss': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
