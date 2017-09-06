from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

#读取数据
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
kdd_data_10percent = pandas.read_csv("kddcup.data_10_percent.csv", header=None, names = col_names)
kdd_data_10percent.describe()

#数据预处理：1.随机抽样
X_sample=kdd_data_10percent.sample(1000)


num_features = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"]
X_raw=X_sample[num_features].astype(float)
y = X_sample['label'].copy()
y[y!='normal.'] = '1'
y[y=='normal.'] = '0'
#数据预处理：2.归一化
MinMaxScaler=MinMaxScaler()
X_Scale=MinMaxScaler.fit_transform(X_raw)




#RF
clf_1 = RandomForestClassifier(min_samples_split=5, bootstrap=False,max_depth=None,criterion='entropy',n_estimators=56,max_features=6)
train_sizes,train_loss,test_loss=learning_curve(clf_1,X_Scale,y,cv=10,
scoring='accuracy',train_sizes=[0.1,0.33,0.55,0.78,1])

train_loss_mean_1=np.mean(train_loss,axis=1)
train_loss_std_1=np.std(train_loss,axis=1)
test_loss_mean_1=np.mean(test_loss,axis=1)
test_loss_std_1=np.std(test_loss,axis=1)
#Xgboost
clf_2 = XGBClassifier(gamma=0.10000000000000001,
                      max_depth=3,
                      colsample_bytree=0.5,
                      min_child_weight=4,
                      subsample=0.89999999999999991,
                      learning_rate=0.16999999999999998)
train_sizes,train_loss,test_loss=learning_curve(clf_2,X_Scale,y,cv=10,
    scoring='accuracy',train_sizes=[0.1,0.33,0.55,0.78,1])
train_loss_mean_2=np.mean(train_loss,axis=1)
train_loss_std_2=np.std(train_loss,axis=1)
test_loss_mean_2=np.mean(test_loss,axis=1)
test_loss_std_2=np.std(test_loss,axis=1)


plt.ylim(0.96,1.01)
plt.fill_between(train_sizes,train_loss_mean_1-train_loss_std_1,
                 train_loss_mean_1 +train_loss_std_2,alpha=0.1,color='r')
plt.fill_between(train_sizes,test_loss_mean_1-test_loss_std_1,
                 test_loss_mean_1 +test_loss_std_1,alpha=0.1,color='g')
plt.plot(train_sizes,train_loss_mean_1,'o-',color="r",label="RF_train")
plt.plot(train_sizes,test_loss_mean_1,'o-',color="g",label="RF_cv")
# plt.fill_between(train_sizes,train_loss_mean_2-train_loss_std_2,
#                  train_loss_mean_2 +train_loss_std_2,alpha=0.1,color='r')
# plt.fill_between(train_sizes,test_loss_mean_2-test_loss_std_2,
#                  test_loss_mean_2 +test_loss_std_2,alpha=0.1,color='g')
plt.plot(train_sizes,train_loss_mean_2,'o-',color="c",label="XGBOOST_train")
plt.plot(train_sizes,test_loss_mean_2,'o-',color="b",label="XGBOOST_cv")
# # plt.plot(train_sizes,train_loss_mean_3,'o-',color="m",label="K-Means_train")
# # plt.plot(train_sizes,test_loss_mean_3,'o-',color="y",label="K-Means_cv")
# # plt.plot(train_sizes,train_loss_mean_4,'o-',color="k",label="KNN_train")
# plt.plot(train_sizes,test_loss_mean_4,'x-',color="g",label="KNN_cv")
plt.title("图像对比")
plt.xlabel("training examples")
plt.ylabel("loss")
plt.legend(loc="best")
plt.show()