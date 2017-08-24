#10000个样本，Xgboost

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
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
X_sample_1=kdd_data_10percent.sample(10000)


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
X_raw_2=X_sample_1[num_features].astype(float)
y_2 = X_sample_1['label'].copy()
y_2[y_2!='normal.'] = '1'
y_2[y_2=='normal.'] = '0'
#数据预处理：2.归一化
MinMaxScaler=MinMaxScaler()
X_Scale_2=MinMaxScaler.fit_transform(X_raw_2)

# #随机采样式超参数优化方法：
# #用于报告超参数搜索的最好结果的函数
# def report(results,n_top=3):
#     for i in range(1,n_top + 1):
#         candidates=np.flatnonzero(results['rank_test_score']==i)
#         for candidate in candidates:
#             print("Model with rank:{0}".format(i))
#             print("Mean validation score:{0:.3f}±{1:.3f}".format(
#                 results['mean_test_score'][candidate],
#                 results['std_test_score'][candidate]))
#             print("Parameter:{0}".format(results['params'][candidate]))
#             print("")
#
#
# #构建分类器
# clf= XGBClassifier()
# #设置想要优化的超参数以及他们的取值分布
# param_dist={"max_depth":np.arange(3,10),
#             "gamma":np.arange(0.1,0.2,0.1),
#             "subsample":np.arange(0.5,1,0.1),
#             "learning_rate":np.arange(0.01,0.2,0.02),
#             "min_child_weight":np.arange(4,6,0.5),
#             "colsample_bytree":np.arange(0.5,1,0.1)}
# #开启超参数空间的随机搜索
# n_iter_search=20
# random_search=RandomizedSearchCV(clf,param_distributions=param_dist, n_iter=n_iter_search,error_score=0)  #,n_jobs=2
# start=time()
# random_search.fit(X_Scale_2,y_2)
# print("RandomizedSearchCV took %.2f seconds for %d candidates"
#       " parameter settings." %((time()-start),n_iter_search))
# report(random_search.cv_results_)

# Learning_curve

clf_2 = XGBClassifier(gamma=0.10000000000000001,
                      max_depth=3,
                      colsample_bytree=0.5,
                      min_child_weight=4,
                      subsample=0.89999999999999991,
                      learning_rate=0.16999999999999998)
train_sizes,train_loss,test_loss=learning_curve(clf_2,X_Scale_2,y_2,cv=10,
scoring='accuracy',train_sizes=[0.1,0.2,0.3,
                                0.4,0.5,0.6,0.7,0.8,0.90,1])
train_loss_mean_2=-np.mean(train_loss,axis=1)
test_loss_mean_2=-np.mean(test_loss,axis=1)
plt.title("Xgboost")
plt.plot(train_sizes,train_loss_mean_2,'o-',color="r",label="Training")
plt.plot(train_sizes,test_loss_mean_2,'o-',color="g",label="cross-validation")
plt.xlabel("training examples")
plt.ylabel("loss")
plt.legend(loc="best")
plt.show()