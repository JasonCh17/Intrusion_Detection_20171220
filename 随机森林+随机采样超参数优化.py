#10000个样本，RF

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy as np
from time import time
from scipy.stats import  randint as sp_randint
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
X_raw_1=X_sample_1[num_features].astype(float)
y_1 = X_sample_1['label'].copy()
y_1[y_1!='normal.'] = '1'
y_1[y_1=='normal.'] = '0'
#数据预处理：2.归一化
MinMaxScaler=MinMaxScaler()
X_Scale_1=MinMaxScaler.fit_transform(X_raw_1)

#随机采样式超参数优化方法：
#用于报告超参数搜索的最好结果的函数
def report(results,n_top=3):
    for i in range(1,n_top + 1):
        candidates=np.flatnonzero(results['rank_test_score']==i)
        for candidate in candidates:
            print("Model with rank:{0}".format(i))
            print("Mean validation score:{0:.3f}±{1:.3f}".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameter:{0}".format(results['params'][candidate]))
            print("")


#构建分类器
clf=RandomForestClassifier()
#设置想要优化的超参数以及他们的取值分布
param_dist={"max_depth":[3,None],
            "max_features":sp_randint(1,11),
            "min_samples_split":sp_randint(2,11),
            "bootstrap":[True,False],
            "criterion":["gini","entropy"],
            "n_estimators":sp_randint(10,100)}
#开启超参数空间的随机搜索
n_iter_search=20
random_search=RandomizedSearchCV(clf,param_distributions=param_dist, n_iter=n_iter_search,error_score=0)  #,n_jobs=2
start=time()
random_search.fit(X_Scale_1,y_1)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings."% ((time()-start),n_iter_search))
report(random_search.cv_results_)

# Learning_curve

clf_1 = RandomForestClassifier(min_samples_split=5, bootstrap=False,max_depth=None,criterion='entropy',n_estimators=56,max_features=6)
train_sizes,train_loss,test_loss=learning_curve(clf_1,X_Scale_1,y_1,cv=10,
scoring='accuracy',train_sizes=[0.1,0.2,0.3,
                                0.4,0.5,0.6,0.7,0.8,0.90,1])
train_loss_mean_1=-np.mean(train_loss,axis=1)
test_loss_mean_1=-np.mean(test_loss,axis=1)
plt.title("Random Forest")
plt.plot(train_sizes,train_loss_mean_1,'o-',color="r",label="Training")
plt.plot(train_sizes,test_loss_mean_1,'o-',color="g",label="cross-validation")
plt.xlabel("training examples")
plt.ylabel("loss")
plt.legend(loc="best")
plt.show()