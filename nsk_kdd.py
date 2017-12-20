# 读取数据

import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","learners"]
data_1= pd.read_csv("E:\Pycharm\Intrusion_Detection\KDDTrain+.csv",  header=None,names = col_names)
data_1.head()
y_1=data_1['label'].copy()   #一维
##y的处理
u2r=["buffer_overflow","loadmodule","perl","rootkit"]
r2l=["ftp_write","imap","guess_passwd","phf","spy","multihop","warezmaster","warezclient"]
dos=["back","land","pod","smurf","teardrop",'neptune']
probe=["satan","portsweep","ipsweep","nmap"]
for i in u2r:
    y_1[y_1==i]="u2r"
for i in r2l:
    y_1[y_1==i]="r2l"
for i in dos:
    y_1[y_1==i]="dos"
for i in probe:
    y_1[y_1==i]="probe"
y_1[y_1=="normal."]="normal"
print(y_1.value_counts())
#去重
# import matplotlib.pyplot as plt
# IsDuplicated=data.duplicated()
#
# IsDuplicated.value_counts().plot(kind='bar')
# plt.show()
# data_1=data.drop_duplicates()

#one-hot
dummies_protocol = pd.get_dummies(data_1["protocol_type"], prefix='protocol')
dummies_flag = pd.get_dummies(data_1["flag"], prefix='flag')
# dummies_service = pd.get_dummies(data_1["service"], prefix='service')
# data_2 = pd.concat([data_1, dummies_protocol,dummies_flag,dummies_service], axis=1)
data_2 = pd.concat([data_1, dummies_protocol,dummies_flag], axis=1)
# data_2
#特征选择(by "A feature reduced intrusion detection system using ANN classifier",25个特征）
#建立X,y
feature_selection=["duration",
    "dst_bytes","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_access_files",
    "num_outbound_cmds","is_guest_login","rerror_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "protocol_icmp","protocol_tcp","protocol_udp",
    "flag_OTH","flag_REJ","flag_RSTO","flag_RSTOS0","flag_RSTR",
    "flag_S0","flag_S1","flag_S2","flag_S3","flag_SF","flag_SH"]
X_3=data_2[feature_selection]
y_3=data_2['label'].copy()   #一维
#baseline
#标准化
from sklearn.preprocessing import StandardScaler
scaler_base=StandardScaler().fit(X_3)
X=scaler_base.transform(X_3)  #X是ndarray
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_3,y_3,test_size=0.1,random_state=0)  #切分样本
#
#
# y_c=y_3.copy()
# y_c[y_1=='normal']='b'
# y_c[y_1=='probe']='r'
# y_c[y_1=='dos']='y'
# y_c[y_1=='u2r']='m'
# y_c[y_1=='r2l']='g'
# # print(y_c.value_counts())
#
# label=["normal","dos","probe","r2l","u2r"]
# from sklearn.manifold import TSNE
# X_embedded = TSNE(n_components=3).fit_transform(X_3)
# import matplotlib.pyplot as plt
# plt.figure()
#
# plt.scatter(X_embedded[:,0],X_embedded[:,1],c=y_c,marker='o')
#
# plt.show()

from sklearn.ensemble import RandomForestClassifier
##建立模型
# clf_1 = LogisticRegression(random_state=1)
clf_1 = RandomForestClassifier(n_estimators=100)
from sklearn.model_selection import cross_val_score
cs=cross_val_score(clf_1,X_3, y_3)
print(cs)
print(np.mean(cs))