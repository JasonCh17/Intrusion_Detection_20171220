# 读取数据
import pandas as pd #数据分析

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
data= pd.read_csv("E:\Pycharm\Intrusion_Detection\kddcup.data_10_percent.csv",  header=None,names = col_names)

#去重
# import matplotlib.pyplot as plt
# IsDuplicated=data.duplicated()
#
# IsDuplicated.value_counts().plot(kind='bar')
# plt.show()
data_1=data.drop_duplicates()

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
# feature_selection=["duration","src_bytes",
#     "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
#     "logged_in","num_compromised","root_shell","su_attempted","num_root",
#     "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
#     "is_host_login","is_guest_login","count","srv_count","serror_rate",
#     "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
#     "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
#     "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
#     "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
#     "dst_host_rerror_rate","dst_host_srv_rerror_rate","protocol_icmp","protocol_tcp","protocol_udp",
#     "flag_OTH","flag_REJ","flag_RSTO","flag_RSTOS0","flag_RSTR","flag_S0","flag_S1","flag_S2","flag_S3","flag_SF","flag_SH"]
X_3=data_2[feature_selection]
y_3=data_1['label'].copy()   #一维

##y的处理
u2r=["buffer_overflow.","loadmodule.","perl.","rootkit."]
r2l=["ftp_write.","imap.","guess_passwd.","phf.","spy.","multihop.","warezmaster.","warezclient."]
dos=["back.","land.","pod.","neptune.","smurf.","teardrop."]
probe=["satan.","portsweep.","ipsweep.","nmap."]
for i in u2r:
    y_3[y_3==i]="u2r"
for i in r2l:
    y_3[y_3==i]="r2l"
for i in dos:
    y_3[y_3==i]="dos"
for i in probe:
    y_3[y_3==i]="probe"
y_3[y_3=="normal."]="normal"





#过采样smote
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=42)
X_smote,y=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_smote)
X=scaler.transform(X_smote)  #X是ndarray
#分类器
from sklearn.ensemble import RandomForestClassifier
##建立模型
clf_1 = RandomForestClassifier()

# #分类器
# from sklearn.ensemble import ExtraTreesClassifier
# ##建立模型
# clf_1 = ExtraTreesClassifier()

#划分数据集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#学习曲线
import numpy as np
from sklearn.model_selection import learning_curve
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_1,
                                X=X_train,y=y_train,
                                train_sizes=np.linspace(0.01,1,10),
                                 cv=10, n_jobs=1,random_state=0)
train_mean_1=np.mean(train_scores,axis=1)
test_mean_1=np.mean(test_scores,axis=1)
train_std_1=np.std(train_scores,axis=1)
test_std_1=np.std(train_scores,axis=1)






#欠采样ENN
from imblearn.under_sampling import EditedNearestNeighbours
oversampler=EditedNearestNeighbours(random_state=42)
X_enn,y=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_enn)
X=scaler.transform(X_enn)

#划分数据集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#学习曲线
import numpy as np
from sklearn.model_selection import learning_curve
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_1,
                                X=X_train,y=y_train,
                                train_sizes=np.linspace(0.01,1,10),
                                 cv=10, n_jobs=1,random_state=0)
train_mean_2=np.mean(train_scores,axis=1)
test_mean_2=np.mean(test_scores,axis=1)
train_std_2=np.std(train_scores,axis=1)
test_std_2=np.std(train_scores,axis=1)



#过采样+欠采样
from imblearn.combine import SMOTEENN
oversampler=SMOTEENN(random_state=42)
X_4,y=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_4)
X=scaler.transform(X_4)  #X是ndarray

#划分数据集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#学习曲线
import numpy as np
from sklearn.model_selection import learning_curve
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_1,
                                X=X_train,y=y_train,
                                train_sizes=np.linspace(0.01,1,10),
                                 cv=10, n_jobs=1,random_state=0)
train_mean_3=np.mean(train_scores,axis=1)
test_mean_3=np.mean(test_scores,axis=1)
train_std_3=np.std(train_scores,axis=1)
test_std_3=np.std(train_scores,axis=1)






import matplotlib.pyplot as plt

plt.figure()
plt.grid()
plt.subplot(131)
plt.plot(train_sizes,train_mean_1,color='blue',marker='o',markersize=5,
         label=' smote training accuracy')
plt.fill_between(train_sizes,train_mean_1+train_std_1,train_mean_1-train_std_1,
         color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean_1,color='green',linestyle='--',
         marker='s',markersize=5)
plt.fill_between(train_sizes,test_mean_1+test_std_1,test_mean_1-test_std_1,color='green',alpha=0.25)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.subplot(132)
plt.plot(train_sizes,train_mean_2,color='blue',marker='o',markersize=5,
         label=' Enn training accuracy')
plt.fill_between(train_sizes,train_mean_2+train_std_2,train_mean_2-train_std_2,
         color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean_2,color='green',linestyle='--',
         marker='s',markersize=5,
         label='Enn')
plt.fill_between(train_sizes,test_mean_2+test_std_2,test_mean_2-test_std_2,
         color='green',alpha=0.25)

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.subplot(133)
plt.plot(train_sizes,train_mean_3,color='blue',marker='o',markersize=5,
         label=' smote+enn training accuracy')
plt.fill_between(train_sizes,train_mean_3+train_std_3,train_mean_3-train_std_3,
         color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean_3,color='green',linestyle='--',
         marker='s',markersize=5,
         label='smote+enn')
plt.fill_between(train_sizes,test_mean_3+test_std_3,test_mean_3-test_std_3,color='green',alpha=0.25)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()

#三种采样的比较
plt.figure()
plt.grid()
# plt.plot(train_sizes,train_mean_1,color='blue',marker='o',markersize=5,
#          label='smote training accuracy')
plt.plot(train_sizes,test_mean_1,color='green',linestyle='--',
         marker='s',markersize=5,
         label='smote validation accuracy')
# plt.fill_between(train_sizes,train_mean_1+train_std_1,train_mean_1-train_std_1,
#          color='blue',alpha=0.25)
plt.fill_between(train_sizes,test_mean_1+test_std_1,test_mean_1-test_std_1,
         color='green',alpha=0.25)
#
# plt.plot(train_sizes,train_mean_2,color='r',marker='o',markersize=5,
#          label=' Enn training accuracy')
plt.plot(train_sizes,test_mean_2,color='y',linestyle='--',
         marker='s',markersize=5,
         label=' Enn validation accuracy')
# plt.fill_between(train_sizes,train_mean_2+train_std_2,train_mean_2-train_std_2,
#          color='blue',alpha=0.25)
plt.fill_between(train_sizes,test_mean_2+test_std_2,test_mean_2-test_std_2,
         color='green',alpha=0.25)

# plt.plot(train_sizes,train_mean_3,color='c',marker='o',markersize=5,
#          label=' smote+enn training accuracy')
plt.plot(train_sizes,test_mean_3,color='m',linestyle='--',
         marker='s',markersize=5,
         label=' smote+enn validation accuracy')
# plt.fill_between(train_sizes,train_mean_3+train_std_3,train_mean_3-train_std_3,
#          color='blue',alpha=0.25)
plt.fill_between(train_sizes,test_mean_3+test_std_3,test_mean_3-test_std_3,
         color='green',alpha=0.25)


plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()