# 读取数据

import pandas as pd #数据分析
import numpy as np #科学计算
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
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
data.head()
#去重
data_1=data.drop_duplicates()
# print(data_1.info())
#one-hot
dummies_protocol = pd.get_dummies(data_1["protocol_type"], prefix='protocol')
dummies_flag = pd.get_dummies(data_1["flag"], prefix='flag')
data_2 = pd.concat([data_1, dummies_protocol,dummies_flag], axis=1)
# data_2
#特征选择
#建立X,y
feature_selection=["duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","protocol_icmp","protocol_tcp","protocol_udp",
    "flag_OTH","flag_REJ","flag_RSTO","flag_RSTOS0","flag_RSTR","flag_S0","flag_S1","flag_S2","flag_S3","flag_SF","flag_SH"]
X_3=data_2[feature_selection]
y_3=data_2['label'].copy()   #一维
##y的处理
u2r=["buffer_overflow.","loadmodule.","perl.","rootkit."]
r2l=["ftp_write.","imap.","guess_passwd.","phf.","spy.","multihop.","warezmaster.","warezclient."]
dos=["back.","land.","pod.","neptune.","smurf.","teardrop."]
probe=["satan.","portsweep.","ipsweep.","nmap."]
# for i in u2r:
#     y_3[y_3==i]="u2r"
# for i in r2l:
#     y_3[y_3==i]="r2l"
# for i in dos:
#     y_3[y_3==i]="dos"
# for i in probe:
#     y_3[y_3==i]="probe"
# y_3[y_3=="normal."]="normal"

for i in u2r:
    y_3[y_3==i]=4 #u2r
for i in r2l:
    y_3[y_3==i]=3 #r2l
for i in dos:
    y_3[y_3==i]=1  #dos
for i in probe:
    y_3[y_3==i]=2 #probe
y_3[y_3=="normal."]=0 #normal
y_3=np.array(y_3)  #变成array格式，一维
#过采样 smote
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=42)
X_smote,y=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_smote)
X=scaler.transform(X_smote)  #X是ndarray
#可视化
y_smote=pd.DataFrame(y)
from sklearn.model_selection import train_test_split
X_train_1,X_test_1,y_train_1,y_test_1=train_test_split(X,y_smote,test_size=0.2,random_state=0)  #切分样本
#不同类别不同颜色
# y_c_1=y_test_1.copy()
# y_c_1[y_test_1=='normal']='b'
# y_c_1[y_test_1=='probe']='r'
# y_c_1[y_test_1=='dos']='y'
# y_c_1[y_test_1=='u2r']='m'
# y_c_1[y_test_1=='r2l']='g'
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X_test_1)
plt.figure()
plt.title("Smote")
for label,color in zip(range(len(classes)),colors):
    plt.scatter(X_embedded[y_test_1==label,0],
                X_embedded[y_test_1==label,1],
                label=classes[label],
                c=color)
plt.legend(loc='best')
# plt.scatter(X_embedded[:,0],X_embedded[:,1],c=y_c_1[:,0],marker='o')
plt.show()
