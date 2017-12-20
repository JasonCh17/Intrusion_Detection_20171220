# 读取数据
import time
import pandas as pd  # 数据分析
import matplotlib.pyplot as plt
import numpy as np
col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
             "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
             "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
             "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
             "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
             "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
data = pd.read_csv("E:\Pycharm\Intrusion_Detection\kddcup.data_10_percent.csv", header=None, names=col_names)

# 去重
# import matplotlib.pyplot as plt
# IsDuplicated=data.duplicated()
#
# IsDuplicated.value_counts().plot(kind='bar')
# plt.show()
data_1 = data.drop_duplicates()

# one-hot
dummies_protocol = pd.get_dummies(data_1["protocol_type"], prefix='protocol')
dummies_flag = pd.get_dummies(data_1["flag"], prefix='flag')
# dummies_service = pd.get_dummies(data_1["service"], prefix='service')
# data_2 = pd.concat([data_1, dummies_protocol,dummies_flag,dummies_service], axis=1)
data_2 = pd.concat([data_1, dummies_protocol, dummies_flag], axis=1)
# data_2
# 特征选择(by "A feature reduced intrusion detection system using ANN classifier",25个特征）
# 建立X,y
feature_selection = ["duration",
                     "dst_bytes", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                     "logged_in", "num_compromised", "root_shell", "su_attempted", "num_access_files",
                     "num_outbound_cmds", "is_guest_login", "rerror_rate", "srv_diff_host_rate",
                     "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
                     "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                     "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
                     "protocol_icmp", "protocol_tcp", "protocol_udp",
                     "flag_OTH", "flag_REJ", "flag_RSTO", "flag_RSTOS0", "flag_RSTR",
                     "flag_S0", "flag_S1", "flag_S2", "flag_S3", "flag_SF", "flag_SH"]
X_3 = data_2[feature_selection]
y_3 = data_2['label'].copy()  # 一维
##y的处理
u2r = ["buffer_overflow.", "loadmodule.", "perl.", "rootkit."]
r2l = ["ftp_write.", "imap.", "guess_passwd.", "phf.", "spy.", "multihop.", "warezmaster.", "warezclient."]
dos = ["back.", "land.", "pod.", "neptune.", "smurf.", "teardrop."]
probe = ["satan.", "portsweep.", "ipsweep.", "nmap."]
for i in u2r:
    y_3[y_3 == i] = 'U2R'  # u2r
for i in r2l:
    y_3[y_3 == i] = 'R2L'  # r2l
for i in dos:
    y_3[y_3 == i] = 'DOS'  # dos
for i in probe:
    y_3[y_3 == i] = 'Probing'  # probe
y_3[y_3 == "normal."] = 'Normal'  # normal
y_3 = np.array(y_3)  # 变成array格式，一维
classes=['Normal','Probing','DOS','U2R','R2L']
colors=['blue','red','y','m','g']
#欠采样 ENN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import EditedNearestNeighbours
params=[3,6,9,12,15,18]
for i in params:
    oversampler = EditedNearestNeighbours(random_state=42, n_neighbors=i)
    X_enn, y_e = oversampler.fit_sample(X_3, y_3)
#标准化
    scaler=StandardScaler().fit(X_enn)
    X_e=scaler.transform(X_enn)
    #可视化
    X_train_2,X_test_2,y_train_2,y_test_2=train_test_split(X_e,y_e,test_size=0.2,random_state=0)  #切分样本
    X_embedded = TSNE(n_components=2).fit_transform(X_test_2)
    plt.figure()
    plt.title("ENN")
    for index,label,color in zip(range(len(classes)),classes,colors):
        plt.scatter(X_embedded[y_test_2==label,0],X_embedded[y_test_2==label,1],label=classes[index],c=color)
    plt.legend(loc='best')
plt.show()