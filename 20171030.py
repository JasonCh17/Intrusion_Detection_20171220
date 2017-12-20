# 读取数据
import time
import pandas as pd #数据分析
import matplotlib.pyplot as plt
import numpy as np
time_start=time.time()
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
X_3=data_2[feature_selection]
y_3=data_2['label'].copy()   #一维

# feature_selection=["duration","src_bytes",
#     "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
#     "logged_in","num_compromised","root_shell","su_attempted","num_root",
#     "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
#     "is_host_login","is_guest_login","count","srv_count","serror_rate",
#     "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
#     "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
#     "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
#     "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
#     "dst_host_rerror_rate","dst_host_srv_rerror_rate"]
# X_3=data_1[feature_selection]
# y_3=data_1['label'].copy()   #一维

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



# # 读取数据
# import time
# import pandas as pd #数据分析
# import matplotlib.pyplot as plt
# import numpy as np
# time_start=time.time()
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
data_test= pd.read_csv("E:\Pycharm\Intrusion_Detection\corrected.csv",  header=None,names = col_names)

#去重
# import matplotlib.pyplot as plt
# IsDuplicated=data.duplicated()
#
# IsDuplicated.value_counts().plot(kind='bar')
# plt.show()
data_test_1=data_test.drop_duplicates()

#one-hot
dummies_protocol = pd.get_dummies(data_test_1["protocol_type"], prefix='protocol')
dummies_flag = pd.get_dummies(data_test_1["flag"], prefix='flag')
# dummies_service = pd.get_dummies(data_1["service"], prefix='service')
# data_2 = pd.concat([data_1, dummies_protocol,dummies_flag,dummies_service], axis=1)
data_test_2 = pd.concat([data_test_1, dummies_protocol,dummies_flag], axis=1)
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
X_test_3=data_test_2[feature_selection]
y_test=data_test_2['label'].copy()   #一维

# feature_selection=["duration","src_bytes",
#     "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
#     "logged_in","num_compromised","root_shell","su_attempted","num_root",
#     "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
#     "is_host_login","is_guest_login","count","srv_count","serror_rate",
#     "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
#     "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
#     "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
#     "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
#     "dst_host_rerror_rate","dst_host_srv_rerror_rate"]
# X_3=data_1[feature_selection]
# y_3=data_1['label'].copy()   #一维

##y的处理
u2r=["buffer_overflow.","loadmodule.","perl.","rootkit.","httptunnel.","ps.","sqlattack.","xterm."]
r2l=["ftp_write.","imap.","guess_passwd.","phf.","spy.","multihop.","warezmaster.","warezclient.","named.","sendmail."
    ,"snmpgetattack.","snmpguess.","worm.","xlock.","xsnoop."]
dos=["back.","land.","pod.","neptune.","smurf.","teardrop.","apache2.","mailbomb.","processtable.","udpstorm."]
probe=["satan.","portsweep.","ipsweep.","nmap.","mscan.","saint."]
for i in u2r:
    y_test[y_test==i]="u2r"
for i in r2l:
    y_test[y_test==i]="r2l"
for i in dos:
    y_test[y_test==i]="dos"
for i in probe:
    y_test[y_test==i]="probe"
y_test[y_test=="normal."]="normal"



#混淆矩阵
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#########################################################################
#无处理
#标准化
from sklearn.preprocessing import StandardScaler
scaler_1=StandardScaler().fit(X_3)
X_1=scaler_1.transform(X_3)  #X是ndarray
#分类器
from sklearn.ensemble import RandomForestClassifier
##建立模型
clf_1 = RandomForestClassifier(oob_score=True)
clf_1.fit(X_1,y_3)
#testdata标准化
X_test=scaler_1.transform(X_test_3)  #X是ndarray
#Predition
preditions=clf_1.predict(X_test)
#########################################################################
#过采样 smote
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=42)
X_smote,y_smote=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler_smote=StandardScaler().fit(X_smote)
X_smote_1=scaler_smote.transform(X_smote)  #X是ndarray
#分类器
from sklearn.ensemble import RandomForestClassifier
##建立模型
clf_2 = RandomForestClassifier(oob_score=True)
clf_2.fit(X_smote_1,y_smote)
#testdata标准化
X_test_smote=scaler_smote.transform(X_test_3)  #X是ndarray
#Predition
preditions_smote=clf_2.predict(X_test_smote)
###########################################################################
#欠采样 ENN
from imblearn.under_sampling import EditedNearestNeighbours
oversampler=EditedNearestNeighbours(random_state=42)
X_enn,y_enn=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler_enn=StandardScaler().fit(X_enn)
X_enn_1=scaler_enn.transform(X_enn)
#分类器
from sklearn.ensemble import RandomForestClassifier
##建立模型
clf_3 = RandomForestClassifier(oob_score=True)
clf_3.fit(X_enn_1,y_enn)
#testdata标准化
X_test_enn=scaler_enn.transform(X_test_3)  #X是ndarray
#Predition
preditions_enn=clf_3.predict(X_test_enn)

###########################################################################
#过采样+欠采样 smote+enn
from imblearn.combine import SMOTEENN
oversampler=SMOTEENN(random_state=42)
X_SMOTEENN,y_smoteenn=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler_smoteenn=StandardScaler().fit(X_SMOTEENN)
X_smoteenn_1=scaler_smoteenn.transform(X_SMOTEENN)  #X是ndarray
#分类器
from sklearn.ensemble import RandomForestClassifier
##建立模型
clf_4 = RandomForestClassifier(oob_score=True)
clf_4.fit(X_smoteenn_1,y_smoteenn)
#testdata标准化
X_test_smoteenn=scaler_smoteenn.transform(X_test_3)  #X是ndarray
#Predition
preditions_smoteenn=clf_4.predict(X_test_smoteenn)
###########################################################################
#混淆矩阵
from sklearn.metrics import confusion_matrix
cnf_matrix=confusion_matrix(y_test,preditions)
cnf_matrix_smote=confusion_matrix(y_test,preditions_smote)
cnf_matrix_enn=confusion_matrix(y_test,preditions_enn)
cnf_matrix_smoteenn=confusion_matrix(y_test,preditions_smoteenn)
class_names=['dos','normal','probe','r2l','u2r']
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_names,title=' Baseline Confusion matrix')
plt.figure()
plot_confusion_matrix(cnf_matrix_smote,classes=class_names,title='SMOTE Confusion matrix')
plt.figure()
plot_confusion_matrix(cnf_matrix_enn,classes=class_names,title='ENN Confusion matrix')
plt.figure()
plot_confusion_matrix(cnf_matrix_smoteenn,classes=class_names,title='SMOTE+ENN Confusion matrix')
# plt.show()
#分类报告
from sklearn.metrics import classification_report
class_names=['dos','normal','probe','r2l','u2r']
print(" Baseline")
print(clf_1.oob_score_)
print(classification_report(y_test,preditions,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
print(" SMOTE")
print(clf_2.oob_score_)
print(classification_report(y_test,preditions_smote,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
print("ENN")
print(clf_3.oob_score_)
print(classification_report(y_test,preditions_enn,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
print(" SMOTE+ENN")
print(clf_4.oob_score_)
print(classification_report(y_test,preditions_smoteenn,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
###########################################################################

time_end=time.time()

print(time_end-time_start,'s')
plt.show()