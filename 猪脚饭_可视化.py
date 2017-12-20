# 读取数据
import time
import pandas as pd #数据分析
import matplotlib.pyplot as plt
import numpy as np
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
##y的处理
u2r=["buffer_overflow.","loadmodule.","perl.","rootkit."]
r2l=["ftp_write.","imap.","guess_passwd.","phf.","spy.","multihop.","warezmaster.","warezclient."]
dos=["back.","land.","pod.","neptune.","smurf.","teardrop."]
probe=["satan.","portsweep.","ipsweep.","nmap."]
for i in u2r:
    y_3[y_3==i]='U2R' #u2r
for i in r2l:
    y_3[y_3==i]='R2L' #r2l
for i in dos:
    y_3[y_3==i]='DOS'  #dos
for i in probe:
    y_3[y_3==i]='Probing' #probe
y_3[y_3=="normal."]='Normal' #normal
y_3=np.array(y_3)  #变成array格式，一维
#baseline
# #标准化
# from sklearn.preprocessing import StandardScaler
# scaler_base=StandardScaler().fit(X_3)
# X_b=scaler_base.transform(X_3)  #X是ndarray，二维
#
# #可视化
classes_1=['Probing','Normal','U2R','DOS','R2L']
y_b=pd.DataFrame(y_3,columns=['label'])
print(y_b.label.value_counts())
y_b['label'].value_counts().plot(kind='bar',rot=45)
plt.figure()
plt.title("Baseline")


y_c=[2131,87832,52,54572,999]
colors_1= ["pink","coral","yellow","orange",'blue']
# # Pie Plot
# plt.pie(y_c, colors=colors_1, labels=classes_1,autopct='%1.2f%%',pctdistance=0.7, shadow=True)
plt.pie(y_c, colors=colors_1, shadow=True)
plt.annotate('R2L(0.69%)',xy=(0.88,-0.0229),xytext=(1.15,-0.16),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.annotate('U2R(0.04%)',xy=(-0.7,-0.643367),xytext=(-1.2,-0.9),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.annotate('Probing(1.46%)',xy=(0.88,0.029),xytext=(1,0.32),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.text(0,0.55,'Normal')
plt.text(0,-0.46,'37.48%')
plt.text(0,-0.33,'DOS')
plt.text(0,0.22,'60.33%')


###################################################################
# #过采样 smote
# from imblearn.over_sampling import SMOTE
# oversampler=SMOTE(random_state=42)
# X_smote,y_s=oversampler.fit_sample(X_3,y_3)
# #标准化
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler().fit(X_smote)
# X_s=scaler.transform(X_smote)  #X是ndarray
#
# #可视化
# y_smote=pd.DataFrame(y_s,columns=['label'])
# print(y_smote.label.value_counts())

classes_1=['Probing','Normal','U2R','DOS','R2L']
plt.figure()
plt.title("SMOTE")
# y_smote['label'].value_counts().plot(kind='bar',rot=45)
y_sc=[87832,87832,87832,87832,87832]
plt.pie(y_sc, colors=colors_1, labels=classes_1,autopct='%1.2f%%',pctdistance=0.7, shadow=True)



#########################################################################
# #欠采样 ENN
# from imblearn.under_sampling import EditedNearestNeighbours
# oversampler=EditedNearestNeighbours(random_state=42,n_neighbors=30)
# X_enn,y_e=oversampler.fit_sample(X_3,y_3)
# #标准化
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler().fit(X_enn)
# X_e=scaler.transform(X_enn)
# #可视化
# y_enn=pd.DataFrame(y_e,columns=['label'])
# print(y_enn.label.value_counts())
plt.figure()
plt.title("ENN")
# y_enn['label'].value_counts().plot(kind='bar',rot=45)
y_ec=[1036,80347,52,52897,316]
# plt.pie(y_ec, colors=colors_1, labels=classes_1,autopct='%1.2f%%',pctdistance=0.7, shadow=True)
plt.pie(y_ec, colors=colors_1, shadow=True)
plt.annotate('R2L(0.23%)',xy=(0.83,-0.006),xytext=(1.15,-0.16),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.annotate('U2R(0.04%)',xy=(-0.78,-0.6),xytext=(-1.2,-0.9),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.annotate('Probing(0.77%)',xy=(0.927,0.0229),xytext=(1,0.32),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.text(0,0.55,'Normal')
plt.text(0,0.22,'59.67%')
plt.text(0,-0.33,'DOS')
plt.text(0,-0.46,'39.29%')


##############################################################################
# #过采样+欠采样 smote+enn
# from imblearn.combine import SMOTEENN
# oversampler=SMOTEENN(random_state=42,n_neighbors=30)
# X_SMOTEENN,y_se=oversampler.fit_sample(X_3,y_3)
# #标准化
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler().fit(X_SMOTEENN)
# X_se=scaler.transform(X_SMOTEENN)  #X是ndarray
# #可视化
# y_smoteenn=pd.DataFrame(y_se,columns=['label'])
# print(y_smoteenn.label.value_counts())
plt.figure()
plt.title("SMOTE+ENN")
# y_smoteenn['label'].value_counts().plot(kind='bar',rot=45)
y_sec=[82190,77276,82130,85498,85129]
plt.pie(y_sec, colors=colors_1, labels=classes_1,autopct='%1.2f%%',pctdistance=0.7, shadow=True)


###########################################################################

plt.show()
