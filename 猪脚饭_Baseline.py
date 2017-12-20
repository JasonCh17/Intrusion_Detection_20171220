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
#标准化
from sklearn.preprocessing import StandardScaler
scaler_base=StandardScaler().fit(X_3)
X_b=scaler_base.transform(X_3)  #X是ndarray，二维

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_b,y_3,test_size=0.2,random_state=0)  #切分样本
# #可视化
# classes_1=['Probing','Normal','U2R','DOS','R2L']
# y_b=pd.DataFrame(y_3,columns=['label'])
# print(y_b.label.value_counts())
# plt.figure()
# plt.title("Baseline")
# # y_b['label'].value_counts().plot(kind='bar',rot=45)
#
# y_c=[2131,87832,52,54572,999]
# colors_1= ["pink","coral","yellow","orange",'blue']
# # # Pie Plot
# # plt.pie(y_c, colors=colors_1, labels=classes_1,autopct='%1.2f%%',pctdistance=0.7, shadow=True)
# plt.pie(y_c, colors=colors_1, shadow=True)
# plt.annotate('R2L(0.23%)',xy=(0.88,0.0055),xytext=(1.15,-0.16),
#              arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
# plt.annotate('U2R(0.01%)',xy=(0.2748,0.9525),xytext=(0.36,1.25),
#              arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
# plt.annotate('Probing(0.83%)',xy=(0.88,0.029),xytext=(1,0.32),
#              arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
# plt.text(0.8,0.70,'Normal')
# plt.text(0.52,0.46,'19.69%')
# plt.text(-1.18,0.088,'DOS')
# plt.text(-0.44,0.24,'79.24%')
# from sklearn.manifold import TSNE
# X_embedded = TSNE(n_components=2).fit_transform(X_test)
# classes=['Normal','Probing','DOS','U2R','R2L']
# colors=['blue','red','y','m','w']
# plt.figure()
# plt.title("Baseline")
#
# for index,label,color in zip(range(len(classes)),classes,colors):
#     plt.scatter(X_embedded[y_test==label,0],
#                 X_embedded[y_test==label,1],
#                 label=classes[index],
#                 c=color)
# plt.legend(loc='best')
#
# # plt.show()
#分类器
from sklearn.ensemble import RandomForestClassifier
##建立模型
clf_1 = RandomForestClassifier()
# #Cv
# from sklearn.model_selection import  cross_val_score
# cv=cross_val_score(clf_1,X_train,y_train,scoring='f1_macro')
#学习曲线
import numpy as np
from sklearn.model_selection import learning_curve
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_1,
                                X=X_train,y=y_train,
                                train_sizes=np.linspace(0.05,1,20),
                                 cv=10, n_jobs=1,random_state=0,scoring='f1_macro')
train_mean_f=np.mean(train_scores,axis=1)
test_mean_f=np.mean(test_scores,axis=1)

train_sizes,train_scores,test_scores=learning_curve(estimator=clf_1,
                                X=X_train,y=y_train,
                                train_sizes=np.linspace(0.05,1,20),
                                 cv=10, n_jobs=1,random_state=0,scoring='precision_macro')
train_mean_p=np.mean(train_scores,axis=1)
test_mean_p=np.mean(test_scores,axis=1)

train_sizes,train_scores,test_scores=learning_curve(estimator=clf_1,
                                X=X_train,y=y_train,
                                train_sizes=np.linspace(0.05,1,20),
                                 cv=10, n_jobs=1,random_state=0,scoring='recall_macro')
train_mean_r=np.mean(train_scores,axis=1)
test_mean_r=np.mean(test_scores,axis=1)

# print('cv',np.mean(cv))

print('F1',test_mean_f)
print('Precison',test_mean_p)
print('Recall',test_mean_r)
print('train_sizes',train_sizes)
#验证测试样本
clf_1.fit(X_train,y_train)
preditions_base=clf_1.predict(X_test)

# #混淆矩阵
class_names=['DOS','Normal','Probing','R2L','U2R']
from sklearn.metrics import confusion_matrix
cnf_matrix=confusion_matrix(y_test,preditions_base)
# cnf_matrix_smote=confusion_matrix(y_test_1,preditions_smote)
# cnf_matrix_enn=confusion_matrix(y_test_2,preditions_enn)
# cnf_matrix_smoteenn=confusion_matrix(y_test_3,preditions_smoteenn)
# class_names=['DOS','Normal','Probing','R2L','U2R']
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_names,title=' Baseline Confusion matrix')
# plt.figure()
# plot_confusion_matrix(cnf_matrix_smote,classes=class_names,title='SMOTE Confusion matrix')
# plt.figure()
# plot_confusion_matrix(cnf_matrix_enn,classes=class_names,title='ENN Confusion matrix')
# plt.figure()
# plot_confusion_matrix(cnf_matrix_smoteenn,classes=class_names,title='SMOTE+ENN Confusion matrix')
# #分类报告
from sklearn.metrics import classification_report

print(" Baseline")
# print(test_mean)
print(classification_report(y_test,preditions_base,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
# print(" SMOTE")
# print(test_mean_1)
# print(classification_report(y_test_1,preditions_smote,target_names=class_names,digits=6))
# print("--------------------------------------------------------------------------------")
# print("ENN")
# print(test_mean_2)
# print(classification_report(y_test_2,preditions_enn,target_names=class_names,digits=6))
# print("--------------------------------------------------------------------------------")
# print(" SMOTE+ENN")
# print(test_mean_3)
# print(classification_report(y_test_3,preditions_smoteenn,target_names=class_names,digits=6))
# print("--------------------------------------------------------------------------------")
# ###########################################################################
#三种采样的比较
plt.figure()

plt.grid()
# plt.plot(train_sizes,train_mean,color='blue',marker='p',markersize=5,
#          label='Baseline training F-value')
# plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,
#          color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean_f,color='green',linestyle='--',
         marker='s',markersize=5,label="Baseline validation F-value")
# plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,color='green',alpha=0.25)
# plt.xlabel('Number of training samples')
# plt.ylabel('F-value')
# plt.legend(loc='lower right')
# # plt.show()
plt.plot(train_sizes,test_mean_p,color='k',linestyle='--',
         marker='+',markersize=5,
         label='Baseline validation precison')
plt.plot(train_sizes,test_mean_r,color='y',linestyle='--',
         marker='*',markersize=5,
         label=' Baseline validation recall')
# plt.plot(train_sizes,test_mean_3,color='m',linestyle='--',
#          marker='o',markersize=5,
#          label=' SMOTE+ENN validation F-value')
# plt.xlabel('Number of training samples')
plt.ylabel('Score')
plt.legend(loc='lower right')
plt.show()
