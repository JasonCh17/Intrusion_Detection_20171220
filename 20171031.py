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
#####################################################################
#baseline
#标准化
from sklearn.preprocessing import StandardScaler
scaler_base=StandardScaler().fit(X_3)
X=scaler_base.transform(X_3)  #X是ndarray
#分类器
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
##建立模型
clf_1 = LogisticRegression(random_state=1)
# clf_1 = RandomForestClassifier(n_estimators=100,oob_score=True)
#验证测试样本
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y_3,test_size=0.2,random_state=0)  #切分样本
clf_1.fit(X_train,y_train)
preditions_base=clf_1.predict(X_test)
#学习曲线
import numpy as np
from sklearn.model_selection import learning_curve
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_1,
                                X=X_train,y=y_train,
                                train_sizes=np.linspace(0.05,1,10),
                                 cv=10, n_jobs=1,random_state=0)
train_mean=np.mean(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_std=np.std(train_scores,axis=1)

###################################################################
#过采样 smote
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
# clf_2 = RandomForestClassifier(n_estimators=100,oob_score=True)
clf_2 = LogisticRegression(random_state=1)
#验证测试样本
from sklearn.model_selection import train_test_split
X_train_1,X_test_1,y_train_1,y_test_1=train_test_split(X,y,test_size=0.2,random_state=0)  #切分样本
clf_2.fit(X_train_1,y_train_1)
preditions_smote=clf_2.predict(X_test_1)
#学习曲线
import numpy as np
from sklearn.model_selection import learning_curve
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_2,
                                X=X_train_1,y=y_train_1,
                                train_sizes=np.linspace(0.05,1,10),
                                 cv=10, n_jobs=1,random_state=0)
train_mean_1=np.mean(train_scores,axis=1)
test_mean_1=np.mean(test_scores,axis=1)
train_std_1=np.std(train_scores,axis=1)
test_std_1=np.std(train_scores,axis=1)
#########################################################################
#欠采样 ENN
from imblearn.under_sampling import EditedNearestNeighbours
oversampler=EditedNearestNeighbours(random_state=42)
X_enn,y=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_enn)
X=scaler.transform(X_enn)
##建立模型
clf_3 = LogisticRegression(random_state=1)
# clf_3 = RandomForestClassifier(n_estimators=100,oob_score=True)
#验证测试样本
from sklearn.model_selection import train_test_split
X_train_2,X_test_2,y_train_2,y_test_2=train_test_split(X,y,test_size=0.2,random_state=0)  #切分样本
clf_3.fit(X_train_2,y_train_2)
preditions_enn=clf_3.predict(X_test_2)
#学习曲线
import numpy as np
from sklearn.model_selection import learning_curve
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_3,
                                X=X_train_2,y=y_train_2,
                                train_sizes=np.linspace(0.05,1,10),
                                 cv=10, n_jobs=1,random_state=0)
train_mean_2=np.mean(train_scores,axis=1)
test_mean_2=np.mean(test_scores,axis=1)
train_std_2=np.std(train_scores,axis=1)
test_std_2=np.std(train_scores,axis=1)
##############################################################################
#过采样+欠采样 smote+enn
from imblearn.combine import SMOTEENN
oversampler=SMOTEENN(random_state=42)
X_SMOTEENN,y=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_SMOTEENN)
X=scaler.transform(X_SMOTEENN)  #X是ndarray
##建立模型
# clf_4 = RandomForestClassifier(n_estimators=100,oob_score=True)
clf_4 = LogisticRegression(random_state=1)
#验证测试样本
from sklearn.model_selection import train_test_split
X_train_3,X_test_3,y_train_3,y_test_3=train_test_split(X,y,test_size=0.2,random_state=0)  #切分样本
clf_4.fit(X_train_3,y_train_3)
preditions_smoteenn=clf_4.predict(X_test_3)
#学习曲线
import numpy as np
from sklearn.model_selection import learning_curve
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_4,
                                X=X_train_3,y=y_train_3,
                                train_sizes=np.linspace(0.05,1,10),
                                 cv=10, n_jobs=1,random_state=0)
train_mean_3=np.mean(train_scores,axis=1)
test_mean_3=np.mean(test_scores,axis=1)
train_std_3=np.std(train_scores,axis=1)
test_std_3=np.std(train_scores,axis=1)




# #欠采样 Tomeklinks
# from imblearn.under_sampling import TomekLinks
# undersampler=TomekLinks(random_state=42)
# X_TL,y=undersampler.fit_sample(X_3,y_3)
# #标准化
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler().fit(X_TL)
# X=scaler.transform(X_TL)  #X是ndarray
# ##建立模型
# clf_4 = RandomForestClassifier()
# #验证测试样本
# from sklearn.model_selection import train_test_split
# X_train_4,X_test_4,y_train_4,y_test_4=train_test_split(X,y,test_size=0.2,random_state=0)  #切分样本
# clf_4.fit(X_train_4,y_train_4)
# preditions_4=clf_4.predict(X_test_4)
# #混淆矩阵
# from sklearn.metrics import confusion_matrix
# cnf_matrix=confusion_matrix(y_test_4,preditions_4)
# class_names=['dos','normal','probe','r2l','u2r']
# plt.figure()
# plot_confusion_matrix(cnf_matrix,classes=class_names,title='Tomek links Confusion matrix')
# # plt.show()
# #分类报告
# from sklearn.metrics import classification_report
# class_names=['dos','normal','probe','r2l','u2r']
# print(" Tomek links")
# print(classification_report(y_test_4,preditions_4,target_names=class_names,digits=6))
# print("--------------------------------------------------------------------------------")
# #学习曲线
# import numpy as np
# from sklearn.model_selection import learning_curve
# train_sizes,train_scores,test_scores=learning_curve(estimator=clf_1,
#                                 X=X_train_4,y=y_train_4,
#                                 train_sizes=np.linspace(0.01,1,10),
#                                  cv=10, n_jobs=1,random_state=0)
# train_mean_4=np.mean(train_scores,axis=1)
# test_mean_4=np.mean(test_scores,axis=1)
# train_std_4=np.std(train_scores,axis=1)
# test_std_4=np.std(train_scores,axis=1)
#
#
#
#
# #过采样欠采样 SMOTE+Tomeklinks
# from imblearn.combine import SMOTETomek
# oversampler=SMOTETomek(random_state=42)
# X_SMOTETL,y=oversampler.fit_sample(X_3,y_3)
# #标准化
# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler().fit(X_SMOTETL)
# X=scaler.transform(X_SMOTETL)  #X是ndarray
# ##建立模型
# clf_5 = RandomForestClassifier()
# #验证测试样本
# from sklearn.model_selection import train_test_split
# X_train_5,X_test_5,y_train_5,y_test_5=train_test_split(X,y,test_size=0.2,random_state=0)  #切分样本
# clf_5.fit(X_train_5,y_train_5)
# preditions_5=clf_5.predict(X_test_5)
# #混淆矩阵
# from sklearn.metrics import confusion_matrix
# cnf_matrix=confusion_matrix(y_test_5,preditions_5)
# class_names=['dos','normal','probe','r2l','u2r']
# plt.figure()
# plot_confusion_matrix(cnf_matrix,classes=class_names,title='SMOTE+Tomek links Confusion matrix')
# # plt.show()
# #分类报告
# from sklearn.metrics import classification_report
# class_names=['dos','normal','probe','r2l','u2r']
# print("SMOTE+Tomek links")
# print(classification_report(y_test_5,preditions_5,target_names=class_names,digits=6))
# print("--------------------------------------------------------------------------------")
# #学习曲线
# import numpy as np
# from sklearn.model_selection import learning_curve
# train_sizes,train_scores,test_scores=learning_curve(estimator=clf_1,
#                                 X=X_train_5,y=y_train_5,
#                                 train_sizes=np.linspace(0.01,1,10),
#                                  cv=10, n_jobs=1,random_state=0)
# train_mean_5=np.mean(train_scores,axis=1)
# test_mean_5=np.mean(test_scores,axis=1)
# train_std_5=np.std(train_scores,axis=1)
# test_std_5=np.std(train_scores,axis=1)

time_end=time.time()

print(time_end-time_start,'s')
###########################################################################
#混淆矩阵
from sklearn.metrics import confusion_matrix
cnf_matrix=confusion_matrix(y_test,preditions_base)
cnf_matrix_smote=confusion_matrix(y_test_1,preditions_smote)
cnf_matrix_enn=confusion_matrix(y_test_2,preditions_enn)
cnf_matrix_smoteenn=confusion_matrix(y_test_3,preditions_smoteenn)
class_names=['dos','normal','probe','r2l','u2r']
plt.figure()
plot_confusion_matrix(cnf_matrix,classes=class_names,title=' Baseline Confusion matrix')
plt.figure()
plot_confusion_matrix(cnf_matrix_smote,classes=class_names,title='SMOTE Confusion matrix')
plt.figure()
plot_confusion_matrix(cnf_matrix_enn,classes=class_names,title='ENN Confusion matrix')
plt.figure()
plot_confusion_matrix(cnf_matrix_smoteenn,classes=class_names,title='SMOTE+ENN Confusion matrix')
#分类报告
from sklearn.metrics import classification_report
class_names=['dos','normal','probe','r2l','u2r']
print(" Baseline")
# print(clf_1.oob_score_)
print(classification_report(y_test,preditions_base,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
print(" SMOTE")
# print(clf_2.oob_score_)
print(classification_report(y_test_1,preditions_smote,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
print("ENN")
# print(clf_3.oob_score_)
print(classification_report(y_test_2,preditions_enn,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
print(" SMOTE+ENN")
# print(clf_4.oob_score_)
print(classification_report(y_test_3,preditions_smoteenn,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
###########################################################################

plt.figure()
plt.grid()
# plt.subplot(131)
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,
         label='Baseline training accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,
         color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean,color='green',linestyle='--',
         marker='s',markersize=5,label="Baseline validation accuracy")
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,color='green',alpha=0.25)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
# plt.show()

plt.figure()
plt.grid()
# plt.subplot(131)
plt.plot(train_sizes,train_mean_1,color='blue',marker='o',markersize=5,
         label='SMOTE training accuracy')
plt.fill_between(train_sizes,train_mean_1+train_std_1,train_mean_1-train_std_1,
         color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean_1,color='green',linestyle='--',
         marker='s',markersize=5,label="SMOTE validation accuracy")
plt.fill_between(train_sizes,test_mean_1+test_std_1,test_mean_1-test_std_1,color='green',alpha=0.25)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
# plt.show()


plt.figure()
plt.grid()
# plt.subplot(132)
plt.plot(train_sizes,train_mean_2,color='blue',marker='o',markersize=5,
         label=' ENN training accuracy')
plt.fill_between(train_sizes,train_mean_2+train_std_2,train_mean_2-train_std_2,
         color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean_2,color='green',linestyle='--',
         marker='s',markersize=5,
         label='ENN validation accuracy')
plt.fill_between(train_sizes,test_mean_2+test_std_2,test_mean_2-test_std_2,
         color='green',alpha=0.25)

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
# plt.show()


plt.figure()
plt.grid()
# plt.subplot(133)
plt.plot(train_sizes,train_mean_3,color='blue',marker='o',markersize=5,
         label='SMOTE+ENN training accuracy')
plt.fill_between(train_sizes,train_mean_3+train_std_3,train_mean_3-train_std_3,
         color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean_3,color='green',linestyle='--',
         marker='s',markersize=5,
         label='SMOTE+ENN validation accuracy')
plt.fill_between(train_sizes,test_mean_3+test_std_3,test_mean_3-test_std_3,color='green',alpha=0.25)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
# plt.show()


# plt.figure()
# plt.grid()
# # plt.subplot(133)
# plt.plot(train_sizes,train_mean_4,color='blue',marker='o',markersize=5,
#          label='Tomek links training accuracy')
# plt.fill_between(train_sizes,train_mean_4+train_std_4,train_mean_4-train_std_4,
#          color='blue',alpha=0.25)
# plt.plot(train_sizes,test_mean_4,color='green',linestyle='--',
#          marker='s',markersize=5,
#          label='Tomek links validation accuracy')
# plt.fill_between(train_sizes,test_mean_4+test_std_4,test_mean_4-test_std_4,color='green',alpha=0.25)
# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# # # plt.show()
# #
# #
# plt.figure()
# plt.grid()
# # plt.subplot(133)
# plt.plot(train_sizes,train_mean_5,color='blue',marker='o',markersize=5,
#          label='SMOTE+Tomek links training accuracy')
# plt.fill_between(train_sizes,train_mean_5+train_std_5,train_mean_5-train_std_5,
#          color='blue',alpha=0.25)
# plt.plot(train_sizes,test_mean_5,color='green',linestyle='--',
#          marker='s',markersize=5,
#          label='SMOTE+Tomek links validation accuracy')
# plt.fill_between(train_sizes,test_mean_5+test_std_5,test_mean_5-test_std_5,color='green',alpha=0.25)
# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# # plt.show()

#三种采样的比较
plt.figure()

plt.grid()
plt.plot(train_sizes,test_mean,color='b',linestyle='--',
         marker='p',markersize=5,
         label=' Baseline validation accuracy')
# plt.plot(train_sizes,train_mean_1,color='blue',marker='o',markersize=5,
#          label='smote training accuracy')
plt.plot(train_sizes,test_mean_1,color='k',linestyle='--',
         marker='+',markersize=5,
         label='SMOTE validation accuracy')
# plt.fill_between(train_sizes,train_mean_1+train_std_1,train_mean_1-train_std_1,
#          color='blue',alpha=0.25)
# plt.fill_between(train_sizes,test_mean_1+test_std_1,test_mean_1-test_std_1,
#          color='green',alpha=0.25)
# #
# plt.plot(train_sizes,train_mean_2,color='r',marker='o',markersize=5,
#          label=' Enn training accuracy')
plt.plot(train_sizes,test_mean_2,color='y',linestyle='--',
         marker='*',markersize=5,
         label=' ENN validation accuracy')
# plt.fill_between(train_sizes,train_mean_2+train_std_2,train_mean_2-train_std_2,
# #          color='blue',alpha=0.25)
# plt.fill_between(train_sizes,test_mean_2+test_std_2,test_mean_2-test_std_2,
#          color='green',alpha=0.25)

# plt.plot(train_sizes,train_mean_3,color='c',marker='o',markersize=5,
#          label=' smote+enn training accuracy')
plt.plot(train_sizes,test_mean_3,color='m',linestyle='--',
         marker='o',markersize=5,
         label=' SMOTE+ENN validation accuracy')
# # plt.fill_between(train_sizes,train_mean_3+train_std_3,train_mean_3-train_std_3,
# # #          color='blue',alpha=0.25)
# # plt.fill_between(train_sizes,test_mean_3+test_std_3,test_mean_3-test_std_3,
# #          color='green',alpha=0.25)
# plt.plot(train_sizes,test_mean_4,color='b',linestyle='--',
#          marker='s',markersize=5,
#          label=' Tomek links validation accuracy')
# plt.plot(train_sizes,test_mean_5,color='r',linestyle='--',
#          marker='p',markersize=5,
#          label=' SMOTE+Tomek links validation accuracy')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

