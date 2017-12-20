
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
import time
import pandas as pd #数据分析
import matplotlib.pyplot as plt
import numpy as np
import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
time_start=time.time()
#读取训练数据
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
train_raw= pd.read_csv("E:\Pycharm\Intrusion_Detection\kddcup.data_10_percent.csv",  header=None,names = col_names)
#去重
train_1=train_raw.drop_duplicates()
#one-hot
dummies_protocol = pd.get_dummies(train_1["protocol_type"], prefix='protocol')
dummies_flag = pd.get_dummies(train_1["flag"], prefix='flag')
train_2 = pd.concat([train_1, dummies_protocol,dummies_flag], axis=1)
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
X_train_fe=train_2[feature_selection]
y_train=train_2['label'].copy()   #一维
##y的处理
u2r=["buffer_overflow.","loadmodule.","perl.","rootkit."]
r2l=["ftp_write.","imap.","guess_passwd.","phf.","spy.","multihop.","warezmaster.","warezclient."]
dos=["back.","land.","pod.","neptune.","smurf.","teardrop."]
probe=["satan.","portsweep.","ipsweep.","nmap."]
for i in u2r:
    y_train[y_train==i]="u2r"
for i in r2l:
    y_train[y_train==i]="r2l"
for i in dos:
    y_train[y_train==i]="dos"
for i in probe:
    y_train[y_train==i]="probe"
y_train[y_train=="normal."]="normal"
#########################################################
#读取数据测试数据
test= pd.read_csv("E:\Pycharm\Intrusion_Detection\corrected.csv",  header=None,names = col_names)
#去重
test_1=test.drop_duplicates()
#one-hot
dummies_protocol = pd.get_dummies(test_1["protocol_type"], prefix='protocol')
dummies_flag = pd.get_dummies(test_1["flag"], prefix='flag')
test_2 = pd.concat([test_1, dummies_protocol,dummies_flag], axis=1)
#特征选择(by "A feature reduced intrusion detection system using ANN classifier",25个特征）
#建立X,y
X_test_fe=test_2[feature_selection]
y_test=test_2['label'].copy()   #一维
#y的处理
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
#标准化
scaler_base=StandardScaler().fit(X_train_fe)
X_train=scaler_base.transform(X_train_fe)  #训练数据标准化
X_test=scaler_base.transform(X_test_fe)    #测试数据标准化
#####################################################################
##Baseline
##建立模型
clf_base = RandomForestClassifier(oob_score=True)
clf_base.fit(X_train,y_train)
preditions_base=clf_base.predict(X_test)
#学习曲线
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_base,
                                X=X_train,y=y_train,
                                train_sizes=np.linspace(0.05,1,10),
                                 cv=10, n_jobs=1,random_state=0)
train_mean_base=np.mean(train_scores,axis=1)
test_mean_base=np.mean(test_scores,axis=1)
train_std_base=np.std(train_scores,axis=1)
test_std_base=np.std(train_scores,axis=1)
###################################################################
##Smote
from imblearn.over_sampling import SMOTE
smote=SMOTE(random_state=42)
X_smote,y_smote=smote.fit_sample(X_train,y_train)
##建立模型
clf_smote = RandomForestClassifier(oob_score=True)
clf_smote.fit(X_smote,y_smote)
preditions_smote=clf_smote.predict(X_test)
#学习曲线
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_smote,
                                X=X_smote,y=y_smote,
                                train_sizes=np.linspace(0.05,1,10),
                                 cv=10, n_jobs=1,random_state=0)
train_mean_smote=np.mean(train_scores,axis=1)
test_mean_smote=np.mean(test_scores,axis=1)
train_std_smote=np.std(train_scores,axis=1)
test_std_smote=np.std(train_scores,axis=1)
###################################################################
##ENN
from imblearn.under_sampling import EditedNearestNeighbours
ENN=EditedNearestNeighbours(random_state=42)
X_enn,y_enn=ENN.fit_sample(X_train,y_train)
##建立模型
clf_enn = RandomForestClassifier(oob_score=True)
clf_enn.fit(X_enn,y_enn)
preditions_enn=clf_enn.predict(X_test)
#学习曲线
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_enn,
                                X=X_enn,y=y_enn,
                                train_sizes=np.linspace(0.05,1,10),
                                 cv=10, n_jobs=1,random_state=0)
train_mean_enn=np.mean(train_scores,axis=1)
test_mean_enn=np.mean(test_scores,axis=1)
train_std_enn=np.std(train_scores,axis=1)
test_std_enn=np.std(train_scores,axis=1)
###################################################################
##SMOTE+ENN
from imblearn.combine import SMOTEENN
smoteenn=SMOTEENN(random_state=42)
X_smoteenn,y_smoteenn=smoteenn.fit_sample(X_train,y_train)
##建立模型
clf_smoteenn = RandomForestClassifier(oob_score=True)
clf_smoteenn.fit(X_smoteenn,y_smoteenn)
preditions_smoteenn=clf_smoteenn.predict(X_test)
#学习曲线
train_sizes,train_scores,test_scores=learning_curve(estimator=clf_smoteenn,
                                X=X_smoteenn,y=y_smoteenn,
                                train_sizes=np.linspace(0.05,1,10),
                                 cv=10, n_jobs=1,random_state=0)
train_mean_smoteenn=np.mean(train_scores,axis=1)
test_mean_smoteenn=np.mean(test_scores,axis=1)
train_std_smoteenn=np.std(train_scores,axis=1)
test_std_smoteenn=np.std(train_scores,axis=1)
###################################################################
#混淆矩阵
class_names=['dos','normal','probe','r2l','u2r']
cnf_matrix_base=confusion_matrix(y_test,preditions_base)
cnf_matrix_smote=confusion_matrix(y_test,preditions_smote)
cnf_matrix_enn=confusion_matrix(y_test,preditions_enn)
cnf_matrix_smoteenn=confusion_matrix(y_test,preditions_smoteenn)
plt.figure()
plot_confusion_matrix.plot_confusion_matrix(cnf_matrix_base,classes=class_names,title=' Baseline Confusion matrix')
plt.figure()
plot_confusion_matrix.plot_confusion_matrix(cnf_matrix_smote,classes=class_names,title=' SMOTE Confusion matrix')
plt.figure()
plot_confusion_matrix.plot_confusion_matrix(cnf_matrix_enn,classes=class_names,title=' ENN Confusion matrix')
plt.figure()
plot_confusion_matrix.plot_confusion_matrix(cnf_matrix_smoteenn,classes=class_names,title=' SMOTE+ENN Confusion matrix')
#分类报告
print(" Baseline")
print(clf_base.oob_score_)
print(classification_report(y_test,preditions_base,target_names=class_names,digits=6))
print(" SMOTE")
print(clf_smote.oob_score_)
print(classification_report(y_test,preditions_smote,target_names=class_names,digits=6))
print(" ENN")
print(clf_enn.oob_score_)
print(classification_report(y_test,preditions_enn,target_names=class_names,digits=6))
print(" SMOTE+ENN")
print(clf_smoteenn.oob_score_)
print(classification_report(y_test,preditions_smoteenn,target_names=class_names,digits=6))
time_end=time.time()
print(time_end-time_start,'s')
#plot learning curve
#Baseline
plt.figure()
plt.grid()
plt.plot(train_sizes,train_mean_base,color='blue',marker='o',markersize=5,
         label='Baseline training accuracy')
plt.fill_between(train_sizes,train_mean_base+train_std_base,train_mean_base-train_std_base,
         color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean_base,color='green',linestyle='--',
         marker='s',markersize=5,label="Baseline validation accuracy")
plt.fill_between(train_sizes,test_mean_base+test_std_base,test_mean_base-test_std_base,color='green',alpha=0.25)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
#SMOTE
plt.figure()
plt.grid()
plt.plot(train_sizes,train_mean_smote,color='blue',marker='o',markersize=5,
         label='SMOTE training accuracy')
plt.fill_between(train_sizes,train_mean_smote+train_std_smote,train_mean_smote-train_std_smote,
         color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean_smote,color='green',linestyle='--',
         marker='s',markersize=5,label="SMOTE validation accuracy")
plt.fill_between(train_sizes,test_mean_smote+test_std_smote,test_mean_smote-test_std_smote,color='green',alpha=0.25)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
#ENN
plt.figure()
plt.grid()
plt.plot(train_sizes,train_mean_enn,color='blue',marker='o',markersize=5,
         label='ENN training accuracy')
plt.fill_between(train_sizes,train_mean_enn+train_std_enn,train_mean_enn-train_std_enn,
         color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean_enn,color='green',linestyle='--',
         marker='s',markersize=5,label="ENN validation accuracy")
plt.fill_between(train_sizes,test_mean_enn+test_std_enn,test_mean_enn-test_std_enn,color='green',alpha=0.25)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
#SMOTE+ENN
plt.figure()
plt.grid()
plt.plot(train_sizes,train_mean_smoteenn,color='blue',marker='o',markersize=5,
         label='SMOTE+ENN training accuracy')
plt.fill_between(train_sizes,train_mean_smoteenn+train_std_smoteenn,train_mean_smoteenn-train_std_smoteenn,
         color='blue',alpha=0.25)
plt.plot(train_sizes,test_mean_smoteenn,color='green',linestyle='--',
         marker='s',markersize=5,label="SMOTE+ENN validation accuracy")
plt.fill_between(train_sizes,test_mean_smoteenn+test_std_smoteenn,test_mean_smoteenn-test_std_smoteenn,color='green',alpha=0.25)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()