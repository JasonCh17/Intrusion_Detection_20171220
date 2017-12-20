# 读取数据
import time
import pandas as pd #数据分析
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
#读取training data
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
data_1=data.drop_duplicates()

#one-hot
dummies_protocol = pd.get_dummies(data_1["protocol_type"], prefix='protocol')
dummies_flag = pd.get_dummies(data_1["flag"], prefix='flag')
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

#读取test data
data_test= pd.read_csv("E:\Pycharm\Intrusion_Detection\corrected.csv",  header=None,names = col_names)
#去重
data_test=data_test.drop_duplicates()
#one-hot
dummies_protocol = pd.get_dummies(data_test["protocol_type"], prefix='protocol')
dummies_flag = pd.get_dummies(data_test["flag"], prefix='flag')
data_test_2 = pd.concat([data_test, dummies_protocol,dummies_flag], axis=1)
#特征选择(by "A feature reduced intrusion detection system using ANN classifier",25个特征）
#建立X,y
X_test=data_test_2[feature_selection]
y_test=data_test_2['label'].copy()   #一维
##y的处理
u2r=["buffer_overflow.","loadmodule.","perl.","rootkit.","httptunnel.","ps.","sqlattack.","xterm."]
r2l=["ftp_write.","imap.","guess_passwd.","phf.","spy.","multihop.","warezmaster.","warezclient.","named.","sendmail."
    ,"snmpgetattack.","snmpguess.","worm.","xlock.","xsnoop."]
dos=["back.","land.","pod.","neptune.","smurf.","teardrop.","apache2.","mailbomb.","processtable.","udpstorm."]
probe=["satan.","portsweep.","ipsweep.","nmap.","mscan.","saint."]
for i in u2r:
    y_test[y_test==i]="U2R"
for i in r2l:
    y_test[y_test==i]="R2L"
for i in dos:
    y_test[y_test==i]="DOS"
for i in probe:
    y_test[y_test==i]="Probing"
y_test[y_test=="normal."]="Normal"
y_test=np.array(y_test)  #变成array格式，一维


#baseline
# #标准化
# from sklearn.preprocessing import StandardScaler
# scaler_base=StandardScaler().fit(X_3)
# X_b=scaler_base.transform(X_3)  #X是ndarray，二维
# X_test_b=scaler_base.transform(X_test)
# from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test=train_test_split(X_b,y_3,test_size=0.2,random_state=0)  #切分样本
# #可视化
# y_b=pd.DataFrame(y_3,columns=['label'])
# plt.figure()
# plt.title("Baseline")
# y_b['label'].value_counts().plot(kind='bar',rot=45)
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
# #分类器
# from sklearn.ensemble import RandomForestClassifier
# ##建立模型
# clf_1 = RandomForestClassifier(n_estimators=100)
#
# #学习曲线
# import numpy as np
# from sklearn.model_selection import learning_curve
# train_sizes,train_scores,test_scores=learning_curve(estimator=clf_1,
#                                 X=X_train,y=y_train,
#                                 train_sizes=np.linspace(0.05,1,10),
#                                  cv=10, n_jobs=1,random_state=0)
# train_mean=np.mean(train_scores,axis=1)
# test_mean=np.mean(test_scores,axis=1)
# train_std=np.std(train_scores,axis=1)
# test_std=np.std(train_scores,axis=1)

# #验证测试样本
# clf_1.fit(X_b,y_3)
# #Prediction
#
# preditions_base=clf_1.predict(X_test_b)

# ###################################################################
# #过采样 smote
# from imblearn.over_sampling import SMOTE
# oversampler=SMOTE(random_state=42)
# X_smote,y_s=oversampler.fit_sample(X_3,y_3)
# #标准化
# from sklearn.preprocessing import StandardScaler
# scaler_s=StandardScaler().fit(X_smote)
# X_s=scaler_s.transform(X_smote)  #X是ndarray
# # #可视化
# # y_smote=pd.DataFrame(y_s,columns=['label'])
# # from sklearn.model_selection import train_test_split
# # X_train_1,X_test_1,y_train_1,y_test_1=train_test_split(X_s,y_s,test_size=0.2,random_state=0)  #切分样本
# # plt.figure()
# # plt.title("SMOTE")
# # y_smote['label'].value_counts().plot(kind='bar',rot=45)
# # from sklearn.manifold import TSNE
# # X_embedded = TSNE(n_components=2).fit_transform(X_test_1)
# # plt.figure()
# # plt.title("SMOTE")
# # for index,label,color in zip(range(len(classes)),classes,colors):
# #     plt.scatter(X_embedded[y_test_1==label,0],
# #                 X_embedded[y_test_1==label,1],
# #                 label=classes[index],
# #                 c=color)
# # plt.legend(loc='best')
# # # plt.show()
#
# #分类器
# from sklearn.ensemble import RandomForestClassifier
# # ##建立模型
# clf_2 = RandomForestClassifier()
# # #验证测试样本
# clf_2.fit(X_s,y_s)
# preditions_smote=clf_2.predict(X_test)
# # #学习曲线
# # import numpy as np
# # from sklearn.model_selection import learning_curve
# # train_sizes,train_scores,test_scores=learning_curve(estimator=clf_2,
# #                                 X=X_train_1,y=y_train_1,
# #                                 train_sizes=np.linspace(0.05,1,10),
# #                                  cv=10, n_jobs=1,random_state=0)
# # train_mean_1=np.mean(train_scores,axis=1)
# # test_mean_1=np.mean(test_scores,axis=1)
# # train_std_1=np.std(train_scores,axis=1)
# # test_std_1=np.std(train_scores,axis=1)
# #########################################################################
# #欠采样 ENN
# from imblearn.under_sampling import EditedNearestNeighbours
# oversampler=EditedNearestNeighbours(random_state=42,n_neighbors=30)
# X_enn,y_e=oversampler.fit_sample(X_3,y_3)
# #标准化
# from sklearn.preprocessing import StandardScaler
# scaler_e=StandardScaler().fit(X_enn)
# X_e=scaler_e.transform(X_enn)
# #可视化
# y_enn=pd.DataFrame(y_e,columns=['label'])
# plt.figure()
# plt.title("ENN")
# y_enn['lable'].value_counts().plot(kind='bar',rot=45)
# from sklearn.model_selection import train_test_split
# X_train_2,X_test_2,y_train_2,y_test_2=train_test_split(X_e,y_e,test_size=0.2,random_state=0)  #切分样本
#
#
# from sklearn.manifold import TSNE
# X_embedded = TSNE(n_components=2).fit_transform(X_test_2)
# plt.figure()
# plt.title("ENN")
# for index,label,color in zip(range(len(classes)),classes,colors):
#     plt.scatter(X_embedded[y_test_2==label,0],
#                 X_embedded[y_test_2==label,1],
#                 label=classes[index],
#                 c=color)
# plt.legend(loc='best')
# # plt.show()
# ##建立模型
# clf_3 = RandomForestClassifier(n_estimators=100)
# #验证测试样本
# clf_3.fit(X_train_2,y_train_2)
# preditions_enn=clf_3.predict(X_test_2)
# #学习曲线
# import numpy as np
# from sklearn.model_selection import learning_curve
# train_sizes,train_scores,test_scores=learning_curve(estimator=clf_3,
#                                 X=X_train_2,y=y_train_2,
#                                 train_sizes=np.linspace(0.05,1,10),
#                                  cv=10, n_jobs=1,random_state=0)
# train_mean_2=np.mean(train_scores,axis=1)
# test_mean_2=np.mean(test_scores,axis=1)
# train_std_2=np.std(train_scores,axis=1)
# test_std_2=np.std(train_scores,axis=1)

##############################################################################
#过采样+欠采样 smote+enn
##建立模型
clf_4 = RandomForestClassifier(n_estimators=100)
from imblearn.combine import SMOTEENN
pa=[10,50,100,150,200,250,300,350,400,450,500]
score=[]
for i in pa:
    oversampler=SMOTEENN(random_state=42,n_neighbors=i)
    X_SMOTEENN,y_se=oversampler.fit_sample(X_3,y_3)
    X_test_SMOTEENN,y_test_se=oversampler.fit_sample(X_test,y_test)
#标准化
    from sklearn.preprocessing import StandardScaler
    scaler_se=StandardScaler().fit(X_SMOTEENN)
    X_se=scaler_se.transform(X_SMOTEENN)  #X是ndarray
    X_test_se=scaler_se.transform(X_test_SMOTEENN)
#验证测试样本
    clf_4.fit(X_se,y_se)
    preditions_smoteenn=clf_4.predict(X_test_se)

#混淆矩阵
    from sklearn.metrics import confusion_matrix
    cnf_matrix_smoteenn=confusion_matrix(y_test_se,preditions_smoteenn)
    class_names=['DOS','Normal','Probing','R2L','U2R']
    plt.figure()
    plot_confusion_matrix(cnf_matrix_smoteenn,classes=class_names,title='SMOTE+ENN Confusion matrix')
#分类报告
    from sklearn.metrics import classification_report
    print(" SMOTE+ENN")
    print(classification_report(y_test_se,preditions_smoteenn,target_names=class_names,digits=6))
    # plt.show()
    score.append(clf_4.score(X_test_se, y_test_se))
print(score)
plt.figure()
plt.grid()
plt.plot(pa,score,color='b',linestyle='--',
         marker='p',markersize=5)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()
