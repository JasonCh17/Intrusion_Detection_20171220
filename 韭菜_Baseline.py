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
#可视化
classes_1=['Probing','Normal','U2R','DOS','R2L']
y_b=pd.DataFrame(y_3,columns=['label'])
print(y_b.label.value_counts())
plt.figure()
plt.title("Baseline")
# y_b['label'].value_counts().plot(kind='bar',rot=45)

y_c=[2131,87832,52,54572,999]
colors_1= ["pink","coral","yellow","orange",'blue']
# # Pie Plot
# plt.pie(y_c, colors=colors_1, labels=classes_1,autopct='%1.2f%%',pctdistance=0.7, shadow=True)
plt.pie(y_c, colors=colors_1, shadow=True)
plt.annotate('R2L(0.23%)',xy=(0.88,0.0055),xytext=(1.15,-0.16),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.annotate('U2R(0.01%)',xy=(0.2748,0.9525),xytext=(0.36,1.25),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.annotate('Probing(0.83%)',xy=(0.88,0.029),xytext=(1,0.32),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.text(0.8,0.70,'Normal')
plt.text(0.52,0.46,'19.69%')
plt.text(-1.18,0.088,'DOS')
plt.text(-0.44,0.24,'79.24%')
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X_test)
classes=['Normal','Probing','DOS','U2R','R2L']
colors=['blue','red','y','m','w']
plt.figure()
plt.title("Baseline")

for index,label,color in zip(range(len(classes)),classes,colors):
    plt.scatter(X_embedded[y_test==label,0],
                X_embedded[y_test==label,1],
                label=classes[index],
                c=color)
plt.legend(loc='best')

# plt.show()
#分类器
from sklearn.ensemble import RandomForestClassifier
##建立模型
clf_1 = RandomForestClassifier(n_estimators=100)

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

#验证测试样本
clf_1.fit(X_train,y_train)
preditions_base=clf_1.predict(X_test)

###################################################################
#过采样 smote
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=42)
X_smote,y_s=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_smote)
X_s=scaler.transform(X_smote)  #X是ndarray

#可视化
y_smote=pd.DataFrame(y_s,columns=['label'])
print(y_smote.label.value_counts())

classes_1=['Probing','Normal','U2R','DOS','R2L']
plt.figure()
plt.title("SMOTE")
# y_smote['label'].value_counts().plot(kind='bar',rot=45)
y_sc=[87832,87832,87832,87832,87832]
plt.pie(y_sc, colors=colors_1, labels=classes_1,autopct='%1.2f%%',pctdistance=0.7, shadow=True)
from sklearn.model_selection import train_test_split
X_train_1,X_test_1,y_train_1,y_test_1=train_test_split(X_s,y_s,test_size=0.2,random_state=0)  #切分样本
from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X_test_1)
plt.figure()
plt.title("SMOTE")
for index,label,color in zip(range(len(classes)),classes,colors):
    plt.scatter(X_embedded[y_test_1==label,0],
                X_embedded[y_test_1==label,1],
                label=classes[index],
                c=color)
plt.legend(loc='best')
# plt.show()

#分类器
from sklearn.ensemble import RandomForestClassifier
##建立模型
clf_2 = RandomForestClassifier(n_estimators=100)
#验证测试样本
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
oversampler=EditedNearestNeighbours(random_state=42,n_neighbors=30)
X_enn,y_e=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_enn)
X_e=scaler.transform(X_enn)
#可视化
y_enn=pd.DataFrame(y_e,columns=['label'])
print(y_enn.label.value_counts())
plt.figure()
plt.title("ENN")
# y_enn['label'].value_counts().plot(kind='bar',rot=45)
y_ec=[1036,80347,52,52897,316]
# plt.pie(y_ec, colors=colors_1, labels=classes_1,autopct='%1.2f%%',pctdistance=0.7, shadow=True)
plt.pie(y_ec, colors=colors_1, shadow=True)
plt.annotate('R2L(0.19%)',xy=(0.927,0.00287),xytext=(1.174,-0.123),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.annotate('U2R(0.03%)',xy=(-0.953,0.0037),xytext=(-1.12,0.0374),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.annotate('Probing(0.63%)',xy=(0.927,0.0258),xytext=(1.17,0.175),
             arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
plt.text(0,1.10,'Normal')
plt.text(0,0.70,'48.80%')
plt.text(0,-1.10,'DOS')
plt.text(0,-0.70,'50.35%')
from sklearn.model_selection import train_test_split
X_train_2,X_test_2,y_train_2,y_test_2=train_test_split(X_e,y_e,test_size=0.2,random_state=0)  #切分样本


from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X_test_2)
plt.figure()
plt.title("ENN")
for index,label,color in zip(range(len(classes)),classes,colors):
    plt.scatter(X_embedded[y_test_2==label,0],
                X_embedded[y_test_2==label,1],
                label=classes[index],
                c=color)
plt.legend(loc='best')
# plt.show()
##建立模型
clf_3 = RandomForestClassifier(n_estimators=100)
#验证测试样本
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
oversampler=SMOTEENN(random_state=42,n_neighbors=30)
X_SMOTEENN,y_se=oversampler.fit_sample(X_3,y_3)
#标准化
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_SMOTEENN)
X_se=scaler.transform(X_SMOTEENN)  #X是ndarray
#可视化
y_smoteenn=pd.DataFrame(y_se,columns=['label'])
print(y_smoteenn.label.value_counts())
plt.figure()
plt.title("SMOTE+ENN")
# y_smoteenn['label'].value_counts().plot(kind='bar',rot=45)
y_sec=[82179,77791,82130,85496,85118]
plt.pie(y_sec, colors=colors_1, labels=classes_1,autopct='%1.2f%%',pctdistance=0.7, shadow=True)
from sklearn.model_selection import train_test_split
X_train_3,X_test_3,y_train_3,y_test_3=train_test_split(X_se,y_se,test_size=0.2,random_state=0)  #切分样本

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X_test_3)
plt.figure()
plt.title("SMOTE+ENN")
for index,label,color in zip(range(len(classes)),classes,colors):
    plt.scatter(X_embedded[y_test_3==label,0],
                X_embedded[y_test_3==label,1],
                label=classes[index],
                c=color)
plt.legend(loc='best')
# plt.show()
##建立模型
clf_4 = RandomForestClassifier(n_estimators=100)
#验证测试样本
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

time_end=time.time()
print(time_end-time_start,'s')

###########################################################################
#混淆矩阵
from sklearn.metrics import confusion_matrix
cnf_matrix=confusion_matrix(y_test,preditions_base)
cnf_matrix_smote=confusion_matrix(y_test_1,preditions_smote)
cnf_matrix_enn=confusion_matrix(y_test_2,preditions_enn)
cnf_matrix_smoteenn=confusion_matrix(y_test_3,preditions_smoteenn)
class_names=['DOS','Normal','Probing','R2L','U2R']
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
class_names=['DOS','Normal','Probing','R2L','U2R']
print(" Baseline")
print(test_mean)
print(classification_report(y_test,preditions_base,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
print(" SMOTE")
print(test_mean_1)
print(classification_report(y_test_1,preditions_smote,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
print("ENN")
print(test_mean_2)
print(classification_report(y_test_2,preditions_enn,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
print(" SMOTE+ENN")
print(test_mean_3)
print(classification_report(y_test_3,preditions_smoteenn,target_names=class_names,digits=6))
print("--------------------------------------------------------------------------------")
###########################################################################
#三种采样的比较
plt.figure()

plt.grid()
plt.plot(train_sizes,test_mean,color='b',linestyle='--',
         marker='p',markersize=5,
         label=' Baseline validation accuracy')
plt.plot(train_sizes,test_mean_1,color='k',linestyle='--',
         marker='+',markersize=5,
         label='SMOTE validation accuracy')
plt.plot(train_sizes,test_mean_2,color='y',linestyle='--',
         marker='*',markersize=5,
         label=' ENN validation accuracy')
plt.plot(train_sizes,test_mean_3,color='m',linestyle='--',
         marker='o',markersize=5,
         label=' SMOTE+ENN validation accuracy')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
