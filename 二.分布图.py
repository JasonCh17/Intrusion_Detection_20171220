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
y_1=data['label'].copy()   #一维
##y的处理
u2r=["buffer_overflow.","loadmodule.","perl.","rootkit."]
r2l=["ftp_write.","imap.","guess_passwd.","phf.","spy.","multihop.","warezmaster.","warezclient."]
dos=["back.","land.","pod.","smurf.","teardrop.",'neptune.']
probe=["satan.","portsweep.","ipsweep.","nmap."]
for i in u2r:
    y_1[y_1==i]="U2R"
for i in r2l:
    y_1[y_1==i]="R2L"
for i in dos:
    y_1[y_1==i]="DOS"
for i in probe:
    y_1[y_1==i]="Probing"
y_1[y_1=="normal."]="Normal"
#去重没做#
print(y_1.value_counts())
import matplotlib.pyplot as plt
y_1.value_counts().plot(kind='bar',rot=45)
# plt.figure()
# y_c=[4107,97278,52,391458,1126]
# classes=['Probing','Normal','U2R','DOS','R2L']
# colors  = ["pink","coral","yellow","orange",'blue']
# Pie Plot
# autopct: format of "percent" string;
# plt.pie(y_c, colors=colors, labels=classes,autopct='%1.2f%%',pctdistance=0.7, shadow=True)
# plt.pie(y_c, colors=colors, shadow=True)
# plt.annotate('R2L(0.23%)',xy=(0.88,0.0055),xytext=(1.15,-0.16),
#              arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
# plt.annotate('U2R(0.01%)',xy=(0.2748,0.9525),xytext=(0.36,1.25),
#              arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
# plt.annotate('Probing(0.83%)',xy=(0.88,0.029),xytext=(1,0.32),
#              arrowprops=dict(arrowstyle='<-', connectionstyle="arc3,rad=.2"))
# plt.text(0.8,0.70,'Normal')
# plt.text(0.52,0.46,'19.69%')
#
# plt.text(-1.18,0.088,'DOS')
# plt.text(-0.44,0.24,'79.24%')
# # plt.annotate()
plt.show()