baseline_F1= [ 0.78907761,  0.8355581 ,  0.79922861,  0.83033056 , 0.83559669 , 0.87135632,
  0.87249762,  0.88483035 , 0.85835904 , 0.85816815 , 0.86856201 , 0.88059487,
  0.88503366 , 0.88710555 , 0.89120769 , 0.88033023,  0.89329652 , 0.89814154,
  0.90345483 , 0.88338819]
baseline_Precison =[ 0.85902802  ,0.85676731 , 0.92760008,  0.90200586 , 0.9117712 ,  0.94644652,
  0.91537288 , 0.93739853 , 0.959948 ,   0.9559059 ,  0.9523177,  0.92626594,
  0.93922553 , 0.93341136 , 0.95412179 , 0.95144623  ,0.95687444 , 0.94925817,
  0.96051529 , 0.94605064]
baseline_Recall=  [ 0.74037671  ,0.79362673 , 0.79933972 , 0.8119757  , 0.80535805 , 0.82745009,
  0.83514727 , 0.84870877 , 0.83490002  ,0.84137803 , 0.85926981 , 0.84194116,
  0.85338975 , 0.86869935 , 0.85884678 , 0.85564322 , 0.8654266 ,  0.8608832,
  0.8439074  , 0.86332866]

enn_F1= [ 0.86952272 , 0.86124273 , 0.88981063  ,0.88660733 , 0.8872364  , 0.89183214,
  0.88669466 , 0.89232325 , 0.88506181 , 0.8935101  , 0.89697957 , 0.90607908,
  0.904395  ,  0.8939812 ,  0.90800501 , 0.90984922 , 0.91648819 , 0.91022679,
  0.91787976 , 0.90243834]
enn_Precison =[ 0.93624623 , 0.95796765 , 0.95355144 , 0.95980014,  0.94839461 , 0.9596189,
  0.95960947 , 0.97987685 , 0.95986591 , 0.96986439 , 0.96987995 , 0.97323197,
  0.94964229 , 0.97989872 , 0.99989081,  0.97322885  ,0.97987676 , 0.96488929,
  0.94588778 , 0.96655916]
enn_Recall =[ 0.83200285 , 0.84543007, 0.86293535  ,0.85901343  ,0.87184004 , 0.86738945,
  0.86005411 , 0.87134565  ,0.86631668,  0.87159266  ,0.87738646 , 0.87712028,
  0.89235569 , 0.87682502,  0.88134263 , 0.88183653,  0.88534585 , 0.87158182,
  0.88634114  ,0.88711037]
smote_F1 =[ 0.9934512  , 0.99481991,  0.9953899  , 0.99567193 , 0.9959543 ,  0.99614711,
  0.99621832  ,0.99625262  ,0.99630672 , 0.99628116 , 0.99641773 , 0.99642053,
  0.99647476 , 0.9964377  , 0.99653743,  0.99650612  ,0.99653733 , 0.99657737,
  0.99655162 , 0.99658581]
smote_Precison= [ 0.99320774 , 0.99476246 , 0.99538685 , 0.99565346,  0.99595532 , 0.99613809,
  0.99626648 , 0.99625006 , 0.99635306 , 0.9963894 ,  0.99648668 , 0.99643002,
  0.99653564 , 0.99649247 , 0.99653528 , 0.9965807 ,  0.99659477 , 0.99657543,
  0.99658915 , 0.99662355]
smote_Recall =[ 0.99320745 , 0.9946345 ,  0.99534062,  0.99571641 , 0.99590726 , 0.99623242,
  0.99621463 , 0.9962381 ,  0.99626092 , 0.99640035 , 0.99643741  ,0.99643183,
  0.99651145 , 0.99649431 , 0.99651998 , 0.99649439 , 0.9965285 ,  0.99653983,
  0.99653978 , 0.99661381]
smoteenn_F1 =[ 0.99859344 , 0.9995305 ,  0.99966092 , 0.9997342 , 0.99981439  ,0.99982759,
  0.99984528 , 0.99986994  ,0.99988549 , 0.99989165 , 0.99992245 , 0.99992543,
  0.99993178 , 0.9999536  , 0.99993465  ,0.99993192 , 0.99994433 , 0.99995647,
  0.99995047 , 0.99995062]
smoteenn_Precison =[ 0.99871447 , 0.99955476 , 0.99965315  ,0.99974495 , 0.99977695 , 0.99980656,
  0.99989528 , 0.99987276 , 0.99988912 , 0.99990962 , 0.99991266 , 0.99993426,
  0.99992483 , 0.99993808 , 0.99993759,  0.99992817 , 0.99991967 , 0.99994744,
  0.99994032 , 0.99995592]
smoteenn_Recall= [ 0.99858439 , 0.99953233 , 0.99966532,  0.99975378 , 0.99982035,  0.9998319,
  0.99987952 , 0.99987961 , 0.99987561 , 0.99991647  ,0.99991246,  0.99992515,
  0.99992312 , 0.99993822 , 0.99995981 , 0.99995351 , 0.99993633,  0.99994334,
  0.99996344 , 0.99995424]
#三种采样的比较
import matplotlib.pyplot as plt
import numpy as np
# train_sizes =[  5240 , 10481  ,15722 , 20963  ,26204  ,31445  ,36686  ,41927 , 47168  ,52409,
#   57650 , 62891  ,68132 , 73373  ,78614  ,83855, 89096  ,94337 , 99578 ,104819]
train_sizes = np.linspace(0.05, 1, 20)
plt.figure()
plt.grid()
plt.plot(train_sizes,baseline_Precison,color='green',linestyle='--',
         marker='s',markersize=5,label="Baseline validation precision")

plt.plot(train_sizes,smote_Precison,color='k',linestyle='--',
         marker='+',markersize=5,
         label='SMOTE validation precison')
plt.plot(train_sizes,enn_Precison,color='y',linestyle='--',
         marker='*',markersize=5,
         label=' ENN validation precison')
plt.plot(train_sizes,smoteenn_Precison,color='r',linestyle='--',
         marker='*',markersize=5,
         label=' SMOTE+ENN validation precison')
plt.xlabel('Percentage of training samples')
plt.ylabel('Precision')
plt.legend(loc='lower right')

plt.figure()
plt.grid()
plt.plot(train_sizes,baseline_Recall,color='green',linestyle='--',
         marker='s',markersize=5,label="Baseline validation recall")

plt.plot(train_sizes,smote_Recall,color='k',linestyle='--',
         marker='+',markersize=5,
         label='SMOTE validation recall')
plt.plot(train_sizes,enn_Recall,color='y',linestyle='--',
         marker='*',markersize=5,
         label=' ENN validation recall')
plt.plot(train_sizes,smoteenn_Recall,color='r',linestyle='--',
         marker='*',markersize=5,
         label=' SMOTE+ENN validation recall')
plt.xlabel('Percentage of training samples')
plt.ylabel('Recall')
plt.legend(loc='lower right')

plt.figure()
plt.grid()
plt.plot(train_sizes,baseline_F1,color='green',linestyle='--',
         marker='s',markersize=5,label="Baseline validation f1-value")

plt.plot(train_sizes,smote_F1,color='k',linestyle='--',
         marker='+',markersize=5,
         label='SMOTE validation f1-value')
plt.plot(train_sizes,enn_F1,color='y',linestyle='--',
         marker='*',markersize=5,
         label=' ENN validation f1-value')
plt.plot(train_sizes,smoteenn_F1,color='r',linestyle='--',
         marker='*',markersize=5,
         label=' SMOTE+ENN validation f1-value')
plt.xlabel('Percentage of training samples')
plt.ylabel('F1-Value')
plt.legend(loc='lower right')
plt.show()