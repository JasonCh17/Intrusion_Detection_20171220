import numpy as np
b_dos_p=[0.881720
,0.934579
,0.991063
,0.998556
,0.928177
,0.997117
,0.995814
,0.999494
,0.997932
,0.998513

]
b_dos_r=[0.881720
,1.000000
,0.991302
,1.000000
,0.928177
,0.996160
,0.999183
,0.999422
,0.994334
,0.997624

]
b_dos_f=[0.881720
,0.966184
,0.991183
,0.999278
,0.928177
,0.996639
,0.997496
,0.999458
,0.996130
,0.998068

]
b_normal_p=[0.999101
,0.995471
,0.992387
,0.994857
,0.997783
,0.922042
,0.989606
,0.986861
,0.992616
,0.997398

]
b_normal_r=[0.998894
,0.999161
,0.999410
,0.996871
,0.999284
,0.999439
,0.991595
,0.992658
,0.996294
,0.996490

]
b_normal_f=[0.998997
,0.997312
,0.995886
,0.995863
,0.998533
,0.959181
,0.990600
,0.989751
,0.994451
,0.996944

]
b_probling_p=[0.000000
,0.967033
,0.976471
,0.943925
,1.000000
,0.979719
,0.967742
,0.975000
,0.777778
,0.850000

]
b_probling_r=[0.000000
,0.814815
,0.718615
,0.990196
,0.944039
,0.927622
,0.904523
,0.886364
,0.945946
,0.967480
]
b_probling_f=[0.000000
,0.884422
,0.827930
,0.966507
,0.971214
,0.952959
,0.935065
,0.928571
,0.853659
,0.904943
]
b_r2l_p=[0.000000
,0.400000
,0.500000
,0.666667
,0.333333
,0.989583
,0.979381
,0.000000
,0.333333
,0.000000
]
b_r2l_r=[0.000000
,0.038462
,0.066667
,0.060606
,1.000000
,0.119048
,0.969388
,0.000000
,1.000000
,0.000000
]
b_r2l_f=[0.000000
,0.070175
,0.117647
,0.111111
,0.500000
,0.212528
,0.974359
,0.000000
,0.500000
,0.000000
]
b_u2r_p=[1.000000
,0.000000
,0.500000
,0.000000
,1.000000
,0.000000
,1.000000
,1.000000
,0.000000
,1.000000
]
b_u2r_r=[0.500000
,0.000000
,1.000000
,0.000000
,1.000000
,0.000000
,0.277778
,0.500000
,0.000000
,0.444444
]
b_u2r_f=[0.666667,
0.000000,
0.666667,
0.000000,
1.000000,
0.000000,
0.434783,
0.666667,
0.000000,
0.615385
]
dos_p=np.mean(b_dos_p)
dos_r=np.mean(b_dos_r)
dos_f=2*dos_p*dos_r/(dos_p+dos_r)
normal_p=np.mean(b_normal_p)
normal_r=np.mean(b_normal_r)
normal_f=2*normal_p*normal_r/(normal_p+normal_r)
probling_p=np.mean(b_probling_p)
probling_r=np.mean(b_probling_r)
probling_f=2*probling_p*probling_r/(probling_p+probling_r)
u2r_p=np.mean(b_u2r_p)
u2r_r=np.mean(b_u2r_r)
u2r_f=2*u2r_p*u2r_r/(u2r_p+u2r_r)
r2l_p=np.mean(b_r2l_p)
r2l_r=np.mean(b_r2l_r)
r2l_f=2*r2l_p*r2l_r/(r2l_p+r2l_r)
print('b_dos_p',np.mean(b_dos_p))
print('b_dos_r',np.mean(b_dos_r))
# print('b_dos_f',np.mean(b_dos_f))
print('b_dos_f',dos_f)
print('b_normal_p',np.mean(b_normal_p))
print('b_normal_r',np.mean(b_normal_r))
# print('b_normal_f',np.mean(b_normal_f))
print('b_normal_f',normal_f)
print('b_probling_p',np.mean(b_probling_p))
print('b_probling_r',np.mean(b_probling_r))
# print('b_probling_f',np.mean(b_probling_f))
print('b_probling_f',probling_f)
print('b_r2l_p',np.mean(b_r2l_p))
print('b_r2l_r',np.mean(b_r2l_r))
# print('b_r2l_f',np.mean(b_r2l_f))
print('b_r2l_f',r2l_f)
print('b_u2r_p',np.mean(b_u2r_p))
print('b_u2r_r',np.mean(b_u2r_r))
# print('b_u2r_f',np.mean(b_u2r_f))
print('b_u2r_f',u2r_f)
print('#############################################')
