import numpy as np
import matplotlib.pyplot as plt
from utils import *



train_x,train_y,test_x,test_y = load_dataset()
print('shape of train_x is '+str(train_x.shape))#(209, 12288)
print('shape of train_y is '+str(train_y.shape))#(209,1)
print('shape of test_x is '+str(test_x.shape))#(50, 12288)
print('shape of test_y is '+str(test_y.shape))#(50,1)

#train from scratch,run these two line
W,B = initial_W_B()
params = optimize(train_x.T,train_y.T,W,B,0.01,3000)

#train from trained weights,run these two line
# dic_param=np.load('myparams.npy',allow_pickle=True).item()
# params = optimize(train_x.T,train_y.T,dic_param['w'],dic_param['b'],0.01,200)


acc = predict(test_x.T,test_y.T,params["w"],params["b"])
print("测试集准确率为"+str(acc*100)+"%")


