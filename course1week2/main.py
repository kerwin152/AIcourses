import numpy as np
import matplotlib.pyplot as plt
from utils import *



train_x,train_y,test_x,test_y = load_dataset()
print('shape of train_x is '+str(train_x.shape))#(209, 12288)
print('shape of train_y is '+str(train_y.shape))#(209,1)
print('shape of test_x is '+str(test_x.shape))#(50, 12288)
print('shape of test_y is '+str(test_y.shape))#(50,1)

W,B = initial_W_B()
# propagate(train_x.T,train_y.T,W,B)
params,lossList = optimize(train_x.T,train_y.T,W,B,0.01,2000)
Ypre = predict(test_x.T,params["w"],params["b"])
print(Ypre)
print(test_y.T)

