# -*- coding: UTF-8 -*-
'''
@Date ：2021/3/19 15:03 
@Author ：Kerwin
@Project ：course1week3
@File ：main.py.py
'''
from utils import *



train_x,train_y,test_x,test_y = load_dataset()
print('shape of train_x is '+str(train_x.shape))#(209, 12288)
print('shape of train_y is '+str(train_y.shape))#(209,1)
print('shape of test_x is '+str(test_x.shape))#(50, 12288)
print('shape of test_y is '+str(test_y.shape))#(50,1)

pp={
    "W1":np.random.random((3,12288)),
    "W2":np.random.random((1,3)),
    "B1":0,
    "B2":0
}
grads,loss = propagate(train_x.T,train_y.T,pp)
print(grads)
print(loss)