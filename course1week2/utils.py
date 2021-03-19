# -*- coding: UTF-8 -*-
'''
@Date ：2021/3/17 8:59 
@Author ：Kerwin
@Project ：course1week2
@File ：utils.py
'''

import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    train_dataset = h5py.File('./datasets/train_catvnoncat.h5','r')
    train_set_x = np.array(train_dataset['train_set_x'][:])#shape (209, 64, 64, 3)
    train_set_y = np.array(train_dataset['train_set_y'][:])#shape (209,)
    train_set_y = train_set_y.reshape(209,1)#这个必须要有，不然后面转置转不过来

    test_dataset = h5py.File('./datasets/test_catvnoncat.h5','r')
    test_set_x = np.array(test_dataset['test_set_x'][:])#shape (50, 64, 64, 3)
    test_set_y = np.array(test_dataset['test_set_y'][:])#shape (50,)
    test_set_y = test_set_y.reshape(50,1)

    train_set_x_reshape = train_set_x.reshape(-1,64*64*3)/255#(209, 12288)
    test_set_x_reshape = test_set_x.reshape(-1,64*64*3)/255#(50, 12288)

    return train_set_x_reshape,train_set_y,test_set_x_reshape,test_set_y

def sigmoid(z):
    '''

    :param z: array
    :return: after sigmoid process
    '''
    return 1/(1+np.exp(-z))

def initial_W_B():#用0来初始化，0-1随机初始化，由于特征太多，导致sigmoid无线趋近于1
    W = np.zeros((12288,1))
    B = 0
    return W,B

def propagate(X,Y,W,B):
    '''
    :param X: (特征数，图片数)
    :param Y: (1，图片数)
    :param W: (特征数,下一层节点数)=(特征数,1)
    :param B: 常量
    :return: grads-梯度信息  loss-损失值
    '''
    Z = np.dot(W.T,X)+B
    A = sigmoid(Z)
    cost = -Y*np.log(A)-(1-Y)*np.log(1-A)
    m = Y.shape[1]
    loss = cost.sum()/m

    dZ=A-Y
    dW=(1/m)*np.dot(X,dZ.T)#(12288,1)
    dB=(1/m)*(dZ.sum()) #常数

    grads={
        "dw":dW,
        "db":dB
    }
    return (grads,loss)


def optimize(X,Y,W,B,learn_rate=0.001,iteration=10):
    '''
    :param X: (特征数，图片数)
    :param Y: (1，图片数)
    :param W: (下一层节点数,特征数)=(1,特征数)
    :param B: 常量
    :param learn_rate: 学习率
    :param iteration: 迭代次数
    :return:
    '''
    losslist=[]
    for i in range(iteration):
        grads,loss = propagate(X,Y,W,B)
        W = W - learn_rate*grads["dw"]
        B = B - learn_rate*grads["db"]

        if i%10==0 :
            losslist.append(loss)
            print("迭代的次数: %i ， 误差值： %f" % (i, loss))

    params={
        "w":W,
        "b":B
    }
    #save weights
    np.save('myparams.npy', params)

    plt.plot(losslist)
    plt.ylabel('loss')
    plt.xlabel('iteration(per tens)')
    plt.title('learning rate='+str(learn_rate))
    plt.show()


    return params

def predict(X,Y,W,B):
    m = X.shape[1]
    Y_pred = np.zeros((1,m))

    A = sigmoid(np.dot(W.T,X)+B)

    for i in range(X.shape[1]):
        Y_pred[0,i] = 1 if A[0,i] > 0.5 else 0
    accuracy = Y-Y_pred #0为准确，不为0就不准确
    cur=0
    for i in range(accuracy.shape[1]):
        if accuracy[0,i] == 0:
            cur = cur+1
    acc = cur / accuracy.shape[1]

    return acc









