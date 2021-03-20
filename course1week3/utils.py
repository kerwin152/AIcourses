# -*- coding: UTF-8 -*-
'''
@Date ：2021/3/19 15:04 
@Author ：Kerwin
@Project ：course1week3
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
    :return: array after sigmoid process
    '''
    return 1/(1+np.exp(-z))

def tanh(z):
    '''
    :param z: array
    :return:array after tanh process
    '''
    a = (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
    return a

def propagate(X,Y,params):
    '''
    :param X: (特征数,图片数)
    :param Y: (1,图片数)
    :param params{W1： (3,特征数)
                  W2: (1,3)
                  B1: 常数
                  B2: 常数}
    :return:
    '''

    m = X.shape[1] #图片数

    Z1 = np.dot(params["W1"],X)+params["B1"]#(3，图片数)
    A1 = tanh(Z1)#(3，图片数)#隐藏层用tanh激活函数
    Z2 = np.dot(params["W2"],A1)+params["B2"]#(1,图片数)
    A2 = sigmoid(Z2) #输出层用sigmoid函数
    cost = -Y*np.log(A2)-(1-Y)*np.log(1-A2)
    m = Y.shape[1]
    loss = cost.sum()/m

    #反向传播
    dZ2 = A2-Y #(1,图片数)
    dW2 = (1/m)*np.dot(dZ2,A1.T) #（1,3）
    dB2 = dZ2.sum()/m
    dZ1 = np.multiply(np.dot(params["W2"].T,dZ2),1-np.power(A1,2))#(3,图片数)
    dW1 = (1/m)*np.dot(dZ1,X.T)#(3,特征数)
    dB1 = dZ1.sum()/m

    grads={
        "dW2":dW2,
        "dW1":dW1,
        "dB1":dB1,
        "dB2":dB2
    }
    return (grads,loss)

def optimize(X,Y,initial_WB,learning_rate=0.001,iteration=100):
    losslist=[]
    params=initial_WB
    for i in range(iteration):
        grads,loss = propagate(X,Y,params)
        params["W1"] = params["W1"] - learning_rate * grads["dW1"]
        params["W2"] = params["W2"] - learning_rate * grads["dW2"]
        params["B1"] = params["B1"] - learning_rate * grads["dB1"]
        params["B2"] = params["B2"] - learning_rate * grads["dB2"]

        if i%10 == 0:
            losslist.append(loss)
            print("迭代的次数: %i ， 误差值： %f" % (i, loss))

    #save weights
    np.save('myparams.npy', params)

    plt.plot(losslist)
    plt.ylabel('loss')
    plt.xlabel('iteration(per tens)')
    plt.title('learning rate='+str(learning_rate))
    plt.show()

    return params

def predict(X,Y,params):
    m = X.shape[1]
    Y_pred = np.zeros((1,m))

    Z1 = np.dot(params["W1"], X) + params["B1"]  # (3，图片数)
    A1 = tanh(Z1)  # (3，图片数)
    Z2 = np.dot(params["W2"], A1) + params["B2"]  # (1,图片数)
    A2 = sigmoid(Z2)  # 输出层用sigmoid函数

    for i in range(X.shape[1]):
        Y_pred[0,i] = 1 if A2[0,i] > 0.5 else 0
    accuracy = Y-Y_pred #0为准确，不为0就不准确
    cur=0
    for i in range(accuracy.shape[1]):
        if accuracy[0,i] == 0:
            cur = cur+1
    acc = cur / accuracy.shape[1]

    return acc