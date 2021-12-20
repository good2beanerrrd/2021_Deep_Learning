# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
import warnings
 
#ignore warning
warnings.filterwarnings("ignore", category = Warning )

#read and convert the training data to numpy array
train_Data = pd.read_csv('./train.csv')
train_Data = pd.DataFrame(train_Data).to_numpy()
train_Data_Y = pd.DataFrame(train_Data[:,0]).to_numpy()    #y array
train_Data = np.delete(train_Data, 0, 1)

#read and convert the training data to numpy array
test_Data = pd.read_csv('./test.csv')
test_Data = pd.DataFrame(test_Data).to_numpy()

#initailize w
def init_weight():
    w = np.random.uniform(-1.0, 1.0, 784)
    return w

def sigmoid(n):
	return 1.0 / (1.0 + np.exp(-n))

def Cross_Entropy(y, y_Hat):
	return -(y * np.log(y_Hat) + (1 - y) * np.log(1 - y_Hat))

def Gradient_Descent(trainData, trainData_Y, init_w, b, Error, learningRate, epoch):
    w = init_w.copy()
    count = 0
    _error = 1
    while count <= epoch:
        if _error < Error:                                   #小於容忍誤差，停止訓練
            break
        
        _error = 0                                         #歸零，重新計算此世代的誤差
        for x, y in zip(trainData, train_Data_Y):
            w += learningRate * (y - sigmoid(b + w.T.dot(x))) * x
            b += learningRate * (y - sigmoid(b + w.T.dot(x)))
            _error += Cross_Entropy(y, sigmoid(b + w.T.dot(x)))
        
        count += 1
        _error /= -len(train_Data)
    
    if count > epoch:
        count -= 1

    print("{}{}".format("The maximum number of epoches is ", count))
    print("{}{}".format("Learning Rate: ", learningRate))
    print("{}{}".format("Bias: ", b))
    print("{}\n{}".format("Weights: ", w))

    return w, b
    
def predict(testData, w, b):
    cmp = sigmoid(b + w.T.dot(train_Data[0]))
    ans = []
    for x in testData:
        if sigmoid(b + w.T.dot(x)) - cmp >= 0.999:
            label = 5
        else:
            label = 2
        ans.append(label)
    return ans

def main(epoch, LearningRate, tau):
    init_b = 1
    init_w = init_weight()
    w, b = Gradient_Descent(train_Data, train_Data_Y, init_w, init_b, tau, LearningRate, epoch)
    ans = predict(test_Data, w, b)
    ans = pd.DataFrame(ans)
    ans.to_csv('./test_ans.csv', index=False)
    
if __name__ == '__main__':
    # main(epoch, learningRate, tau)
    main(30, 0.5, 0.01)