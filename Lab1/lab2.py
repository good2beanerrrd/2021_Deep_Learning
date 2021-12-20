import pandas as pd
import numpy as np
import random

trainData = pd.read_csv('Iris_training.txt', header=None)
data = np.array(trainData)

weight = []

#guess the weights and bias
bias = random.randrange(0, 1)
for i in range(2):
    weight.append(random.randrange(-1, 1))

#check if the update is needed for the weight vectors and bias
for i in range(10):

    signValue = 0
    for j in range(2):
        signValue += data[i][j] * weight[j]
    signValue += bias

    #if signValue!=y, then the update is required
    if signValue != data[i][j+1]:
        for j in range(2):
            weight[j] = weight[j] + data[i][j+1]*data[i][j]
        bias += data[i][j+1]

print("{}{}".format("w1 = ", weight[0]))
print("{}{}".format("w2 = ", weight[1]))
print("{}{}".format("bias = ", bias))

#test
testData = pd.read_csv('Iris_test.txt', header=None)
test = np.array(testData)
result = []

for i in range(len(test)):
    for j in range(2):
        signValue += test[i][j] * weight[j]
        signValue += bias
    
    if signValue >= 0:
        result.append(1)
    else:
        result.append(-1)

#calculate the accuracy
ctr = 0
for i in range(len(test)):
    if result[i] == int(test[i][2]):
        ctr += 1

print("{}{}{}".format("The accurascy rate is ", ctr, " out of 10."))