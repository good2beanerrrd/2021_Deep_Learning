import random

x = [
    [1, 0],
    [1,3],
    [2,-6],
    [-1,-3],
    [-5,5],
    [5,2],
    [-2,2],
    [-7,2],
    [4,-4],
    [-5,-1]
]
y = [ 1, -1, 1, 1, -1, 1, -1, -1, 1, -1]
weight = []

#guess the weights and bias
bias = random.randrange(0, 1)
for i in range(2):
    weight.append(random.randrange(-5, 5))

#check if the update is needed for the weight vectors and bias
for i in range(10):

    signValue = 0
    for j in range(2):
        signValue += x[i][j] * weight[j]
    signValue += bias

    #if signValue!=y, then the update is required
    if signValue != y[i]:
        for j in range(2):
            weight[j] = weight[j] + y[i]*x[i][j]
        bias += y[i]

print("{}{}".format("w1 = ", weight[0]))
print("{}{}".format("w2 = ", weight[1]))
print("{}{}".format("bias = ", bias))

#test
print("Testing data:")
test = [
    [2, -4],
    [-5, 1],
    [-2, -2]
]

for i in range(len(test)):
    for j in range(2):
        signValue += test[i][j] * weight[j]
        signValue += bias
    
    if signValue >= 0:
        result = 1
    else:
        result = -1

    print("{}{}{}{}".format("Class of example ", i+1, ": ", result))