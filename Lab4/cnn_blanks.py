'''Information
This template is for Deep-learning class 2021.
Please read the assignment requirements in detail and fill in the blanks below.
Any part that fails to meet the requirements will deduct the points.
Please answer carefully.
'''

# Please import the required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.dataset import random_split
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# Define NeuralNetwork
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # image shape is 3 * 32 * 32, which
        # 3 is for RGB three-color channel, 32 * 32 is for image size

        # ------- convalution layer -------
        # please add at least one more layer of conv
        # ::: your code :::
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5) #output(8*28*28)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5) #output(16*24*24)
        # ::: end of code :::

        # ------- pooling layer and activation function -------
        # ::: your code :::
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # ::: end of code ::: 

        # ------- fully connected layers -------
        # ::: your code :::
        self.fc1 = nn.Linear(16*5*5 , 100)    #input_shape=(16*5*5)
        self.fc2 = nn.Linear(100 , 50)
        self.fc3 = nn.Linear(50, 10)
        # The number of neurons in the last fully connected layer 
        # should be the same as number of classes
        # ::: end of code :::
        

    def forward(self, x):
        # first conv
        x = self.pool(F.relu(self.conv1(x))) # please count out the size for each layer
        # example: x = self.pool(self.relu(self.conv1(x))) # output size = 10x25x25
        # second conv
        # ::: your code :::
        x = self.pool(F.relu(self.conv2(x)))             #input size = 8x14x14
                                                            #output size = 16x7x7
        # ::: end of code :::

        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        # fully connection layers
        # ::: your code :::
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # ::: end of code :::

        return x


def train():
    # Device configuration
    # ::: your code :::
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ::: end of code :::

    # set up basic parameters
    # ::: your code :::
    learning_rate = 0.001
    batch_size = 100
    num_epochs = 5
    # ::: end of code :::

    # step 0: import the data and set it as Pytorch dataset
    # Dataset: CIFAR10 dataset
    # Cifar-10 data
    CIFAR10_train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    CIFAR10_test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    
    #將訓練資料分為訓練和驗證資料
    train_size = int(0.8 * len(CIFAR10_train_data))
    valid_size = len(CIFAR10_train_data) - train_size
    train_data, valid_data = random_split(CIFAR10_train_data, [train_size, valid_size])

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size) 
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size)
    test_loader = DataLoader(dataset=CIFAR10_test_data, batch_size=batch_size)

    # step 1: set up models, criterion and optimizer
    model = ConvolutionalNeuralNetwork().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # step 2: start training
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        valid_loss = float(0)
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            # ::: your code :::
            # init optimizer
            optimizer.zero_grad()

            # forward -> backward -> update
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()       #update weigths and bias
            # ::: end of code :::
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')

        # step 3: Validation loop
        # ::: your code :::
        for i, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss_v = loss_func(outputs, labels)
            valid_loss += loss_v.item()
        valid_loss = valid_loss / len(valid_loader)
        print('\tValidation Loss : {:.4f}'.format(valid_loss))
        # ::: end of code :::
    print('Finished Training')

    # set model to Evaluation Mode
    model = model
    

    # step 4: Testing loop
    # no grad here
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        # run through testing data
        
        # ::: your code :::
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            # ::: end of code :::
            outputs = model(images)
            # max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {i}: {acc} %')

    # save your model
    # ::: your code :::
    FILE = 'model_all.pt'
    torch.save(model, FILE)
    torch.save(model.state_dict(), FILE)
    # ::: end of code :::


if __name__ == '__main__':
    train()
