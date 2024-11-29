
#AUTHOR: AARON WHITAKER
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#import tensorflow as tf
from tensorflow import keras

class ConvolutedSudokuModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvolutedSudokuModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

#Loads sudoku.csv dataset
quizzes = np.zeros((1000000, 81), np.int32)
solutions = np.zeros((1000000, 81), np.int32)
for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
    quiz, solution = line.split(",")
    for j, q_s in enumerate(zip(quiz, solution)):
        q, s = q_s
        quizzes[i, j] = q
        solutions[i, j] = s
quizzes = quizzes.reshape((-1, 9, 9))
solutions = solutions.reshape((-1, 9, 9))




    

print(quizzes)
print(solutions)

df = pd.read_csv('sudoku.csv')
df.head()



X = np.array(df.quizzes.map(lambda x: list(map(int, x))).to_list())
Y = np.array(df.solutions.map(lambda x: list(map(int, x))).to_list())

#print(X)
#print(Y)

X = X.reshape(-1, 9, 9, 1)
Y = Y.reshape(-1, 9, 9) - 1

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

print(X_train.shape)

model = ConvolutedSudokuModel()
criterion = nn.CrossEntropyLoss()
custom_optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.005)
model_loss = []

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for x in X_train:
        
        outputs = model(x.flatten())
        #loss = criterion(outputs, labels)
        
        #custom_optimizer.zero_grad()
        #loss.backward()
        #custom_optimizer.step()

    #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    #model_loss.append(loss.item())