
#AUTHOR: AARON WHITAKER
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Activation, Reshape, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D,BatchNormalization
#from tensorflow.keras.optimizers import RMSprop
#from tensorflow.keras.optimizers import Adam, SGD, RMSprop

class ConvolutedSudokuModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvolutedSudokuModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(True)
        self.norm1 = nn.BatchNorm2d(64)
        
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(True)
        self.norm2 = nn.BatchNorm2d(64)
        
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True)
        self.relu3 = nn.ReLU(True)
        self.norm3 = nn.BatchNorm2d(64)
        
        self.last_conv = nn.Conv2d(64, 9, 1)
        #self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 1)
        
        #self.fc1 = nn.Linear(1, 128)
        
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        
        out = self.conv_layer2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        
        out = self.conv_layer3(out)
        out = self.norm3(out)
        out = self.relu3(out)

        out = self.last_conv(out)
        return out

#Loads sudoku.csv dataset
#quizzes = np.zeros((1000000, 81), np.int32)
#solutions = np.zeros((1000000, 81), np.int32)
#for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
#    quiz, solution = line.split(",")
#    for j, q_s in enumerate(zip(quiz, solution)):
#        q, s = q_s
#        quizzes[i, j] = q
#        solutions[i, j] = s
#quizzes = quizzes.reshape((-1, 9, 9))
#solutions = solutions.reshape((-1, 9, 9))   

#print(quizzes)
#print(solutions)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('sudoku.csv')
df_x = df['quizzes']
df_y = df['solutions']

#X = np.array(df.quizzes.map(lambda x: list(map(int, x))).to_list())
#Y = np.array(df.solutions.map(lambda x: list(map(int, x))).to_list())

X=[]
Y=[]
for i in df_x:
    x = np.array([int(j) for j in i]).reshape((1,9,9))
    X.append(x)
        
X = np.array(X)
X = X/9
X -= .5  

for i in df_y:
    
        y = np.array([int(j) for j in i]).reshape((9,9)) - 1
        Y.append(y)   
    
Y = np.array(Y)

#print(X)
#print(Y)

#X = X.reshape(-1, 9, 9)
#Y = Y.reshape(-1, 9, 9) - 1

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.99)
_, X_test, _, y_test = train_test_split(X_test, y_test, test_size = 0.99)
train_dat    = torch.utils.data.TensorDataset(torch.tensor(np.float32(X_train)), torch.tensor(np.float32(y_train)))
train_loader = torch.utils.data.DataLoader(dataset = train_dat,
                                           batch_size = 64,
                                           shuffle = True,
                                           )

test_dat    = torch.utils.data.TensorDataset(torch.tensor(np.float32(X_test)), torch.tensor(np.float32(y_test)))
test_loader = torch.utils.data.DataLoader(dataset = train_dat,
                                           batch_size = 64,
                                           shuffle = True
                                           )
print(X_train.shape)

model = ConvolutedSudokuModel()
criterion = nn.CrossEntropyLoss()
custom_optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.005)


num_epochs = 50
model_train_loss = []
model_train_accuracy = []
model_test_loss = []
model_test_accuracy = []
print("Starting training")

for epoch in range(num_epochs):
    model.train()
    y_true = []
    y_pred = []
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        
        #print(x)
        loss = criterion(outputs, (y).long())
        custom_optimizer.zero_grad()
        loss.backward()
        custom_optimizer.step()
        
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Accuracy: {np.equal(np.argmax((y_true), axis=-1), np.argmax(y_pred, axis=-1)).mean()}')
    model_train_loss.append(loss.item())
    model_train_accuracy.append(np.equal(np.argmax((y_true), axis=-1), np.argmax(y_pred, axis=-1)).mean())
    
    model.eval()
    y_true = []
    y_pred = []
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(y.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())
        #print(x)
        loss = criterion(outputs, (y).long())
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {loss.item():.4f}, Test Accuracy: {np.equal(np.argmax((y_true), axis=-1), np.argmax(y_pred, axis=-1)).mean()}')
    model_test_loss.append(loss.item())
    model_test_accuracy.append(np.equal(np.argmax((y_true), axis=-1), np.argmax(y_pred, axis=-1)).mean())
    

print("Model train loss: ", model_train_loss)
print("Model train accuracy: ", model_train_accuracy)
print("Model test loss: ", model_test_loss)
print("Model test accuracy: ", model_test_accuracy)

y_axis = []
i=1
for epoch in range(num_epochs):
    y_axis.append(i)
    i += 1
plt.plot(y_axis, model_train_loss)
plt.ylabel('CNN Train loss')  
plt.xlabel('Epochs')  
plt.title('Train losses over epochs')  
plt.show()

plt.plot(y_axis, model_test_loss)
plt.ylabel('CNN Test loss')  
plt.xlabel('Epochs')  
plt.title('Test losses over epochs')  
plt.show()

plt.plot(y_axis, model_train_accuracy)
plt.ylabel('CNN Train Accuracy')  
plt.xlabel('Epochs')  
plt.title('Train accuracy over epochs')  
plt.show()

plt.plot(y_axis, model_test_accuracy)
plt.ylabel('CNN Test Accuracy')  
plt.xlabel('Epochs')  
plt.title('Test accuracy over epochs')  
plt.show()
torch.save(model.state_dict(), "sudoku_cnn.pth")
#torch.save({
#            'epoch': 3,
#            'model_state_dict': model.state_dict(),
#            'optimizer_state_dict': custom_optimizer.state_dict(),
#            'loss': model_loss[-1],
#            }, 'sudoku_cnn.pth')

#model = ConvolutedSudokuModel()
#torch.serialization.add_safe_globals(['scalar'])
#model.load_state_dict(torch.load("sudoku_cnn.pth", weights_only=False))

#def evaluate_model(model, data_loader):
#    model.eval()
#    y_true = []
#    y_pred = []
#    with torch.no_grad():
#        for x, y in data_loader:
#            x, y = x.to(device), y.to(device)  # Move data to GPU
#            outputs = model(x)
#            _, predicted = torch.max(outputs, 1)
#            print(predicted)
#            y_true.extend(y.cpu().numpy())
#            y_pred.extend(predicted.cpu().numpy())
#    return np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)).mean()
#
#y_true_custom, y_pred_custom = evaluate_model(model, test_loader)
#print(y_true_custom)
#print(y_pred_custom)
#print("Custom Network Accuracy: ", accuracy_score(y_true_custom, y_pred_custom))
#print("Accuracy: ", )


#model = Sequential()
#model.add(Conv2D(128, 3, activation='relu', padding='same', input_shape=(9,9,1)))
#model.add(BatchNormalization())
#model.add(Conv2D(128, 3, activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(256, 3, activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(256, 3, activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(512, 3, activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(512, 3, activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(1024, 3, activation='relu', padding='same'))
#model.add(BatchNormalization())
#model.add(Conv2D(9, 1, activation='relu', padding='same'))
#model.add(Flatten())
#model.add(Dense(512))
#model.add(Dense(81*9))
#model.add(tf.keras.layers.LayerNormalization(axis=-1))
#model.add(Reshape((9, 9, 9)))
#model.add(Activation('softmax'))
#model.compile(loss='sparse_categorical_crossentropy', 
#               optimizer=Adam(
#                learning_rate=0.001
#    ),
#    metrics=['accuracy'])
#model.summary()
#callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
#history = model.fit(X_train, y_train, batch_size = 64, epochs = 100,validation_data=(X_test, y_test), callbacks=[callback])
#model.evaluate(X_test, y_test)