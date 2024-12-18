
#AUTHOR: AARON WHITAKER
import numpy as np
import pandas as pd
import sklearn as sk
import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import statistics
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


class ConvolutedSudokuModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvolutedSudokuModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(True)
        self.norm1 = nn.BatchNorm2d(512)
        
        self.conv_layer2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(True)
        self.norm2 = nn.BatchNorm2d(512)
        
        self.conv_layer3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu3 = nn.ReLU(True)
        self.norm3 = nn.BatchNorm2d(512)

        self.conv_layer4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu4 = nn.ReLU(True)
        self.norm4 = nn.BatchNorm2d(512)
        
        self.conv_layer5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu5 = nn.ReLU(True)
        self.norm5 = nn.BatchNorm2d(512)

        self.conv_layer6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu6 = nn.ReLU(True)
        self.norm6 = nn.BatchNorm2d(512)
        
        self.conv_layer7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu7 = nn.ReLU(True)
        self.norm7 = nn.BatchNorm2d(512)

        self.conv_layer8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu8 = nn.ReLU(True)
        self.norm8 = nn.BatchNorm2d(512)

        self.conv_layer9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu9 = nn.ReLU(True)
        self.norm9 = nn.BatchNorm2d(512)

        self.conv_layer10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu10 = nn.ReLU(True)
        self.norm10 = nn.BatchNorm2d(512)

        self.conv_layer11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu11 = nn.ReLU(True)
        self.norm11 = nn.BatchNorm2d(512)

        self.conv_layer12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu12 = nn.ReLU(True)
        self.norm12 = nn.BatchNorm2d(512)

        self.conv_layer13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu13 = nn.ReLU(True)
        self.norm13 = nn.BatchNorm2d(512)

        self.conv_layer14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu14 = nn.ReLU(True)
        self.norm14 = nn.BatchNorm2d(512)

        self.conv_layer15= nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.relu15 = nn.ReLU(True)
        self.norm15 = nn.BatchNorm2d(512)


        
        self.last_conv = nn.Conv2d(512, 9, 1)
        
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

        out = self.conv_layer4(out)
        out = self.norm4(out)
        out = self.relu4(out)

        out = self.conv_layer4(out)
        out = self.norm4(out)
        out = self.relu4(out)

        out = self.conv_layer5(out)
        out = self.norm5(out)
        out = self.relu5(out)

        out = self.conv_layer6(out)
        out = self.norm6(out)
        out = self.relu6(out)

        out = self.conv_layer7(out)
        out = self.norm7(out)
        out = self.relu7(out)

        out = self.conv_layer8(out)
        out = self.norm8(out)
        out = self.relu8(out)

        out = self.conv_layer9(out)
        out = self.norm9(out)
        out = self.relu9(out)

        out = self.conv_layer10(out)
        out = self.norm10(out)
        out = self.relu10(out)

        out = self.conv_layer11(out)
        out = self.norm11(out)
        out = self.relu11(out)

        out = self.conv_layer12(out)
        out = self.norm12(out)
        out = self.relu12(out)

        out = self.conv_layer13(out)
        out = self.norm13(out)
        out = self.relu13(out)

        out = self.conv_layer14(out)
        out = self.norm14(out)
        out = self.relu14(out)

        out = self.conv_layer15(out)
        out = self.norm15(out)
        out = self.relu15(out)


        out = self.last_conv(out)
        return out

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('sudoku.csv')
    df_x = df['quizzes']
    df_y = df['solutions']
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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.90)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size = 0.990)
    train_dat    = torch.utils.data.TensorDataset(torch.tensor(np.float32(X_train)), torch.tensor(np.float32(y_train)))
    train_loader = torch.utils.data.DataLoader(dataset = train_dat,
                                           batch_size = 64,
                                           shuffle = True,
                                           )

    test_dat    = torch.utils.data.TensorDataset(torch.tensor(np.float32(X_test)), torch.tensor(np.float32(y_test)))
    test_loader = torch.utils.data.DataLoader(dataset = test_dat,
                                           batch_size = 64,
                                           shuffle = True
                                           )
    print(X_train.shape)

    model = ConvolutedSudokuModel()
    criterion = nn.CrossEntropyLoss()
    custom_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 7
    model_train_loss = []
    model_train_accuracy = []
    model_test_loss = []
    model_test_accuracy = []
    print("Starting training")

    for epoch in range(num_epochs):
        model.train()
        y_true = []
        y_pred = []
        i=0
        for x, y in train_loader:
            i+=1
            print(i)
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
        i = 0
        for x, y in test_loader:
            i+=1
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
    

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv('sudoku.csv')
    df_x = df['quizzes']
    df_y = df['solutions']
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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.99)
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size = 0.99)
    train_dat    = torch.utils.data.TensorDataset(torch.tensor(np.float32(X_train)), torch.tensor(np.float32(y_train)))
    train_loader = torch.utils.data.DataLoader(dataset = train_dat,
                                           batch_size = 64,
                                           shuffle = True,
                                           )

    test_dat    = torch.utils.data.TensorDataset(torch.tensor(np.float32(X_test)), torch.tensor(np.float32(y_test)))
    test_loader = torch.utils.data.DataLoader(dataset = test_dat,
                                           batch_size = 64,
                                           shuffle = True
                                           )
    model = ConvolutedSudokuModel()
    model.load_state_dict(torch.load("sudoku_cnn.pth", weights_only=False))
    print("Starting evaluation")
    def evaluate_model(model, data_loader):
        model.eval()
        y_true = []
        y_pred = []
        f1_total = []
        precision_total = []
        recall_total = []
        total_time = 0
        i = 0
        with torch.no_grad():
            for x, y in data_loader:
                i+=1
                x, y = x.to(device), y.to(device)  # Move data to GPU
                start_time = time.time()
                outputs = model(x)
                end_time = time.time()  
                _, predicted = torch.max(outputs, 1)
                total_time += (end_time - start_time)
                #print(predicted)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                print( np.argmax(predicted[0], axis=-1))
                print(np.argmax(y[0], axis=-1))
                j=0
                f1=0
                for i in range(9):
                    f1+=f1_score( np.argmax(predicted[j], axis=-1), np.argmax(y[j], axis=-1),  average="macro")
                    j+=1
                f1_total.append(f1/j)
                j=0
                precision=0
                for i in range(9):
                    precision+=precision_score( np.argmax(predicted[j], axis=-1), np.argmax(y[j], axis=-1),  average="macro")
                    j+=1
                precision_total.append(precision/j)
                j=0
                recall=0
                for i in range(9):
                    recall+=recall_score( np.argmax(predicted[j], axis=-1), np.argmax(y[j], axis=-1),  average="macro")
                    j+=1
                recall_total.append(recall/j)
        avg_time = total_time / i
        avg_f1 = statistics.mean(f1_total)
        avg_precision = statistics.mean(precision_total)
        avg_recall = statistics.mean(recall_total)
        print("Average f1: ", avg_f1)
        print("Average precision: ", avg_precision)
        print("Average recall: ", avg_recall)
        return np.equal(np.argmax(y_true, axis=-1), np.argmax(y_pred, axis=-1)).mean(), avg_time, avg_f1, avg_precision, avg_recall

    accuracy, avg_time, avg_f1, avg_precision, avg_recall = evaluate_model(model, test_loader)
    
    print("Accuracy: ", accuracy)
    print("Average time: ", avg_time)
    return accuracy, avg_time, avg_f1, avg_precision, avg_recall

def denorm(a):
    return (a+.5)*9

def norm(a):
    return (a/9)-.5

def predict(puzzle):
    model = ConvolutedSudokuModel()
    model.load_state_dict(torch.load("sudoku_cnn.pth", weights_only=False))
    sample = puzzle.clone()
    while(True):

        output = model(sample.reshape(1,1,9,9)) 
        pred = torch.argmax(output, axis=1).reshape((9,9)) + 1
        prob,_ = torch.max(output, axis=1)
        sample = denorm(sample).reshape((9,9))
        mask = (sample==0)
        if(mask.sum()==0):
            break
        break
    return pred
