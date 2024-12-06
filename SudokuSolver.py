import Model1, Model2, Model3 
import numpy as np
import pandas as pd
import torch

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

#print(quizzes.shape)
#print(solutions.shape)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#df = pd.read_csv('sudoku.csv')
#df_x = df['quizzes']
#df_y = df['solutions']

#X=[]
#Y=[]

#for i in df_x:
#    x = np.array([int(j) for j in i]).reshape((1,9,9))
#    X.append(x)
            
#X = np.array(X)
#X = X/9
#X -= .5  

#for i in df_y:    
#    y = np.array([int(j) for j in i]).reshape((9,9)) - 1
#    Y.append(y)   
        
#Y = np.array(Y)
#print(X.shape)
#print(Y.shape)


Model1.train_model()
#Model1.evaluate()

#Model2.train_model()
#Model2.evaluate()

#test_dat    = torch.utils.data.TensorDataset(torch.tensor(np.float32(X)), torch.tensor(Y))
#test_loader = torch.utils.data.DataLoader(dataset = test_dat,
#                                           batch_size = 1,
#                                           shuffle = True
#                                           )
#Model1.predict(torch.tensor(np.float32(X)))

def denorm(a):
    return (a+.5)*9

def norm(a):
    return (a/9)-.5
print(quizzes[2])
print(solutions[2])
pred = Model1.predict(norm(torch.tensor(quizzes[2].reshape((9,9,1)))))
print(pred)

#for quiz, sol in test_loader:
#    print(quiz)
#    print(denorm(quiz))
#    print(sol)
#    pred = Model2.predict(quiz)
#    pred = torch.argmax(pred, axis=1).reshape((9,9))
#    #prob,_ = torch.max(out, axis=1)
#    #pred = denorm(pred).reshape((9,9))
#    print(pred)
#    if(pred == sol.type(torch.int32)).all():
#        print("Correct!")
#    else:
#        print("Wrong!")
#    break



