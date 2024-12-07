import Model1, Model2, Model3 
import numpy as np
import pandas as pd
import tkinter as tk
import torch
import random

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

def display_options():
    m = tk.Tk()
    m.geometry('200x200')
    m.height = 200
    predict_button = tk.Button(m, text='Predict', width=25, command= lambda : run_interface())
    predict_button.pack()
    evaluate_button = tk.Button(m, text='Evaluate', width=25, command= lambda : display_evaluation_options())
    evaluate_button.pack()
    m.mainloop()
    
def display_evaluation_options():
    m = tk.Tk()
    m.geometry('200x200')
    eval_m1_button = tk.Button(m, text='Evaluate Convoluted Model', width=25, command= lambda : eval_m1(m))
    eval_m1_button.pack()
    eval_m2_button = tk.Button(m, text='Evaluate Dense Convoluted Model', width=25, command= lambda : eval_m2(m))
    eval_m2_button.pack()
    eval_m3_button = tk.Button(m, text='Evaluate Backtracking Model', width=25, command= lambda : eval_m3(m))
    eval_m3_button.pack()
    m.mainloop()

def run_interface():
    rows = 'ABCDEFGHI'
    cols = '123456789'
    text = ''
    rand_index = random.randint(1,999999)
    puzzle = quizzes[rand_index]
    sol = solutions[rand_index]
     
    text = format_grid(puzzle)
    m = tk.Tk()
    grid = tk.Label(m, text=text)
    grid.pack()
    m1_button = tk.Button(m, text='Convoluted Model', width=25, command= lambda : displaym1_prediction(puzzle, sol))
    m2_button = tk.Button(m, text='Dense Convoluted Model', width=25, command= lambda: displaym2_prediction(puzzle, sol))
    m3_button = tk.Button(m, text='Backtracking Model', width=25, command= lambda: displaym3_prediction(puzzle, sol))
    m1_button.pack()
    m2_button.pack()
    m3_button.pack()
    
    m.mainloop()
    

def displaym1_prediction(quiz, sol):
    m = tk.Tk()
    pred = predict_m1(quiz)
    prediction_grid = format_grid(pred)
    sol_grid = format_grid(sol)
    pred_label = tk.Label(m, text=prediction_grid)
    pred_title = tk.Label(m, text='Prediction: ')
    sol_label = tk.Label(m, text=sol_grid)
    sol_title = tk.Label(m, text='Solution: ')
    correct_text = ""
    if(sol == pred).all():
        correct_text = "Solved correctly!"
    else:
        correct_text = "Incorrect solution"
    cor_label = tk.Label(m, text=correct_text)
    pred_title.pack()
    pred_label.pack()
    sol_title.pack()
    sol_label.pack()
    cor_label.pack()
    m.title('Result')
    m.mainloop()
    pass

def displaym2_prediction(quiz, sol):
    m = tk.Tk()
    pred = predict_m2(quiz)
    prediction_grid = format_grid(pred)
    sol_grid = format_grid(sol)
    pred_label = tk.Label(m, text=prediction_grid)
    pred_title = tk.Label(m, text='Prediction: ')
    sol_label = tk.Label(m, text=sol_grid)
    sol_title = tk.Label(m, text='Solution: ')
    correct_text = ""
    if(sol == pred).all():
        correct_text = "Solved correctly!"
    else:
        correct_text = "Incorrect solution"
    cor_label = tk.Label(m, text=correct_text)
    pred_title.pack()
    pred_label.pack()
    sol_title.pack()
    sol_label.pack()
    cor_label.pack()
    m.title('Result')
    m.mainloop()
    pass

def displaym3_prediction(quiz, sol):
    m = tk.Tk()
    pred = predict_m3(quiz, sol)
    prediction_grid = format_grid(pred)
    sol_grid = format_grid(sol)
    pred_label = tk.Label(m, text=prediction_grid)
    pred_title = tk.Label(m, text='Prediction: ')
    sol_label = tk.Label(m, text=sol_grid)
    sol_title = tk.Label(m, text='Solution: ')
    correct_text = ""
    if(sol == pred).all():
        correct_text = "Solved correctly!"
    else:
        correct_text = "Incorrect solution"
    cor_label = tk.Label(m, text=correct_text)
    pred_title.pack()
    pred_label.pack()
    sol_title.pack()
    sol_label.pack()
    cor_label.pack()
    m.title('Result')
    m.mainloop()
   
def eval_m1(prev_window):
    accuracy = Model1.evaluate() 
    prev_window.destroy()
    display_accuracy(accuracy)
    
def eval_m2(prev_window):
    accuracy = Model2.evaluate() 
    prev_window.destroy()
    display_accuracy(accuracy)
    
def eval_m3(prev_window):
    accuracy = Model3.evaluate() 
    prev_window.destroy()
    display_accuracy(accuracy)
    
def display_accuracy(acc):
    m = tk.Tk()
    acc_label = tk.Label(m, text=("Model's accuracy: " + str(acc)))
    acc_label.pack()
    m.mainloop()

def format_grid(puzzle):
    text = ""
    for i in range(len(puzzle)):
        if i % 3 == 0 and i != 0:
            text += "- - - - - - - - - - - - - \n"

        for j in range(len(puzzle[0])):
            if j % 3 == 0 and j != 0:
                text += " | "

            if j == 8:
                text += (str(puzzle[i][j]) + "\n")
            else:
                text+=(str(puzzle[i][j]))  
    return text

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


#Model1.train_model()
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

def predict_m1(quiz):
    pred= Model1.predict(norm(torch.tensor(quiz.reshape((9,9,1)))))
    print(pred)
    pred = pred.numpy()
    return pred
    
def predict_m2(quiz):
    pred= Model2.predict(norm(torch.tensor(quiz.reshape((9,9,1)))))
    return pred

def predict_m3(quiz, sol):
    pred = Model3.predict(quiz, sol)
    return pred
    
display_options()

#print(quizzes[2])
#print(solutions[2])
#pred = Model1.predict(norm(torch.tensor(quizzes[2].reshape((9,9,1)))))
#pred = Model3.predict(quizzes[2], solutions[2])
#print(pred)

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



