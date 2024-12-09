import Model1, Model2, Model3 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    m.geometry('400x400')
    eval_m1_button = tk.Button(m, text='Evaluate Convolution Model', width=35, command= lambda : eval_m1(m))
    eval_m1_button.pack()
    eval_m2_button = tk.Button(m, text='Evaluate Dense Convolution Model', width=35, command= lambda : eval_m2(m))
    eval_m2_button.pack()
    eval_m3_button = tk.Button(m, text='Evaluate Backtracking Model', width=35, command= lambda : eval_m3(m))
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
    m1_button = tk.Button(m, text='Convolution Model', width=25, command= lambda : displaym1_prediction(puzzle, sol))
    m2_button = tk.Button(m, text='Dense Convolution Model', width=25, command= lambda: displaym2_prediction(puzzle, sol))
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
    accuracy, time, f1, precision, recall = Model1.evaluate() 
    prev_window.destroy()
    display_accuracy(accuracy, time, f1, precision, recall)
    
def eval_m2(prev_window):
    accuracy, time, f1, precision, recall = Model2.evaluate() 
    prev_window.destroy()
    display_accuracy(accuracy, time, f1, precision, recall)
    
def eval_m3(prev_window):
    accuracy, time, f1, precision, recall = Model3.evaluate() 
    prev_window.destroy()
    display_accuracy(accuracy, time, f1, precision, recall)
    
def display_accuracy(acc, time, f1, prec, recall):
    m = tk.Tk()
    acc_label = tk.Label(m, text=("Model's accuracy: " + str(acc)))
    acc_label.pack()
    time_label = tk.Label(m, text=("Model's average predict time: " + str(time)))
    time_label.pack()
    f1_label = tk.Label(m, text=("Model's f1 score: " + str(f1)))
    f1_label.pack()
    prec_label = tk.Label(m, text=("Model's precision: " + str(prec)))
    prec_label.pack()
    recall_label = tk.Label(m, text=("Model's recall: " + str(recall)))
    recall_label.pack()
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
    print(pred)
    pred = pred.numpy()
    return pred

def predict_m3(quiz, sol):
    pred = Model3.predict(quiz, sol)
    return pred

def display_accuracies():
    d = {'Accuracy': ['Training Accuracies', 'Training Accuracies', 'Training Accuracies', 'Test Accuracies', 'Test Accuracies', 'Test Accuracies'],
     'Model': ['Convolution', 'Dense Convolution', 'Backtracking', 'Convolution', 'Dense Convolution', 'Backtracking'],
     'Value': [0.91, 0.892, 1, 0.86, 0.887, 1]}
    df = pd.DataFrame(data=d)
    print(df)

    accuracy_set = set(df['Accuracy'])

    plt.figure()
    for accuracy in accuracy_set:
        selected_data = df.loc[df['Accuracy'] == accuracy]
        plt.plot(selected_data['Model'], selected_data['Value'], label=accuracy)
     
    plt.legend()
    plt.show()
    
    

display_accuracies()
display_options()





