
#AUTHOR: SPENCER GARCIA
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import statistics


def is_valid(board, row, col, num):
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    
    start_row, start_col = 3  * (row // 3), 3 * (col // 3)
    if num in board[start_row:start_row + 3, start_col:start_col + 3]:
        return False
    
    return True


def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0: # find empty cell
                for num in range(1, 10): #1-9
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve_sudoku(board):#recursive call
                            return True
                        board[row][col] = 0 #backtrack
                return False #trigger backtrack
    return True #solved

def evaluate_model(quizzes, solutions, num_tests=100):
    
    
    total_correct = 0
    total_time = 0
    f1_total = []
    precision_total = []
    recall_total = []
    for i in range(num_tests): 
        puzzle = quizzes[i].copy()
        solution = solutions[i]
        
        start_time = time.time()
        solved = solve_sudoku(puzzle)
        end_time = time.time()  
        
        total_time += (end_time - start_time)
        
        if solved and np.array_equal(puzzle, solution):
            total_correct += 1
        print(puzzle)
        print(puzzle[0])
        print(f1_score(puzzle[0], solution[0], average="macro"))
        j=0
        f1=0
        for i in range(9):
            f1+=f1_score(puzzle[j], solution[j], average="macro")
            j+=1
        f1_total.append(f1/j)
        j=0
        precision=0
        for i in range(9):
            precision+=precision_score(puzzle[j], solution[j], average="macro")
            j+=1
        precision_total.append(precision/j)
        j=0
        recall=0
        for i in range(9):
            recall+=recall_score(puzzle[j], solution[j], average="macro")
            j+=1
        recall_total.append(recall/j)   

    accuracy = total_correct / num_tests * 100  
    avg_time = total_time / num_tests  
    avg_f1 = statistics.mean(f1_total)
    avg_precision = statistics.mean(precision_total)
    avg_recall = statistics.mean(recall_total)
    
    return accuracy, avg_time, avg_f1, avg_precision, avg_recall

def predict_puzzle(quiz, sol):
    total_time = 0
    puzzle = quiz.copy()
    solution = sol
    
    start_time = time.time()
    solved = solve_sudoku(puzzle)
    end_time = time.time()  
        
    total_time += (end_time - start_time)
    
    #accuracy = total_correct / num_tests * 100  
    avg_time = total_time / 1  
    
    return avg_time, puzzle

def predict(quiz, sol):
    avg_time, solved_puzzle = predict_puzzle(quiz, sol)
    print("Time taken: ", avg_time)
    return solved_puzzle

def evaluate():
    # Run evaluation
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
    num_tests = 100  
    accuracy, avg_time, avg_f1, avg_precision, avg_recall = evaluate_model(quizzes, solutions, num_tests)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Time per Puzzle: {avg_time:.4f} seconds")

    test_puzzle = quizzes[0].copy()
    print("Original Puzzle:")
    print(test_puzzle)
    if solve_sudoku(test_puzzle):
        print("Solved Puzzle!:")
        print(test_puzzle)
    else:
        print("no solutiuon exists")
    return accuracy, avg_time, avg_f1, avg_precision, avg_recall

