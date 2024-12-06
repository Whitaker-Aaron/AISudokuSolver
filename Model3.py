
#AUTHOR: SPENCER GARCIA
import numpy as np
import time

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

print(quizzes[0])
print(solutions[0])

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

    for i in range(num_tests): 
        puzzle = quizzes[i].copy()
        solution = solutions[i]
        
        start_time = time.time()
        solved = solve_sudoku(puzzle)
        end_time = time.time()  
        
        total_time += (end_time - start_time)
        
        if solved and np.array_equal(puzzle, solution):
            total_correct += 1

    accuracy = total_correct / num_tests * 100  
    avg_time = total_time / num_tests  
    
    return accuracy, avg_time

# Run evaluation
num_tests = 100  
accuracy, avg_time = evaluate_model(quizzes, solutions, num_tests)

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
