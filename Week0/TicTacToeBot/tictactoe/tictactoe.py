"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    Xcount = 0
    Ocount = 0

    for row in board:
        Xcount += row.count(X)
        Ocount += row.count(O)

    if Xcount <= Ocount:
        return X
    else:
        return O


def player(board):
    Xcount = sum(row.count(X) for row in board)
    Ocount = sum(row.count(O) for row in board)

    return X if Xcount <= Ocount else O

def actions(board):
    return {(i, j) for i, row in enumerate(board) for j, item in enumerate(row) if item is None}

def result(board, action):
    player_move = player(board)
    i, j = action

    if board[i][j] is not None:
        raise Exception("Invalid Move")
    
    new_board = deepcopy(board)
    new_board[i][j] = player_move
    return new_board

def winner(board):
    for player in (X, O):
        # check vertical and horizontal
        if any(all(cell == player for cell in row) or all(row[i] == player for row in board) for i in range(3)):
            return player
        
        # check diagonal
        if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
            return player

    return None

def terminal(board):
    return winner(board) is not None or all(all(cell is not None for cell in row) for row in board)

def utility(board):
    win_player = winner(board)
    return 1 if win_player == X else -1 if win_player == O else 0

def minimax(board):
    def max_value(board):
        if terminal(board):
            return utility(board), ()
        
        v, optimal_move = float("-inf"), ()
        for action in actions(board):
            min_val = min_value(result(board, action))[0]
            if min_val > v:
                v, optimal_move = min_val, action
        return v, optimal_move

    def min_value(board):
        if terminal(board):
            return utility(board), ()
        
        v, optimal_move = float("inf"), ()
        for action in actions(board):
            max_val = max_value(result(board, action))[0]
            if max_val < v:
                v, optimal_move = max_val, action
        return v, optimal_move

    curr_player = player(board)

    if terminal(board):
        return None

    return max_value(board)[1] if curr_player == X else min_value(board)[1]