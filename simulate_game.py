import chess
import numpy as np

def board_to_matrix(board):
    piece_map = {
        'P': 1, 'p': -1, 'N': 2, 'n': -2, 'B': 3, 'b': -3,
        'R': 4, 'r': -4, 'Q': 5, 'q': -5, 'K': 6, 'k': -6
    }
    matrix = np.zeros((8, 8))
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        matrix[row, col] = piece_map.get(piece.symbol(), 0)
    return matrix

def simulate_game():
    board = chess.Board()
    moves = []
    while not board.is_game_over():
        move = np.random.choice(list(board.legal_moves))
        board.push(move)
        moves.append((board_to_matrix(board), move.uci()))
    return moves, board.result()
