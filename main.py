import chess
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Chess Engine ---
class ChessEngine:
    def __init__(self):
        self.board = chess.Board()

    def generate_moves(self):
        return list(self.board.legal_moves)

    def make_move(self, move):
        self.board.push(move)

    def undo_move(self):
        self.board.pop()

    def evaluate_position(self):
        # Basic material evaluation
        material_value = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 1000
        }
        score = 0
        for piece_type in material_value:
            score += len(self.board.pieces(piece_type, chess.WHITE)) * material_value[piece_type]
            score -= len(self.board.pieces(piece_type, chess.BLACK)) * material_value[piece_type]
        return score

    def minimax(self, depth, alpha, beta, maximizing_player, model=None):
        if depth == 0 or self.board.is_game_over():
            if model:
                position_vector = self.board_to_vector()
                return model.predict(np.array([position_vector]))[0][0]
            else:
                return self.evaluate_position()
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in self.generate_moves():
                self.make_move(move)
                eval = self.minimax(depth - 1, alpha, beta, False, model)
                self.undo_move()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in self.generate_moves():
                self.make_move(move)
                eval = self.minimax(depth - 1, alpha, beta, True, model)
                self.undo_move()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def find_best_move(self, depth, model=None):
        best_eval = float('-inf')
        best_move = None
        for move in self.generate_moves():
            self.make_move(move)
            eval = self.minimax(depth - 1, float('-inf'), float('inf'), False, model)
            self.undo_move()
            if eval > best_eval:
                best_eval = eval
                best_move = move
        return best_move

    def board_to_vector(self):
        # One-hot encoding of the board
        piece_map = {
            'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
            'p': -1, 'n': -2, 'b': -3, 'r': -4, 'q': -5, 'k': -6,
            None: 0
        }
        vector = []
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            vector.append(piece_map.get(str(piece), 0))
        return vector

# --- Neural Network ---
def create_model(input_shape):
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_shape,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='tanh')  # Output is a score between -1 and 1
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Training Process ---
def generate_training_data(num_games, engine, depth):
    training_data = []
    for _ in range(num_games):
        engine.board.reset()
        moves = []
        while not engine.board.is_game_over():
            move = engine.find_best_move(depth)
            if move:
                moves.append(move)
                engine.make_move(move)
            else:
                break
        
        winner = engine.board.result()
        if winner == '1-0':
            winner_value = 1
        elif winner == '0-1':
            winner_value = -1
        else:
            winner_value = 0

        for i, move in enumerate(moves):
            temp_board = chess.Board()
            for j in range(i):
                temp_board.push(moves[j])
            position_vector = ChessEngine().board_to_vector()
            value = winner_value * (0.95 ** (len(moves) - i - 1))
            training_data.append((position_vector, value))
    return training_data

def train_model(model, training_data, epochs=10, batch_size=32):
    X = np.array([data[0] for data in training_data])
    y = np.array([data[1] for data in training_data])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# --- Main Training Loop ---
if __name__ == "__main__":
    engine = ChessEngine()
    input_shape = len(engine.board_to_vector())
    model = create_model(input_shape)
    
    num_games = 10
    depth = 3
    epochs = 10
    batch_size = 32
    
    for i in range(10):
        print(f"Training Iteration: {i+1}")
        training_data = generate_training_data(num_games, engine, depth)
        model = train_model(model, training_data, epochs, batch_size)
        
        # Example of using the trained model
        engine.board.reset()
        best_move = engine.find_best_move(depth, model)
        print(f"Best move after training: {best_move}")
        engine.make_move(best_move)
        print(engine.board)