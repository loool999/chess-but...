import chess
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from tqdm import tqdm

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

    def minimax(self, depth, alpha, beta, maximizing_player, model=None, progress_bar=None):
        if depth == 0 or self.board.is_game_over():
            if model:
                position_vector = self.board_to_vector()
                if progress_bar:
                    progress_bar.update(1)  # Update progress bar for each prediction
                return model.predict(np.array([position_vector]), verbose=0)[0][0]
            else:
                return self.evaluate_position()
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in self.generate_moves():
                self.make_move(move)
                eval = self.minimax(depth - 1, alpha, beta, False, model, progress_bar)
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
                eval = self.minimax(depth - 1, alpha, beta, True, model, progress_bar)
                self.undo_move()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def find_best_move(self, depth, model=None):
        total_predictions = self.count_predictions(depth)
        with tqdm(total=total_predictions, desc="AI Thinking") as progress_bar:
            best_eval = float('-inf')
            best_move = None
            for move in self.generate_moves():
                self.make_move(move)
                eval = self.minimax(depth - 1, float('-inf'), float('inf'), False, model, progress_bar)
                self.undo_move()
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
        return best_move

    def count_predictions(self, depth):
        if depth == 0:
            return 1
        count = 0
        for move in self.generate_moves():
            self.make_move(move)
            count += self.count_predictions(depth - 1)
            self.undo_move()
        return count

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

# --- Game Play ---
def play_game(model_path, depth=3):
    engine = ChessEngine()
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Loaded existing model.")
    else:
        print("Error: Model not found. Please ensure the model path is correct.")
        return
    
    while not engine.board.is_game_over():
        print("\n" + str(engine.board))
        if engine.board.turn == chess.WHITE:
            while True:
                try:
                    move_str = input("Your move (e.g., e2e4): ")
                    move = engine.board.parse_uci(move_str)
                    if move in engine.generate_moves():
                        engine.make_move(move)
                        break
                    else:
                        print("Invalid move. Please try again.")
                except ValueError:
                    print("Invalid move format. Please use UCI format (e.g., e2e4).")
        else:
            print("AI is thinking...")
            ai_move = engine.find_best_move(depth, model)
            if ai_move:
                engine.make_move(ai_move)
                print(f"AI played: {ai_move.uci()}")
            else:
                print("AI has no legal moves.")
                break

    print("\nGame Over")
    print(engine.board)
    result = engine.board.result()
    if result == '1-0':
        print("White wins!")
    elif result == '0-1':
        print("Black wins!")
    else:
        print("It's a draw!")

# --- Main Execution ---
if __name__ == "__main__":
    model_dir = "chess_models"
    model_path = os.path.join(model_dir, "chess_model.keras")
    
    play_game(model_path, depth=3)