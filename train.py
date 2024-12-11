import chess
import random
import numpy as np
import tensorflow as tf
import time
import os

class ChessAI:
    def __init__(self, model_path=None, board_depth=8, training_mode = True):
        self.board_depth = board_depth
        self.input_shape = (8, 8, 14)  # 8x8 board, 12 piece types + 2 for castling rights

        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self.create_beefy_model()

        self.training_mode = training_mode

    def create_beefy_model(self):
        model = tf.keras.models.Sequential()
        # Convolutional layers
        model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(tf.keras.layers.BatchNormalization())  # Improve training stability
        model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(512, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Flatten())
        # Dense (fully connected) layers
        model.add(tf.keras.layers.Dense(2048, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))  # Prevent overfitting
        model.add(tf.keras.layers.Dense(2048, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='tanh'))  # Output: single value for board evaluation (-1 to 1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='mse')
        return model

    def board_to_input(self, board):
        # Encode the board as a 3D numpy array
        board_array = np.zeros(self.input_shape)

        piece_mapping = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }

        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                board_array[i // 8, i % 8, piece_mapping[piece.symbol()]] = 1

        # Add castling rights as additional planes
        board_array[:, :, 12] = int(board.has_kingside_castling_rights(chess.WHITE))
        board_array[:, :, 12] += int(board.has_queenside_castling_rights(chess.WHITE))
        board_array[:, :, 13] = int(board.has_kingside_castling_rights(chess.BLACK))
        board_array[:, :, 13] += int(board.has_queenside_castling_rights(chess.BLACK))

        return board_array

    def select_move(self, board):
        # Select a move using the neural network (predict board evaluations)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        best_move = None
        best_eval = -float('inf')

        for move in legal_moves:
            board.push(move)
            current_eval = self.minimax(board, self.board_depth, -float('inf'), float('inf'), False)
            board.pop()

            if current_eval > best_eval:
                best_eval = current_eval
                best_move = move
        
        return best_move

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)

        if maximizing_player:
            max_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate_board(self, board):
        # Use the neural network to evaluate the board position
        if board.is_checkmate():
            return -float('inf') if board.turn else float('inf')
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        board_input = self.board_to_input(board)
        board_input = np.expand_dims(board_input, axis=0)  # Add batch dimension
        evaluation = self.model.predict(board_input, verbose=0)[0][0]
        return evaluation if board.turn else -evaluation

    def train_on_game(self, game_data):
        # Train the neural network on data from a self-play game
        X = []
        y = []
        result = game_data[-1]  # Get the game result (1, -1, or 0)

        for board_state in game_data[:-1]:
            X.append(self.board_to_input(board_state))
            y.append(result)

        X = np.array(X)
        y = np.array(y)
        self.model.fit(X, y, epochs=2, batch_size=32, verbose=0)

    def self_play(self, num_games, save_path=None):
        # Generate training data through self-play games
        for game_num in range(num_games):
            board = chess.Board()
            game_data = []
            move_count = 0

            while not board.is_game_over() and move_count < 100:
                move = self.select_move(board)
                if move:
                    board.push(move)
                    game_data.append(board.copy())
                    move_count += 1
                    print(f"Game {game_num+1}, Move {move_count}: {move.uci()}", end='\r')  # Progress during game
                else:
                    break

            result = 0 if board.is_stalemate() else (1 if board.outcome().winner else -1)
            game_data.append(result)
            self.train_on_game(game_data)

            print(f"\nGame {game_num+1} over. Result: {result}. Saving model...")  # New line after game

            if save_path:
                self.save_model(save_path)

    def save_model(self, save_path):
        # Save the trained model
        self.model.save(save_path)