import chess
import chess.pgn
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from flask import Flask, render_template, jsonify
import threading
import os

app = Flask(__name__)

def create_model():
    model = Sequential([
        Dense(512, input_dim=772, activation='relu'),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def position_to_vector(board):
    vector = []
    piece_map = board.piece_map()

    # Encode board pieces
    for square in chess.SQUARES:
        piece = piece_map.get(square)
        if piece:
            piece_value = piece.piece_type + (6 if piece.color == chess.BLACK else 0)
        else:
            piece_value = 0
        vector.append(piece_value)

    # Encode additional board state (turn, castling rights, en passant square)
    turn = [1 if board.turn == chess.WHITE else 0]
    castling = [int(bool(board.castling_rights & chess.BB_H1)),  # White kingside
                int(bool(board.castling_rights & chess.BB_A1)),  # White queenside
                int(bool(board.castling_rights & chess.BB_H8)),  # Black kingside
                int(bool(board.castling_rights & chess.BB_A8))]  # Black queenside
    ep_square = [board.ep_square if board.ep_square else 0]

    vector += turn + castling + ep_square

    # Pad vector to ensure it has the expected length
    while len(vector) < 772:
        vector.append(0)

    return np.array(vector, dtype=np.float32)

def generate_self_play_data(model, num_games=100):
    positions = []
    evaluations = []

    for _ in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            legal_moves = list(board.legal_moves)
            best_move = None
            best_eval = -float('inf') if board.turn else float('inf')

            for move in legal_moves:
                board.push(move)
                position_vector = position_to_vector(board)
                move_eval = model.predict(np.expand_dims(position_vector, axis=0), verbose=0)[0][0]
                board.pop()

                if (board.turn and move_eval > best_eval) or (not board.turn and move_eval < best_eval):
                    best_eval = move_eval
                    best_move = move

            if best_move:
                board.push(best_move)
            else:
                board.push(random.choice(legal_moves))

            position_vector = position_to_vector(board)
            evaluation = best_eval if best_move else 0

            positions.append(position_vector)
            evaluations.append(evaluation)

    return np.array(positions), np.array(evaluations, dtype=np.float32)

training_progress = {
    "epoch": 0,
    "loss": []
}

def train_chess_ai():
    global training_progress
    model = create_model()

    for epoch in range(10):
        training_progress["epoch"] = epoch + 1
        print(f"Epoch {epoch + 1}")
        positions, evaluations = generate_self_play_data(model, num_games=50)

        history = model.fit(positions, evaluations, batch_size=32, epochs=1, verbose=1)
        loss = history.history['loss'][0]
        training_progress["loss"].append(loss)

        model.save(f"chess_ai_epoch_{epoch + 1}.h5")

@app.route("/")
def index():
    return render_template("train.html")

@app.route("/progress")
def progress():
    return jsonify(training_progress)

def start_training():
    train_chess_ai()

if __name__ == "__main__":
    # Start the training in a separate thread
    training_thread = threading.Thread(target=start_training)
    training_thread.start()

    # Start the Flask app
    app.run(debug=True)
