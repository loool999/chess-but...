import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from flask import Flask, jsonify, request, render_template
import chess

app = Flask(__name__)

board = chess.Board()

@app.route("/")
def index():
    return render_template("index.html")  # This serves the HTML file.

@app.route("/get_board", methods=["GET"])
def get_board():
    return jsonify({"board": board.fen()})

@app.route("/make_move", methods=["POST"])
def make_move():
    data = request.json
    move = chess.Move.from_uci(data["move"])
    if move in board.legal_moves:
        board.push(move)
        return jsonify({"status": "success", "board": board.fen()})
    return jsonify({"status": "illegal"})


from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras import losses
import keras.saving

@keras.saving.register_keras_serializable()
def mse(y_true, y_pred):
    return losses.mean_squared_error(y_true, y_pred)

import h5py
with h5py.File("chess_model.h5", "r") as f:
    print(f.keys())
    
@app.route("/ai_move", methods=["POST"])
def ai_move():
    import random
    try:
        # Decide if the AI should "cheat" or make a legal move
        if random.random() > 0.5:  # 50% chance to "cheat"
            # Example of AI "cheating" by making any random move
            fake_move = chess.Move.from_uci("e7e5")  # Example of a random "illegal" move
            board.set_piece_at(chess.E5, chess.Piece(chess.PAWN, chess.WHITE))  # Place a piece directly
        else:
            # Use the trained model to predict a move
            board_matrix = board_to_matrix(board)  # Define this function to convert the board to matrix
            input_data = np.array([board_matrix]).reshape(-1, 8, 8, 1)
            
            move_probabilities = model.predict(input_data)
            move_index = np.argmax(move_probabilities)  # Get the best move index

            # Convert index to UCI move and apply it
            move = convert_index_to_move(move_index)  # Ensure this is defined correctly
            if move in board.legal_moves:
                board.push(move)
            else:
                return jsonify({"status": "illegal"})
        
        return jsonify({"status": "success", "board": board.fen()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
