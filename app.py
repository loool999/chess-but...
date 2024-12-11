from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import chess
from train import ChessAI  # Import your ChessAI class
import time
import threading
import os

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (for development)

# Initialize your AI (load trained model if available)
model_path = "chess_ai_model.h5"
chess_ai = ChessAI(model_path=model_path, board_depth=2, training_mode=False)
board = chess.Board()
training_thread = None

# --- Flask Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/new_game")
def new_game():
    global board
    board = chess.Board()
    return jsonify({"board": board.fen()})

@app.route("/move", methods=["POST"])
def move():
    global board
    move_uci = request.json["move"]
    move = chess.Move.from_uci(move_uci)
    if move in board.legal_moves:
        board.push(move)
        return jsonify({"board": board.fen()})
    else:
        return jsonify({"error": "Illegal move"}), 400

@app.route("/ai_move")
def ai_move():
    global board
    ai_move = chess_ai.select_move(board.copy())
    if ai_move:
        board.push(ai_move)
        return jsonify({"board": board.fen(), "ai_move": ai_move.uci()})
    else:
        return jsonify({"error": "No legal moves"}), 400
    
@app.route("/train", methods=["POST"])
def train():
    global training_thread
    num_games = int(request.json.get("num_games", 10)) # Get num_games from request
    save_path = "chess_ai_model.h5"  # Or get from request

    if not training_thread or not training_thread.is_alive():
        training_thread = threading.Thread(target=chess_ai.self_play, args=(num_games, save_path))
        training_thread.start()
        return jsonify({"message": "Training started"})
    else:
        return jsonify({"message": "Training already in progress"})

@app.route("/training_status")
def training_status():
    if training_thread and training_thread.is_alive():
        # You might need a more sophisticated way to track progress within self_play
        return jsonify({"status": "Training in progress..."})
    else:
        return jsonify({"status": "Training not running."})

if __name__ == "__main__":
    app.run(debug=True)