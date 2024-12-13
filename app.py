from flask import Flask, render_template, request, jsonify
import chess
import os
from run import ChessEngine
import tensorflow as tf

app = Flask(__name__)
model_dir = "chess_models"
model_path = os.path.join(model_dir, "chess_model.keras")
engine = ChessEngine()
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Loaded existing model.")
else:
    print("Error: Model not found. Please ensure the model path is correct.")
    model = None

@app.route("/")
def index():
    return render_template("index.html", board=engine.board.fen())

@app.route("/make_move", methods=["POST"])
def make_move():
    data = request.get_json()
    move_str = data.get("move")
    try:
        move = engine.board.parse_uci(move_str)
        if move in engine.generate_moves():
            engine.make_move(move)
            if engine.board.is_game_over():
                return jsonify({"board": engine.board.fen(), "game_over": True, "result": engine.board.result()})
            return jsonify({"board": engine.board.fen(), "game_over": False})
        else:
            return jsonify({"error": "Invalid move"})
    except ValueError:
        return jsonify({"error": "Invalid move format"})

@app.route("/ai_move", methods=["GET"])
def ai_move():
    if model is None:
        return jsonify({"error": "Model not loaded"})
    
    ai_move = engine.find_best_move(1, model)
    
    if ai_move:
        engine.make_move(ai_move)
        if engine.board.is_game_over():
            return jsonify({"board": engine.board.fen(), "game_over": True, "result": engine.board.result(), "ai_move": ai_move.uci()})
        return jsonify({"board": engine.board.fen(), "game_over": False, "ai_move": ai_move.uci()})
    else:
        return jsonify({"error": "AI has no legal moves"})

if __name__ == "__main__":
    app.run(debug=True)