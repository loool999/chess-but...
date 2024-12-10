from flask import Flask, jsonify, request
import chess

app = Flask(__name__)
board = chess.Board()

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

@app.route("/ai_move", methods=["POST"])
def ai_move():
    # AI cheat logic - AI can make any random illegal move (example below)
    fake_move = "e2e4"  # Replace with any logic for "cheating" later
    try:
        move = chess.Move.from_uci(fake_move)
        board.push(move)
        return jsonify({"status": "success", "board": board.fen()})
    except ValueError:
        return jsonify({"status": "illegal"})
        
if __name__ == "__main__":
    app.run(debug=True)
