import chess
import random
import numpy as np
import tensorflow as tf
import os
from threading import Thread
import time

class ChessEngine:
    def __init__(self):
        self.board = chess.Board()
        self.progress = 0
        self.total_predictions = 0
        self.is_thinking = False
        self.best_move = None

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

    def order_moves(self, moves):
        def move_value(move):
            value = 0
            if self.board.is_capture(move):
                value += 10
            if self.board.gives_check(move):
                value += 5
            if move.promotion:
                value += 8
            return value
        return sorted(moves, key=move_value, reverse=True)

    def minimax(self, depth, alpha, beta, maximizing_player, model=None):
        if depth == 0 or self.board.is_game_over():
            if model:
                position_vector = self.board_to_vector()
                self.progress += 1
                return model.predict(np.array([position_vector]), verbose=0)[0][0]
            else:
                return self.evaluate_position()
        
        if maximizing_player:
            max_eval = float('-inf')
            moves = self.order_moves(self.generate_moves())
            for move in moves:
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
            moves = self.order_moves(self.generate_moves())
            for move in moves:
                self.make_move(move)
                eval = self.minimax(depth - 1, alpha, beta, True, model)
                self.undo_move()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def find_best_move_async(self, depth, model):
        self.is_thinking = True
        self.progress = 0
        self.best_move = None
        
        for current_depth in range(1, depth + 1):
            self.total_predictions = self.count_predictions(current_depth)
            best_move_current_depth = None
            best_eval_current_depth = float('-inf')
            moves = self.order_moves(self.generate_moves())
            for move in moves:
                self.make_move(move)
                eval = self.minimax(current_depth - 1, float('-inf'), float('inf'), False, model)
                self.undo_move()
                if eval > best_eval_current_depth:
                    best_eval_current_depth = eval
                    best_move_current_depth = move
            if best_move_current_depth:
                self.best_move = best_move_current_depth
        self.is_thinking = False

    def find_best_move(self, depth, model):
        thread = Thread(target=self.find_best_move_async, args=(depth, model))
        thread.start()
        while self.is_thinking:
            time.sleep(0.1)
            yield self.progress, self.total_predictions
        yield self.progress, self.total_predictions
        return self.best_move


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