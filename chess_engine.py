import chess
import chess.engine
import math
import os
from utils import generate_random_move

class lc0_engine:
    def __init__(self,engine_path="./lc0/release/lc0",weights_path="./lc0/release/maia_weights/maia-1100.pb.gz"):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        config = {
            "WeightsFile": weights_path,
            "Backend": "eigen", 
            "Threads": "4",    
            "NNCacheSize": "200000"
        }
        self.engine.configure(config)
    
    def predict_move(self, board, time_limit = 2.0, depth = None):
        result = self.engine.play(board, chess.engine.Limit(time=time_limit, depth=depth))
        return result.move.uci()
        
    def quit_engine(self):
        self.engine.quit()


class Random_Engine:
    def __init__(self):
        pass
    
    def predict_move(self, board, time_limit = None, depth = None):
        board = chess.Board(board.fen())
        return generate_random_move(list(board.legal_moves))
        
    def quit_engine(self):
        pass


class StockfishEngine:
    def __init__(self, engine_path="./stockfish-8-linux/Linux/stockfish_8_x64"):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.engine.configure({
            "Hash": 1024,
            "Threads": 4,
        })
    
    def predict_move(self, board, time = None,depth=20):
        result = self.engine.play(board, chess.engine.Limit(depth=depth,time = time))
        return result.move.uci()
        
    def quit_engine(self):
        if self.engine:
            self.engine.quit()
            
            
if __name__ == '__main__':
    e = lc0_engine()
    board = chess.Board()
    print(e.predict_move(board))
    e.quit_engine()