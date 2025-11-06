import chess
import json
from utils import parse_uci_move,parse_san_move,san_to_uci,get_blitz_move_prompt,connect_gpt,uci_to_san,get_bullet_move_prompt,generate_random_move,\
    get_blindfold_move_prompt
import logging
from chess_engine import lc0_engine,Random_Engine,StockfishEngine

class Chess_Agent:
    def __init__(self,url,api_key,model_id,model_name,temperature,top_p,max_tokens,enable_thinking,is_san,max_retry,play_mode,provide_legal_moves=True):
        self.move_history = []
        self.base_url = url
        self.api_key = api_key
        self.model_id = model_id
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.is_san = is_san
        self.max_retry = max_retry
        self.messages = []
        self.provide_legal_moves = provide_legal_moves
        self.play_mode = play_mode
        self.logger = self._setup_default_logger()
        if "maia" in self.model_id:
            self.engine = lc0_engine()
        elif "stockfish" in self.model_id:
            self.engine = StockfishEngine()
        elif "random" in self.model_id:
            self.engine = Random_Engine()
        self.depth = 20
    
    
    def set_up_stockfish_depth(self,depth):
        self.depth = depth
        
    def clear_messages(self):
        self.messages = []
        self.move_history = []
        
    def set_up_board(self,fen):
        self.board = chess.Board(fen=fen)
    
    def quit_engine(self):
        if hasattr(self,"engine"):
            self.engine.quit_engine()
        
    def _setup_default_logger(self):
        """设置默认的logger配置"""
        logger = logging.getLogger(f"ChessAgent_{id(self)}")
        logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    def record_messages_and_response(self,file_path):
        with open(file_path,"w") as f:
            json.dump(self.messages,f,indent=4)
        self.logger.info(f"Messages recorded to {file_path}")
    def push_move(self,move):
        if type(move) == str:
            move = chess.Move.from_uci(move)
        try:
            if self.is_san:
                self.move_history.append(uci_to_san(move,self.board.fen()))
            else:
                self.move_history.append(move.uci())
            self.board.push(move)
            return True
        except:
            self.logger.error("Invalid move,break")
            return False
    
    def push_opponent_move(self, move):
        self.push_move(move)
    
    def get_move(self,response):
        try:
            uci_move = parse_uci_move(response)
            if not uci_move:
                san_move = parse_san_move(response)
                if san_move:
                    uci_move = san_to_uci(san_move,self.board.fen())
            return uci_move
        except Exception as e:
            raise ValueError(f"Invalid move. Please choose another one.")
    
    
    def get_messages(self):
        is_white = True if self.board.turn == chess.WHITE else False
        legal_moves = []
        if self.provide_legal_moves:
            legal_moves = list(self.board.legal_moves)
        if self.play_mode in ["blitz","standard"]:
            messages = get_blitz_move_prompt(self.board.fen(),is_white,legal_moves,self.move_history,self.is_san)
        elif self.play_mode == "bullet":
            messages = get_bullet_move_prompt(self.board.fen(),is_white,legal_moves,self.move_history,self.is_san)
        else:
            #Blindfold play mode is not implemented. It's hard to compose prompt for blindfold play mode.
            raise NotImplementedError
        self.messages.extend(messages)
        return messages
    
    def _validate_uci_move(self, uci_move: str) -> bool:
        """Validate that the UCI move is legal in the current board position."""
        try:
            move = self.board.parse_uci(uci_move)
            if move not in self.board.legal_moves:
                feedback = f"Illegal move: {uci_move} is not a legal move in the current position. Please choose another one. Note that we have given you all legal moves." if self.provide_legal_moves else f"Illegal move: {uci_move} is not a legal move in the current position. Please choose another one."
                raise ValueError(feedback)
            return True
        except Exception as e:
            raise ValueError(f"Invalid move: {uci_move}. Please choose another one.")
        
    def call_loop(self):
        messages = self.get_messages()
        for i in range(self.max_retry):
            if hasattr(self,"engine"):
                move = self.engine.predict_move(self.board)
                record_move = f"""
```
{move}
```
"""
                self.messages.append({"role":"assistant","content":record_move})
                return move
            
            
            response = connect_gpt(self.model_id, self.base_url, messages, self.max_tokens, 
                                self.temperature, self.top_p, self.api_key, self.enable_thinking)
            messages.append({"role":"assistant","content":response})
            self.messages.append({"role":"assistant","content":response})
            try:
                move = self.get_move(response)
            except Exception as e:
                feedback = str(e)
                move = ""
            if not move:
                if self.is_san:
                    feedback = "Parsing error: No valid SAN move found in response. Rethink the valid SAN move notation and example we give you. Your answer must follow our example(i.e., adding ``` in the beginning and ending)."
                else:
                    feedback = "Parsing error: No valid UCI move found in response. Rethink the valid UCI move notation and example we give you. Your answer must follow our example(i.e., adding ``` in the beginning and ending). Don't add \"x\" or \"+\" in UCI of your answer."
            else:
                try:
                    if self._validate_uci_move(move): #if valid
                        return move
                except Exception as e:
                    feedback = str(e)
            self.logger.debug(f"{move}: {feedback}")
            messages.append({"role":"user","content":feedback})
            self.messages.append({"role":"user","content":feedback})
            
        return ""
    
    def step(self):
        '''
        generate a move and push it to the board
        '''
        move = self.call_loop()
        if not move:
            return False
        self.logger.info(f"fen {self.board.fen()} Client's Move: {move}")
        f = self.push_move(chess.Move.from_uci(move))
        if not f:
            return False
        return move


if __name__ == '__main__':
    fen = "8/pp4Rp/7k/2p5/2qp4/7P/PP4RK/8 b - - 0 38"
    url = ""
    api_key = ""
    model_id = ""
    model_name = ""
    temperature = 0.2
    top_p = 0.95
    max_tokens = 2048
    enable_thinking = False
    is_san = True
    max_retry = 5
    provide_leagl_moves = True
    agent = Chess_Agent(fen,url,api_key,   model_id,   model_name,   temperature,   top_p,   max_tokens,   enable_thinking,   is_san,   max_retry)
    
    response = """
Let's analyze the position:

- Black to move.
- Material: Black is up a queen for a rook, but White has two active rooks and a pawn majority on the kingside.
- Black's king is on h6, White's rooks are on g7 and h2, White's king is on h2.
- Black's queen is on d4, pawns on a7, b7, c5, d4.
- White's pawns: a2, b2, h3.
- Black's king is a bit exposed, but White's rooks are not coordinated for an immediate mate.
- Black's queen is very active on d4, controlling many squares.
- White's threat: Rg6+ followed by h4-h5 could be dangerous, but currently, the queen covers g7 and h6.

Candidate moves:
1. Qxg7 is not possible.
2. Qe5+ is not possible.
3. Qd6 defends the 6th rank.
4. Qe3 threatens mate on g3.
5. Qf4+ is not possible.
6. Kh5 steps out of the pin and threatens to bring the king to safety, but could be exposed.
7. Qe2 threatens Qe5+ and Qh2+ ideas.
8. Qd3 attacks the b2 pawn.
9. Qxa2 wins a pawn.

Let's check for tactics:
- If Qxa2, White has Rg6+ Kh5, Rg5+! Kxg5, h4+ Kf6, Rf2+ Ke7, b3 and White is not mating, but Black's king is exposed.
- If Qd3, attacking b2, White has Rg6+ Kh5, Rg5+ Kxg5, h4+ Kf6, Rf2+ Ke7, b3.
- If Qe3, threatening mate on g3, White can play Rg6+ Kh5, Rg5+ Kxg5, h4+ Kf6, Rf2+ Ke7, b3.

But let's check if there is any immediate threat from White. The only check is Rg6+, after which Kh5 is possible.

Let's check Qe3:
- Qe3 threatens Qg1#.
- If White plays Rg3, then Qf4 and the queen is very active.
- If White plays h4, then Qf4+ and the king is exposed.

But Qe3 also threatens Qg1# and Qf4+.

Therefore, the best move is:
```
Qe3
```
"""
    print(agent.get_move(response))