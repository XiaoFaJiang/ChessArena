import re
import chess
import random
import json
import requests
import time
import copy
import chess.pgn
from datetime import datetime



def parse_fen_from_user_prompt(prompt):
    FEN_REGEX = r'Current board position in FEN notation:\s*([^\s]+(?:\s+[^\s]+){5})'
    fen_match = re.search(FEN_REGEX, prompt)
    fen = fen_match.group(1).strip() if fen_match else ""
    return fen

def judge_thinking(content):
    """judge if there are any forbidden thinking process"""
    content = content.strip()
    pattern = re.compile(r"\s*([a-h][1-8][a-h][1-8](?:[qrbnQRBN])?)\s*",re.DOTALL) #judge any UCI moves
    content = pattern.sub('',content) #substitue UCI move into empty string
    content_lst = content.split('```')
    real_lst = []
    for v in content_lst:
        if v.strip():
            real_lst.append(v)
    if len(real_lst) > 0: #judge if there any other words
        return True
    return False


def connect_gpt(model, url, messages, max_tokens,temperature,top_p,api_key,enable_thinking):
    headers = {
            "Content-Type": "application/json",
            "Authorization": api_key
        }
    
    
    payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            }
    ret = None
    url = url.rstrip('/')
    if "chat/completions" not in url:
        url = url + "/chat/completions"
    while ret is None or len(ret) == 0:
        try:
            response = requests.post(
                        url,
                        headers=headers,
                        json=payload,
                        timeout=3600
                    )
            reasoning_content = response.json()['choices'][0]['message'].get('reasoning_content',"")
            if not reasoning_content:
                reasoning_content = ""
            ret = reasoning_content + \
                response.json()['choices'][0]['message']['content']
            ret = ret.strip()
        except requests.exceptions.Timeout:
            print("Timeout error. Waiting...")
            print(e)
            time.sleep(20)
        except requests.exceptions.RequestException as e:
            print("RequestException error. Waiting...")
            print(e)
            time.sleep(20)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(20)
    return ret

def parse_json(content):
    """Parse json data from LLM response."""
    pattern = re.compile(r"```json(.*?)```",re.DOTALL)
    jsons = pattern.findall(content)
    try:
        if jsons:
            return json.loads(jsons[-1])
    except:
        pass
    return {}

def parse_uci_move(content,static_eval=False):
    """Parse UCI chess move from LLM response."""
    # 找到所有匹配项，取最后一个
    moves = re.findall(r'```\s*([a-h][1-8][a-h][1-8](?:[qrbnQRBN])?)\s*```', content, re.DOTALL)
    if moves:
        return moves[-1]  # 返回最后一个匹配项
    if static_eval:
        moves = re.findall(r'\s*([a-h][1-8][a-h][1-8](?:[qrbnQRBN])?)\s*', content, re.DOTALL) #去除```的影响
        if moves:
            return moves[-1]

def parse_san_move(content,static_eval=False):
    """Parse SAN chess move from LLM response."""
    pattern = r'```\s*(?P<move>O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[KQRBN])?[+#]?)\s*```'
    move_match = re.findall(pattern, content, re.DOTALL)
    if move_match:
        return move_match[-1]
    if static_eval:
        move_match = re.findall(r'\s*(?P<move>O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[KQRBN])?[+#]?)\s*', content, re.DOTALL)
        if move_match:
            return move_match[-1]


def san_to_uci(san_move, board_position=None):
    """
    将SAN格式转换为UCI格式
    
    Args:
        san_move: SAN格式的走法，如 "Nf3", "exd5", "O-O"
        board_position: 棋盘位置，如果None则使用初始位置
    
    Returns:
        UCI格式的走法，如 "g1f3", "e5d5", "e1g1"
    """
    if type(san_move) != str:
        san_move = str(san_move)
    if board_position is None:
        board = chess.Board()  # 使用初始位置
    else:
        board = chess.Board(fen=board_position)  # 使用指定位置
    
    try:
        # 解析SAN走法
        move = board.parse_san(san_move)
        # 转换为UCI格式
        return move.uci()
    except Exception as e:
        return san_move
        #raise Exception(f"Error parsing SAN move: {san_move}") from e

def uci_to_san(uci_move, board_position=None):
    """
    将UCI格式转换为SAN格式
    
    Args:
        uci_move: UCI格式的走法，如 "g1f3", "e2e4"
        board_position: 棋盘位置，如果None则使用初始位置
    
    Returns:
        SAN格式的走法，如 "Nf3", "e4"
    """
    if type(uci_move) != str:
        uci_move = str(uci_move)
    if board_position is None:
        board = chess.Board()  # 使用初始位置
    else:
        board = chess.Board(fen=board_position)  # 使用指定位置
    
    try:
        # 解析UCI走法
        move = chess.Move.from_uci(uci_move)
        # 转换为SAN格式
        return board.san(move)
    except Exception as e:
        return uci_move
        #raise Exception(f"Error parsing UCI move: {uci_move}") from e

def remove_all_empty_lines(text):
    return re.sub(r'\n\s*\n', '\n', text)


def sp_blitz(is_white,merl=False):
    merl_str = ""
    if merl:
        merl_str = """
Your reasoning process and answer must be enclosed within <think> </think> and <answer> </answer> tags, respectively.
You will get evaluated following Evaluation Scoring Rules:
# Format Score:
- ```<move>``` is the most important tag, if you output your move inside ```, score 0.5
- If you output special tokens like <think>, each token will add 0.125 points
- So, if you follow the format exactly as above, format score is 1
- Otherwise, score 0

# Correctness Score:
- If your final move is legal but not good enough, score 3
- An additional move quality score, where the score is higher when your move is better, with a maximum score of 6 and a minimum score of 0.
- So, if you get a best move, correctness score is 9
- Otherwise, score 0

You will get the final score as their sum. Only when you have got the format score can you potentially obtain the correct score. 
You should try your best to analysis the chess board and get the optimal move.
"""
    system_prompt = f"""
You are an expert chess player. You are playing a game of chess. {'You are playing as White.' if is_white else 'You are playing as Black.'}
You must thoroughly analyze the position and play with utmost caution. When you have the advantage, press it relentlessly and aim for a swift checkmate. Carefully evaluate every move to eliminate any chance of a counterplay or draw by your opponent.
When at a disadvantage, strive to turn the tide and win if possible. If victory is unattainable, exhaust all possible means to force a draw.
Meticulously analyze legal moves, then select the absolute best one. You need to determine whether you are playing as Black or White. Then, you need to observe the positions of your pieces and choose one of your own pieces to move; make sure that your move follows the rules of chess.

Considering the long-term strategy and short-term tactic. Analyze the position carefully. You may think through the position and consider multiple candidate moves.
When you have decided on your final move, output it in UCI notation (e.g., 'e2e4', 'g8f6' , 'e7e8q') in the following format:
```
<move>
```

Note: UCI notation represents chess moves using only start and end positions like "e2e4" or "g1f3", treating captures the same as regular moves without "x" or "+" symbols, and adds a letter like "q" for pawn promotion (e.g., "e7e8q").
For example:
```
e2e4
```

Reminder of chess rules:
- Bishops move diagonally.
- Rooks move horizontally or vertically.
- Knights jump in an L-shape.
- Queens combine rook and bishop movement.
- Kings move one square in any direction.
- Pawns move forward, capture diagonally, and can promote.
{merl_str}
You can think and reason as much as you want(step by step), but your final move must be formatted exactly as shown above.
"""
    return system_prompt


def sp_bullet(is_white):
    system_prompt = f"""
You are an expert chess player. You are playing a game of chess. {'You are playing as White.' if is_white else 'You are playing as Black.'}
You must thoroughly analyze the position and play with utmost caution. When you have the advantage, press it relentlessly and aim for a swift checkmate. Carefully evaluate every move to eliminate any chance of a counterplay or draw by your opponent.
When at a disadvantage, strive to turn the tide and win if possible. If victory is unattainable, exhaust all possible means to force a draw.
Meticulously analyze legal moves, then select the absolute best one. You need to determine whether you are playing as Black or White. Then, you need to observe the positions of your pieces and choose one of your own pieces to move; make sure that your move follows the rules of chess.

Considering the long-term strategy and short-term tactic. Analyze the position carefully. You may think through the position and consider multiple candidate moves.
When you have decided on your final move, output it in UCI notation (e.g., 'e2e4', 'g8f6' , 'e7e8q') in the following format:
```
<move>
```

Note: UCI notation represents chess moves using only start and end positions like "e2e4" or "g1f3", treating captures the same as regular moves without "x" or "+" symbols, and adds a letter like "q" for pawn promotion (e.g., "e7e8q").
For example:
```
e2e4
```

Reminder of chess rules:
- Bishops move diagonally.
- Rooks move horizontally or vertically.
- Knights jump in an L-shape.
- Queens combine rook and bishop movement.
- Kings move one square in any direction.
- Pawns move forward, capture diagonally, and can promote.

You must give me your answer directly without using any other words. I will not accept your answer if there exists any other words. Only output ```<move>```. 
Your final move must be formatted exactly as shown above.
"""
    return system_prompt


def sp_blindfold(is_white):
    system_prompt = f"""
You are an expert chess player. You are playing a game of chess. {'You are playing as White.' if is_white else 'You are playing as Black.'}
We have the move history of you and your opponent. You must reconstruct the game and choose a best move from the legal moves.

This is a critically urgent match. You must thoroughly analyze the position and play with utmost caution. Leverage your wisdom to secure victory.
When you have the advantage, press it relentlessly and aim for a swift checkmate. Carefully evaluate every move to eliminate any chance of a counterplay or draw by your opponent.
When at a disadvantage, strive to turn the tide and win if possible. If victory is unattainable, exhaust all possible means to force a draw.
Meticulously analyze legal moves, then select the absolute best one. You need to determine whether you are playing as Black or White. Then, you need to observe the positions of your pieces and choose one of your own pieces to move; make sure that your move follows the rules of chess.

Considering the long-term strategy and short-term tactic. Analyze the position carefully. You may think through the position and consider multiple candidate moves.
When you have decided on your final move, output it in UCI notation (e.g., 'e2e4', 'g8f6' , 'e7e8q') in the following format:
```
<move>
```

Note: UCI notation represents chess moves using only start and end positions like "e2e4" or "g1f3", treating captures the same as regular moves without "x" or "+" symbols, and adds a letter like "q" for pawn promotion (e.g., "e7e8q").
For example:
```
e2e4
```

Reminder of chess rules:
- Bishops move diagonally.
- Rooks move horizontally or vertically.
- Knights jump in an L-shape.
- Queens combine rook and bishop movement.
- Kings move one square in any direction.
- Pawns move forward, capture diagonally, and can promote.

You can think and reason as much as you want(step by step), but your final move must be formatted exactly as shown above.
""" 
    return system_prompt


def sp_chess_modeling():
    system_prompt = """
You are an expert chess player. I need you to help me model a chessboard. The specific steps are as follows:
I will provide you with a FEN string representing the current board state, and then give you a position. You need to identify the piece at that position from the FEN and output all legal moves for that piece.
You must carefully analyze the board, consider the rules of chess, and provide the final answer. 

Your answer should be format as follows(output a json):
```json
{
    "piece": <piece symbol>,
    "legal_moves": [<list of legal moves>]
}
```

For example:
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Position: g1
Answer:
```json
{
    "piece": "N",
    "legal_moves": ["g1h3", "g1f3"]
}
```
Note:
If the given position has no piece, directly output empty(i.e. None), and the corresponding legal moves should also be empty(i.e. []).
When it's White's turn to move, if the position contains a Black piece, you should identify the piece but its legal moves must be empty (and vice versa for Black's turn).
You can think and reason as much as you want(step by step), but your final answer must be formatted exactly as shown above.
"""
    return system_prompt


def get_blitz_move_prompt(fen, is_white, legal_moves=None,move_history=None,is_san=False):
    '''
    Generate a prompt for the LLM to make a blitz move.
    given the current board position in FEN notation, and whether it is white or black's turn, and a list of legal moves
    enable thinking and reason.
    '''
    # Format the legal moves string if provided, otherwise empty string
    system_prompt = sp_blitz(is_white,is_san)
    legal_moves_str = ""
    if legal_moves:
        if not is_san:
            moves_list = ", ".join(move.uci() for move in legal_moves)
            legal_moves_str = f"\nLegal moves in UCI notation: {moves_list}\n"
        else:
            moves_list = ", ".join([uci_to_san(move,fen) for move in legal_moves])
            legal_moves_str = f"\nLegal moves in SAN notation: {moves_list}\n"
    
    move_history_str = ""
    if move_history:
        if not is_san:
            move_history_str = f"\nPartial move history in UCI notation:{', '.join(move_history)}\n"
        else:
            move_history_str = f"\nPartial move history in SAN notation:{', '.join(move_history)}\n"
    # Complete prompt as a single f-string
    if legal_moves_str:
        end_hint = "What is the best move to make out of the list of legal moves? Think it step by step."
    else:
        end_hint = "What is the best move to make out of the legal moves? Think it step by step."
    user_prompt = f"""
Current board position in FEN notation:
{fen}
{move_history_str}
{legal_moves_str}
{end_hint}
"""
    return [{"role":"system","content":system_prompt},{"role": "user", "content": remove_all_empty_lines(user_prompt)}]

def generate_random_move(legal_moves):
    return random.choice(legal_moves)
def get_bullet_move_prompt(fen, is_white, legal_moves=None, move_history=None):
    '''
    Generate a prompt for the LLM to make a bullet move.
    given the current board position in FEN notation, and whether it is white or black's turn, and a list of legal moves
    unable thinking and reason.
    '''
    # Format the legal moves string if provided, otherwise empty string
    system_prompt = sp_bullet(is_white)
    legal_moves_str = ""
    if legal_moves:
        moves_list = ", ".join(move.uci() for move in legal_moves)
        legal_moves_str = f"\nLegal moves in UCI notation: {moves_list}\n"
    
    move_history_str = ""
    if move_history:
        move_history_str = f"\nPartial move history in UCI notation:{', '.join(move_history)}\n"
    # Complete prompt as a single f-string
    user_prompt = f"""
Current board position in FEN notation:
{fen}
{legal_moves_str}
{move_history_str}
"""
    
    return [{"role":"system","content":system_prompt},{"role": "user", "content": remove_all_empty_lines(user_prompt)}]

def get_blindfold_move_prompt(fen, is_white, legal_moves=None, move_history=None):
    '''
    Generate a prompt for the LLM to make a blindfold move.
    given the move history in UCI notation, and whether it is white or black's turn, and a list of legal moves
    enable thinking and reason.
    '''
    # Format the legal moves string if provided, otherwise empty string
    system_prompt = f"""
You are an expert chess player. You are playing a game of chess. {'You are playing as White.' if is_white else 'You are playing as Black.'}
We only have the move history of you and your opponent. You must reconstruct the game and choose a best move from the legal moves.

This is a critically urgent match. You must thoroughly analyze the position and play with utmost caution. Leverage your wisdom to secure victory.
When you have the advantage, press it relentlessly and aim for a swift checkmate. Carefully evaluate every move to eliminate any chance of a counterplay or draw by your opponent.
When at a disadvantage, strive to turn the tide and win if possible. If victory is unattainable, exhaust all possible means to force a draw.
Meticulously analyze legal moves, then select the absolute best one. You need to determine whether you are playing as Black or White. Then, you need to observe the positions of your pieces and choose one of your own pieces to move; make sure that your move follows the rules of chess.

Considering the long-term strategy and short-term tactic. Analyze the position carefully. You may think through the position and consider multiple candidate moves.
When you have decided on your final move, output it in UCI notation (e.g., 'e2e4', 'g8f6' , 'e7e8q') in the following format:
```
<move>
```

For example:
```
e2e4
```
You can think and reason as much as you want(step by step), but your final move must be formatted exactly as shown above.
"""
    legal_moves_str = ""
    if legal_moves:
        moves_list = ", ".join(move.uci() for move in legal_moves)
        legal_moves_str = f"\nLegal moves in UCI notation: {moves_list}\n"
    
    move_history_str = "\nMove history in UCI notation: None(It's the begining of the game).\n"
    if move_history:
        move_history_str = f"\nMove history in UCI notation:{', '.join(move_history)}\n"
    # Complete prompt as a single f-string
    user_prompt = f"""
{move_history_str}
{legal_moves_str}
"""
    
    return [{"role":"system","content":system_prompt},{"role": "user", "content": remove_all_empty_lines(user_prompt)}]

def sp_blindfold_board_reduction(is_white):
    system_prompt = f"""
You are an expert chess player. You are playing a game of chess. {'You are playing as White.' if is_white else 'You are playing as Black.'}
We only have the move history of you and your opponent. You must reconstruct the game and choose a best move from the legal moves.
You should reconstruct the chess board and give me the board as FEN notation in the following format.
```
<fen>
```

For example:
```
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```
You can think and reason as much as you want(step by step), but your final move must be formatted exactly as shown above.
"""
    return system_prompt


def extract_fen(input_text):
    '''
    Extract the FEN string from the input text.
    '''
    # FEN格式：棋盘位置 行棋方 易位权利 吃过路兵 半回合计数 全回合计数
    # 棋盘位置包含斜杠，所以需要特殊处理
    fen_pattern = r'([rnbqkpRNBQKP1-8/]+)\s+([wb])\s+([KQkq\-]+)\s+([a-h][36]\-|[a-h][36]|\-)\s+(\d+)\s+(\d+)'
    
    # 先尝试从代码块中提取
    code_block_pattern = r'```(?:\w+)?\s*([^`]+)\s*```'
    code_blocks = re.finditer(code_block_pattern, input_text, re.MULTILINE | re.DOTALL)
    
    for block in code_blocks:
        content = block.group(1).strip()
        fen_match = re.search(fen_pattern, content)
        if fen_match:
            return fen_match.group(0)
    
    # 如果代码块中没找到，直接在全文中搜索
    fen_match = re.search(fen_pattern, input_text)
    if fen_match:
        return fen_match.group(0)
    
    return None


def get_multi_turn_blindfold_move_prompt(is_white, legal_moves=None, chat_history=None, last_opponent_move=None):
    '''
    Multi Turn blindfold move prompt. Give the move history of a player and its opponent.
    Model can think and reason as much as it wants.
    '''
    
    system_prompt = sp_blindfold(is_white)
    pattern = re.compile(r"Legal moves in UCI notation: (.*)\n")
    if chat_history:
        chat_history = copy.deepcopy(chat_history)
        for i, v in enumerate(chat_history):
            if v and v.get('role') == 'user' and 'content' in v:
                chat_history[i]['content'] = pattern.sub('', v['content']) #delete history legal moves.

    legal_moves_str = ""
    if legal_moves:
        moves_list = ", ".join(move.uci() for move in legal_moves)
        legal_moves_str = f"\nLegal moves in UCI notation: {moves_list}\n"
    if last_opponent_move == None or last_opponent_move == 'None':
        user_prompt = f"""
It's the begining of the game. 
"""
    else:
        user_prompt = f"""
The opponent's last move is {last_opponent_move}.
{legal_moves_str}
""" 

    if not chat_history:
        chat_history = []
    chat_history.append({'role':'user','content':remove_all_empty_lines(user_prompt)})
    return system_prompt,chat_history

def get_piece_and_legal_moves(fen_string, position):
    """
    获取指定位置的棋子和该棋子的所有合法走法
    
    Args:
        fen_string: FEN字符串，表示棋盘状态
        position: 位置字符串，如"a1", "e4"等
    
    Returns:
        dict: 包含棋子信息和合法走法的字典
        {
            "piece": str or None,  # 棋子符号，如"P", "r", None表示空
            "legal_moves": list,   # UCI格式的合法走法列表
            "current_turn": str,   # "white" 或 "black"
            "error": str or None   # 错误信息
        }
    """
    try:
        # 从FEN字符串创建棋盘
        board = chess.Board(fen_string)
        
        # 将位置字符串转换为chess.Square对象
        square = chess.parse_square(position)
        
        # 获取指定位置的棋子
        piece = board.piece_at(square)
        
        # 获取当前行棋方
        current_turn = "white" if board.turn == chess.WHITE else "black"
        
        # 初始化结果
        result = {
            "piece": None,
            "legal_moves": [],
            "current_turn": current_turn,
            "error": None
        }
        
        # 如果位置为空，直接返回
        if piece is None:
            return result
        
        # 设置棋子信息
        result["piece"] = piece.symbol()
        
        # 检查棋子颜色是否与当前行棋方匹配
        piece_color = "white" if piece.color == chess.WHITE else "black"
        
        # 只有当棋子颜色与当前行棋方匹配时，才计算合法走法
        if piece_color == current_turn:
            # 获取所有合法走法
            legal_moves = list(board.legal_moves)
            
            # 筛选出从指定位置开始的走法，并转换为UCI格式
            moves_from_position = [
                move.uci() for move in legal_moves 
                if move.from_square == square
            ]
            
            result["legal_moves"] = moves_from_position
        
        return result
        
    except ValueError as e:
        return {
            "piece": None,
            "legal_moves": [],
            "current_turn": None,
            "error": f"Invalid fen or position: {str(e)}"
        }
    except Exception as e:
        return {
            "piece": None,
            "legal_moves": [],
            "current_turn": None,
            "error": f"Error in processing: {str(e)}"
        }
        
       
def get_random_piece_position_by_color(fen_string, color=None):
    """
    从给定的FEN字符串中随机返回一个指定颜色棋子的位置
    
    Args:
        fen_string: FEN字符串，表示棋盘状态
        color: 棋子颜色，"white"/"black"/None(任意颜色)
    
    Returns:
        str: 随机选择的有棋子的位置
        None: 如果没有找到指定颜色的棋子或FEN无效
    """
    try:
        board = chess.Board(fen_string)
        occupied_squares = []
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                if color is None:
                    occupied_squares.append(square)
                elif color == "white" and piece.color == chess.WHITE:
                    occupied_squares.append(square)
                elif color == "black" and piece.color == chess.BLACK:
                    occupied_squares.append(square)
        
        if not occupied_squares:
            return None
        
        random_square = random.choice(occupied_squares)
        return chess.square_name(random_square)
        
    except (ValueError, Exception) as e:
        print(f"错误: 无效的FEN字符串 - {e}")
        return None 

def get_random_empty_position(fen_string):
    """
    从给定的FEN字符串中随机返回一个不存在棋子的位置
    
    Args:
        fen_string: FEN字符串，表示棋盘状态
    
    Returns:
        str: 随机选择的空位置 (如"a1", "e4"等)
        None: 如果棋盘已满或FEN无效
    """
    try:
        # 从FEN字符串创建棋盘
        board = chess.Board(fen_string)
        
        # 找到所有空位置
        empty_squares = []
        
        # 遍历棋盘上的所有64个格子
        for square in chess.SQUARES:
            if board.piece_at(square) is None:
                empty_squares.append(square)
        
        # 如果没有空位置，返回None
        if not empty_squares:
            return None
        
        # 随机选择一个空位置
        random_square = random.choice(empty_squares)
        
        # 将square对象转换为位置字符串
        return chess.square_name(random_square)
        
    except (ValueError, Exception) as e:
        print(f"Error: invalid fen string - {e}")
        return None

def evaluate_sets(pred_set, ground_truth_set):
    """
    计算预测集合和真实集合的Precision、Recall和F1-Score
    
    Args:
        pred_set (set): 预测集合
        ground_truth_set (set): 真实集合（Ground Truth）
    
    Returns:
        dict: 包含precision、recall、f1_score和详细统计信息的字典
    """
    # 转换为集合，确保输入正确
    pred_set = set(pred_set)
    ground_truth_set = set(ground_truth_set)
    # 计算交集
    intersection = pred_set & ground_truth_set
    # 计算各项指标
    true_positives = len(intersection)
    # 计算Precision
    if len(pred_set) == 0:
        precision = 1.0 if len(ground_truth_set) == 0 else 0.0
    else:
        precision = true_positives / len(pred_set)
    
    # 计算Recall
    if len(ground_truth_set) == 0:
        recall = 1.0 if len(pred_set) == 0 else 0.0
    else:
        recall = true_positives / len(ground_truth_set)
    
    # 计算F1-Score
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }

def uci_to_pgn(uci_moves, event="Chess Game", site="Online", white="Player1", black="Player2", result="*"):
    """
    将UCI格式的棋步转换为PGN格式
    
    Args:
        uci_moves: UCI格式的棋步列表，如 ["e2e4", "g8f6", ...]
        event: 比赛事件名称
        site: 比赛地点
        white: 白方棋手名称
        black: 黑方棋手名称
        result: 比赛结果 ("1-0", "0-1", "1/2-1/2", "*")
    
    Returns:
        PGN格式的字符串
    """
    
    # 创建新的棋局
    game = chess.pgn.Game()
    
    # 设置PGN头部信息
    game.headers["Event"] = event
    game.headers["Site"] = site
    game.headers["Date"] = "Now"
    game.headers["Round"] = "?"
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Result"] = result
    
    # 创建棋盘
    board = chess.Board()
    node = game
    
    # 逐步添加棋步
    for uci_move in uci_moves:
        try:
            # 将UCI格式转换为Move对象
            move = chess.Move.from_uci(uci_move)
            
            # 检查棋步是否合法
            if move in board.legal_moves:
                # 在棋盘上执行棋步
                board.push(move)
                # 添加到PGN节点
                node = node.add_variation(move)
            else:
                print(f"警告: 非法棋步 {uci_move}")
                break
                
        except ValueError as e:
            print(f"错误: 无法解析棋步 {uci_move}: {e}")
            break
    
    return str(game)

def load_uci_from_json(json_file):
    """从JSON文件加载UCI棋步"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('moves', [])

def save_pgn_to_file(pgn_content, filename):
    """将PGN内容保存到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(pgn_content)
    print(f"PGN文件已保存到: {filename}")
    

def parse_side_to_move_from_fen(fen):
    """
    从FEN字符串中解析轮到哪方走棋
    
    FEN格式说明:
    FEN = "棋盘位置 轮到走棋方 王车易位权利 吃过路兵目标格 半步计数器 全步计数器"
    
    轮到走棋方字段:
    - 'w' = 白方 (White)
    - 'b' = 黑方 (Black)
    
    Parameters:
    - fen: FEN字符串
    
    Returns:
    - 'white' 如果轮到白方
    - 'black' 如果轮到黑方
    - None 如果解析失败
    """
    
    # 正则表达式解释:
    # ^[rnbqkpRNBQKP1-8/]+  : 匹配棋盘位置部分（开头）
    # \s+                   : 匹配一个或多个空格
    # ([wb])                : 捕获组，匹配 'w' 或 'b'
    # .*$                   : 匹配剩余的所有内容到结尾
    
    pattern = r'^[rnbqkpRNBQKP1-8/]+\s+([wb]).*$'
    
    match = re.match(pattern, fen.strip())
    
    if match:
        side_char = match.group(1)
        if side_char == 'w':
            return 'white'
        elif side_char == 'b':
            return 'black'
    
    return None

if __name__ == '__main__':
    content = """Let's analyze the position:

FEN: 8/8/8/1P2k3/8/1P3K2/8/8 w - - 3 49

White: King on f3, pawns on b3 and b5  
Black: King on e5

It's White's move.

**Material:**  
White has two pawns (b3, b5) and King (f3).  
Black has only the King (e5).

**Objective:**  
Convert the material advantage (two pawns) into a win. The plan is to promote one of the pawns while keeping the Black king away.

**Analysis:**
- If b6, Black plays Kd6, and after b7, Kc7, and the pawn is stopped.
- If Ke3, Black can approach the pawns.
- If b4, Black can play Kd4 and try to attack the pawns.

But the best plan is to use the king to support the pawns.

**Candidate moves:**
1. Ke3: Centralizes the king, but Black can play Kd5 and approach the pawns.
2. Kg4: Approaches the pawns from the kingside, but not optimal.
3. b6: Immediate push, but Black can stop the pawn.
4. Ke2: Head towards the queenside to support the pawns.

But let's see what happens if we play Ke3:

- 1. Ke3 Kd5 (Black tries to approach)
- 2. Kd3 (White heads to c4 to support the pawns)
- If Black plays Kc5, then Kc3, and White can support the b-pawns.

Alternatively, let's try b6:

- 1. b6 Kd6 (Black approaches)
- 2. b7 Kc7 (Black stops the pawn)
- 3. Ke4 Kxb7 (Black wins the pawn)

So b6 is not optimal.

What about Kg4?

- 1. Kg4 Kd4 (Black approaches)
- 2. Kf5 Kc3 (Black attacks the pawns)
- 3. Ke5 Kxb3 (Black wins a pawn)

So not optimal.

What about Ke2?

- 1. Ke2 Kd4
- 2. Kd2 (heading to c2/c3)
- If Black plays Kc5, then Kc3, and White can support the pawns.

But the most direct way is to play Ke3, centralizing the king and heading to support the pawns.

**Final Decision:**  
The best move is to centralize the king and head towards the queenside to support the pawns.

```
Ke3
```
"""
    fen = "8/8/8/1P2k3/8/1P3K2/8/8 w - - 3 49"
    parsed_san = parse_san_move(content)
    if parsed_san:
        print(parsed_san)
        print(san_to_uci(parsed_san,fen))