import re
import chess
import json

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
    
def get_move(response, fen):
    try:
        uci_move = parse_uci_move(response)
        if not uci_move:
            san_move = parse_san_move(response)
            if san_move:
                uci_move = san_to_uci(san_move, fen)
        return uci_move
    except Exception as e:
        return None
    
def legal_reward_func(response, ground_truth, extra_info):
    reward = 0
    try:
        current_fen = extra_info["fen"]
        legal_moves = extra_info["legal_moves"]
        move = get_move(response, current_fen)
        if move and move in legal_moves:
            reward = 1
        else:
            reward = 0
    except:
            pass
    return reward,len(legal_moves)
    
def optimal_move_rank(response, ground_truth, extra_info):
    try:
        current_fen = extra_info["fen"]
        top_moves = extra_info["top_moves"]
        move = get_move(response, current_fen)
        if move in top_moves:
            return top_moves.index(move),len(top_moves) #[0,1,2]
        return -1,len(top_moves)
    except:
            pass
    return -1,len(top_moves)


def format_reward_func(response, ground_truth, extra_info):
    pattern = r'```\s*.*?\s*```'
    reward = 1 if re.search(pattern, response, re.DOTALL) else 0.0
    return reward


def compute_score_with_legal_moves(response, ground_truth, extra_info):
    provide_legal_moves = extra_info["provide_legal_moves"]
    format_score = format_reward_func(response, ground_truth,extra_info)
    legal_score,len_legal_moves = legal_reward_func(response, ground_truth,extra_info)
    optimal_rank,len_top_moves = optimal_move_rank(response, ground_truth,extra_info)
    
    reward = 0
    #format:0.1
    #legal:0.3
    #optimal:0.6
    if len_legal_moves > 10:
        if optimal_rank == 0:
            reward += 0.6
        elif optimal_rank == 1:
            reward += 0.4
        elif optimal_rank == 2:
            reward += 0.2
    else:
        if optimal_rank == 0:
            reward += 0.6
            
    return reward
        
def compute_score_without_legal_moves(response, ground_truth, extra_info):
    provide_legal_moves = extra_info["provide_legal_moves"]
    format_score = format_reward_func(response, ground_truth,extra_info)
    legal_score,len_legal_moves = legal_reward_func(response, ground_truth,extra_info)
    optimal_rank,len_top_moves = optimal_move_rank(response, ground_truth,extra_info)
    
    reward = 0
    
    if len_legal_moves > 10:
        if optimal_rank == 0:
            reward += 0.5
        elif optimal_rank == 1:
            reward += 0.3
        elif optimal_rank == 2:
            reward += 0.1
    else:
        if optimal_rank == 0:
            reward += 0.5
            
    return reward