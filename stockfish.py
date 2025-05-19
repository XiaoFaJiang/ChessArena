import chess
import chess.engine
import math
import os

def calculate_win_rate(board, stockfish_path="./stockfish-8-linux/Linux/stockfish_8_x64", depth=20):
    """
    Calculate approximate win percentage based on current board position
    
    Parameters:
    - board: chess.Board object representing current position
    - stockfish_path: Path to Stockfish executable
    - depth: How deep the engine should analyze (higher = more accurate but slower)
    
    Returns:
    - Win percentage for the side to move (0-100)
    """
    # Initialize the engine with the provided path
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    try:
        # Get position evaluation
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].relative.score(mate_score=10000)
        
        # Convert centipawn score to win percentage using sigmoid function
        win_percentage = 50 + 50 * (2 / (1 + math.exp(-0.004 * score)) - 1)
        
        return win_percentage
    finally:
        # Always close the engine properly
        engine.quit()


def get_best_moves_and_evaluate(board, candidate_move=None, stockfish_path="./stockfish-8-linux/Linux/stockfish_8_x64", n=3, depth=20):
    """
    Calculate the top n moves with their resulting win percentages and evaluate a specific candidate move if provided
    
    Parameters:
    - board: chess.Board object representing current position
    - candidate_move: Optional specific move to evaluate (chess.Move object or UCI string)
    - stockfish_path: Path to Stockfish executable
    - n: Number of best moves to return
    - depth: How deep the engine should analyze (higher = more accurate but slower)
    
    Returns:
    - Tuple containing:
      1. List of tuples (move, win_percentage) for the top n moves
      2. Tuple (candidate_move, win_percentage, ranking) for the candidate move if provided, otherwise None
         where ranking is the move's position in the top n list (1-based) or None if not in top n
    """
    # Initialize the engine with the provided path
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    try:
        # Convert string move to chess.Move if needed
        candidate_move_obj = None
        if candidate_move:
            if isinstance(candidate_move, str):
                try:
                    # Try parsing as UCI
                    candidate_move_obj = chess.Move.from_uci(candidate_move)
                except ValueError:
                    # Try parsing as SAN
                    try:
                        candidate_move_obj = board.parse_san(candidate_move)
                    except ValueError:
                        raise ValueError(f"Invalid move format: {candidate_move}")
            else:
                candidate_move_obj = candidate_move
            
            # Check if the move is legal
            if candidate_move_obj not in board.legal_moves:
                raise ValueError(f"Illegal move: {candidate_move}")
        
        # Get multipv analysis (multiple principal variations)
        info = engine.analyse(
            board, 
            chess.engine.Limit(depth=depth),
            multipv=n
        )
        
        # Process results
        moves_with_win_rates = []
        for pv_info in info:
            move = pv_info["pv"][0]  # First move in the principal variation
            score = pv_info["score"].relative.score(mate_score=10000)
            
            # Convert score to win percentage
            win_percentage = 50 + 50 * (2 / (1 + math.exp(-0.004 * score)) - 1)
            
            # Add to our list
            moves_with_win_rates.append((move, win_percentage))
        
        # Evaluate candidate move if provided and not already in top moves
        candidate_result = None
        if candidate_move_obj:
            # Check if candidate move is in top moves
            candidate_ranking = None
            for i, (move, win_rate) in enumerate(moves_with_win_rates, 1):
                if move == candidate_move_obj:
                    candidate_ranking = i
                    candidate_result = (candidate_move_obj, win_rate, candidate_ranking)
                    break
            
            # If candidate move is not in top moves, evaluate it separately
            if candidate_ranking is None:
                # Make a copy of the board
                board_copy = board.copy()
                
                # Make the move
                board_copy.push(candidate_move_obj)
                
                # Analyze the resulting position
                candidate_info = engine.analyse(board_copy, chess.engine.Limit(depth=depth))
                candidate_score = candidate_info["score"].relative.score(mate_score=10000)
                
                # Convert opponent's score to our win percentage (negate because it's from opponent's perspective)
                candidate_win_rate = 50 + 50 * (2 / (1 + math.exp(0.004 * candidate_score)) - 1)
                
                candidate_result = (candidate_move_obj, candidate_win_rate, None)  # None means not in top n
        
        return moves_with_win_rates, candidate_result
    finally:
        # Always close the engine properly
        engine.quit()
