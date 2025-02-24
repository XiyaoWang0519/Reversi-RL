import torch
import os
import random
from src.board import ReversiBoard
from src.agent import RLAgent

def test_auto_play():
    """Test an automated game between the agent and a random player."""
    print("Testing automated game...")
    
    # Load the agent
    device = torch.device('cpu')
    agent = RLAgent(board_size=6, num_channels=32, device=device)
    
    # Check if model exists
    model_path = "models/reversi_agent_final.pt"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        agent.load_model(model_path)
    else:
        print(f"Model file not found: {model_path}")
        print("Running with untrained agent")
    
    # Create board
    board = ReversiBoard(size=6)
    
    # Assign roles
    agent_player = 1  # Black
    random_player = -1  # White
    
    print("Starting game: Agent (Black) vs Random (White)")
    print("Initial board:")
    print(board)
    
    move_count = 0
    # Play until game is over
    while not board.is_game_over():
        current_player = board.current_player
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            print(f"No valid moves for {'Black' if current_player == 1 else 'White'}, skipping turn")
            continue
        
        if current_player == agent_player:
            # Agent's turn
            move = agent.select_move(board, valid_moves, training=False)
            player_name = "Agent"
        else:
            # Random player's turn
            move = random.choice(valid_moves)
            player_name = "Random"
        
        # Make move
        row, col = move
        print(f"Move {move_count + 1}: {player_name} ({row},{col})")
        board.make_move(row, col)
        move_count += 1
        
        # Print board every few moves
        if move_count % 5 == 0:
            print(board)
    
    # Game over
    print("\nFinal board:")
    print(board)
    
    black_count, white_count = board.count_pieces()
    print(f"Final score - Agent (Black): {black_count}, Random (White): {white_count}")
    
    winner = board.get_result()
    if winner == 0:
        print("Game result: Draw")
    elif winner == agent_player:
        print("Game result: Agent wins!")
    else:
        print("Game result: Random player wins!")
    
    print("Test game completed successfully!")

if __name__ == "__main__":
    test_auto_play()