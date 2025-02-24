import torch
import numpy as np
import os
from src.board import ReversiBoard
from src.agent import RLAgent, ReversiNet

def test_board():
    """Test the Reversi board implementation."""
    print("Testing board implementation...")
    board = ReversiBoard(size=8)
    
    # Display initial board
    print("Initial board:")
    print(board)
    
    # Test valid moves for black (initial state)
    valid_moves = board.get_valid_moves()
    print("Valid moves for Black:", valid_moves)
    
    # Make a move
    if valid_moves:
        move = valid_moves[0]
        print(f"Making move: {move}")
        board.make_move(move[0], move[1])
        print(board)
    
    # Test valid moves for white
    valid_moves = board.get_valid_moves()
    print("Valid moves for White:", valid_moves)
    
    # Make a move for white
    if valid_moves:
        move = valid_moves[0]
        print(f"Making move: {move}")
        board.make_move(move[0], move[1])
        print(board)
    
    print("Board test complete!")
    print()

def test_agent():
    """Test the agent implementation."""
    print("Testing agent implementation...")
    
    # Create agent
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    agent = RLAgent(board_size=8, device=device)
    
    # Test forward pass
    board = ReversiBoard(size=8)
    valid_moves = board.get_valid_moves()
    
    # Select a move
    move = agent.select_move(board, valid_moves, training=True)
    print(f"Agent selected move: {move}")
    
    # Make the move
    board.make_move(move[0], move[1])
    print(board)
    
    # Test training update (requires some data in replay buffer)
    state = board.get_state()
    player = board.current_player
    policy = np.zeros(64)
    policy[move[0] * 8 + move[1]] = 1
    reward = 0.0
    
    agent.store_experience(state, player, policy, reward)
    
    # Need to add more experiences to meet batch size
    for i in range(130):
        agent.store_experience(state, player, policy, reward)
    
    # Test model update
    loss = agent.update_model()
    print(f"Training loss: {loss}")
    
    # Test save/load
    os.makedirs('models', exist_ok=True)
    agent.save_model('models/test_model.pt')
    print("Model saved to models/test_model.pt")
    
    try:
        agent.load_model('models/test_model.pt')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    print("Agent test complete!")
    print()

if __name__ == "__main__":
    test_board()
    test_agent()
    print("All tests completed successfully!")