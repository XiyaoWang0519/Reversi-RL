import numpy as np
import torch
import argparse
import os
from src.board import ReversiBoard
from src.agent import RLAgent

def get_human_move(board, valid_moves):
    """Get a move from the human player."""
    print("Valid moves:", valid_moves)
    
    while True:
        try:
            move_input = input("Enter your move (row,col): ")
            row, col = map(int, move_input.strip().split(','))
            
            # Check if move is valid
            if (row, col) in valid_moves:
                return row, col
            else:
                print("Invalid move! Please choose from valid moves.")
        except ValueError:
            print("Invalid input! Please enter row,col (e.g., 2,3)")
        except KeyboardInterrupt:
            print("\nExiting game.")
            exit(0)

def play_against_agent(agent, human_player=1, board_size=8):
    """Play a game of Reversi against the trained agent."""
    board = ReversiBoard(board_size)
    
    print("Starting a new game of Reversi!")
    print("You are playing as", "Black" if human_player == 1 else "White")
    print("Black goes first")
    print()
    
    while not board.is_game_over():
        # Display the board
        print(board)
        black_count, white_count = board.count_pieces()
        print(f"Black: {black_count}, White: {white_count}")
        print()
        
        current_player = board.current_player
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            print(f"No valid moves for {'Black' if current_player == 1 else 'White'}, skipping turn")
            continue
        
        if current_player == human_player:
            # Human's turn
            print("Your turn!")
            row, col = get_human_move(board, valid_moves)
        else:
            # Agent's turn
            print("Agent is thinking...")
            row, col = agent.select_move(board, valid_moves, training=False)
            print(f"Agent chose: {row},{col}")
        
        # Make the move
        board.make_move(row, col)
        print()
    
    # Game over
    print(board)
    print("Game over!")
    black_count, white_count = board.count_pieces()
    print(f"Final score - Black: {black_count}, White: {white_count}")
    
    winner = board.get_result()
    if winner == 0:
        print("It's a draw!")
    elif winner == human_player:
        print("You win!")
    else:
        print("Agent wins!")

def main():
    parser = argparse.ArgumentParser(description="Play Reversi against a trained agent")
    parser.add_argument("--model", type=str, default="models/reversi_agent_latest.pt", 
                        help="Path to the trained model file")
    parser.add_argument("--color", type=str, default="black", choices=["black", "white"],
                        help="Play as black or white")
    parser.add_argument("--size", type=int, default=8,
                        help="Board size (default: 8)")
    args = parser.parse_args()
    
    # Map color to player number
    human_player = 1 if args.color.lower() == "black" else -1
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Please train an agent first or specify a valid model path.")
        return
    
    # Load the agent with the same parameters used for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For our mini-train, we used board_size=6 and num_channels=32
    # In a real scenario, we'd save these parameters with the model
    if args.model == "models/reversi_agent_latest.pt" or args.model == "models/reversi_agent_final.pt":
        print("Loading mini-trained model with board_size=6, num_channels=32")
        agent = RLAgent(board_size=6, num_channels=32, device=device)
    else:
        agent = RLAgent(board_size=args.size, num_channels=128, device=device)
    
    agent.load_model(args.model)
    print(f"Loaded model from {args.model}")
    
    # Start the game
    try:
        # Use board_size=6 for mini-trained model
        if args.model == "models/reversi_agent_latest.pt" or args.model == "models/reversi_agent_final.pt":
            play_against_agent(agent, human_player, board_size=6)
        else:
            play_against_agent(agent, human_player, args.size)
    except KeyboardInterrupt:
        print("\nGame interrupted. Exiting.")

if __name__ == "__main__":
    main()