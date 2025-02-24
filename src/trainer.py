import torch
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from src.board import ReversiBoard
from src.agent import RLAgent

class SelfPlayTrainer:
    """Trains a Reversi agent through self-play reinforcement learning."""
    
    def __init__(self, board_size=8, num_channels=128, device='cpu', use_wandb=True):
        self.board_size = board_size
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize agent
        self.agent = RLAgent(board_size, num_channels, device)
        
        # Training parameters
        self.num_episodes = 1000
        self.evaluation_interval = 50
        self.save_interval = 100
        self.temperature_threshold = 10  # Number of moves before temperature drops
        
        # Statistics
        self.training_stats = {
            'episode_rewards': [],
            'losses': [],
            'evaluation_scores': []
        }
        
        # Initialize Weights & Biases
        if self.use_wandb:
            config = {
                "board_size": board_size,
                "num_channels": num_channels,
                "device": str(device),
                "num_episodes": self.num_episodes,
                "evaluation_interval": self.evaluation_interval,
                "save_interval": self.save_interval,
                "temperature_threshold": self.temperature_threshold,
                "batch_size": self.agent.batch_size,
                "gamma": self.agent.gamma,
            }
            wandb.init(project="reversi-rl", config=config)
    
    def play_game(self, training=True):
        """Play a full game of Reversi using the current agent."""
        board = ReversiBoard(self.board_size)
        states_history = []
        policies_history = []
        players_history = []
        
        # Play until game is over
        while not board.is_game_over():
            current_player = board.current_player
            valid_moves = board.get_valid_moves()
            
            # Skip turn if no valid moves
            if not valid_moves:
                continue
            
            # Store current state
            states_history.append(board.get_state().copy())
            players_history.append(current_player)
            
            # Decide temperature based on move number
            temperature = 1.0 if len(states_history) < self.temperature_threshold else 0.5
            
            # Select move based on current policy
            move = self.agent.select_move(board, valid_moves, training, temperature)
            
            # Create one-hot policy vector
            policy = np.zeros(self.board_size * self.board_size)
            policy[move[0] * self.board_size + move[1]] = 1
            policies_history.append(policy)
            
            # Make move
            board.make_move(move[0], move[1])
        
        # Game is over, determine rewards
        winner = board.get_result()
        rewards = []
        
        for player in players_history:
            if winner == 0:  # Draw
                rewards.append(0.0)
            elif winner == player:  # Win
                rewards.append(1.0)
            else:  # Loss
                rewards.append(-1.0)
        
        # Store experiences in replay buffer
        if training:
            for state, player, policy, reward in zip(states_history, players_history, policies_history, rewards):
                self.agent.store_experience(state, player, policy, reward)
        
        return winner, rewards
    
    def evaluate_agent(self, num_games=10):
        """Evaluate the current agent by playing against itself without exploration."""
        wins = {1: 0, -1: 0, 0: 0}  # Black wins, White wins, Draws
        
        for _ in range(num_games):
            winner, _ = self.play_game(training=False)
            wins[winner] += 1
        
        win_rate = (wins[1] + wins[-1]) / num_games
        draw_rate = wins[0] / num_games
        
        return win_rate, draw_rate, wins
    
    def train(self):
        """Train the agent through self-play."""
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        print("Starting training...")
        
        for episode in tqdm(range(1, self.num_episodes + 1)):
            # Self-play
            winner, rewards = self.play_game(training=True)
            avg_reward = np.mean(rewards)
            self.training_stats['episode_rewards'].append(avg_reward)
            
            # Update model
            loss = self.agent.update_model()
            if loss is not None:
                self.training_stats['losses'].append(loss)
            
            # Log to W&B
            if self.use_wandb:
                metrics = {
                    "episode": episode,
                    "avg_reward": avg_reward,
                    "loss": loss if loss is not None else 0,
                    "buffer_size": len(self.agent.replay_buffer)
                }
                wandb.log(metrics)
            
            # Evaluation
            if episode % self.evaluation_interval == 0:
                win_rate, draw_rate, wins = self.evaluate_agent()
                print(f"Episode {episode}, Win rate: {win_rate:.2f}, Draw rate: {draw_rate:.2f}")
                print(f"Black wins: {wins[1]}, White wins: {wins[-1]}, Draws: {wins[0]}")
                self.training_stats['evaluation_scores'].append(win_rate)
                
                # Log evaluation metrics to W&B
                if self.use_wandb:
                    eval_metrics = {
                        "win_rate": win_rate,
                        "draw_rate": draw_rate,
                        "black_wins": wins[1],
                        "white_wins": wins[-1],
                        "draws": wins[0]
                    }
                    wandb.log(eval_metrics)
                
                # Plot statistics
                self._plot_statistics()
            
            # Save model
            if episode % self.save_interval == 0:
                model_path = f"models/reversi_agent_episode_{episode}.pt"
                self.agent.save_model(model_path)
                # Save latest model
                self.agent.save_model("models/reversi_agent_latest.pt")
                
                # Log model to W&B
                if self.use_wandb:
                    wandb.save(model_path)
        
        # Save final model
        final_model_path = "models/reversi_agent_final.pt"
        self.agent.save_model(final_model_path)
        if self.use_wandb:
            wandb.save(final_model_path)
            wandb.finish()
        
        print("Training completed.")
    
    def _plot_statistics(self):
        """Plot training statistics."""
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot episode rewards
        ax1.plot(self.training_stats['episode_rewards'])
        ax1.set_title('Average Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        ax1.grid(True)
        
        # Plot losses
        if self.training_stats['losses']:
            ax2.plot(self.training_stats['losses'])
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        # Save figure
        plt.tight_layout()
        rewards_plot_path = 'plots/training_stats.png'
        plt.savefig(rewards_plot_path)
        
        # Log figure to W&B
        if self.use_wandb:
            wandb.log({"training_stats": wandb.Image(rewards_plot_path)})
            
        plt.close()
        
        # Plot evaluation scores
        if self.training_stats['evaluation_scores']:
            plt.figure(figsize=(10, 6))
            plt.plot(range(self.evaluation_interval, 
                          self.evaluation_interval * (len(self.training_stats['evaluation_scores']) + 1), 
                          self.evaluation_interval), 
                    self.training_stats['evaluation_scores'])
            plt.title('Evaluation Win Rate')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            plt.grid(True)
            
            eval_plot_path = 'plots/evaluation_scores.png'
            plt.savefig(eval_plot_path)
            
            # Log figure to W&B
            if self.use_wandb:
                wandb.log({"evaluation_scores": wandb.Image(eval_plot_path)})
                
            plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Reversi reinforcement learning agent')
    parser.add_argument('--board_size', type=int, default=8, help='Size of the Reversi board')
    parser.add_argument('--num_channels', type=int, default=128, help='Number of channels in neural network')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging')
    args = parser.parse_args()
    
    # Handle device selection
    if args.device:
        device = torch.device(args.device)
    else:
        # Auto-detect best available device
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create trainer
    trainer = SelfPlayTrainer(
        board_size=args.board_size,
        num_channels=args.num_channels,
        device=device,
        use_wandb=not args.no_wandb
    )
    
    # Override number of episodes if specified
    if args.episodes != 1000:
        trainer.num_episodes = args.episodes
        print(f"Training for {args.episodes} episodes")
    
    # Start training
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    print(f"Training took {end_time - start_time:.2f} seconds")