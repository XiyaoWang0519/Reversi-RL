import torch
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
import copy
from src.board import ReversiBoard
from src.agent import RLAgent, ReversiNet
from src.elo import ModelPool, ELOTracker

class ELOSelfPlayTrainer:
    """Trains a Reversi agent through ELO-based self-play reinforcement learning."""
    
    def __init__(self, board_size=8, num_channels=128, device='cpu', use_wandb=True):
        self.board_size = board_size
        self.device = device
        self.use_wandb = use_wandb
        self.num_channels = num_channels
        
        # Initialize agent with improved parameters
        self.agent = RLAgent(board_size, num_channels, device)
        
        # Initialize model pool for opponents with larger size
        self.model_pool = ModelPool(save_dir="models/pool", max_pool_size=50)
        
        # Add current model to the pool
        self.current_agent_id = self.model_pool.add_model(self.agent.model)
        
        # Initialize improved ELO tracker with larger window
        self.elo_tracker = ELOTracker(window_size=200)
        
        # Training parameters
        self.num_episodes = 1000
        self.evaluation_interval = 50
        self.save_interval = 100
        self.pool_update_interval = 100  # Add models less frequently for more stable ratings
        self.temperature_threshold = 15  # Increased for more exploration
        self.opponent_pool_temperature = 1.5  # Lower for more selective opponent sampling
        
        # Additional training parameters
        self.evaluation_games = 20  # More evaluation games for better statistics
        self.min_buffer_size = 1000  # Minimum buffer size before learning starts
        self.log_interval = 20  # How often to log detailed metrics
        
        # Statistics
        self.training_stats = {
            'episode_rewards': [],
            'losses': [],
            'evaluation_scores': [],
            'elo_ratings': [],
            'smoothed_elo_ratings': [],
            'opponent_strengths': [],
            'win_rates': []
        }
        
        # Initialize Weights & Biases with more config parameters
        if self.use_wandb:
            config = {
                # Board & model settings
                "board_size": board_size,
                "num_channels": num_channels,
                "device": str(device),
                
                # Training settings
                "num_episodes": self.num_episodes,
                "evaluation_interval": self.evaluation_interval,
                "evaluation_games": self.evaluation_games,
                "save_interval": self.save_interval,
                "pool_update_interval": self.pool_update_interval,
                "temperature_threshold": self.temperature_threshold,
                "opponent_pool_temperature": self.opponent_pool_temperature,
                "min_buffer_size": self.min_buffer_size,
                
                # Agent settings
                "batch_size": self.agent.batch_size,
                "gamma": self.agent.gamma,
                "learning_rate": self.agent.optimizer.param_groups[0]['lr'],
                "value_weight": self.agent.value_weight,
                "policy_weight": self.agent.policy_weight,
                "entropy_weight": self.agent.entropy_weight,
                
                # ELO settings
                "elo_k_factor": self.elo_tracker.elo_system.base_k_factor,
                "elo_smoothing_factor": self.elo_tracker.smoothing_factor
            }
            wandb.init(project="reversi-rl-elo-improved", config=config)
    
    def play_game_against_opponent(self, opponent_model, training=True):
        """Play a full game against a specific opponent model."""
        board = ReversiBoard(self.board_size)
        states_history = []
        policies_history = []
        players_history = []
        
        # Randomly decide which player goes first
        current_agent_as_black = np.random.random() < 0.5
        current_agent_player = 1 if current_agent_as_black else -1
        opponent_player = -current_agent_player
        
        # Play until game is over
        while not board.is_game_over():
            current_player = board.current_player
            valid_moves = board.get_valid_moves()
            
            # Skip turn if no valid moves
            if not valid_moves:
                continue
            
            # Store current state
            if training and current_player == current_agent_player:
                states_history.append(board.get_state().copy())
                players_history.append(current_player)
            
            # Decide temperature based on move number
            temperature = 1.0 if len(states_history) < self.temperature_threshold else 0.5
            
            if current_player == current_agent_player:
                # Current agent's turn
                move = self.agent.select_move(board, valid_moves, training, temperature)
            else:
                # Opponent's turn
                with torch.no_grad():
                    state = np.array(board.get_state())
                    policy, _ = opponent_model.predict(state, current_player)
                    policy = policy.cpu().numpy()
                    
                    # Create a mask for valid moves
                    valid_moves_mask = np.zeros(self.board_size * self.board_size)
                    for row, col in valid_moves:
                        valid_moves_mask[row * self.board_size + col] = 1
                    
                    # Apply mask to policy
                    masked_policy = policy * valid_moves_mask
                    masked_policy = np.abs(masked_policy)  # Ensure non-negative
                    
                    # Choose best valid move
                    if np.sum(masked_policy) > 0:
                        move_idx = np.argmax(masked_policy)
                        move = (move_idx // self.board_size, move_idx % self.board_size)
                    else:
                        # Fallback to random valid move
                        move = np.random.choice(len(valid_moves))
                        move = valid_moves[move]
            
            # Create one-hot policy vector
            if training and current_player == current_agent_player:
                policy = np.zeros(self.board_size * self.board_size)
                policy[move[0] * self.board_size + move[1]] = 1
                policies_history.append(policy)
            
            # Make move
            board.make_move(move[0], move[1])
        
        # Game is over, determine rewards and winner
        winner = board.get_result()
        rewards = []
        
        # Calculate result from current agent's perspective
        if winner == 0:  # Draw
            result = 0.5
        elif winner == current_agent_player:  # Win
            result = 1.0
        else:  # Loss
            result = 0.0
        
        # Calculate rewards for each state in the game
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
        
        return winner, result, rewards
    
    def evaluate_agent(self, num_games=10):
        """Evaluate the current agent by playing against opponents from the pool."""
        wins = {1: 0, -1: 0, 0: 0}  # Black wins, White wins, Draws
        
        for _ in range(num_games):
            # Sample an opponent from the pool
            opponent_info = self.model_pool.sample_opponent(agent_id=self.current_agent_id, 
                                                           temperature=self.opponent_pool_temperature)
            
            if opponent_info is None:
                # No opponents in pool, use a random agent
                opponent_model = ReversiNet(self.board_size, self.num_channels, self.device).to(self.device)
                opponent_id = 0
            else:
                # Load opponent model
                model_path, opponent_id, _ = opponent_info
                opponent_model = ReversiNet(self.board_size, self.num_channels, self.device).to(self.device)
                opponent_model, _ = self.model_pool.load_model(model_path, opponent_model)
            
            # Play a game against this opponent
            winner, result, _ = self.play_game_against_opponent(opponent_model, training=False)
            wins[winner] += 1
        
        win_rate = (wins[1] + wins[-1]) / num_games
        draw_rate = wins[0] / num_games
        
        return win_rate, draw_rate, wins
    
    def train(self):
        """Train the agent through improved ELO-based self-play with curriculum learning."""
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        print("Starting improved ELO-based self-play training...")
        
        # Track various metrics for analysis
        last_elo_print = 0
        best_elo = 1000
        stagnation_counter = 0
        training_started = False
        temperature_schedule = np.linspace(2.0, 0.5, self.num_episodes)  # Gradually decrease temperature
        
        for episode in tqdm(range(1, self.num_episodes + 1)):
            # Only start training after collecting enough experience
            if not training_started and len(self.agent.replay_buffer) >= self.min_buffer_size:
                print(f"Episode {episode}: Starting training with {len(self.agent.replay_buffer)} experiences")
                training_started = True
            
            # Get temperature for this episode (annealing schedule)
            current_temperature = temperature_schedule[episode-1]
            
            # Sample an opponent from the pool with improved selection
            opponent_info = self.model_pool.sample_opponent(
                agent_id=self.current_agent_id, 
                temperature=self.opponent_pool_temperature
            )
            
            if opponent_info is None or len(self.model_pool.models) < 2:
                # No opponents or only self in pool, play against a random agent for baseline
                opponent_model = ReversiNet(self.board_size, self.num_channels, self.device).to(self.device)
                opponent_id = 0
                opponent_elo = 1000
            else:
                # Load opponent model with proper logging
                model_path, opponent_id, opponent_elo = opponent_info
                opponent_model = ReversiNet(self.board_size, self.num_channels, self.device).to(self.device)
                opponent_model, _ = self.model_pool.load_model(model_path, opponent_model)
            
            # Play a game against this opponent with current temperature
            _, result, rewards = self.play_game_against_opponent(opponent_model, training=True)
            
            # Update both agents' ELO ratings - current agent and opponent
            self.model_pool.update_rating(self.current_agent_id, opponent_id, result)
            
            # Update ELO tracker with more sophisticated tracking
            self.elo_tracker.add_game_result(opponent_id, result, episode)
            current_elo = self.elo_tracker.get_current_elo()
            raw_elo = self.elo_tracker.get_raw_elo()
            
            # Track if agent is improving
            if current_elo > best_elo:
                best_elo = current_elo
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Store detailed metrics
            self.training_stats['elo_ratings'].append(raw_elo)
            self.training_stats['smoothed_elo_ratings'].append(current_elo)
            self.training_stats['opponent_strengths'].append(opponent_elo)
            self.training_stats['win_rates'].append(1.0 if result > 0.5 else 0.0 if result < 0.5 else 0.5)
            
            # Calculate average reward
            avg_reward = np.mean(rewards) if rewards else 0
            self.training_stats['episode_rewards'].append(avg_reward)
            
            # Update model with multiple batches if there's enough data
            if training_started:
                loss = self.agent.update_model()
                if loss is not None:
                    self.training_stats['losses'].append(loss)
                    
                    # If loss is too high, additional model updates to stabilize
                    if loss > 10.0 and episode > 100:
                        for _ in range(2):  # Extra training steps if loss is high
                            extra_loss = self.agent.update_model()
                            if extra_loss is not None and extra_loss < loss:
                                loss = extra_loss  # Use the lowest loss for logging
            else:
                loss = None
            
            # Print ELO progress periodically without waiting for evaluation
            if episode - last_elo_print >= 20:
                print(f"Episode {episode}: ELO {current_elo:.1f}, WR {self.elo_tracker.get_win_rate(20):.2f}, " +
                      f"vs opponent: {opponent_id} (ELO: {opponent_elo:.1f}) - Result: {'Win' if result == 1.0 else 'Draw' if result == 0.5 else 'Loss'}")
                last_elo_print = episode
            
            # Log detailed metrics to W&B
            if self.use_wandb:
                # Basic episode metrics
                metrics = {
                    "episode": episode,
                    "temperature": current_temperature,
                    "avg_reward": avg_reward,
                    "loss": loss if loss is not None else 0,
                    "buffer_size": len(self.agent.replay_buffer),
                    
                    # ELO metrics
                    "elo_rating_raw": raw_elo,
                    "elo_rating_smoothed": current_elo,
                    "win_rate": self.elo_tracker.get_win_rate(20),  # Recent win rate
                    "opponent_elo": opponent_elo,
                    
                    # Pool metrics
                    "models_in_pool": len(self.model_pool.models),
                    "stagnation_counter": stagnation_counter,
                    
                    # Game result
                    "game_result": result
                }
                
                # Add more detailed metrics less frequently to avoid cluttering
                if episode % self.log_interval == 0:
                    # Progress metrics
                    progress = self.elo_tracker.get_progress_metrics()
                    for key, value in progress.items():
                        metrics[key] = value
                
                wandb.log(metrics)
            
            # Full evaluation on multiple opponents
            if episode % self.evaluation_interval == 0:
                win_rate, draw_rate, wins = self.evaluate_agent(num_games=self.evaluation_games)
                print(f"Episode {episode}, ELO: {current_elo:.1f}, Win rate: {win_rate:.2f}, Draw rate: {draw_rate:.2f}")
                print(f"Black wins: {wins[1]}, White wins: {wins[-1]}, Draws: {wins[0]}")
                self.training_stats['evaluation_scores'].append(win_rate)
                
                # Log detailed evaluation metrics to W&B
                if self.use_wandb:
                    eval_metrics = {
                        "eval_win_rate": win_rate,
                        "eval_draw_rate": draw_rate,
                        "eval_black_wins": wins[1],
                        "eval_white_wins": wins[-1],
                        "eval_draws": wins[0],
                        "eval_total_games": sum(wins.values())
                    }
                    wandb.log(eval_metrics)
                
                # Plot statistics
                self._plot_statistics()
                
                # If agent is stagnating for too long, adjust learning rate
                if stagnation_counter > 300:
                    # Reduce learning rate to stabilize training
                    for param_group in self.agent.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    
                    print(f"Reducing learning rate to {self.agent.optimizer.param_groups[0]['lr']:.6f} due to stagnation")
                    stagnation_counter = 0
            
            # Add current model to the pool periodically
            if episode % self.pool_update_interval == 0:
                # Create a copy of the current model
                new_model = copy.deepcopy(self.agent.model)
                
                # Add to the pool
                self.current_agent_id = self.model_pool.add_model(new_model, current_episode=episode)
                print(f"Added model to pool, new agent_id: {self.current_agent_id}")
                
                if self.use_wandb:
                    wandb.log({"pool_size": len(self.model_pool.models)})
            
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
            
            # Log final statistics
            wandb.run.summary.update({
                "final_elo": current_elo,
                "best_elo": best_elo,
                "final_win_rate": self.elo_tracker.get_win_rate(),
                "final_models_in_pool": len(self.model_pool.models),
                "total_games_played": len(self.elo_tracker.game_results)
            })
            
            wandb.finish()
        
        print(f"Training completed. Final ELO: {current_elo:.1f}, Best ELO: {best_elo:.1f}")
    
    def _plot_statistics(self):
        """Plot enhanced training statistics with more detailed analytics."""
        # Create figure with 4 subplots for more comprehensive visualization
        fig, axes = plt.subplots(4, 1, figsize=(12, 20))
        
        # Plot episode rewards with moving average
        ax1 = axes[0]
        rewards = self.training_stats['episode_rewards']
        ax1.plot(rewards, alpha=0.3, color='blue', label='Raw')
        
        # Add moving average (window=50) for smoother trend visualization
        if len(rewards) > 50:
            window_size = 50
            weights = np.ones(window_size) / window_size
            ma_rewards = np.convolve(rewards, weights, mode='valid')
            ma_x = range(window_size-1, len(rewards))
            ax1.plot(ma_x, ma_rewards, linewidth=2, color='darkblue', label='Moving Avg')
        
        ax1.set_title('Average Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Reward')
        ax1.grid(True)
        ax1.legend()
        
        # Plot losses with moving average
        ax2 = axes[1]
        if self.training_stats['losses']:
            losses = self.training_stats['losses']
            ax2.plot(losses, alpha=0.3, color='red', label='Raw')
            
            # Add moving average for losses
            if len(losses) > 20:
                window_size = 20
                weights = np.ones(window_size) / window_size
                ma_losses = np.convolve(losses, weights, mode='valid')
                ma_x = range(window_size-1, len(losses))
                ax2.plot(ma_x, ma_losses, linewidth=2, color='darkred', label='Moving Avg')
            
            ax2.set_title('Training Loss')
            ax2.set_xlabel('Update')
            ax2.set_ylabel('Loss')
            ax2.set_yscale('log')  # Log scale for better visualization
            ax2.grid(True)
            ax2.legend()
        
        # Plot both raw and smoothed ELO ratings
        ax3 = axes[2]
        
        # Plot raw ELO ratings
        if self.training_stats['elo_ratings']:
            raw_elos = self.training_stats['elo_ratings']
            ax3.plot(raw_elos, alpha=0.3, color='green', label='Raw ELO')
        
        # Plot smoothed ELO ratings
        if self.training_stats['smoothed_elo_ratings']:
            smoothed_elos = self.training_stats['smoothed_elo_ratings']
            ax3.plot(smoothed_elos, linewidth=2, color='darkgreen', label='Smoothed ELO')
            
        # Plot opponent strengths
        if self.training_stats['opponent_strengths']:
            # Plot with lower opacity and downsampling for clarity if many points
            opponents = self.training_stats['opponent_strengths']
            if len(opponents) > 500:
                # Downsample for clarity
                step = len(opponents) // 500
                ax3.scatter(range(0, len(opponents), step), opponents[::step], 
                           alpha=0.1, color='purple', label='Opponent ELO')
            else:
                ax3.scatter(range(len(opponents)), opponents, 
                           alpha=0.1, color='purple', label='Opponent ELO')
                
        ax3.set_title('ELO Rating Progression')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('ELO Rating')
        ax3.grid(True)
        ax3.legend()
        
        # Add horizontal line at 1000 (base ELO)
        ax3.axhline(y=1000, color='gray', linestyle='--', alpha=0.5)
        
        # Plot win rates over time (4th subplot)
        ax4 = axes[3]
        if self.training_stats['win_rates']:
            win_rates = self.training_stats['win_rates']
            
            # Calculate moving average win rate
            if len(win_rates) > 100:
                window_size = 100
                weights = np.ones(window_size) / window_size
                ma_win_rates = np.convolve(win_rates, weights, mode='valid')
                ma_x = range(window_size-1, len(win_rates))
                
                # Plot raw win/loss/draw as a scatter plot (1=win, 0.5=draw, 0=loss)
                ax4.scatter(range(len(win_rates)), win_rates, alpha=0.1, color='blue', label='Game Results')
                # Plot moving average as a line
                ax4.plot(ma_x, ma_win_rates, linewidth=2, color='darkblue', label='Win Rate (MA)')
            else:
                ax4.scatter(range(len(win_rates)), win_rates, alpha=0.3, color='blue', label='Game Results')
                
        # Add evaluation points as red X markers
        if self.training_stats['evaluation_scores']:
            eval_x = range(self.evaluation_interval, 
                          self.evaluation_interval * (len(self.training_stats['evaluation_scores']) + 1), 
                          self.evaluation_interval)
            ax4.plot(eval_x, self.training_stats['evaluation_scores'], 'rx-', 
                    markersize=8, linewidth=1, label='Evaluation Win Rate')
            
        ax4.set_title('Win Rate Progression')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Win Rate')
        ax4.set_ylim(-0.05, 1.05)  # Set y-axis limits for win rate
        ax4.grid(True)
        ax4.legend()
        
        # Add horizontal line at 0.5 (50% win rate)
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # Save figure with better layout
        plt.tight_layout()
        rewards_plot_path = 'plots/elo_training_stats.png'
        plt.savefig(rewards_plot_path, dpi=300)  # Higher DPI for better quality
        
        # Log figure to W&B
        if self.use_wandb:
            wandb.log({"training_stats": wandb.Image(rewards_plot_path)})
            
        plt.close()
        
        # Create separate plot for evaluation scores with more details
        if self.training_stats['evaluation_scores']:
            plt.figure(figsize=(12, 8))
            
            # Plot evaluation win rate as a line with markers
            eval_x = range(self.evaluation_interval, 
                          self.evaluation_interval * (len(self.training_stats['evaluation_scores']) + 1), 
                          self.evaluation_interval)
            
            plt.plot(eval_x, self.training_stats['evaluation_scores'], 'b-', 
                    marker='o', markersize=8, linewidth=2, label='Win Rate')
            
            # Add smoothed ELO at evaluation points if available
            if len(self.training_stats['smoothed_elo_ratings']) >= max(eval_x):
                eval_elos = [self.training_stats['smoothed_elo_ratings'][i-1] for i in eval_x]
                
                # Normalize ELO to 0-1 scale for comparison with win rate
                min_elo = min(eval_elos)
                max_elo = max(eval_elos)
                elo_range = max_elo - min_elo
                
                if elo_range > 0:  # Avoid division by zero
                    normalized_elos = [(elo - min_elo) / elo_range * 0.8 + 0.1 for elo in eval_elos]
                    
                    # Create a second y-axis for ELO
                    ax_elo = plt.gca().twinx()
                    ax_elo.plot(eval_x, eval_elos, 'r-', marker='s', markersize=6, 
                              linewidth=1.5, label='ELO Rating')
                    ax_elo.set_ylabel('ELO Rating', color='red')
                    ax_elo.tick_params(axis='y', colors='red')
                    ax_elo.spines['right'].set_color('red')
            
            plt.title('Evaluation Performance')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            plt.ylim(-0.05, 1.05)  # Set y-axis limits for win rate
            plt.grid(True)
            plt.legend(loc='upper left')
            
            # Add horizontal line at 0.5 (50% win rate)
            plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            
            eval_plot_path = 'plots/elo_evaluation_scores.png'
            plt.savefig(eval_plot_path, dpi=300)  # Higher DPI for better quality
            
            # Log figure to W&B
            if self.use_wandb:
                wandb.log({"evaluation_scores": wandb.Image(eval_plot_path)})
                
            plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Reversi reinforcement learning agent with ELO self-play')
    parser.add_argument('--board_size', type=int, default=8, help='Size of the Reversi board')
    parser.add_argument('--num_channels', type=int, default=128, help='Number of channels in neural network')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--pool_size', type=int, default=20, help='Maximum number of models in the pool')
    parser.add_argument('--pool_update', type=int, default=50, help='How often to add a model to the pool')
    parser.add_argument('--temp', type=float, default=2.0, help='Temperature for opponent sampling')
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
    trainer = ELOSelfPlayTrainer(
        board_size=args.board_size,
        num_channels=args.num_channels,
        device=device,
        use_wandb=not args.no_wandb
    )
    
    # Override training parameters if specified
    if args.episodes != 1000:
        trainer.num_episodes = args.episodes
        print(f"Training for {args.episodes} episodes")
    
    if args.pool_size != 20:
        trainer.model_pool.max_pool_size = args.pool_size
        print(f"Model pool size set to {args.pool_size}")
    
    if args.pool_update != 50:
        trainer.pool_update_interval = args.pool_update
        print(f"Pool update interval set to {args.pool_update}")
    
    if args.temp != 2.0:
        trainer.opponent_pool_temperature = args.temp
        print(f"Opponent sampling temperature set to {args.temp}")
    
    # Start training
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    print(f"Training took {end_time - start_time:.2f} seconds")