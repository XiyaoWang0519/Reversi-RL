import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import copy

class ReversiNet(nn.Module):
    """Neural network for Reversi playing agent."""
    
    def __init__(self, board_size=8, num_channels=128, device='cpu'):
        super(ReversiNet, self).__init__()
        
        # Board will be represented as 3 planes:
        # - Current player's pieces
        # - Opponent's pieces
        # - Empty spaces
        self.input_channels = 3
        self.device = device
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.input_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_channels)
        
        # Policy head (action probabilities)
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head (state evaluation)
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Shared representation
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value
    
    def _prepare_input(self, board_state, current_player):
        """Convert board state to input tensor format."""
        # Create 3-channel input
        batch_size = board_state.shape[0] if len(board_state.shape) > 2 else 1
        board_size = board_state.shape[-1]
        
        # Reshape if needed
        if len(board_state.shape) < 3:
            board_state = board_state.reshape(1, board_size, board_size)
        
        # Initialize 3 planes: current player, opponent, empty
        input_tensor = torch.zeros(batch_size, 3, board_size, board_size, device=self.device)
        
        for i in range(batch_size):
            state = board_state[i]
            # Current player pieces
            input_tensor[i, 0] = torch.tensor((state == current_player), dtype=torch.float, device=self.device)
            # Opponent pieces
            input_tensor[i, 1] = torch.tensor((state == -current_player), dtype=torch.float, device=self.device)
            # Empty spaces
            input_tensor[i, 2] = torch.tensor((state == 0), dtype=torch.float, device=self.device)
        
        return input_tensor
    
    def predict(self, board_state, current_player):
        """Make a prediction based on board state."""
        # Prepare input
        x = self._prepare_input(board_state, current_player)
        
        # Forward pass
        with torch.no_grad():
            policy, value = self.forward(x)
        
        return policy.squeeze(), value.squeeze()


class RLAgent:
    """Reinforcement learning agent for Reversi."""
    
    def __init__(self, board_size=8, num_channels=128, device='cpu'):
        self.board_size = board_size
        self.device = device
        
        # Initialize neural network
        self.model = ReversiNet(board_size, num_channels, device).to(device)
        # Use a smaller learning rate for more stable learning
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=1e-4)
        
        # Improved replay buffer - larger to retain more diverse experiences
        self.replay_buffer = deque(maxlen=50000)
        
        # Training parameters
        self.batch_size = 256  # Larger batch size for better gradient estimates
        self.gamma = 0.99  # Discount factor
        self.value_weight = 1.0  # Weight for value loss
        self.policy_weight = 1.0  # Weight for policy loss
        self.entropy_weight = 0.01  # Weight for entropy regularization
        
    def select_move(self, board, valid_moves, training=True, temperature=1.0):
        """Select a move based on the current policy with improved exploration."""
        if not valid_moves:
            return None
        
        # Get board state and current player
        state = np.array(board.get_state())
        player = board.current_player
        
        # Get policy and value from model
        policy, value = self.model.predict(state, player)
        policy = policy.cpu().numpy()
        
        # Create a mask for valid moves
        valid_moves_mask = np.zeros(self.board_size * self.board_size)
        for row, col in valid_moves:
            valid_moves_mask[row * self.board_size + col] = 1
        
        # Apply mask to policy
        masked_policy = policy * valid_moves_mask
        
        # Add a small positive constant to ensure all valid moves have some probability
        # This helps with exploration
        epsilon = 1e-8
        masked_policy = np.abs(masked_policy) + epsilon * valid_moves_mask
        
        # Handle exploration vs exploitation
        if training:
            # Adaptive temperature based on game progress:
            # - Higher temp at beginning for more exploration
            # - Lower temp later for exploitation
            num_pieces = np.sum(np.abs(board.board))
            total_squares = self.board_size * self.board_size
            game_progress = num_pieces / total_squares
            
            # Adjust temperature based on game progress
            # Early game: more exploration, late game: more exploitation
            adjusted_temp = temperature * (1.0 - 0.5 * game_progress)
            
            # Apply temperature to control exploration
            if adjusted_temp > 0:
                masked_policy = np.power(masked_policy, 1.0 / adjusted_temp)
            
            # Normalize probabilities
            if np.sum(masked_policy) > 0:
                masked_policy /= np.sum(masked_policy)
            else:
                # Fallback to uniform distribution
                masked_policy = valid_moves_mask / np.sum(valid_moves_mask)
            
            # Sample move according to policy
            move_idx = np.random.choice(self.board_size * self.board_size, p=masked_policy)
            move = (move_idx // self.board_size, move_idx % self.board_size)
        else:
            # During evaluation, choose the best move with a small chance of exploration
            if np.random.random() < 0.05:  # 5% chance of exploration during evaluation
                move_idx = np.random.choice(self.board_size * self.board_size, p=masked_policy/np.sum(masked_policy))
            else:
                move_idx = np.argmax(masked_policy)
            
            move = (move_idx // self.board_size, move_idx % self.board_size)
        
        return move
    
    def store_experience(self, state, player, policy, reward):
        """Store an experience in the replay buffer."""
        self.replay_buffer.append((state, player, policy, reward))
    
    def update_model(self):
        """Update model weights using experiences from replay buffer with improved learning."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Perform multiple updates per call for more efficient learning
        num_updates = min(5, len(self.replay_buffer) // self.batch_size)
        total_loss_avg = 0
        
        for _ in range(num_updates):
            # Sample batch with prioritization:
            # - More weight to recent experiences (they're likely more relevant)
            # - More diverse states for better generalization
            
            # Simple prioritization: Recent experiences have higher probability
            probs = np.linspace(0.5, 1.0, len(self.replay_buffer))
            probs = probs / np.sum(probs)
            
            # Sample batch indices
            batch_indices = np.random.choice(
                len(self.replay_buffer), 
                size=self.batch_size, 
                replace=False, 
                p=probs
            )
            
            # Get batch data
            mini_batch = [self.replay_buffer[i] for i in batch_indices]
            
            # Prepare batch data
            states = []
            players = []
            target_policies = []
            target_values = []
            
            for state, player, policy, reward in mini_batch:
                states.append(state)
                players.append(player)
                target_policies.append(policy)
                target_values.append(reward)
            
            states = np.array(states)
            players = np.array(players)
            target_policies = np.array(target_policies)
            target_values = np.array(target_values)
            
            # Convert to tensors
            inputs = [self.model._prepare_input(s, p) for s, p in zip(states, players)]
            x = torch.cat(inputs, dim=0)  # Already on device from _prepare_input
            target_policies = torch.tensor(target_policies, dtype=torch.float32, device=self.device)
            target_values = torch.tensor(target_values, dtype=torch.float32, device=self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            policy_pred, value_pred = self.model(x)
            
            # Loss calculation with improved components
            
            # 1. Policy loss with better handling of distributions
            # KL divergence between predicted policy and target policy
            log_policy_pred = F.log_softmax(policy_pred, dim=1)
            policy_loss = -torch.sum(target_policies * log_policy_pred, dim=1).mean()
            
            # 2. Value loss remains MSE
            value_loss = F.mse_loss(value_pred, target_values)
            
            # 3. Add entropy regularization to encourage exploration
            # Higher entropy = more exploration
            policy_entropy = -torch.sum(F.softmax(policy_pred, dim=1) * log_policy_pred, dim=1).mean()
            
            # Combine losses with weights
            total_loss = (
                self.policy_weight * policy_loss + 
                self.value_weight * value_loss - 
                self.entropy_weight * policy_entropy  # Minus because we want to maximize entropy
            )
            
            # Backward pass with gradient clipping for stability
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss_avg += total_loss.item()
        
        # Return average loss
        return total_loss_avg / num_updates
    
    def save_model(self, path):
        """Save model weights to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load model weights from file."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()  # Set to evaluation mode