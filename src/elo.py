import math
import numpy as np
import random
import os
import torch
import copy
from collections import deque
from datetime import datetime

# Add OpenMP environment variable
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

class ELOSystem:
    """Implements an improved ELO rating system for Reversi agents."""
    
    def __init__(self, base_elo=1000, k_factor=40):
        """Initialize the ELO rating system.
        
        Args:
            base_elo: Starting ELO rating for new agents
            k_factor: Base K-factor (will be adjusted dynamically)
        """
        self.base_elo = base_elo
        self.base_k_factor = k_factor
        self.ratings = {}  # Maps agent_id to ELO rating
        self.game_counts = {}  # Maps agent_id to number of games played
        self.confidence = {}  # Maps agent_id to confidence factor (0-1)
        self.win_streaks = {}  # Maps agent_id to current win streak
        self.provisional_threshold = 20
    
    def get_rating(self, agent_id):
        """Get the ELO rating for an agent."""
        return self.ratings.get(agent_id, self.base_elo)
    
    def get_game_count(self, agent_id):
        """Get the number of games played by an agent."""
        return self.game_counts.get(agent_id, 0)
    
    def is_provisional(self, agent_id):
        """Check if an agent's rating is provisional (fewer than threshold games)."""
        return self.get_game_count(agent_id) < self.provisional_threshold
    
    def get_confidence(self, agent_id):
        """Get the confidence factor for an agent's rating."""
        return self.confidence.get(agent_id, 0.5)
    
    def calculate_expected_score(self, rating_a, rating_b):
        """Calculate the expected score for player A when playing against player B."""
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
    
    def calculate_dynamic_k_factor(self, agent_id, opponent_id):
        """Calculate a dynamic K-factor based on the agents' characteristics."""
        # Higher K-factor for:
        # 1. Newer models (fewer games played) to help them rise/fall faster
        # 2. Models on win streaks (they may be underrated)
        # 3. Lower confidence ratings
        
        # Base K-factor
        k = self.base_k_factor
        
        # Adjust for game count (provisional status)
        agent_games = self.get_game_count(agent_id)
        if agent_games < 5:
            # Very new agent, allow rapid rating change
            k *= 3.0
        elif agent_games < 15:
            # New but has some games, allow faster rating change
            k *= 2.0
        elif agent_games < self.provisional_threshold:
            # Still provisional, slightly higher K
            k *= 1.5
            
        # Adjust for win streak
        win_streak = self.win_streaks.get(agent_id, 0)
        if win_streak > 3:
            # Agent is consistently winning, may be underrated
            k *= min(1.0 + 0.1 * win_streak, 1.2)
            
        # Adjust for confidence
        conf = self.get_confidence(agent_id)
        if conf < 0.7:
            # Lower confidence means we should adjust rating more aggressively
            k *= min(1.0 + (0.7 - conf), 1.2)
            
        return k
    
    def update_rating(self, agent_id, opponent_id, score):
        """Update the ELO rating for an agent after a game with dynamic adjustments.
        
        Args:
            agent_id: ID of the agent to update
            opponent_id: ID of the opponent
            score: Actual score (1.0 for win, 0.5 for draw, 0.0 for loss)
        
        Returns:
            The new rating for the agent
        """
        # Get current ratings
        rating_a = self.get_rating(agent_id)
        rating_b = self.get_rating(opponent_id)
        
        # Calculate expected score
        expected_score = self.calculate_expected_score(rating_a, rating_b)
        
        # Calculate dynamic K-factor
        k_factor = self.calculate_dynamic_k_factor(agent_id, opponent_id)
        
        # Update rating
        rating_delta = k_factor * (score - expected_score)
        new_rating = rating_a + rating_delta
        
        # Update game count
        self.game_counts[agent_id] = self.get_game_count(agent_id) + 1
        
        # Update win streak
        if score == 1.0:  # Win
            self.win_streaks[agent_id] = self.win_streaks.get(agent_id, 0) + 1
        else:
            self.win_streaks[agent_id] = 0
            
        # Update confidence based on performance vs. expectation
        confidence = self.get_confidence(agent_id)
        # If result matches expectation, increase confidence
        error = abs(score - expected_score)
        if error < 0.3:
            confidence = min(1.0, confidence + 0.02)
        else:
            confidence = max(0.1, confidence - 0.01)
        self.confidence[agent_id] = confidence
        
        # Store updated rating
        self.ratings[agent_id] = new_rating
        
        return new_rating
    
    def add_agent(self, agent_id, rating=None, confidence=0.5):
        """Add a new agent to the ELO system with improved initialization."""
        if rating is None:
            rating = self.base_elo
            
        # Initialize all agent tracking fields
        self.ratings[agent_id] = rating
        self.game_counts[agent_id] = 0
        self.confidence[agent_id] = confidence
        self.win_streaks[agent_id] = 0


class ModelPool:
    """Manages a pool of models at different skill levels for self-play training with improved selection."""
    
    def __init__(self, save_dir="models/pool", max_pool_size=20, base_elo=1000):
        """Initialize the model pool.
        
        Args:
            save_dir: Directory to save model pool
            max_pool_size: Maximum number of models to keep in the pool
            base_elo: Base ELO rating for new models
        """
        self.save_dir = save_dir
        self.max_pool_size = max_pool_size
        self.base_elo = base_elo  # Store base_elo as instance variable
        self.models = []  # List of (model_path, agent_id, elo) tuples
        self.elo_system = ELOSystem(base_elo=base_elo, k_factor=40)  # Higher k-factor for faster adjustments
        self.latest_agent_id = 0
        self.model_metadata = {}  # Additional info about models (win rate, creation episode, etc.)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def add_model(self, model, current_episode=None):
        """Add a new model to the pool with progressive initialization.
        
        Args:
            model: The model to add
            current_episode: Current training episode number (for naming)
            
        Returns:
            agent_id of the added model
        """
        # Generate a unique ID for the new model
        self.latest_agent_id += 1
        agent_id = self.latest_agent_id
        
        # Save the model
        if current_episode is not None:
            filename = f"agent_{agent_id}_episode_{current_episode}.pt"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_{agent_id}_{timestamp}.pt"
            
        model_path = os.path.join(self.save_dir, filename)
        
        # Save the model state with metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'agent_id': agent_id,
            'creation_episode': current_episode
        }, model_path)
        
        # Initialize new model's ELO
        # If this is not the first model, boost its initial rating 
        # based on episode progression to help it compete
        initial_rating = self.base_elo
        if current_episode and self.latest_agent_id > 1 and current_episode > 100:
            # Progressively increase starting rating based on episode
            # This assumes newer models are generally better
            base_boost = min(100, current_episode / 50)  # Cap the boost at 100 points
            initial_rating = self.base_elo + base_boost
        
        # Add model to ELO system with lower initial confidence
        # to allow faster rating adjustments
        self.elo_system.add_agent(agent_id, rating=initial_rating, confidence=0.3)
        
        # Add model to pool with metadata
        self.models.append((model_path, agent_id, self.elo_system.get_rating(agent_id)))
        self.model_metadata[agent_id] = {
            'creation_episode': current_episode,
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'win_rate': 0.0
        }
        
        # If pool is too big, manage which models to keep:
        # - Always keep the latest model
        # - Always keep some models from early training (diversity)
        # - Keep a range of ELO ratings for progressive learning
        # - Remove underperforming models
        if len(self.models) > self.max_pool_size:
            self._prune_model_pool()
        
        return agent_id
    
    def _prune_model_pool(self):
        """Strategically prune the model pool to maintain diversity and quality."""
        # Always keep the latest model (highest agent_id)
        latest_id = max([id for _, id, _ in self.models])
        
        # Sort remaining models by various criteria
        self.models.sort(key=lambda x: (
            # First sorting key: Is this the latest model? (0 if latest, 1 otherwise)
            0 if x[1] == latest_id else 1,
            # Second sorting key: Negative games played (keep models with more experience)
            -self.elo_system.get_game_count(x[1]),
            # Third sorting key: Negative ELO rating (higher rating = higher priority to keep)
            -x[2]
        ))
        
        # Remove the lowest ranking model based on these criteria
        removed_model = self.models.pop(-1)  # Remove last (worst) model
        
        # Optionally delete the file
        # os.remove(removed_model[0])
    
    def sample_opponent(self, agent_id=None, temperature=1.0):
        """Sample an opponent from the model pool with improved selection.
        
        Args:
            agent_id: ID of the agent requesting an opponent (to avoid self-play)
            temperature: Temperature parameter for sampling (higher = more uniform)
                         Controls how much to weight by ELO difference
        
        Returns:
            Tuple of (model_path, opponent_id, opponent_elo)
        """
        if not self.models:
            return None
        
        # If only one model in the pool
        if len(self.models) == 1:
            return self.models[0]
        
        # Filter out the requesting agent
        candidates = [model for model in self.models if model[1] != agent_id]
        
        if not candidates:
            # If no candidates (only the requesting agent is in the pool)
            return random.choice(self.models)
        
        # Curriculum learning strategy:
        # 10% chance of selecting a random opponent for exploration
        if random.random() < 0.1:
            return random.choice(candidates)
        
        # Get requesting agent's ELO
        agent_elo = self.elo_system.get_rating(agent_id)
        
        # Check if agent is still new (provisional rating)
        is_provisional = self.elo_system.is_provisional(agent_id)
        
        if is_provisional:
            # For provisional agents, focus on opponents slightly weaker or equal
            # This helps new models build confidence and learn fundamentals
            suitable_opponents = [
                model for model in candidates 
                if model[2] <= agent_elo + 50  # Maximum 50 points stronger
            ]
            
            if suitable_opponents:
                # From suitable opponents, bias toward those closest to agent's level
                elo_diffs = np.array([abs(model[2] - agent_elo) for model in suitable_opponents])
                probs = 1.0 / (elo_diffs + 1.0)  # Add 1 to avoid division by zero
                probs = probs / np.sum(probs)
                
                idx = np.random.choice(len(suitable_opponents), p=probs)
                return suitable_opponents[idx]
            
        # For established agents, implement a dynamic opponent selection:
        
        # Extract ELOs and calculate differences from agent's ELO
        elos = np.array([model[2] for model in candidates])
        elo_diffs = elos - agent_elo
        
        # Calculate probabilities based on challenge level
        # Higher probability for opponents that are:
        # 1. Slightly stronger than the agent (positive learning gradient)
        # 2. Not too far beyond the agent's level (appropriate challenge)
        
        # Target a band of ELO differences centered slightly above agent's ELO
        target_diff = 30  # Prefer opponents ~30 points stronger
        
        # Convert differences to probabilities using a Gaussian-like function
        # centered at target_diff with width controlled by temperature
        probabilities = np.exp(-0.5 * ((elo_diffs - target_diff) / (temperature * 100))**2)
        
        # Normalize to create valid probability distribution
        sum_probs = np.sum(probabilities)
        if sum_probs > 0:
            probabilities = probabilities / sum_probs
        else:
            # Fallback to uniform distribution
            probabilities = np.ones(len(elos)) / len(elos)
        
        # Sample an opponent
        idx = np.random.choice(len(candidates), p=probabilities)
        return candidates[idx]
    
    def update_rating(self, agent_id, opponent_id, score):
        """Update the ELO rating after a game with improved tracking.
        
        Args:
            agent_id: ID of the agent
            opponent_id: ID of the opponent
            score: Game result (1.0 for win, 0.5 for draw, 0.0 for loss)
        """
        # Update ELO ratings with dynamic K-factor
        new_rating = self.elo_system.update_rating(agent_id, opponent_id, score)
        
        # Update opponent's rating too if needed
        opponent_score = 1.0 - score if score != 0.5 else 0.5
        self.elo_system.update_rating(opponent_id, agent_id, opponent_score)
        
        # Update the rating in the model pool
        for i, (path, id, _) in enumerate(self.models):
            if id == agent_id:
                self.models[i] = (path, id, new_rating)
            elif id == opponent_id:
                # Also update opponent's rating in the model list
                opponent_rating = self.elo_system.get_rating(opponent_id)
                self.models[i] = (path, id, opponent_rating)
        
        # Update metadata for both models
        for id, result in [(agent_id, score), (opponent_id, opponent_score)]:
            if id in self.model_metadata:
                metadata = self.model_metadata[id]
                metadata['games_played'] += 1
                
                if result == 1.0:
                    metadata['wins'] += 1
                elif result == 0.5:
                    metadata['draws'] += 1
                else:
                    metadata['losses'] += 1
                    
                metadata['win_rate'] = metadata['wins'] / metadata['games_played']
    
    def get_best_model(self):
        """Get the highest rated model from the pool."""
        if not self.models:
            return None
        
        return max(self.models, key=lambda x: x[2])
    
    def load_model(self, model_path, model):
        """Load a model from the pool.
        
        Args:
            model_path: Path to the model file
            model: The model to load into
            
        Returns:
            The loaded model and its agent_id
        """
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        agent_id = checkpoint.get('agent_id', 0)
        return model, agent_id


class ELOTracker:
    """Tracks the ELO progression of a main agent during training with improved metrics."""
    
    def __init__(self, window_size=100):
        """Initialize the ELO tracker with enhanced monitoring.
        
        Args:
            window_size: Size of the sliding window for calculating rolling ELO
        """
        self.elo_history = []  # List of (episode, elo) tuples
        self.game_results = deque(maxlen=window_size)  # Recent game results
        self.opponent_strengths = deque(maxlen=window_size)  # Track opponent ELO
        self.elo_system = ELOSystem(k_factor=40)  # Higher k-factor for training agent
        self.main_agent_id = "main"
        self.opponent_history = {}  # Track results against specific opponents
        self.smoothed_elo = 1000  # Exponentially smoothed ELO for stability
        self.smoothing_factor = 0.7  # Reduced from 0.9 for more responsive changes
        
        # Add the main agent to the ELO system
        self.elo_system.add_agent(self.main_agent_id)
    
    def add_game_result(self, opponent_id, result, current_episode):
        """Add a game result and update the ELO history with improved tracking.
        
        Args:
            opponent_id: ID of the opponent
            result: Game result (1.0 for win, 0.5 for draw, 0.0 for loss)
            current_episode: Current training episode
        """
        # Add opponent to ELO system if not present
        if opponent_id not in self.elo_system.ratings:
            self.elo_system.add_agent(opponent_id)
        
        # Get opponent's rating before update
        opponent_rating = self.elo_system.get_rating(opponent_id)
        self.opponent_strengths.append(opponent_rating)
        
        # Update main agent's rating
        new_rating = self.elo_system.update_rating(self.main_agent_id, opponent_id, result)
        
        # Also update opponent's rating for consistency
        opponent_score = 1.0 - result if result != 0.5 else 0.5
        self.elo_system.update_rating(opponent_id, self.main_agent_id, opponent_score)
        
        # Apply exponential smoothing to ratings to reduce noise
        self.smoothed_elo = (self.smoothing_factor * self.smoothed_elo + 
                             (1 - self.smoothing_factor) * new_rating)
        
        # Record the game result
        self.game_results.append(result)
        
        # Track performance against this specific opponent
        if opponent_id not in self.opponent_history:
            self.opponent_history[opponent_id] = {
                'wins': 0, 
                'losses': 0, 
                'draws': 0, 
                'episodes': []
            }
            
        history = self.opponent_history[opponent_id]
        if result == 1.0:
            history['wins'] += 1
        elif result == 0.5:
            history['draws'] += 1
        else:
            history['losses'] += 1
        history['episodes'].append(current_episode)
        
        # Update ELO history with both raw and smoothed values
        self.elo_history.append((current_episode, new_rating, self.smoothed_elo))
    
    def get_current_elo(self):
        """Get the current ELO rating of the main agent (smoothed for stability)."""
        return self.smoothed_elo
    
    def get_raw_elo(self):
        """Get the current raw (unsmoothed) ELO rating."""
        return self.elo_system.get_rating(self.main_agent_id)
    
    def get_win_rate(self, window=None):
        """Get the current win rate based on recent games.
        
        Args:
            window: Optional window size for calculating win rate
                   (default: use all stored results)
        """
        results = self.game_results
        if window and window < len(results):
            results = list(results)[-window:]
            
        if not results:
            return 0.0
        
        return sum(results) / len(results)
    
    def get_avg_opponent_strength(self, window=None):
        """Get the average ELO of recent opponents.
        
        Args:
            window: Optional window size for calculation
                   (default: use all stored values)
        """
        strengths = self.opponent_strengths
        if window and window < len(strengths):
            strengths = list(strengths)[-window:]
            
        if not strengths:
            return 1000.0
        
        return sum(strengths) / len(strengths)
    
    def get_progress_metrics(self, window=20):
        """Get key metrics about agent's progress.
        
        Returns:
            Dictionary of progress metrics
        """
        return {
            'current_elo': self.get_current_elo(),
            'raw_elo': self.get_raw_elo(),
            'recent_win_rate': self.get_win_rate(window),
            'overall_win_rate': self.get_win_rate(),
            'avg_opponent_strength': self.get_avg_opponent_strength(window),
            'num_opponents_faced': len(self.opponent_history),
            'games_played': len(self.game_results)
        }
    
    def get_elo_history(self):
        """Get the ELO history as lists of episodes, ratings, and smoothed ratings."""
        if not self.elo_history:
            return [], [], []
        
        episodes, ratings, smoothed = zip(*self.elo_history)
        return episodes, ratings, smoothed