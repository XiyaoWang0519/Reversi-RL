import torch
import numpy as np
import os
from src.board import ReversiBoard
from src.agent import RLAgent, ReversiNet
from src.elo import ELOSystem, ModelPool, ELOTracker

def test_elo_system():
    """Test the ELO rating system."""
    print("Testing ELO rating system...")
    
    # Create ELO system
    elo = ELOSystem(base_elo=1000, k_factor=32)
    
    # Add agents
    elo.add_agent("agent1")
    elo.add_agent("agent2")
    
    # Initial ratings
    print(f"Initial rating agent1: {elo.get_rating('agent1')}")
    print(f"Initial rating agent2: {elo.get_rating('agent2')}")
    
    # Test update for agent1 win
    new_rating = elo.update_rating("agent1", "agent2", 1.0)
    print(f"After agent1 win: agent1 = {new_rating}, agent2 = {elo.get_rating('agent2')}")
    
    # Test update for agent2 win
    new_rating = elo.update_rating("agent2", "agent1", 1.0)
    print(f"After agent2 win: agent1 = {elo.get_rating('agent1')}, agent2 = {new_rating}")
    
    # Test draw
    new_rating1 = elo.update_rating("agent1", "agent2", 0.5)
    new_rating2 = elo.get_rating("agent2")
    print(f"After draw: agent1 = {new_rating1}, agent2 = {new_rating2}")
    
    print("ELO rating system test complete!")
    print()

def test_model_pool():
    """Test the model pool functionality."""
    print("Testing model pool...")
    
    # Create model pool
    pool = ModelPool(save_dir="models/test_pool", max_pool_size=5)
    
    # Create dummy models
    device = torch.device('cpu')
    board_size = 6
    num_channels = 32
    
    # Add models to pool
    for i in range(3):
        model = ReversiNet(board_size, num_channels).to(device)
        agent_id = pool.add_model(model, current_episode=i+1)
        print(f"Added model {i+1}, agent_id: {agent_id}")
    
    # Test sampling
    for temp in [0.0, 1.0, 2.0]:
        opponent = pool.sample_opponent(temperature=temp)
        if opponent:
            print(f"Sampled opponent at temp={temp}: agent_id={opponent[1]}, elo={opponent[2]}")
    
    # Test updating ratings
    pool.update_rating(1, 2, 1.0)  # Agent 1 beats Agent 2
    pool.update_rating(3, 1, 0.5)  # Agent 3 draws with Agent 1
    
    print("Model pool ratings after games:")
    for path, agent_id, elo in pool.models:
        print(f"Agent {agent_id}: ELO = {elo}")
    
    # Test getting best model
    best = pool.get_best_model()
    if best:
        print(f"Best model: agent_id={best[1]}, elo={best[2]}")
    
    print("Model pool test complete!")
    print()

def test_elo_tracker():
    """Test the ELO tracker."""
    print("Testing ELO tracker...")
    
    # Create tracker
    tracker = ELOTracker(window_size=10)
    
    # Add some game results
    print("Initial ELO:", tracker.get_current_elo())
    
    # Win against weak opponent
    tracker.add_game_result("weak_opponent", 1.0, 1)
    print(f"After beating weak opponent: {tracker.get_current_elo()}")
    
    # Loss against strong opponent
    tracker.add_game_result("strong_opponent", 0.0, 2)
    print(f"After losing to strong opponent: {tracker.get_current_elo()}")
    
    # Several games with mixed results
    for i in range(3, 13):
        # Alternate wins and losses
        result = 1.0 if i % 2 == 0 else 0.0
        opponent = f"opponent_{i}"
        tracker.add_game_result(opponent, result, i)
    
    # Check win rate
    print(f"Win rate over last 10 games: {tracker.get_win_rate()}")
    
    # Check history
    episodes, ratings = tracker.get_elo_history()
    print(f"ELO history: {len(episodes)} entries")
    print(f"Current ELO: {tracker.get_current_elo()}")
    
    print("ELO tracker test complete!")

if __name__ == "__main__":
    os.makedirs("models/test_pool", exist_ok=True)
    test_elo_system()
    test_model_pool()
    test_elo_tracker()