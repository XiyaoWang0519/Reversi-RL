import torch
import os
from src.trainer_elo import ELOSelfPlayTrainer

def run_mini_elo_training():
    """Run a minimal ELO-based training session to test the pipeline."""
    print("Running mini ELO training session...")
    
    # Use CPU for testing
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create trainer with reduced parameters
    trainer = ELOSelfPlayTrainer(board_size=6, num_channels=32, device=device, use_wandb=True)
    
    # Override training parameters for quick test
    trainer.num_episodes = 10
    trainer.evaluation_interval = 5
    trainer.save_interval = 10
    trainer.pool_update_interval = 2  # Add model to pool frequently for testing
    
    # Start mini training
    print("Starting mini ELO training (10 episodes)...")
    trainer.train()
    
    # Verify model file was created
    model_path = "models/reversi_agent_final.pt"
    if os.path.exists(model_path):
        print(f"Model file created: {model_path}")
    else:
        print(f"ERROR: Model file not created: {model_path}")
    
    # Verify plots were created
    plots_path = "plots/elo_training_stats.png"
    if os.path.exists(plots_path):
        print(f"Training plot created: {plots_path}")
    else:
        print(f"ERROR: Training plot not created: {plots_path}")
    
    # Verify model pool
    pool_size = len(trainer.model_pool.models)
    print(f"Model pool size: {pool_size}")
    
    # Print ELO progression
    current_elo = trainer.elo_tracker.get_current_elo()
    print(f"Final ELO rating: {current_elo}")
    
    print("Mini ELO training completed!")

if __name__ == "__main__":
    run_mini_elo_training()