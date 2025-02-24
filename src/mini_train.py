import torch
import os
from src.trainer import SelfPlayTrainer

def run_mini_training():
    """Run a minimal training session to test the pipeline."""
    print("Running mini training session...")
    
    # Use CPU for testing
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create trainer with reduced parameters
    trainer = SelfPlayTrainer(board_size=6, num_channels=32, device=device, use_wandb=True)
    
    # Override training parameters for quick test
    trainer.num_episodes = 5
    trainer.evaluation_interval = 2
    trainer.save_interval = 5
    
    # Start mini training
    print("Starting mini training (5 episodes)...")
    trainer.train()
    
    # Verify model file was created
    model_path = "models/reversi_agent_final.pt"
    if os.path.exists(model_path):
        print(f"Model file created: {model_path}")
    else:
        print(f"ERROR: Model file not created: {model_path}")
    
    # Verify plots were created
    plots_path = "plots/training_stats.png"
    if os.path.exists(plots_path):
        print(f"Training plot created: {plots_path}")
    else:
        print(f"ERROR: Training plot not created: {plots_path}")
    
    print("Mini training completed!")

if __name__ == "__main__":
    run_mini_training()