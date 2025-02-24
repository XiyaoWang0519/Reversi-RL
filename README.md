# Reversi Reinforcement Learning

A reinforcement learning project for training an agent to play Reversi (Othello) through self-play with advanced ELO-based curriculum learning.

## Project Structure

- `src/board.py`: Implementation of the Reversi game board and rules
- `src/agent.py`: Reinforcement learning agent with neural network policy
- `src/elo.py`: ELO rating system and model pool management
- `src/trainer.py`: Standard self-play training loop
- `src/trainer_elo.py`: Enhanced ELO-based training with curriculum learning
- `src/play.py`: Interactive play against the trained agent

## Features

- **Advanced Agent Learning**: Improved neural network training with KL divergence loss, entropy regularization, and adaptive temperature-based exploration
- **ELO Rating System**: Dynamic K-factor, confidence tracking, and exponential smoothing for more meaningful agent evaluation
- **Curriculum Learning**: Progressive opponent selection based on agent skill level for optimal growth
- **Model Pool Management**: Smart model selection, strategic pruning, and diverse opponent sampling
- **Comprehensive Visualizations**: Detailed statistics tracking, monitoring, and analysis

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

### Standard Self-Play Training

To train the agent through standard self-play:

```bash
python -m src.trainer
```

Command line arguments for training:
```
--board_size INTEGER    Size of the Reversi board (default: 8)
--num_channels INTEGER  Number of channels in neural network (default: 128)
--episodes INTEGER      Number of training episodes (default: 1000)
--device STR            Device to use: 'cpu', 'cuda', or 'mps' (for Apple Silicon)
--no_wandb              Disable Weights & Biases logging
```

Example with custom parameters:
```bash
python -m src.trainer --board_size 6 --num_channels 64 --episodes 500 --device mps
```

### Enhanced ELO-based Self-Play Training

For advanced training using the improved ELO rating system and model pool:

```bash
python -m src.trainer_elo
```

Additional command line arguments for ELO training:
```
--pool_size INTEGER     Maximum number of models in the pool (default: 50)
--pool_update INTEGER   How often to add a model to the pool (default: 100)
--temp FLOAT            Temperature for opponent sampling (default: 1.5)
--device STR            Device to use: 'cpu', 'cuda', or 'mps' (for Apple Silicon)
```

Example with custom ELO parameters for Apple Silicon:
```bash
python -m src.trainer_elo --board_size 8 --episodes 5000 --pool_size 50 --temp 1.5 --device mps
```

### Training Process

The enhanced training process:
- Uses improved exploration strategies with adaptive temperature
- Prioritizes recent experiences for more relevant learning
- Implements multiple training updates per episode for efficiency
- Applies gradient clipping for stability
- Dynamically adapts learning rates to overcome stagnation
- Uses curriculum learning to gradually increase difficulty
- Maintains a diverse pool of opponents with varied skill levels
- Provides detailed metrics and visualizations

Training parameters can be modified in the trainer classes or using command line arguments.

## Playing Against the Agent

To play against a trained agent:

```bash
python -m src.play --model models/reversi_agent_latest.pt --color black
```

Optional arguments:
- `--model`: Path to a trained model file (default: models/reversi_agent_latest.pt)
- `--color`: Play as "black" or "white" (default: black)
- `--size`: Board size (default: 8)

## Model Architecture

The agent uses a convolutional neural network with:
- Input: 3-channel representation of the board (current player's pieces, opponent's pieces, empty spaces)
- Convolutional layers with batch normalization
- Dual-headed output:
  - Policy head: Action probabilities for each board position
  - Value head: Evaluation of the current board state

## Reinforcement Learning Approach

The project implements a policy gradient approach with:
- Self-play to generate training data
- Experience replay buffer for training stability
- Temperature-based exploration vs exploitation balance
- Value function to evaluate board positions

## Key Improvements

The improved implementation addresses several challenges in the original code:

1. **Enhanced Learning Algorithm**:
   - Better network update with multiple loss components
   - Improved exploration with adaptive temperature
   - Multiple training updates per episode
   - Prioritized experience sampling

2. **ELO System Enhancements**:
   - Dynamic K-factor based on experience and win streaks
   - Confidence tracking for better rating adjustments
   - Improved smoothing for more stable evaluation

3. **Opponent Selection**:
   - Curriculum learning based on agent skill level
   - Diverse opponent selection for robust learning
   - Strategic model pool pruning to maintain diversity

4. **Training Process**:
   - Delayed training start for better initial experiences
   - Temperature annealing for balanced exploration
   - Automatic learning rate adjustment
   - Comprehensive logging and visualization

## License

MIT