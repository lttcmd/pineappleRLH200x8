# Training Flow - 100k Hands with Checkpoints

## Overview
Training runs for 100,000 hands with checkpoints at 25k, 50k, 75k, and 100k.

## Order of Operations at Each Checkpoint

### Checkpoint Flow (repeats at 25k, 50k, 75k, 100k):

1. **Generate Episodes** → Play hands and collect game data
   - Example: Generate episodes 0 → 25,000
   - Progress bar shows: `Generating episodes: 25%|██████▌ | 25000/100000`

2. **At Checkpoint (e.g., 25,000)**:
   - **Step 1: Training** → Model learns from collected data
     - Runs 1024 gradient updates
     - Progress bar: `Training: 100%|████████| 1024/1024`
   
   - **Step 2: Evaluation** → Test how well model plays
     - Runs 500 test hands using the trained model
     - Progress bar: `Evaluating: 100%|████████| 500/500`
     - Shows: average score, foul rate, royalty rate
   
   - **Step 3: Save Checkpoint** → Save model weights
     - File: `value_net_checkpoint_ep25000.pth`

3. **Continue to Next Checkpoint**
   - Generate episodes: 25,000 → 50,000
   - Repeat checkpoint flow at 50k

## Complete Timeline

```
Episode 0 → 25,000:    Generate episodes
Episode 25,000:         Training → Evaluation → Save checkpoint
Episode 25,001 → 50,000: Generate episodes  
Episode 50,000:         Training → Evaluation → Save checkpoint
Episode 50,001 → 75,000: Generate episodes
Episode 75,000:         Training → Evaluation → Save checkpoint
Episode 75,001 → 100,000: Generate episodes
Episode 100,000:        Training → Evaluation → Save checkpoint (final)
```

## What Happens During Each Phase

### Generating Episodes
- Model plays hands (using its current strategy)
- Game data (states, scores) collected into replay buffer
- Progress bar updates smoothly

### Training Phase
- Model learns from collected game data
- Updates neural network weights via gradient descent
- Improves strategy based on what it learned

### Evaluation Phase
- Model plays 500 test hands
- Measures performance: average score, foul rate, royalty rate
- Shows how well the model is playing

### Save Checkpoint
- Saves model weights to disk
- Allows resuming training later
- Can be used to play games with trained model

## Key Points

- **Training happens at every checkpoint** (25k, 50k, 75k, 100k)
- **Evaluation happens at every checkpoint** to track progress
- **Checkpoints are saved** so you can resume or use the model
- **Flow is always**: Generate → Train → Evaluate → Save → Continue

