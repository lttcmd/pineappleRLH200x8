"""
Debug script to check if model is actually learning.
"""

import torch
import torch.nn.functional as F
from rl_policy_net import RLPolicyNet

# Load a small sample of data
from train_supervised_sfl import load_sfl_dataset, remap_action_data
import numpy as np

print("=" * 60)
print("DEBUGGING MODEL TRAINING")
print("=" * 60)

# Load just 100 shards
print("\n1. Loading dataset...")
states, labels, action_offsets, action_encodings = load_sfl_dataset(
    "data/sfl_dataset_10m", max_shards=100
)
print(f"   Loaded {len(states)} states")

# Take a small subset
indices = torch.arange(min(1000, len(states)), dtype=torch.long)
train_states, train_labels, train_action_offsets, train_action_encodings = remap_action_data(
    indices, states, labels, action_offsets, action_encodings
)
print(f"   Using {len(train_states)} states for testing")

# Check label distribution
print("\n2. Checking label distribution...")
unique_labels, counts = torch.unique(train_labels, return_counts=True)
print(f"   Unique labels: {len(unique_labels)}")
print(f"   Label distribution (first 10):")
for label, count in zip(unique_labels[:10], counts[:10]):
    print(f"     Label {label}: {count} ({count/len(train_labels)*100:.1f}%)")

# Check action counts
print("\n3. Checking action counts per state...")
action_counts = []
for i in range(len(train_states)):
    num_actions = train_action_offsets[i+1].item() - train_action_offsets[i].item()
    action_counts.append(num_actions)
print(f"   Min actions: {min(action_counts)}")
print(f"   Max actions: {max(action_counts)}")
print(f"   Avg actions: {np.mean(action_counts):.1f}")

# Check if labels are valid
print("\n4. Checking label validity...")
invalid_labels = 0
for i in range(len(train_states)):
    num_actions = train_action_offsets[i+1].item() - train_action_offsets[i].item()
    label = train_labels[i].item()
    if label < 0 or label >= num_actions:
        invalid_labels += 1
print(f"   Invalid labels: {invalid_labels} / {len(train_states)}")

# Test model forward pass
print("\n5. Testing model forward pass...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RLPolicyNet().to(device)
model.train()

# Get a batch
batch_size = 32
batch_states = train_states[:batch_size].to(device)
batch_labels = train_labels[:batch_size].to(device)

# Get actions for this batch
batch_action_encodings = []
batch_valid_mask = []
max_actions = 0

for i in range(batch_size):
    start = train_action_offsets[i].item()
    end = train_action_offsets[i+1].item()
    num_actions = end - start
    state_actions = train_action_encodings[start:end].to(device)
    batch_action_encodings.append(state_actions)
    max_actions = max(max_actions, num_actions)

# Pad actions
padded_actions = []
valid_masks = []
for i, actions in enumerate(batch_action_encodings):
    num_actions = actions.shape[0]
    if num_actions < max_actions:
        padding = torch.zeros(max_actions - num_actions, 838, dtype=actions.dtype, device=device)
        actions = torch.cat([actions, padding], dim=0)
    padded_actions.append(actions)
    
    mask = torch.zeros(max_actions, dtype=torch.bool, device=device)
    mask[:num_actions] = True
    valid_masks.append(mask)

batch_action_enc = torch.stack(padded_actions, dim=0)  # (B, max_actions, 838)
batch_valid_mask = torch.stack(valid_masks, dim=0)  # (B, max_actions)

# Forward pass
scores = model(batch_states, batch_action_enc)  # (B, max_actions)
scores = scores.masked_fill(~batch_valid_mask, float('-inf'))

# Check scores
print(f"   Scores shape: {scores.shape}")
print(f"   Scores range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")
print(f"   Scores mean: {scores.mean().item():.3f}")

# Check predictions
preds = scores.argmax(dim=-1)
correct = (preds == batch_labels).sum().item()
print(f"   Accuracy on batch: {correct}/{batch_size} ({100*correct/batch_size:.1f}%)")

# Check loss
loss = F.cross_entropy(scores, batch_labels)
print(f"   Loss: {loss.item():.4f}")

# Check gradients
print("\n6. Testing gradient flow...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss.backward()

total_grad_norm = 0
num_params = 0
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        total_grad_norm += grad_norm
        num_params += 1
        if num_params <= 5:  # Show first 5
            print(f"   {name}: grad_norm={grad_norm:.6f}")
    else:
        print(f"   {name}: NO GRADIENT!")

if num_params > 0:
    print(f"   Average grad norm: {total_grad_norm/num_params:.6f}")
    if total_grad_norm < 1e-6:
        print("   ⚠️  WARNING: Gradients are very small! Model may not be learning.")
    else:
        print("   ✓ Gradients are flowing")
else:
    print("   ❌ ERROR: No gradients!")

# Test a few training steps
print("\n7. Testing training steps...")
initial_loss = loss.item()
for step in range(10):
    optimizer.zero_grad()
    scores = model(batch_states, batch_action_enc)
    scores = scores.masked_fill(~batch_valid_mask, float('-inf'))
    loss = F.cross_entropy(scores, batch_labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    preds = scores.argmax(dim=-1)
    acc = (preds == batch_labels).sum().item() / batch_size
    print(f"   Step {step+1}: loss={loss.item():.4f}, acc={100*acc:.1f}%")

final_loss = loss.item()
if final_loss < initial_loss:
    print(f"   ✓ Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")
else:
    print(f"   ⚠️  Loss did not decrease (initial: {initial_loss:.4f}, final: {final_loss:.4f})")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)

