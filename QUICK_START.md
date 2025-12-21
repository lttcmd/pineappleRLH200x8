# Quick Start Guide - Small Dataset Testing

## Step 1: Generate Small Dataset (5-10 minutes)

```bash
# Activate environment
source .venv/bin/activate

# Generate 50 shards (50,000 examples) - quick test
python3 generate_sfl_dataset.py \
  --output data/sfl_dataset_small \
  --examples-per-shard 1000 \
  --num-shards 50 \
  --seed 42 \
  --fast \
  --num-workers 8
```

**Expected output:** `data/sfl_dataset_small/shard_0000.pt` through `shard_0049.pt`

## Step 2: Train Model (10-20 minutes)

```bash
python3 train_supervised_sfl.py \
  --data data/sfl_dataset_small \
  --output models/test_small.pth \
  --epochs 10 \
  --batch-size 512 \
  --lr 1e-4 \
  --num-workers 4 \
  --seed 42
```

**What to watch for:**
- Training accuracy should increase (start ~8-10%, end ~15-20%)
- Validation accuracy should be close to training (within 2-3%)
- Loss should decrease (start ~3.0, end ~2.5-2.7)

## Step 3: Test Model Performance

```bash
# Evaluate the trained model
python3 analyze_training.py \
  --checkpoint models/test_small.pth \
  --episodes 1000
```

**What to expect:**
- **Good model:** Foul rate < 30%, Royalty rate > 5%
- **Bad model:** Foul rate > 50%, Royalty rate < 2%

## Step 4: Compare with SFL Baseline

```bash
# See what SFL heuristic gets
python3 -c "
import ofc_cpp
stats = ofc_cpp.simulate_sfl_stats(1000, fast_mode=True)
print(f'SFL Baseline:')
print(f'  Foul rate: {stats[\"foul_rate\"]*100:.1f}%')
print(f'  Royalty rate: {stats[\"royalty_rate\"]*100:.1f}%')
print(f'  Avg score: {stats[\"avg_score\"]:.2f}')
"
```

**SFL typically gets:**
- Foul rate: ~25-30%
- Royalty rate: ~8-12%
- Avg score: ~15-20

## Quick Troubleshooting

### If training accuracy stays at ~10%:
- Model might be too small → Check `rl_policy_net.py` (should be 512 hidden dim)
- Learning rate too low → Try `--lr 5e-4`

### If validation accuracy much lower than training:
- Overfitting → Model already has dropout, might need more data
- Corrupted labels → Check for warnings during training about "corrupted label"

### If model performs worse than random:
- Check that model loaded correctly
- Verify dataset was generated correctly
- Try training for more epochs

## Full Example (Copy-Paste)

```bash
# 1. Generate dataset
python3 generate_sfl_dataset.py \
  --output data/sfl_dataset_small \
  --examples-per-shard 1000 \
  --num-shards 50 \
  --seed 42 \
  --fast \
  --num-workers 8

# 2. Train
python3 train_supervised_sfl.py \
  --data data/sfl_dataset_small \
  --output models/test_small.pth \
  --epochs 10 \
  --batch-size 512 \
  --lr 1e-4 \
  --num-workers 4 \
  --seed 42

# 3. Test
python3 analyze_training.py \
  --checkpoint models/test_small.pth \
  --episodes 1000
```

## Expected Timeline

- Dataset generation: 5-10 minutes (50 shards)
- Training: 10-20 minutes (10 epochs)
- Evaluation: 2-5 minutes (1000 episodes)
- **Total: ~20-35 minutes**

## Next Steps After Testing

If small test works:
1. Scale up to 500-1000 shards
2. Train for 30 epochs
3. Evaluate on 2000+ episodes

If small test fails:
1. Check `debug_training.py` for issues
2. Verify C++ extension is built correctly
3. Check for label corruption warnings

