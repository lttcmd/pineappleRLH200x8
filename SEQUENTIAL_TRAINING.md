# Sequential Training - 1000 Shards at a Time

Train on 1000 shards, then continue on the next 1000 using the checkpoint.

## Step 1: Train on First 1000 Shards

```bash
python3 train_supervised_sfl.py \
  --data data/sfl_dataset_10m \
  --output models/supervised_sfl_1000.pth \
  --epochs 10 \
  --batch-size 2048 \
  --lr 1e-3 \
  --num-workers 8 \
  --max-shards 1000 \
  --seed 42
```

## Step 2: Continue on Next 1000 Shards (1000-2000)

```bash
python3 train_supervised_sfl.py \
  --data data/sfl_dataset_10m \
  --output models/supervised_sfl_2000.pth \
  --epochs 5 \
  --batch-size 2048 \
  --lr 1e-3 \
  --num-workers 8 \
  --max-shards 2000 \
  --checkpoint models/supervised_sfl_1000.pth \
  --seed 42
```

## Step 3: Continue on Next 1000 Shards (2000-3000)

```bash
python3 train_supervised_sfl.py \
  --data data/sfl_dataset_10m \
  --output models/supervised_sfl_3000.pth \
  --epochs 5 \
  --batch-size 2048 \
  --lr 1e-3 \
  --num-workers 8 \
  --max-shards 3000 \
  --checkpoint models/supervised_sfl_2000.pth \
  --seed 42
```

## Continue Pattern...

For each additional 1000 shards:
- Use `--max-shards` to include all shards up to that point (e.g., `--max-shards 4000` for shards 0-4000)
- Use `--checkpoint` to load the previous model
- Use fewer epochs (5) since you're fine-tuning
- The model will see all previous shards + new ones

## Full Example: Train on All 2500 Shards

```bash
# Step 1: First 1000
python3 train_supervised_sfl.py \
  --data data/sfl_dataset_10m \
  --output models/supervised_sfl_1000.pth \
  --epochs 10 \
  --batch-size 2048 \
  --lr 1e-3 \
  --num-workers 8 \
  --max-shards 1000 \
  --seed 42

# Step 2: Continue to 2000
python3 train_supervised_sfl.py \
  --data data/sfl_dataset_10m \
  --output models/supervised_sfl_2000.pth \
  --epochs 5 \
  --batch-size 2048 \
  --lr 1e-3 \
  --num-workers 8 \
  --max-shards 2000 \
  --checkpoint models/supervised_sfl_1000.pth \
  --seed 42

# Step 3: Continue to 2500
python3 train_supervised_sfl.py \
  --data data/sfl_dataset_10m \
  --output models/supervised_sfl_2500.pth \
  --epochs 5 \
  --batch-size 2048 \
  --lr 1e-3 \
  --num-workers 8 \
  --max-shards 2500 \
  --checkpoint models/supervised_sfl_2000.pth \
  --seed 42
```

## Notes

- **First run**: Use more epochs (10) to learn from scratch
- **Subsequent runs**: Use fewer epochs (5) since you're fine-tuning
- **Memory**: Each run only loads the specified number of shards into memory
- **Final model**: `models/supervised_sfl_2500.pth` will have seen all 2500 shards

## Evaluate Final Model

```bash
python3 analyze_training.py \
  --checkpoint models/supervised_sfl_2500.pth \
  --episodes 2000
```

