"""
Generate SFL dataset by calling the C++ generate_sfl_dataset function.
Saves data in shards for efficient loading.
Supports parallel generation using multiprocessing.
"""

import argparse
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import torch
from tqdm import tqdm

import ofc_cpp


def generate_single_shard(args):
    """
    Generate a single shard. This function is called by multiprocessing.
    """
    (shard_idx, output_dir, examples_per_shard, seed, fast_mode,
     foul_penalty, pass_penalty, medium_bonus, strong_bonus, monster_mult) = args
    
    # Set SFL shaping parameters
    ofc_cpp.set_sfl_shaping(
        foul_penalty=foul_penalty,
        pass_penalty=pass_penalty,
        medium_bonus=medium_bonus,
        strong_bonus=strong_bonus,
        monster_mult=monster_mult,
    )
    
    if fast_mode:
        os.environ["OFC_SFL_FAST"] = "1"
    else:
        os.environ.pop("OFC_SFL_FAST", None)
    
    # Use different seed for each shard
    shard_seed = seed + shard_idx
    
    # Generate dataset
    encoded, labels, offsets, action_encoded = ofc_cpp.generate_sfl_dataset(
        shard_seed, examples_per_shard
    )
    
    # Convert to tensors if needed
    if isinstance(encoded, np.ndarray):
        encoded = torch.from_numpy(encoded)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    if isinstance(action_encoded, np.ndarray):
        action_encoded = torch.from_numpy(action_encoded)
    
    # Convert offsets list to tensor
    if isinstance(offsets, list):
        offsets = torch.tensor(offsets, dtype=torch.long)
    elif isinstance(offsets, np.ndarray):
        offsets = torch.from_numpy(offsets)
    
    # Save shard
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shard_file = output_path / f"shard_{shard_idx:04d}.pt"
    torch.save(
        {
            "encoded": encoded,
            "labels": labels,
            "action_offsets": offsets,
            "action_encodings": action_encoded,
        },
        shard_file,
    )
    
    return shard_idx


def generate_and_save_shards(
    output_dir: str,
    examples_per_shard: int,
    num_shards: int,
    seed: int,
    fast_mode: bool = False,
    foul_penalty: float = -4.0,
    pass_penalty: float = -3.0,
    medium_bonus: float = 4.0,
    strong_bonus: float = 8.0,
    monster_mult: float = 10.0,
    num_workers: int = None,
):
    """
    Generate SFL dataset and save in shards.
    
    Args:
        output_dir: Directory to save shards
        examples_per_shard: Number of examples per shard
        num_shards: Number of shards to generate
        seed: Random seed (will increment for each shard)
        fast_mode: Use fast SFL rollouts
        foul_penalty: SFL planning-time foul penalty
        pass_penalty: SFL planning-time pass penalty
        medium_bonus: SFL bonus for medium royalties (4-8)
        strong_bonus: SFL bonus for strong royalties (8-12)
        monster_mult: SFL multiplier for monster hands (>=12)
    """
    # Set default workers
    if num_workers is None:
        num_workers = cpu_count()
    
    total_examples = num_shards * examples_per_shard
    print(f"[generate_sfl_dataset] Generating {num_shards} shards with {examples_per_shard} examples each")
    print(f"[generate_sfl_dataset] Total examples: {total_examples:,}")
    print(f"[generate_sfl_dataset] Output directory: {output_dir}")
    print(f"[generate_sfl_dataset] Fast mode: {fast_mode}")
    print(f"[generate_sfl_dataset] Using {num_workers} workers (CPU cores)")
    print(f"[generate_sfl_dataset] SFL shaping parameters:")
    print(f"  foul_penalty={foul_penalty}, pass_penalty={pass_penalty}")
    print(f"  medium_bonus={medium_bonus}, strong_bonus={strong_bonus}, monster_mult={monster_mult}")
    print()
    
    # Prepare arguments for each shard
    shard_args = [
        (shard_idx, output_dir, examples_per_shard, seed, fast_mode,
         foul_penalty, pass_penalty, medium_bonus, strong_bonus, monster_mult)
        for shard_idx in range(num_shards)
    ]
    
    # Generate shards in parallel
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            list(tqdm(
                pool.imap(generate_single_shard, shard_args),
                total=num_shards,
                desc="Generating dataset",
                unit="shard",
                ncols=100,
            ))
    else:
        # Single-threaded fallback
        for args in tqdm(shard_args, desc="Generating dataset", unit="shard", ncols=100):
            generate_single_shard(args)
    
    print()
    print(f"[generate_sfl_dataset] ✓ Generated {num_shards} shards ({total_examples:,} total examples)")
    print(f"[generate_sfl_dataset] ✓ Saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFL dataset and save in shards."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sfl_dataset",
        help="Output directory for shards.",
    )
    parser.add_argument(
        "--examples-per-shard",
        type=int,
        default=1000,
        help="Number of examples per shard (default: 1000).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=20,
        help="Number of shards to generate (default: 20 = 20k examples).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (incremented for each shard).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast SFL rollouts (OFC_SFL_FAST=1).",
    )
    parser.add_argument(
        "--foul-penalty",
        type=float,
        default=-4.0,
        help="SFL planning-time foul penalty (default: -4.0).",
    )
    parser.add_argument(
        "--pass-penalty",
        type=float,
        default=-3.0,
        help="SFL planning-time pass penalty (default: -3.0).",
    )
    parser.add_argument(
        "--medium-bonus",
        type=float,
        default=4.0,
        help="SFL bonus for medium royalties 4-8 (default: 4.0).",
    )
    parser.add_argument(
        "--strong-bonus",
        type=float,
        default=8.0,
        help="SFL bonus for strong royalties 8-12 (default: 8.0).",
    )
    parser.add_argument(
        "--monster-mult",
        type=float,
        default=10.0,
        help="SFL multiplier for monster hands >=12 (default: 10.0).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores).",
    )
    args = parser.parse_args()
    
    generate_and_save_shards(
        output_dir=args.output,
        examples_per_shard=args.examples_per_shard,
        num_shards=args.num_shards,
        seed=args.seed,
        fast_mode=args.fast,
        foul_penalty=args.foul_penalty,
        pass_penalty=args.pass_penalty,
        medium_bonus=args.medium_bonus,
        strong_bonus=args.strong_bonus,
        monster_mult=args.monster_mult,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

