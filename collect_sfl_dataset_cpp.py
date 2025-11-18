"""
Fast SFL dataset generator using the native C++ helper.

This bypasses the Python OfcEnv loop entirely. Each shard call returns
encoded states, action metadata, and the SFL-chosen action index for that state.
"""

import argparse
import os

import torch
from tqdm import tqdm

import ofc_cpp


def collect_shard(seed: int, num_examples: int):
    encoded, labels, action_offsets, action_states = ofc_cpp.generate_sfl_dataset(seed, num_examples)
    encoded = torch.from_numpy(encoded)
    labels = torch.from_numpy(labels)
    action_offsets = torch.tensor(action_offsets, dtype=torch.long)
    action_states = torch.from_numpy(action_states)
    return encoded, labels, action_offsets, action_states


def main():
    parser = argparse.ArgumentParser(description="Collect SFL dataset via C++ helper.")
    parser.add_argument("--output_dir", type=str, default="data/sfl_cpp_dataset")
    parser.add_argument("--shards", type=int, default=20, help="Total number of shards in the full dataset.")
    parser.add_argument("--examples_per_shard", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--shard_start",
        type=int,
        default=0,
        help="Index of first shard to generate (inclusive, 0-based).",
    )
    parser.add_argument(
        "--shard_count",
        type=int,
        default=None,
        help="Number of shards to generate from shard_start. Defaults to all remaining.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    start = max(0, args.shard_start)
    if args.shard_count is None:
        end = args.shards
    else:
        end = min(args.shards, start + max(0, args.shard_count))

    shard_indices = list(range(start, end))
    for shard in tqdm(shard_indices, desc="Shards", unit="shard"):
        print(f"\n=== Collecting shard {shard+1}/{args.shards} ===")
        shard_seed = args.seed + shard
        encoded, labels, action_offsets, action_states = collect_shard(shard_seed, args.examples_per_shard)
        shard_path = os.path.join(args.output_dir, f"shard_{shard:04d}.pt")
        torch.save(
            {
                "encoded": encoded,
                "labels": labels,
                "action_offsets": action_offsets,
                "action_encodings": action_states,
            },
            shard_path,
        )
        print(f"Saved {encoded.shape[0]} examples to {shard_path}")


if __name__ == "__main__":
    main()


