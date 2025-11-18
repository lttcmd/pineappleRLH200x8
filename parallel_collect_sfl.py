import argparse
import math
import os
import subprocess
import sys


def launch_worker(
    output_dir: str,
    total_shards: int,
    examples_per_shard: int,
    seed: int,
    shard_start: int,
    shard_count: int,
) -> subprocess.Popen:
    args = [
        sys.executable,
        "collect_sfl_dataset_cpp.py",
        "--output_dir",
        output_dir,
        "--shards",
        str(total_shards),
        "--examples_per_shard",
        str(examples_per_shard),
        "--seed",
        str(seed),
        "--shard_start",
        str(shard_start),
        "--shard_count",
        str(shard_count),
    ]
    env = os.environ.copy()
    # Respect any OFC_SFL_* flags the user has set in the parent shell.
    return subprocess.Popen(args, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel SFL dataset collection across multiple processes.")
    parser.add_argument("--output_dir", type=str, default="data/sfl_cpp_dataset")
    parser.add_argument("--total_shards", type=int, required=True)
    parser.add_argument("--examples_per_shard", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total = args.total_shards
    workers = max(1, args.num_workers)
    base = total // workers
    rem = total % workers

    procs = []
    shard_start = 0
    for i in range(workers):
        shard_count = base + (1 if i < rem else 0)
        if shard_count <= 0:
            continue
        p = launch_worker(
            output_dir=args.output_dir,
            total_shards=args.total_shards,
            examples_per_shard=args.examples_per_shard,
            seed=args.seed,
            shard_start=shard_start,
            shard_count=shard_count,
        )
        procs.append(p)
        shard_start += shard_count

    if not procs:
        print("No shards assigned; nothing to do.")
        return

    # Wait for all workers to finish
    exit_code = 0
    for p in procs:
        code = p.wait()
        if code != 0:
            exit_code = code
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()


