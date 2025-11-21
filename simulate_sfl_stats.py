import argparse
import json
import os

import ofc_cpp


def main():
    parser = argparse.ArgumentParser(
        description="Fast SFL heuristic stats powered by the native C++ module."
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of hands to simulate.")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast rollouts (OFC_SFL_FAST=1) for quicker/lower-quality evaluation.",
    )
    args = parser.parse_args()

    if args.fast:
        os.environ["OFC_SFL_FAST"] = "1"
    else:
        os.environ.pop("OFC_SFL_FAST", None)

    stats = ofc_cpp.simulate_sfl_stats(args.seed, args.episodes)

    print(
        json.dumps(
            {
                "episodes": int(stats["episodes"]),
                "fast_mode": bool(stats["fast_mode"]),
                "fouls": int(stats["fouls"]),
                "passes": int(stats["passes"]),
                "royalty_boards": int(stats["royalty_boards"]),
                "foul_rate": float(stats["foul_rate"]),
                "pass_rate": float(stats["pass_rate"]),
                "royalty_rate": float(stats["royalty_rate"]),
                "avg_reward": float(stats["avg_reward"]),
                "avg_royalty": float(stats["avg_royalty"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()








