import json
import os
from typing import List, Tuple

import ofc_cpp


Config = Tuple[str, float, float, float, float, float]


def run_config(name: str,
               foul_penalty: float,
               pass_penalty: float,
               medium_bonus: float,
               strong_bonus: float,
               monster_mult: float,
               episodes: int,
               seed: int) -> dict:
    ofc_cpp.set_sfl_shaping(
        foul_penalty=foul_penalty,
        pass_penalty=pass_penalty,
        medium_bonus=medium_bonus,
        strong_bonus=strong_bonus,
        monster_mult=monster_mult,
    )
    stats = ofc_cpp.simulate_sfl_stats(seed, episodes)
    return {
        "name": name,
        "episodes": int(stats["episodes"]),
        "foul_penalty": foul_penalty,
        "pass_penalty": pass_penalty,
        "medium_bonus": medium_bonus,
        "strong_bonus": strong_bonus,
        "monster_mult": monster_mult,
        "fouls": int(stats["fouls"]),
        "passes": int(stats["passes"]),
        "royalty_boards": int(stats["royalty_boards"]),
        "foul_rate": float(stats["foul_rate"]),
        "pass_rate": float(stats["pass_rate"]),
        "royalty_rate": float(stats["royalty_rate"]),
        "avg_reward": float(stats["avg_reward"]),
        "avg_royalty": float(stats["avg_royalty"]),
    }


def main() -> None:
    # Use fast rollouts so many configs are cheap to evaluate.
    os.environ["OFC_SFL_FAST"] = "1"

    # Each config: (name, foul_penalty, pass_penalty, medium_bonus, strong_bonus, monster_mult)
    configs: List[Config] = [
        # Around previous best: light_pass_penalty & safer_foul.
        ("lp_base",
         -6.0, -0.5, 4.0, 8.0, 1.0),
        ("lp_softer_foul",
         -5.0, -0.5, 4.0, 8.0, 1.0),
        ("lp_harder_foul",
         -7.0, -0.5, 4.0, 8.0, 1.0),
        ("lp_more_medium",
         -6.0, -0.5, 6.0, 8.0, 1.0),
        ("lp_more_strong",
         -6.0, -0.5, 4.0, 10.0, 1.0),
        ("lp_more_both",
         -6.0, -0.5, 6.0, 10.0, 1.0),
        ("lp_less_monster",
         -6.0, -0.5, 4.0, 8.0, 0.5),
        ("sf_base",
         -4.0, -3.0, 4.0, 8.0, 1.0),
        ("sf_lighter_pass",
         -4.0, -1.5, 4.0, 8.0, 1.0),
        ("sf_lighter_pass_more_med",
         -4.0, -1.5, 6.0, 8.0, 1.0),
    ]

    episodes_per_config = 300
    base_seed = 12345

    results = []
    for i, cfg in enumerate(configs):
        name, fp, pp, mb, sb, mm = cfg
        seed = base_seed + i
        res = run_config(name, fp, pp, mb, sb, mm, episodes_per_config, seed)
        results.append(res)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


