import json
import os

import ofc_cpp


def main() -> None:
    # 1) Set your shaping knobs here.
    ofc_cpp.set_sfl_shaping(
        foul_penalty=-25.0,  # planning-time reward for a foul (real game still uses -6)
        pass_penalty=-5.0,   # planning-time reward for a 0-point pass
        medium_bonus=15.0,   # added when 4 <= royalties < 8
        strong_bonus=25.0,    # added when 8 <= royalties < 12
        monster_mult=35.0,    # extra = monster_mult * royalties when royalties >= 12
    )

    # 2) Optional: force fast SFL rollouts for speed.
    os.environ["OFC_SFL_FAST"] = "1"

    # 3) Run the stats sim directly (same process) so shaping definitely applies.
    seed = 123
    episodes = 200  # tweak this up/down for more precision vs speed
    stats = ofc_cpp.simulate_sfl_stats(seed, episodes)

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