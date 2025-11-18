import argparse

from ofc_env import OfcEnv
from policy_net_policy import PolicyNetPolicy
from ofc_scoring import score_board, rank_to_string


def format_row(cards):
    # cards is a list of Card or None
    out = []
    for c in cards:
        if c is None:
            out.append("__")
        else:
            # Human-friendly: rank (2-9, T, J, Q, K, A) + suit (h, d, c, s)
            rank_str = rank_to_string(c.rank)
            suit_str = c.suit.name[0].lower()
            out.append(f"{rank_str}{suit_str}")
    return " ".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    env = OfcEnv(soft_mask=True, use_cpp=False)
    policy = PolicyNetPolicy(checkpoint_path=args.checkpoint)

    for ep in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            legal_actions = env.legal_actions(state)
            if not legal_actions:
                break
            action = policy.choose_action(env, state, legal_actions)
            state, _, done = env.step(state, action)

        bottom = state.board[0:5]
        middle = state.board[5:10]
        top = state.board[10:13]

        score, fouled = score_board(
            [c for c in bottom if c is not None],
            [c for c in middle if c is not None],
            [c for c in top if c is not None],
        )

        verdict = "FOUL" if fouled else "PASS"

        print(f"\nEpisode {ep+1}")
        print("Top   :", format_row(top))
        print("Middle:", format_row(middle))
        print("Bottom:", format_row(bottom))
        print(f"Score: {score:.1f}, Result: {verdict}")


if __name__ == "__main__":
    main()