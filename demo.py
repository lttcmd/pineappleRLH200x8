"""
Demo script: Play a single hand of OFC using the trained value network.
"""
import torch
from ofc_env import OfcEnv, State
from state_encoding import get_input_dim
from value_net import ValueNet
from action_selection import choose_best_action_beam_search, load_trained_model, get_device
from ofc_scoring import score_board, total_royalties, validate_board


def format_card(card):
    """Format a card for display."""
    if card is None:
        return "  "
    rank_str = str(card.rank.value) if card.rank.value <= 9 else card.rank.name[0]
    suit_str = card.suit.name[0]
    return f"{rank_str}{suit_str}"


def print_board(state):
    """Print the current board state."""
    bottom = [state.board[i] for i in range(5)]
    middle = [state.board[i] for i in range(5, 10)]
    top = [state.board[i] for i in range(10, 13)]
    
    print("\n" + "="*50)
    print("Current Board:")
    print(f"  Top (3):    {' '.join(format_card(c) for c in top)}")
    print(f"  Middle (5): {' '.join(format_card(c) for c in middle)}")
    print(f"  Bottom (5): {' '.join(format_card(c) for c in bottom)}")
    print("="*50)


def play_single_hand(model_path: str = 'value_net.pth', use_beam: bool = True):
    """
    Play exactly one hand of OFC using the trained value network.
    
    Args:
        model_path: Path to trained model checkpoint
        use_beam: If True, use beam search for better decisions
    """
    env = OfcEnv()
    device = get_device()
    
    # Load model
    try:
        model = load_trained_model(model_path, device=device)
        print(f"✓ Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"⚠ Model not found at {model_path}. Using untrained model.")
        input_dim = get_input_dim()
        model = ValueNet(input_dim, hidden_dim=512)
        model = model.to(device)
        model.eval()
    
    # Reset to start a fresh hand
    state = env.reset()
    step_count = 0
    
    print("\n" + "="*50)
    print("PLAYING ONE HAND OF OFC")
    print("="*50)
    print(f"\nRound 0: Initial 5 cards dealt")
    print(f"  Cards: {' '.join(format_card(c) for c in state.current_draw)}")
    
    while True:
        # Get legal actions
        legal_actions = env.legal_actions(state)
        
        if not legal_actions:
            print("\n⚠ No legal actions available. Hand complete.")
            break
        
        # Choose action
        if use_beam and step_count > 0:  # Use beam search for rounds 1-4
            action = choose_best_action_beam_search(
                state, legal_actions, model, env, 
                beam_width=12, device=device
            )
        else:
            # For round 0 or if beam disabled, use simple value net
            from action_selection import choose_best_action_with_value_net
            action = choose_best_action_with_value_net(
                state, legal_actions, model, env, device=device
            )
        
        # Show what we're about to do
        if state.round == 0:
            print(f"\n  → Placing all 5 cards on board...")
        else:
            kept = [state.current_draw[i] for i in action.keep_indices]
            discarded = [c for i, c in enumerate(state.current_draw) if i not in action.keep_indices]
            print(f"\nRound {state.round}:")
            print(f"  Keeping: {' '.join(format_card(c) for c in kept)}")
            print(f"  Discarding: {' '.join(format_card(c) for c in discarded)}")
        
        # Step environment
        state, reward, done = env.step(state, action)
        step_count += 1
        
        # Show board after placement
        print_board(state)
        
        if done:
            break
    
    # Final scoring
    final_score = env.score(state)
    bottom = [state.board[i] for i in range(5)]
    middle = [state.board[i] for i in range(5, 10)]
    top = [state.board[i] for i in range(10, 13)]
    
    is_valid, reason = validate_board(bottom, middle, top)
    royalties = total_royalties(bottom, middle, top) if is_valid else 0
    
    print("\n" + "="*50)
    print("FINAL RESULT")
    print("="*50)
    print(f"  Final Score: {final_score:.1f}")
    if is_valid:
        print(f"  ✓ Board is VALID")
        print(f"  Royalties: {royalties}")
    else:
        print(f"  ✗ Board is FOULED: {reason}")
        print(f"  Penalty: -6 points")
    print("="*50 + "\n")
    
    return final_score


if __name__ == '__main__':
    # Play one hand
    play_single_hand()

