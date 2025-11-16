"""
Demo script showing how to use the trained value network to play OFC.
"""
import torch
from ofc_env import OfcEnv, State
from state_encoding import get_input_dim
from value_net import ValueNet
from action_selection import choose_best_action_with_value_net, load_trained_model, get_device


def play_game_with_net(model_path: str = 'value_net.pth'):
    """Play a game using the trained value network."""
    env = OfcEnv()
    device = get_device()
    
    # Load model
    try:
        model = load_trained_model(model_path, device=device)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Using untrained model.")
        input_dim = get_input_dim()
        model = ValueNet(input_dim, hidden_dim=256)
        model = model.to(device)
        model.eval()
    
    state = env.reset()
    step_count = 0
    
    print("Starting game with value network...")
    print(f"Initial draw: {len(state.current_draw)} cards")
    print(f"Round: {state.round}")
    
    while True:
        # Get legal actions
        legal_actions = env.legal_actions(state)
        
        if not legal_actions:
            print("No legal actions available. Ending game.")
            break
        
        # Choose action using value network
        action = choose_best_action_with_value_net(state, legal_actions, model, env, device=device)
        
        # Step environment
        state, reward, done = env.step(state, action)
        step_count += 1
        
        print(f"\nStep {step_count}:")
        print(f"  Round: {state.round}")
        print(f"  Cards placed: {sum(1 for card in state.board if card is not None)}/13")
        print(f"  Cards remaining in deck: {len(state.deck)}")
        
        if done:
            break
    
    # Compute final score
    final_score = env.score(state)
    print(f"\nGame finished!")
    print(f"Final score: {final_score:.2f}")
    
    # Show board
    print("\nFinal board:")
    bottom = [state.board[i] for i in range(5)]
    middle = [state.board[i] for i in range(5, 10)]
    top = [state.board[i] for i in range(10, 13)]
    
    print(f"Bottom (5): {[f'{c.rank.name[0]}{c.suit.name[0]}' if c else 'None' for c in bottom]}")
    print(f"Middle (5): {[f'{c.rank.name[0]}{c.suit.name[0]}' if c else 'None' for c in middle]}")
    print(f"Top (3): {[f'{c.rank.name[0]}{c.suit.name[0]}' if c else 'None' for c in top]}")
    
    return final_score


if __name__ == '__main__':
    play_game_with_net()

