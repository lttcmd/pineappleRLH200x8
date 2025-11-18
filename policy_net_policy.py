"""
Policy wrapper that uses the supervised PolicyNet for action selection.
"""

from typing import List, Optional

import torch

from ofc_env import OfcEnv, State, Action
from policy_net import PolicyNet


class PolicyNetPolicy:
    def __init__(self, checkpoint_path: str, device: Optional[str] = None, use_delta: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PolicyNet().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.use_delta = use_delta

    def choose_action(self, env: OfcEnv, state: State, legal_actions: List[Action]) -> Action:
        if not legal_actions:
            raise RuntimeError("No legal actions for PolicyNetPolicy")
        with torch.no_grad():
            logits = self.model.score_actions(
                env=env,
                state=state,
                actions=legal_actions,
                device=self.device,
                use_delta=self.use_delta,
            )
            idx = int(torch.argmax(logits).item()) if logits.numel() > 0 else 0
        return legal_actions[idx]



