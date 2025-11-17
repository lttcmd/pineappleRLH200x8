import numpy as np
import pytest

import train
from train import SelfPlayTrainer
from value_net import ValueNet
from state_encoding import get_input_dim


class DummyTrainer(SelfPlayTrainer):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_cuda", False)
        kwargs.setdefault("num_workers", 1)
        super().__init__(*args, **kwargs)
        self.train_calls = []
    
    def _collect_random_batch(self, batch_size: int, seed: int):
        batch_size = max(1, batch_size)
        encoded = np.zeros((batch_size, 838), dtype=np.float32)
        offsets = np.arange(0, batch_size + 1, dtype=np.int32)
        scores = np.zeros(batch_size, dtype=np.float32)
        self.add_encoded_to_buffer(encoded, offsets, scores)
        return batch_size, scores.tolist()
    
    def _collect_model_batch(self, batch_size: int, seed: int):
        return self._collect_random_batch(batch_size, seed)
    
    def _train_cycle(self, num_updates: int):
        if num_updates <= 0:
            return []
        self.train_calls.append(num_updates)
        return [0.0 for _ in range(num_updates)]
    
    def _evaluate(self, *_, **__):
        # Skip slow evaluation during tests
        return None


@pytest.mark.parametrize("episodes,checkpoint", [(1000, 250)])
def test_checkpoint_cadence(monkeypatch, episodes, checkpoint):
    # Force pure Python path
    monkeypatch.setattr(train, "_CPP", None, raising=False)
    monkeypatch.setattr(train, "_USE_CPP", False, raising=False)
    
    model = ValueNet(get_input_dim(), hidden_dim=32)
    trainer = DummyTrainer(model=model, buffer_size=1024, batch_size=16, learning_rate=1e-3)
    trainer.train(num_episodes=episodes, episodes_per_update=10, eval_frequency=checkpoint, resume=False)
    expected = list(range(checkpoint, episodes + 1, checkpoint))
    assert trainer.checkpoint_history == expected
    # Ensure we actually trained between checkpoints
    assert sum(trainer.train_calls) > 0

