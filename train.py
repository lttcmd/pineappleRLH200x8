"""
Training script for OFC value network using self-play.
Trains through millions of hands to learn good vs bad choices via RL.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
import os
import glob
import re
import math
import multiprocessing as mp

# Try optional C++ backend
_USE_CPP = os.getenv("OFC_USE_CPP", "1") == "1"
try:
    import ofc_cpp as _CPP
except Exception:
    _CPP = None
    _USE_CPP = False

from ofc_env import State, Action, OfcEnv
from state_encoding import encode_state, encode_state_batch, get_input_dim
from value_net import ValueNet
from action_selection import choose_best_action_beam_search


def _random_episode_worker(task):
    seed, num_episodes = task
    encoded, offsets, scores = _CPP.generate_random_episodes(np.uint64(seed), int(num_episodes))
    # Use asarray so NumPy 2.0 can safely copy when needed (e.g. when offsets is a Python list)
    encoded_np = np.asarray(encoded, dtype=np.float32)
    offsets_np = np.asarray(offsets, dtype=np.int32)
    scores_np = np.asarray(scores, dtype=np.float32)
    return encoded_np, offsets_np, scores_np


def _merge_episode_chunks(chunks):
    if not chunks:
        return (
            np.empty((0, 838), dtype=np.float32),
            np.zeros(1, dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )
    encoded_parts = []
    scores_parts = []
    offsets_list = [0]
    state_offset = 0
    for enc_np, offs_np, scores_np in chunks:
        if enc_np.size == 0 or offs_np.size == 0:
            continue
        encoded_parts.append(np.asarray(enc_np, dtype=np.float32))
        scores_parts.append(np.asarray(scores_np, dtype=np.float32))
        offs_arr = np.asarray(offs_np, dtype=np.int32)
        shifted = offs_arr + state_offset
        offsets_list.extend(shifted[1:].tolist())
        state_offset = shifted[-1]
    if not encoded_parts:
        return (
            np.empty((0, 838), dtype=np.float32),
            np.zeros(1, dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )
    encoded = np.concatenate(encoded_parts, axis=0)
    scores = np.concatenate(scores_parts, axis=0)
    offsets = np.array(offsets_list, dtype=np.int32)
    return encoded, offsets, scores


class ParallelEpisodeGenerator:
    def __init__(self, num_workers: int):
        self.num_workers = max(1, num_workers)
        self.ctx = mp.get_context("spawn")
        self.pool = None
        if self.num_workers > 1:
            self.pool = self.ctx.Pool(self.num_workers)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def generate_random(self, total_episodes: int, base_seed: int):
        if total_episodes <= 0:
            return (
                np.empty((0, 838), dtype=np.float32),
                np.zeros(1, dtype=np.int32),
                np.empty((0,), dtype=np.float32),
            )
        if self.pool is None or self.num_workers == 1:
            return _random_episode_worker((base_seed, total_episodes))
        chunk_size = math.ceil(total_episodes / self.num_workers)
        tasks = []
        generated = 0
        for worker_idx in range(self.num_workers):
            remaining = total_episodes - generated
            if remaining <= 0:
                break
            episodes = min(chunk_size, remaining)
            seed = base_seed + worker_idx * 131071 + generated
            tasks.append((seed, episodes))
            generated += episodes
        chunks = self.pool.map(_random_episode_worker, tasks)
        return _merge_episode_chunks(chunks)


@dataclass
class TrainingStats:
    total_hands: int = 0
    total_score: float = 0.0
    total_royalties: int = 0
    total_fouls: int = 0
    total_zero: int = 0
    royalty_scores: List[float] = field(default_factory=list)

    def observe(self, score: float):
        self.total_hands += 1
        self.total_score += float(score)
        if score > 0:
            self.total_royalties += 1
            self.royalty_scores.append(float(score))
        elif score < 0:
            self.total_fouls += 1
        else:
            self.total_zero += 1

    def observe_many(self, scores: List[float]):
        for score in scores:
            self.observe(score)

    @property
    def avg_score(self) -> float:
        return self.total_score / max(1, self.total_hands)

    @property
    def foul_rate(self) -> float:
        if self.total_hands == 0:
            return 0.0
        return (self.total_fouls / self.total_hands) * 100.0


# NOTE: Python worker functions removed - all episode generation now uses C++ workers
# via generate_random_episodes() for random episodes and engine_policy_generate_once()
# for model-guided episodes. This ensures all game logic runs in C++.


class SelfPlayTrainer:
    """Manages self-play training for the value network."""
    
    def __init__(
        self,
        model: ValueNet,
        buffer_size: int = 10000,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        use_cuda: bool = True,
        num_workers: int = None
    ):
        self.model = model
        
        # Set device (CUDA if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.model = self.model.to(self.device)
        self.model.train()
        
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=buffer_size)
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # C++ engine handles multiple environments internally
        # No need for Python OfcEnv objects in training
        self.num_envs = 64  # Used for C++ engine configuration
        
        # Use pinned memory for faster CPU->GPU transfer
        self.pin_memory = True if torch.cuda.is_available() and use_cuda else False
        
        # C++ workers handle parallelism internally via engine
        # Python multiprocessing pool no longer needed for episode generation
        # (C++ generate_random_episodes and engine handle parallelism)
        if num_workers is None:
            num_workers = max(1, mp.cpu_count())  # Used for engine configuration
        self.num_workers = num_workers
        self.pool = None  # Legacy placeholder
        print(f"C++ Workers: Engine will use up to {num_workers} parallel environments")
        self.random_workers = max(1, min(num_workers, mp.cpu_count()))
        self.parallel_random_gen = None
        if _CPP is not None and self.random_workers > 1:
            try:
                self.parallel_random_gen = ParallelEpisodeGenerator(self.random_workers)
                print(f"Parallel random generator using {self.random_workers} processes")
            except Exception as exc:
                print(f"Failed to start parallel generator ({exc}); falling back to single process")
                self.parallel_random_gen = None

        # Configure default engine parameters
        self.engine_num_envs = max(64, min(256, self.num_workers))  # more envs
        self.engine_max_candidates = 192  # stronger exploration
        self.engine_cycles = 400  # push episodes further per run
        # Hybrid exploration schedule - optimized for strategy learning
        # Transition quickly to model-guided play so it learns strategy faster
        self.random_phase_episodes = 5_000            # 100% random for first 5k hands (exploration)
        self.anneal_phase_episodes = 20_000           # linearly anneal next 20k hands (by 25k, mostly model-guided)
        self.min_random_prob = 0.15                   # keep 15% random thereafter (85% model-guided for strategy learning)
        # Foul-aware selection penalty schedule
        self.selection_penalty_start = 12.0
        self.selection_penalty_final = 8.0
        # EMA normalization for targets
        self.target_mean_ema = 0.0
        self.target_var_ema = 1.0
        self.target_ema_decay = 0.99
        self.target_ema_initialized = False
        # Scheduler
        self.scheduler_cursor = 0
        self.scheduler_period = 16
        self.checkpoint_history: List[int] = []
    
    def _foul_penalty_for_episode(self, absolute_episode: int) -> float:
        """
        Linearly anneal foul penalty from selection_penalty_start down to selection_penalty_final
        across random_phase + anneal_phase, then keep constant.
        """
        total_anneal = self.random_phase_episodes + self.anneal_phase_episodes
        if absolute_episode <= 0:
            return self.selection_penalty_start
        if absolute_episode >= total_anneal:
            return self.selection_penalty_final
        # Linear interpolation
        alpha = min(1.0, max(0.0, absolute_episode / max(1, total_anneal)))
        return float(self.selection_penalty_start + alpha * (self.selection_penalty_final - self.selection_penalty_start))
    
    def generate_episode(self, use_random: bool = True, env_idx: int = 0) -> List[Tuple[State, float]]:
        """
        Generate one episode of self-play.
        For random episodes, uses C++ generate_random_episodes.
        For model-guided episodes, uses C++ engine with GPU.
        
        Args:
            use_random: If True, use random actions. If False, use value network via C++ engine.
            env_idx: Which environment to use (for parallel generation, only used if C++ not available)
        
        Returns:
            List of (state, final_score) pairs (or encoded arrays if from C++)
        """
        # For random episodes with C++, use fast C++ path
        if use_random and _USE_CPP and _CPP is not None:
            seed = int(time.time() * 1000000) + env_idx  # Unique seed
            encoded, offsets, scores_np = _CPP.generate_random_episodes(np.uint64(seed), 1)
            encoded_np = np.asarray(encoded, dtype=np.float32)
            offsets_np = np.asarray(offsets, dtype=np.int32)
            scores_np = np.asarray(scores_np, dtype=np.float32)
            if scores_np.shape[0] > 0:
                score = float(scores_np[0])
                s0 = int(offsets_np[0])
                s1 = int(offsets_np[1])
                # Return encoded states (not State objects) for efficiency
                return [(encoded_np[s], score) for s in range(s0, s1)]
            return []
        
        # For model-guided episodes, use C++ engine
        if not use_random and _USE_CPP and _CPP is not None:
            seed = int(time.time() * 1000000) + env_idx
            enc2, offs, scores = self._engine_policy_generate_once(seed)
            if scores.shape[0] > 0:
                score = float(scores[0])
                s0 = int(offs[0])
                s1 = int(offs[1])
                # Return encoded states (not State objects) for efficiency
                return [(enc2[s], score) for s in range(s0, s1)]
            return []
        
        # Fallback: Python path (only if C++ not available)
        # This should rarely be used in production
        env = OfcEnv()
        state = env.reset()
        episode_states = []
        max_steps = 30
        step_count = 0
        
        while step_count < max_steps:
            legal_actions = env.legal_actions(state)
            if not legal_actions:
                episode_states.append(state)
                break
            
            episode_states.append(state)
            step_count += 1
            
            if use_random:
                action = legal_actions[random.randint(0, len(legal_actions) - 1)]
            else:
                from action_selection import choose_best_action_with_value_net
                action = choose_best_action_with_value_net(
                    state=state,
                    legal_actions=legal_actions,
                    model=self.model,
                    env=env,
                    device=self.device
                )
                if action is None:
                    action = legal_actions[random.randint(0, len(legal_actions) - 1)]
            
            state, reward, done = env.step(state, action)
            if done:
                episode_states.append(state)
                break
        
        final_score = env.score(state)
        return [(s, final_score) for s in episode_states]
    
    def generate_episodes_parallel(self, num_episodes: int, use_random: bool = True, base_seed: int = 0) -> List[Tuple[State, float]]:
        """
        Generate multiple episodes using C++ workers.
        For random episodes, uses C++ generate_random_episodes().
        For model-guided episodes, falls back to single-process path (uses C++ engine via _engine_policy_generate_once).
        
        Args:
            num_episodes: Number of episodes to generate
            use_random: If True, use random actions (uses C++ workers)
            base_seed: Base seed for reproducibility
        
        Returns:
            List of all (state, final_score) pairs from all episodes
        """
        # For random episodes, use C++ generate_random_episodes (fast, parallel)
        if use_random and _USE_CPP and _CPP is not None:
            encoded, offsets, scores_np = _CPP.generate_random_episodes(np.uint64(base_seed), int(num_episodes))
            encoded_np = np.asarray(encoded, dtype=np.float32)
            # Offsets are now returned as Python list to avoid pybind11 array bug on Linux
            # Convert list to numpy array (this is safe and works on both platforms)
            offsets_np = np.asarray(offsets, dtype=np.int32)
            scores_np = np.asarray(scores_np, dtype=np.float32)
            
            # Convert to list of (state, score) tuples for compatibility
            # Note: states are pre-encoded, so we return encoded arrays
            all_data = []
            num_episodes_returned = scores_np.shape[0]
            if num_episodes_returned == 0:
                # No episodes generated, return empty list
                return all_data
            # Ensure offsets array has correct length (should be num_episodes + 1)
            if offsets_np.shape[0] != num_episodes_returned + 1:
                # This shouldn't happen, but handle gracefully
                return all_data
            for e in range(num_episodes_returned):
                s0 = int(offsets_np[e])
                s1 = int(offsets_np[e+1])
                score = float(scores_np[e])
                # Store encoded states (not State objects) for efficiency
                # encoded_np is shape (num_states, 838), so we index by state index
                for s in range(s0, s1):
                    if s < encoded_np.shape[0]:
                        all_data.append((encoded_np[s].copy(), score))
            return all_data
        
        # Fallback: For non-random or when C++ not available, use single-process path
        # This path still uses C++ engine for model-guided episodes via generate_episode()
        all_data = []
        for i in range(num_episodes):
            episode_data = self.generate_episode(use_random=use_random, env_idx=i)
            all_data.extend(episode_data)
        return all_data
    
    def _choose_action_with_net(self, state: State, legal_actions: List[Action], env_idx: int = 0) -> Action:
        """
        Choose action using value network (greedy) - batched for GPU efficiency.
        NOTE: This method is only used in fallback Python path when C++ is not available.
        For production, use C++ engine via _engine_policy_generate_once().
        """
        if not legal_actions:
            return None
        
        # Fallback: Use Python OfcEnv only if C++ not available
        env = OfcEnv()
        
        # Temporarily set model to eval mode
        self.model.eval()
        with torch.no_grad():
            # Batch all state encodings for GPU efficiency
            next_states = []
            valid_actions = []
            
            for action in legal_actions:
                # Simulate action to get next state
                next_state, _, done = env.step(state, action)
                next_states.append(next_state)
                valid_actions.append(action)
            
            # Encode all states at once using batch encoding (faster)
            encoded_batch = encode_state_batch(next_states).to(self.device)
            
            # Forward pass on entire batch (much faster on GPU)
            values, foul_logit, _, _ = self.model(encoded_batch)
            values = values.squeeze()
            foul_prob = torch.sigmoid(foul_logit).squeeze()
            # Foul-aware value selection with a conservative fixed penalty during single-process selection
            penalty = self.selection_penalty_final
            combined = values - penalty * foul_prob
            best_idx = combined.argmax().item()
            best_action = valid_actions[best_idx]
        
        self.model.train()
        return best_action
    
    def add_to_buffer(self, episode_data: List[Tuple[State, float]]):
        """Add episode data to replay buffer."""
        for state, score in episode_data:
            self.replay_buffer.append((state, score))

    def add_encoded_to_buffer(self, encoded_states: np.ndarray, episode_offsets: np.ndarray, final_scores: np.ndarray):
        """
        Add pre-encoded states to replay buffer. Each state's target is the episode's final score.
        encoded_states: float32 [S, 838]
        episode_offsets: int32 [E+1]
        final_scores: float32 [E]
        """
        # Minimal validation
        if encoded_states.ndim != 2 or encoded_states.shape[1] != 838:
            raise ValueError("encoded_states must be [S,838]")
        if episode_offsets.ndim != 1 or final_scores.ndim != 1 or episode_offsets.shape[0] != final_scores.shape[0] + 1:
            raise ValueError("episode_offsets must be [E+1] and final_scores [E] with matching sizes")
        # Append each state's encoded vector with its episode score
        for e in range(final_scores.shape[0]):
            s0 = int(episode_offsets[e])
            s1 = int(episode_offsets[e+1])
            score = float(final_scores[e])
            # Slice view; store per-state rows
            for s in range(s0, s1):
                self.replay_buffer.append((encoded_states[s], score))

    def _engine_policy_generate_once(self, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run one engine-policy session: start envs, perform cycles of request->GPU forward->apply,
        then collect encoded states and scores for completed episodes.
        Returns (encoded [S,838], offsets [E+1], scores [E]).
        """
        assert _CPP is not None, "Engine policy requires C++ module"
        # Create engine
        h = _CPP.create_engine(np.uint64(seed))
        try:
            _CPP.engine_start_envs(h, int(self.engine_num_envs))
            # Loop cycles
            for _ in range(self.engine_cycles):
                enc, meta = _CPP.request_policy_batch(h, int(self.engine_max_candidates))
                # If no candidates, break
                if enc.shape[0] == 0:
                    break
                # Forward on GPU
                with torch.no_grad():
                    x = torch.from_numpy(np.asarray(enc, dtype=np.float32))
                    x = x.to(self.device, non_blocking=True)
                    vals, foul_logit, _, _ = self.model(x)
                    vals = vals.squeeze()
                    foul_prob = torch.sigmoid(foul_logit).squeeze()
                    # Use conservative fixed penalty in engine mode (no episode index here)
                    penalty = self.selection_penalty_final
                    combined = vals - penalty * foul_prob
                    vals_cpu = combined.detach().float().cpu().numpy()
                # Pick best per env
                best_by_env = {}
                for i in range(meta.shape[0]):
                    env_id = int(meta[i, 0]); action_id = int(meta[i, 1])
                    v = vals_cpu[i]
                    if (env_id not in best_by_env) or (v > best_by_env[env_id][0]):
                        best_by_env[env_id] = (v, action_id)
                if not best_by_env:
                    break
                chosen = np.array([[e, a] for e, (_, a) in best_by_env.items()], dtype=np.int32)
                _CPP.apply_policy_actions(h, chosen)
            # Collect outputs
            enc2, offs, scores = _CPP.engine_collect_encoded_episodes(h)
            enc2 = np.asarray(enc2, dtype=np.float32)
            offs = np.asarray(offs, dtype=np.int32)
            scores = np.asarray(scores, dtype=np.float32)
            return enc2, offs, scores
        finally:
            _CPP.destroy_engine(h)
    
    def _random_probability(self, absolute_episode: int) -> float:
        total_anneal = self.random_phase_episodes + self.anneal_phase_episodes
        if absolute_episode < self.random_phase_episodes:
            return 1.0
        if absolute_episode < total_anneal:
            t = (absolute_episode - self.random_phase_episodes) / max(1, self.anneal_phase_episodes)
            return 1.0 - t * (1.0 - self.min_random_prob)
        return self.min_random_prob

    def _select_batch_mode(self, absolute_episode: int) -> bool:
        random_prob = self._random_probability(absolute_episode)
        if random_prob >= 0.999:
            return True
        if random_prob <= 0.001:
            return False
        slots = max(2, self.scheduler_period)
        random_slots = max(1, int(round(slots * random_prob)))
        slot = self.scheduler_cursor % slots
        self.scheduler_cursor += 1
        return slot < random_slots
    
    def _collect_random_batch(self, batch_size: int, seed: int) -> Tuple[int, List[float]]:
        if _USE_CPP and _CPP is not None:
            if self.parallel_random_gen is not None:
                encoded_np, offsets_np, scores_np = self.parallel_random_gen.generate_random(int(batch_size), int(seed))
            else:
                encoded, offsets, scores_np = _CPP.generate_random_episodes(np.uint64(seed), int(batch_size))
                encoded_np = np.asarray(encoded, dtype=np.float32)
                offsets_np = np.asarray(offsets, dtype=np.int32)
                scores_np = np.asarray(scores_np, dtype=np.float32)
            if scores_np.size == 0:
                return 0, []
            self.add_encoded_to_buffer(encoded_np, offsets_np, scores_np)
            return scores_np.shape[0], scores_np.tolist()
        episode_data = self.generate_episodes_parallel(batch_size, use_random=True, base_seed=seed)
        scores = self._ingest_python_episode_data(episode_data)
        return len(scores), scores
    
    def _collect_model_batch(self, batch_size: int, seed: int) -> Tuple[int, List[float]]:
        if _CPP is not None:
            collected = 0
            scores_all: List[float] = []
            for i in range(batch_size):
                enc2, offs, scores = self._engine_policy_generate_once(seed + i)
                if scores.shape[0] == 0:
                    continue
                self.add_encoded_to_buffer(enc2, offs, scores)
                collected += scores.shape[0]
                scores_all.extend(scores.tolist())
            return collected, scores_all
        episode_data = []
        for i in range(batch_size):
            episode = self.generate_episode(use_random=False, env_idx=i)
            episode_data.extend(episode)
        scores = self._ingest_python_episode_data(episode_data)
        return len(scores), scores
    
    def _ingest_python_episode_data(self, samples: List[Tuple[State, float]]) -> List[float]:
        scores: List[float] = []
        for state_or_encoded, score in samples:
            if isinstance(state_or_encoded, np.ndarray):
                self.replay_buffer.append((state_or_encoded, float(score)))
            elif torch.is_tensor(state_or_encoded):
                self.replay_buffer.append((state_or_encoded.detach().clone().float(), float(score)))
            else:
                self.replay_buffer.append((state_or_encoded, float(score)))
            scores.append(float(score))
        return scores
    
    def _train_cycle(self, num_updates: int) -> List[float]:
        if len(self.replay_buffer) < self.batch_size or num_updates <= 0:
            return []
        losses = []
        train_pbar = tqdm(range(num_updates), desc="Training", unit="update", leave=False, mininterval=0.5)
        for _ in train_pbar:
            loss = self.train_step()
            losses.append(loss)
        train_pbar.close()
        avg_loss = np.mean(losses) if losses else 0.0
        print(f"Training complete. Average loss: {avg_loss:.4f}\n")
        return losses
    
    def _run_checkpoint(self, episode: int, stats: TrainingStats, avg_loss: Optional[float]):
        print(f"\n{'='*60}")
        print(f"Checkpoint at episode {episode:,}")
        print(f"{'='*60}\n")
        if avg_loss is not None:
            print(f"Average training loss: {avg_loss:.4f}")
        self.checkpoint_history.append(episode)
        training_foul_rate = stats.foul_rate
        avg_score_per_hand = stats.avg_score
        self._evaluate(
            stats=stats,
            training_foul_rate=training_foul_rate,
            avg_score_per_hand=avg_score_per_hand
        )
        checkpoint_path = f'value_net_checkpoint_ep{episode}.pth'
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}\n")
    
    def train_step(self) -> float:
        """
        Perform one training step on a batch from replay buffer.
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch with simple prioritization:
        # - Fouled hands get higher weight
        # - Larger |score| get higher weight (harder examples)
        buf_list = list(self.replay_buffer)
        scores_all = [float(sc) for _, sc in buf_list]
        weights = []
        for sc in scores_all:
            w = 1.0
            if sc < 0:
                w *= 2.0
            w *= (1.0 + min(1.0, abs(sc) / 10.0))
            weights.append(w)
        # Normalize
        total_w = sum(weights) if weights else 1.0
        probs = [w / total_w for w in weights]
        # Draw indices without replacement proportional to probs
        idxs = np.random.choice(len(buf_list), size=self.batch_size, replace=False, p=np.asarray(probs, dtype=np.float64))
        batch = [buf_list[i] for i in idxs]
        
        # Extract states and scores
        states = [s for s, _ in batch]
        scores = [score for _, score in batch]
        
        # Encode states in batch (supports both State objects and pre-encoded vectors)
        # Detect if items are pre-encoded numpy arrays/tensors of shape [838]
        if hasattr(states[0], "board"):
            # Likely a State
            state_batch = encode_state_batch(states)
        else:
            # Assume iterable of numpy arrays or tensors with shape [838]
            if isinstance(states[0], np.ndarray):
                state_batch = torch.from_numpy(np.stack(states, axis=0).astype(np.float32, copy=False))
            elif torch.is_tensor(states[0]):
                state_batch = torch.stack(states, dim=0).to(dtype=torch.float32)
            else:
                # Fallback: try through encoder (handles lists)
                state_batch = encode_state_batch(states)
        targets = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)
        
        # Move to device with pinned memory for faster transfer
        if self.pin_memory:
            state_batch = state_batch.pin_memory().to(self.device, non_blocking=True)
            targets = targets.pin_memory().to(self.device, non_blocking=True)
        else:
            state_batch = state_batch.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
        
        # Forward pass (two-head model: value and foul)
        self.optimizer.zero_grad()
        value_pred, foul_logit, round0_logits, feas_logit = self.model(state_batch)
        # Normalize regression targets and build foul labels
        with torch.no_grad():
            # Clip extreme targets to stabilize
            clipped = torch.clamp(targets, min=-20.0, max=40.0)
            # Update EMA of mean/var
            batch_mean = clipped.mean()
            batch_var = torch.var(clipped, unbiased=False).clamp_min(1e-6)
            if not self.target_ema_initialized:
                self.target_mean_ema = float(batch_mean)
                self.target_var_ema = float(batch_var)
                self.target_ema_initialized = True
            else:
                self.target_mean_ema = float(self.target_ema_decay * self.target_mean_ema + (1 - self.target_ema_decay) * batch_mean)
                self.target_var_ema = float(self.target_ema_decay * self.target_var_ema + (1 - self.target_ema_decay) * batch_var)
            t_mean = torch.as_tensor(self.target_mean_ema, dtype=clipped.dtype, device=clipped.device)
            t_std = torch.sqrt(torch.as_tensor(self.target_var_ema, dtype=clipped.dtype, device=clipped.device)).clamp_min(1e-3)
            t_norm = (clipped - t_mean) / t_std
            foul_labels = (targets < 0).float()
            neg_mask = foul_labels
            high_mask = (targets > 5.0).float()
            weights = 1.0 + 0.5 * neg_mask + 0.25 * high_mask
        mse_per = (value_pred - t_norm) ** 2
        value_loss = (mse_per * weights).mean()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(foul_logit, foul_labels, reduction='none')
        # Upweight fouled examples more strongly
        foul_loss = (bce * (1.0 + 2.0 * foul_labels)).mean()
        # Auxiliary feasibility proxy from encoded capacity monotonicity (empty slots per row)
        # state_batch layout: 13*52 one-hots for slots [0..12]
        # Empty slot => all zeros in its 52 slice. Count empties per row.
        with torch.no_grad():
            boards = state_batch[:, :13*52]
            slots = boards.view(boards.shape[0], 13, 52)
            # empty if sum == 0
            empty_mask = (slots.abs().sum(dim=2) == 0).float()
            empty_bot = empty_mask[:, 0:5].sum(dim=1, keepdim=True)
            empty_mid = empty_mask[:, 5:10].sum(dim=1, keepdim=True)
            empty_top = empty_mask[:, 10:13].sum(dim=1, keepdim=True)
            feas_target = ((empty_bot >= empty_mid) & (empty_mid >= empty_top)).float()
        feas_loss = torch.nn.functional.binary_cross_entropy_with_logits(feas_logit, feas_target)
        # Optional Round-0 imitation on early curriculum: generate heuristic labels
        round0_loss = torch.tensor(0.0, device=self.device)
        try:
            # Only when we actually have State objects to derive actions
            if hasattr(states[0], "board"):
                round0_indices = [i for i, s in enumerate(states) if getattr(s, "round", 1) == 0]
                if round0_indices:
                    env_tmp = OfcEnv(soft_mask=False)
                    labels = []
                    idx_keep = []
                    for i in round0_indices:
                        s = states[i]
                        actions = env_tmp.legal_actions(s)
                        if not actions:
                            continue
                        # Score each action by simple heuristic
                        best_j = 0
                        best_score = None
                        for j, a in enumerate(actions[:12]):
                            sc = 0.0
                            for _, slot in a.placements:
                                # bottom slots 0..4, middle 5..9, top 10..12
                                if 0 <= slot <= 4:
                                    sc += 2.0
                                elif 5 <= slot <= 9:
                                    sc += 1.5
                                else:
                                    sc += 1.0
                            # light penalty if placing two or more on top
                            top_ct = sum(1 for _, slot in a.placements if slot >= 10)
                            if top_ct >= 2:
                                sc -= 1.0
                            if (best_score is None) or (sc > best_score):
                                best_score = sc
                                best_j = j
                        labels.append(best_j)
                        idx_keep.append(i)
                    if idx_keep:
                        idx_tensor = torch.tensor(idx_keep, dtype=torch.long, device=self.device)
                        target_labels = torch.tensor(labels, dtype=torch.long, device=self.device)
                        logits_sel = round0_logits.index_select(0, idx_tensor)
                        round0_loss = torch.nn.functional.cross_entropy(logits_sel, target_labels)
        except Exception:
            pass
        loss = value_loss + 5.0 * foul_loss + 0.5 * feas_loss + 0.2 * round0_loss
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def find_latest_checkpoint(self) -> Optional[Tuple[str, int]]:
        """Find the latest checkpoint file and return (path, episode_number)."""
        checkpoint_files = glob.glob('value_net_checkpoint_ep*.pth')
        if not checkpoint_files:
            return None
        
        # Extract episode numbers and find the latest
        latest_episode = -1
        latest_path = None
        for path in checkpoint_files:
            match = re.search(r'ep(\d+)\.pth', path)
            if match:
                episode = int(match.group(1))
                if episode > latest_episode:
                    latest_episode = episode
                    latest_path = path
        
        return (latest_path, latest_episode) if latest_path else None
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model from checkpoint. Returns episode number."""
        print(f"Loading checkpoint: {checkpoint_path}")
        # Load weights-only to avoid pickle execution and silence FutureWarning
        try:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        except TypeError:
            # Fallback for older torch versions without weights_only
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        # Extract episode number from filename
        match = re.search(r'ep(\d+)\.pth', checkpoint_path)
        episode = int(match.group(1)) if match else 0
        print(f"Resuming from episode {episode:,}")
        return episode
    
    def train(self, num_episodes: int, episodes_per_update: int = 10, eval_frequency: int = 1000, resume: bool = True):
        """
        Train the model using self-play through millions of hands.
        The bot learns what good and bad choices are via RL.
        """
        start_time = time.time()
        stats = TrainingStats()
        start_episode = 0
        checkpoint_loaded = None
        original_target = num_episodes
        updates_per_cycle = 1024
        self.max_updates_per_batch = max(16, updates_per_cycle // 8)
        losses_history: List[float] = []
        
        if resume:
            checkpoint_info = self.find_latest_checkpoint()
            if checkpoint_info:
                checkpoint_path, checkpoint_episode = checkpoint_info
                self.load_checkpoint(checkpoint_path)
                checkpoint_loaded = checkpoint_episode
                start_episode = checkpoint_episode + 1
                if checkpoint_episode >= num_episodes:
                    num_episodes = checkpoint_episode + original_target
        
        print(f"\n{'='*60}")
        print("Starting RL Training")
        print(f"{'='*60}")
        if checkpoint_loaded is not None:
            print(f"Checkpoint Loaded: {checkpoint_loaded:,}")
            print(f"This Training will do (hands): {num_episodes - start_episode:,}")
        else:
            print(f"This Training will do (hands): {num_episodes:,}")
        print(f"Total by end: {num_episodes:,}")
        print(f"Device: {self.device}")
        print(f"Buffer size: {self.buffer_size:,}")
        print(f"Batch size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        pbar = tqdm(
            total=num_episodes,
            desc="Generating episodes",
            unit="hand",
            initial=start_episode,
            mininterval=0.1,
            smoothing=0.1,
        )
        
        episodes_per_batch = max(self.num_workers, episodes_per_update, 32)
        current_episode = start_episode
        update_budget = 0.0
        total_updates_run = 0
        throughput_ema = None
        next_checkpoint = None
        if eval_frequency > 0:
            next_checkpoint = eval_frequency
            while next_checkpoint <= start_episode:
                next_checkpoint += eval_frequency
        
        while current_episode < num_episodes:
            remaining = num_episodes - current_episode
            batch_size_current = min(episodes_per_batch, remaining)
            use_random = self._select_batch_mode(current_episode)
            seed = int(current_episode * 1000)
            batch_start = time.time()
            
            if use_random:
                generated, batch_scores = self._collect_random_batch(batch_size_current, seed)
            else:
                generated, batch_scores = self._collect_model_batch(batch_size_current, seed)
            
            if generated <= 0:
                continue
            
            stats.observe_many(batch_scores[:generated])
            current_episode += generated
            pbar.update(generated)
            
            batch_elapsed = max(1e-6, time.time() - batch_start)
            hands_per_sec = generated / batch_elapsed
            if throughput_ema is None:
                throughput_ema = hands_per_sec
            else:
                throughput_ema = 0.9 * throughput_ema + 0.1 * hands_per_sec
            pbar.set_postfix({
                "mode": "rand" if use_random else "model",
                "h/s": f"{hands_per_sec:6.1f}",
                "avg": f"{throughput_ema:6.1f}",
                "upd": total_updates_run
            })
            
            if eval_frequency > 0:
                update_budget += (generated / eval_frequency) * updates_per_cycle
            else:
                update_budget += generated / max(1, episodes_per_batch) * self.max_updates_per_batch
            
            updates_ready = int(update_budget)
            if updates_ready > 0 and len(self.replay_buffer) >= self.batch_size:
                updates_to_run = min(updates_ready, self.max_updates_per_batch)
                losses = self._train_cycle(updates_to_run)
                total_updates_run += updates_to_run
                update_budget -= updates_to_run
                if losses:
                    losses_history.extend(losses)
            
            if eval_frequency > 0 and next_checkpoint is not None:
                while next_checkpoint <= num_episodes and current_episode >= next_checkpoint:
                    if len(self.replay_buffer) >= self.batch_size and update_budget > 0:
                        extra_updates = max(int(math.ceil(update_budget)), self.max_updates_per_batch)
                        losses = self._train_cycle(extra_updates)
                        total_updates_run += extra_updates
                        update_budget = 0.0
                        if losses:
                            losses_history.extend(losses)
                    recent_loss = float(np.mean(
                        losses_history[-self.max_updates_per_batch:]
                    )) if losses_history else None
                    self._run_checkpoint(next_checkpoint, stats, recent_loss)
                    next_checkpoint += eval_frequency
        
        pbar.close()
        
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {elapsed/3600:.2f} hours ({elapsed:.0f} seconds)")
        print(f"Episodes (target): {num_episodes:,}")
        if num_episodes > 0:
            print(f"Average time per episode: {elapsed/num_episodes*1000:.2f} ms")
        print(f"{'='*60}\n")
        print("Final Training Statistics:")
        print(f"  Total Hands Recorded: {stats.total_hands:,}")
        print(f"  Hands Fouled: {stats.total_fouls}/{max(1, stats.total_hands)} ({stats.foul_rate:.1f}%)")
        zero_rate = (stats.total_zero / stats.total_hands * 100) if stats.total_hands else 0.0
        royalty_rate = (stats.total_royalties / stats.total_hands * 100) if stats.total_hands else 0.0
        print(f"  Hands Scored 0: {stats.total_zero}/{max(1, stats.total_hands)} ({zero_rate:.1f}%)")
        print(f"  Hands with Royalties: {stats.total_royalties}/{max(1, stats.total_hands)} ({royalty_rate:.2f}%)")
        print(f"  Average Score Per Hand: {stats.avg_score:.2f}\n")
    
    def _evaluate(self, stats: Optional[TrainingStats] = None,
                  training_foul_rate: float = 0.0, avg_score_per_hand: float = 0.0):
        """Evaluate model on test episodes."""
        if stats is None:
            stats = TrainingStats()
        self.model.eval()
        test_scores = []
        test_fouls = 0
        incomplete_boards = 0
        complete_boards = 0
        foul_by_round = [0, 0, 0, 0, 0]  # round index when done (0..4)
        royalties_top_ct = 0
        royalties_mid_ct = 0
        royalties_bot_ct = 0
        
        eval_env = OfcEnv(use_cpp=False)
        with torch.no_grad():
            eval_pbar = tqdm(range(500), desc="Evaluating", unit="hand", leave=False, mininterval=0.5)
            for eval_idx in eval_pbar:
                env = eval_env
                state = env.reset()
                
                # Verify reset worked correctly
                if state.round != 0 or len(state.current_draw) != 5:
                    # Reset failed, skip this episode
                    if eval_idx < 3:
                        print(f"  Eval {eval_idx}: RESET FAILED - round={state.round}, draw_len={len(state.current_draw)}")
                    test_scores.append(0.0)
                    incomplete_boards += 1
                    continue
                
                episode_states = []
                max_steps = 50  # Increased from 30 to allow more steps for completion
                step_count = 0
                done = False
                
                while step_count < max_steps and not done:
                    legal_actions = env.legal_actions(state)
                    if not legal_actions:
                        if eval_idx < 3:
                            filled = sum(1 for s in state.board if s is not None)
                            print(f"  Eval {eval_idx}: NO LEGAL ACTIONS at step {step_count}, filled={filled}/13, round={state.round}")
                        episode_states.append(state)
                        break
                    
                    episode_states.append(state)
                    step_count += 1
                    
                    # Debug first few steps of first episode
                    if eval_idx == 0 and step_count <= 3:
                        filled_before = sum(1 for s in state.board if s is not None)
                        print(f"    Step {step_count}: round={state.round}, filled={filled_before}/13, legal_actions={len(legal_actions)}")
                    
                    # Use value network to choose action
                    from action_selection import choose_best_action_with_value_net
                    action = choose_best_action_with_value_net(
                        state=state,
                        legal_actions=legal_actions,
                        model=self.model,
                        env=env,
                        device=self.device
                    )
                    if action is None:
                        # Fallback to random if model fails
                        action = legal_actions[random.randint(0, len(legal_actions) - 1)]
                    
                    # Debug action chosen
                    if eval_idx == 0 and step_count <= 3:
                        if state.round == 0:
                            print(f"      Action: round0, placements={len(action.placements)}")
                        else:
                            print(f"      Action: round{state.round}, keep={action.keep_indices}, placements={len(action.placements)}")
                    
                    state, reward, done = env.step(state, action)
                    
                    # Debug after step
                    if eval_idx == 0 and step_count <= 3:
                        filled_after = sum(1 for s in state.board if s is not None)
                        print(f"      After step: round={state.round}, filled={filled_after}/13, done={done}")
                    
                    if done:
                        episode_states.append(state)
                        break
                    
                    # Safety check: if board is complete, mark as done
                    if all(slot is not None for slot in state.board):
                        # Board is complete, force done
                        done = True
                        break
                
                # Check if board was complete BEFORE scoring
                # (score() returns 0.0 for incomplete boards, so we need to check first)
                is_complete = all(slot is not None for slot in state.board)
                filled_slots = sum(1 for slot in state.board if slot is not None)
                
                if eval_idx < 3:
                    print(f"  Eval {eval_idx}: After {step_count} steps - filled={filled_slots}/13, round={state.round}, done={done}, complete={is_complete}")
                
                final_score = env.score(state)
                test_scores.append(final_score)
                
                # Track completion status
                if is_complete:
                    complete_boards += 1
                else:
                    incomplete_boards += 1
                
                if final_score < 0:  # Foul penalty
                    test_fouls += 1
                    # Count which round finished
                    ridx = int(state.round) if hasattr(state, 'round') else 4
                    ridx = max(0, min(4, ridx))
                    foul_by_round[ridx] += 1
                else:
                    # Count royalties per row if non-negative score
                    try:
                        from ofc_scoring import royalties_five, royalties_top
                        bottom_cards = [state.board[i] for i in range(5)]
                        middle_cards = [state.board[i] for i in range(5, 10)]
                        top_cards = [state.board[i] for i in range(10, 13)]
                        if all(c is not None for c in top_cards) and royalties_top(top_cards) > 0:
                            royalties_top_ct += 1
                        if all(c is not None for c in middle_cards) and royalties_five(middle_cards, True) > 0:
                            royalties_mid_ct += 1
                        if all(c is not None for c in bottom_cards) and royalties_five(bottom_cards, False) > 0:
                            royalties_bot_ct += 1
                    except Exception:
                        pass
        
        if test_scores:
            avg_score = np.mean(test_scores)
            std_score = np.std(test_scores)
            max_score = np.max(test_scores)
            min_score = np.min(test_scores)
            foul_rate = test_fouls / len(test_scores) * 100
            
            if stats.total_hands > 0:
                print(f"{'='*23}")
                print(f"Training Statistics:")
                print(f"    Total Hands: {stats.total_hands:,}")
                print(f"    Hands Fouled: {stats.total_fouls:,}/{stats.total_hands:,} ({training_foul_rate:.1f}%)")
                zero_rate = (stats.total_zero / stats.total_hands * 100) if stats.total_hands else 0.0
                royalty_rate = (stats.total_royalties / stats.total_hands * 100) if stats.total_hands else 0.0
                print(f"    Hands Scored 0: {stats.total_zero:,}/{stats.total_hands:,} ({zero_rate:.1f}%)")
                print(f"    Hands with Royalties: {stats.total_royalties:,}/{stats.total_hands:,} ({royalty_rate:.2f}%)")
                print(f"    Average Score Per Hand: {avg_score_per_hand:.2f}")
                print()
            
            print(f"{'='*23}")
            print(f"Evaluation Statistics: (500 test hands)")
            print(f"  Avg score: {avg_score:.2f} ± {std_score:.2f}")
            print(f"  Range: [{min_score:.1f}, {max_score:.1f}]")
            print(f"  Foul rate: {foul_rate:.1f}%")
            print(f"  Board completion: {complete_boards}/{len(test_scores)} complete, {incomplete_boards} incomplete")
            print(f"  Fouls by round (0..4): {foul_by_round}")
            print(f"  Royalties by row (count over non-negative hands): top={royalties_top_ct}, middle={royalties_mid_ct}, bottom={royalties_bot_ct}")
            
            # Show score distribution
            positive_scores = sum(1 for s in test_scores if s > 0)
            zero_scores = sum(1 for s in test_scores if s == 0)
            negative_scores = sum(1 for s in test_scores if s < 0)
            print(f"  Score breakdown: {positive_scores} positive, {zero_scores} zero, {negative_scores} negative")
            print()
        else:
            print("Warning: No test scores collected during evaluation!")
        
        self.model.train()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.parallel_random_gen is not None:
            self.parallel_random_gen.close()
            self.parallel_random_gen = None


def main():
    """
    Main training function.
    Trains the bot through millions of hands to learn good vs bad choices.
    """
    # Initialize model with larger network for better GPU utilization
    input_dim = get_input_dim()
    model = ValueNet(input_dim, hidden_dim=512)  # Increased from 256 to 512
    
    # Initialize trainer (will auto-detect CUDA)
    trainer = SelfPlayTrainer(
        model=model,
        #buffer_size=200000,  # Back to original size
        #batch_size=64,  # Back to original size
        buffer_size=200000,
        batch_size=64,
        learning_rate=1e-3,
        use_cuda=True  # Will use CUDA if available
    )
    
    try:
        # Train for 100,000 hands with checkpoints and evaluation every 25k
        num_episodes = 100_000
        
        trainer.train(
            num_episodes=num_episodes,
            episodes_per_update=10,
            eval_frequency=25000,  # Evaluate and checkpoint every 25k hands
            resume=True  # Load latest checkpoint if available
        )
        
        # Save final model
        torch.save(model.state_dict(), 'value_net.pth')
        print("Final model saved to value_net.pth")
    finally:
        # Cleanup multiprocessing resources
        trainer.cleanup()


if __name__ == '__main__':
    # Note: No multiprocessing setup needed - C++ workers handle parallelism
    # CUDA is used only in the main Python process for GPU training
    main()

