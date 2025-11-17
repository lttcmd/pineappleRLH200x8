"""
Training script for OFC value network using self-play.
Trains through millions of hands to learn good vs bad choices via RL.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import time
import os
import glob
import re
import multiprocessing as mp
from multiprocessing import Pool, Process, Queue

# Try optional C++ backend
_USE_CPP = os.getenv("OFC_USE_CPP", "1") == "1"
_USE_ENGINE_POLICY = os.getenv("OFC_USE_ENGINE_POLICY", "0") == "1"
try:
    import ofc_cpp as _CPP
except Exception:
    _CPP = None
    _USE_CPP = False

from ofc_env import State, Action
# Note: OfcEnv import removed - training uses C++ workers exclusively
# OfcEnv is only used in demo.py for interactive play
from state_encoding import encode_state, encode_state_batch, get_input_dim
from value_net import ValueNet
from action_selection import choose_best_action_beam_search


# NOTE: Python worker functions removed - all episode generation now uses C++ workers
# via generate_random_episodes() for random episodes and engine_policy_generate_once()
# for model-guided episodes. This ensures all game logic runs in C++.


class ActionServer:
    """
    Central GPU action server that batches requests from workers and runs one large
    forward pass on the GPU, then replies with best indices.
    """
    def __init__(self, model: ValueNet, device: torch.device, request_queue: Queue, response_queues, max_batch: int = 16384):
        self.model = model
        self.device = device
        self.request_queue = request_queue
        self.response_queues = response_queues
        self.max_batch = max_batch
        self._running = True
        self.model.eval()

    def stop(self):
        self._running = False

    def serve_once(self, timeout: float = 0.02) -> int:
        """
        Collect up to max_batch requests and serve them in one GPU forward.
        Returns number of requests served.
        """
        batch = []
        try:
            item = self.request_queue.get(timeout=timeout)
            batch.append(item)
        except Exception:
            return 0

        while len(batch) < self.max_batch:
            try:
                item = self.request_queue.get_nowait()
                batch.append(item)
            except Exception:
                break

        # Build mega batch
        tensors = []
        splits = []
        req_ids = []
        for (request_id, num_actions, encoded_np) in batch:
            tensors.append(torch.from_numpy(encoded_np))
            splits.append(num_actions)
            req_ids.append(request_id)

        mega = torch.cat(tensors, dim=0).to(self.device, non_blocking=True)
        with torch.no_grad():
            vals, foul_logit, _, _ = self.model(mega)
            values = vals.squeeze()
            foul_prob = torch.sigmoid(foul_logit).squeeze()

        # Respond
        offset = 0
        for request_id, num_actions in zip(req_ids, splits):
            slice_vals = values[offset:offset+num_actions]
            best_idx = 0 if slice_vals.dim() == 0 else int(slice_vals.argmax().item())
            offset += num_actions
            worker_id, _ = request_id
            self.response_queues[worker_id].put((request_id, best_idx))

        return len(batch)


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
        self.pool = None  # No longer used - C++ handles parallelism
        print(f"C++ Workers: Engine will use up to {num_workers} parallel environments")
        # Centralized action serving infra (lazy init)
        self.request_queue: Optional[Queue] = None
        self.response_queues = None

        # Engine-policy settings (Phase 2)
        self.use_engine_policy = _USE_ENGINE_POLICY and (_CPP is not None)
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
            encoded_np = np.array(encoded, copy=False)
            offsets_np = np.array(offsets, dtype=np.int32)
            scores_np = np.array(scores_np, copy=False).astype(np.float32, copy=False)
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
        from ofc_env import OfcEnv
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
            encoded_np = np.array(encoded, copy=False)
            # Offsets are now returned as Python list to avoid pybind11 array bug on Linux
            # Convert list to numpy array (this is safe and works on both platforms)
            offsets_np = np.array(offsets, dtype=np.int32)
            scores_np = np.array(scores_np, copy=False).astype(np.float32, copy=False)
            
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
    
    def generate_episodes_parallel_with_server(self, num_episodes: int, base_seed: int = 0) -> List[List[Tuple[State, float]]]:
        """
        Parallel net-based episodes using C++ engine with GPU action server.
        Uses C++ engine's request_policy_batch → GPU forward → apply_policy_actions flow.
        This is the primary method for model-guided episode generation with C++ workers.
        """
        if _CPP is None:
            # Fallback to single-process if C++ not available
            episodes = []
            for i in range(num_episodes):
                episode_data = self.generate_episode(use_random=False, env_idx=i)
                episodes.append(episode_data)
            return episodes
        
        # Use C++ engine for parallel episode generation
        # The engine handles multiple environments internally
        episodes = []
        for i in range(num_episodes):
            seed = base_seed + i
            enc2, offs, scores = self._engine_policy_generate_once(seed)
            # Convert to list of (state, score) tuples per episode
            if scores.shape[0] > 0:
                episode_data = []
                for e in range(scores.shape[0]):
                    s0 = int(offs[e])
                    s1 = int(offs[e+1])
                    score = float(scores[e])
                    # Store encoded states (not State objects) for efficiency
                    for s in range(s0, s1):
                        episode_data.append((enc2[s], score))
                episodes.append(episode_data)
        
        return episodes
    
    def _choose_action_with_net(self, state: State, legal_actions: List[Action], env_idx: int = 0) -> Action:
        """
        Choose action using value network (greedy) - batched for GPU efficiency.
        NOTE: This method is only used in fallback Python path when C++ is not available.
        For production, use C++ engine via _engine_policy_generate_once().
        """
        if not legal_actions:
            return None
        
        # Fallback: Use Python OfcEnv only if C++ not available
        from ofc_env import OfcEnv
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
                    from ofc_env import OfcEnv
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
        
        Args:
            num_episodes: Total number of episodes (hands) to generate
            episodes_per_update: How many episodes to collect before updating
            eval_frequency: How often to evaluate (in episodes)
        """
        start_time = time.time()
        losses = []
        total_royalties = 0
        total_fouls = 0
        total_zero = 0
        royalty_scores = []
        total_score = 0.0  # Track total score for average calculation
        start_episode = 0
        checkpoint_loaded = None
        original_target = num_episodes
        
        # Always try to resume from checkpoint if available
        if resume:
            checkpoint_info = self.find_latest_checkpoint()
            if checkpoint_info:
                checkpoint_path, checkpoint_episode = checkpoint_info
                # Always load the checkpoint to continue training
                self.load_checkpoint(checkpoint_path)
                checkpoint_loaded = checkpoint_episode
                start_episode = checkpoint_episode + 1
                
                # If checkpoint is at or past target, extend training
                if checkpoint_episode >= num_episodes:
                    # Extend target to continue training
                    num_episodes = checkpoint_episode + original_target
        
        # Print training information
        print(f"\n{'='*60}")
        print(f"Starting RL Training")
        print(f"{'='*60}")
        if checkpoint_loaded is not None:
            print(f"Checkpoint Loaded: {checkpoint_loaded:,}")
            print(f"This Training will do (hands): {num_episodes - start_episode:,}")
            print(f"Total by end: {num_episodes:,}")
        else:
            print(f"This Training will do (hands): {num_episodes:,}")
            print(f"Total by end: {num_episodes:,}")
        print(f"Device: {self.device}")
        print(f"Buffer size: {self.buffer_size:,}")
        print(f"Batch size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        # Progress bar for episodes (starting from resume point)
        pbar = tqdm(
            range(start_episode, num_episodes),
            desc="Generating episodes",
            unit="hand",
            initial=start_episode,
            total=num_episodes,
            mininterval=0.1,  # Update more frequently for smoother progress
            smoothing=0.1,   # Add slight smoothing to reduce jitter
            dynamic_ncols=False
        )
        
        # Generate episodes in batches sized to reduce blocking pauses
        episodes_per_batch = max(self.num_workers, 32)  # Smaller batches for smoother progress
        # Train at every checkpoint (25k, 50k, 75k, 100k)
        episodes_per_train_step = eval_frequency  # Train at same frequency as checkpoints
        updates_per_cycle = 1024  # Reasonable number of updates per checkpoint
        
        episode_idx = 0
        # Very smooth progress bar updates
        pbar_update_stride = 1  # Update every episode for maximum smoothness
        since_last_update = 0
        # Track which checkpoints we've already evaluated to avoid duplicates
        evaluated_checkpoints = set()
        # Track which episodes we've already trained at to avoid duplicates
        trained_episodes = set()
        # Throughput-first: 100% random batches for maximum hands/sec
        while start_episode + episode_idx < num_episodes:
            # Calculate how many episodes to generate in this batch
            remaining = num_episodes - (start_episode + episode_idx)
            batch_size_current = min(episodes_per_batch, remaining)
            
            # Calculate absolute episode number at start of batch
            absolute_episode_start = start_episode + episode_idx
            
            # Hybrid exploration schedule - calculate random prob for this batch
            # Use the episode count at the START of the batch to decide exploration
            if absolute_episode_start < self.random_phase_episodes:
                random_prob = 1.0
            elif absolute_episode_start < (self.random_phase_episodes + self.anneal_phase_episodes):
                t = (absolute_episode_start - self.random_phase_episodes) / max(1, self.anneal_phase_episodes)
                # 1.0 -> min_random_prob linearly
                random_prob = 1.0 - t * (1.0 - self.min_random_prob)
            else:
                random_prob = self.min_random_prob
            # Stochastic choice per batch
            use_random_for_batch = (np.random.rand() < random_prob)
            
            # Generate batch of episodes
            if use_random_for_batch:
                episodes_generated = 0
                if _USE_CPP and _CPP is not None:
                    # Single C++ call generates many random episodes efficiently
                    seed = int(absolute_episode_start * 1000)
                    encoded, offsets, scores_np = _CPP.generate_random_episodes(np.uint64(seed), int(batch_size_current))
                    # Numpy arrays
                    encoded_np = np.array(encoded, copy=False)            # [S,838] float32
                    offsets_np = np.array(offsets, dtype=np.int32)  # [E+1]
                    scores_np = np.array(scores_np, dtype=np.float32)  # [E]
                    # Update statistics and buffer
                    self.add_encoded_to_buffer(encoded_np, offsets_np, scores_np)
                    episodes_generated = scores_np.shape[0]
                    # Track stats
                    final_scores_batch = scores_np.tolist()
                    for final_score in final_scores_batch:
                        total_score += float(final_score)
                        if final_score > 0:
                            total_royalties += 1
                            royalty_scores.append(float(final_score))
                        elif final_score < 0:
                            total_fouls += 1
                        else:
                            total_zero += 1
                else:
                    # Fallback: Python path (only if C++ not available)
                    all_episode_data = self.generate_episodes_parallel(
                        batch_size_current,
                        use_random=True,
                        base_seed=absolute_episode_start * 1000
                    )
                    # Process episode data (may be encoded arrays or State objects)
                    for state_or_encoded, score in all_episode_data:
                        # If it's an encoded array, add directly to buffer
                        if isinstance(state_or_encoded, np.ndarray):
                            self.replay_buffer.append((state_or_encoded, score))
                            episodes_generated += 1
                            total_score += float(score)
                            if score > 0:
                                total_royalties += 1
                                royalty_scores.append(float(score))
                            elif score < 0:
                                total_fouls += 1
                            else:
                                total_zero += 1
                        else:
                            # Legacy State object path (shouldn't happen with C++)
                            self.add_to_buffer([(state_or_encoded, score)])
                            episodes_generated += 1
                            total_score += float(score)
                            if score > 0:
                                total_royalties += 1
                                royalty_scores.append(float(score))
                            elif score < 0:
                                total_fouls += 1
                            else:
                                total_zero += 1
            else:
                # Model-guided episode generation (non-random) using C++ engine
                episodes_generated = 0
                if _USE_CPP and _CPP is not None:
                    # Use C++ engine for parallel model-guided episodes
                    # Generate episodes in batches using engine
                    for i in range(batch_size_current):
                        seed = int(absolute_episode_start * 1000) + i
                        enc2, offs, scores = self._engine_policy_generate_once(seed)
                        if scores.shape[0] > 0:
                            # Add encoded states to buffer
                            self.add_encoded_to_buffer(enc2, offs, scores)
                            episodes_generated += scores.shape[0]
                            # Track stats
                            for final_score in scores.tolist():
                                total_score += float(final_score)
                                if final_score > 0:
                                    total_royalties += 1
                                    royalty_scores.append(float(final_score))
                                elif final_score < 0:
                                    total_fouls += 1
                                else:
                                    total_zero += 1
                else:
                    # Fallback: Python path (only if C++ not available)
                    all_episode_data = []
                    for i in range(batch_size_current):
                        episode_data = self.generate_episode(use_random=False, env_idx=i)
                        if episode_data:
                            all_episode_data.extend(episode_data)
                            final_score = episode_data[-1][1]
                            total_score += final_score
                            if final_score > 0:
                                total_royalties += 1
                                royalty_scores.append(float(final_score))
                            elif final_score < 0:
                                total_fouls += 1
                            else:
                                total_zero += 1
                            episodes_generated += 1
                    # Add all to buffer
                    for state_or_encoded, score in all_episode_data:
                        if isinstance(state_or_encoded, np.ndarray):
                            self.replay_buffer.append((state_or_encoded, score))
                        else:
                            self.add_to_buffer([(state_or_encoded, score)])
            
            
            # Update progress and training using episodes_generated
            eval_ran_this_batch = False
            for _ in range(episodes_generated):
                episode_idx += 1
                absolute_episode = start_episode + episode_idx - 1
                since_last_update += 1
                if since_last_update >= pbar_update_stride:
                    pbar.update(since_last_update)
                    since_last_update = 0
                # Train in bursts after large generation windows (but skip if we're at a checkpoint - handled there)
                if len(self.replay_buffer) >= self.batch_size and episode_idx % episodes_per_train_step == 0:
                    # Only train here if NOT at a checkpoint (checkpoints handle training separately)
                    # Also check if we've already trained at this episode
                    if (absolute_episode not in trained_episodes and 
                        not (absolute_episode >= eval_frequency and absolute_episode % eval_frequency == 0)):
                        trained_episodes.add(absolute_episode)
                        pbar.clear()
                        print(f"\n[Episode {absolute_episode:,}] Training cycle ({updates_per_cycle} gradient updates)")
                        train_pbar = tqdm(range(updates_per_cycle), desc="Training", unit="update", leave=False, mininterval=0.5)
                        for update_idx in train_pbar:
                            loss = self.train_step()
                            losses.append(loss)
                        train_pbar.close()
                        avg_loss = np.mean(losses[-updates_per_cycle:])
                        print(f"Training complete. Average loss: {avg_loss:.4f}\n")
                # Periodic evaluation and checkpointing
                # At checkpoints: Training → Evaluation → Save checkpoint
                if absolute_episode >= eval_frequency and absolute_episode % eval_frequency == 0:
                    if absolute_episode not in evaluated_checkpoints:
                        eval_ran_this_batch = True
                        evaluated_checkpoints.add(absolute_episode)
                        pbar.clear()
                        print(f"\n{'='*60}")
                        print(f"Checkpoint at episode {absolute_episode:,}")
                        print(f"{'='*60}\n")
                        import sys
                        sys.stdout.flush()
                        
                        # Step 1: Training (if needed and not already done)
                        if (len(self.replay_buffer) >= self.batch_size and 
                            absolute_episode % episodes_per_train_step == 0 and
                            absolute_episode not in trained_episodes):
                            trained_episodes.add(absolute_episode)
                            print(f"[Episode {absolute_episode:,}] Training cycle ({updates_per_cycle} gradient updates)")
                            train_pbar = tqdm(range(updates_per_cycle), desc="Training", unit="update", leave=False, mininterval=0.5)
                            for update_idx in train_pbar:
                                loss = self.train_step()
                                losses.append(loss)
                            train_pbar.close()
                            avg_loss = np.mean(losses[-updates_per_cycle:])
                            print(f"Training complete. Average loss: {avg_loss:.4f}\n")
                        
                        # Step 2: Evaluation
                        print(f"--- Evaluation at episode {absolute_episode:,} ---\n")
                        training_foul_rate = (total_fouls / (absolute_episode + 1)) * 100 if absolute_episode > 0 else 0
                        avg_score_per_hand = total_score / (absolute_episode + 1) if absolute_episode > 0 else 0.0
                        self._evaluate(total_episodes=absolute_episode+1, total_fouls=total_fouls, 
                                      total_royalties=total_royalties, total_zero=total_zero,
                                      training_foul_rate=training_foul_rate,
                                      avg_score_per_hand=avg_score_per_hand)
                        sys.stdout.flush()
                        
                        # Step 3: Save checkpoint
                        checkpoint_path = f'value_net_checkpoint_ep{absolute_episode}.pth'
                        torch.save(self.model.state_dict(), checkpoint_path)
                        print(f"\nCheckpoint saved: {checkpoint_path}\n")
                        sys.stdout.flush()
            
            # Check if we passed a checkpoint after batch processing (only if we didn't evaluate in loop)
            if not eval_ran_this_batch and episodes_generated > 0:
                episode_after_batch = start_episode + episode_idx - 1
                # Check if we're at or past a checkpoint
                if episode_after_batch >= eval_frequency:
                    last_checkpoint = (episode_after_batch // eval_frequency) * eval_frequency
                    # Only evaluate if we haven't already evaluated this checkpoint
                    if last_checkpoint not in evaluated_checkpoints and episode_after_batch >= last_checkpoint:
                        evaluated_checkpoints.add(last_checkpoint)
                        pbar.clear()
                        print(f"\n{'='*60}")
                        print(f"Checkpoint at episode {last_checkpoint:,}")
                        print(f"{'='*60}\n")
                        import sys
                        sys.stdout.flush()
                        
                        # Step 1: Training (if needed and not already done)
                        if (len(self.replay_buffer) >= self.batch_size and 
                            last_checkpoint % episodes_per_train_step == 0 and
                            last_checkpoint not in trained_episodes):
                            trained_episodes.add(last_checkpoint)
                            print(f"[Episode {last_checkpoint:,}] Training cycle ({updates_per_cycle} gradient updates)")
                            train_pbar = tqdm(range(updates_per_cycle), desc="Training", unit="update", leave=False, mininterval=0.5)
                            for update_idx in train_pbar:
                                loss = self.train_step()
                                losses.append(loss)
                            train_pbar.close()
                            avg_loss = np.mean(losses[-updates_per_cycle:])
                            print(f"Training complete. Average loss: {avg_loss:.4f}\n")
                        
                        # Step 2: Evaluation
                        print(f"--- Evaluation at episode {last_checkpoint:,} ---\n")
                        training_foul_rate = (total_fouls / (episode_after_batch + 1)) * 100 if episode_after_batch > 0 else 0
                        avg_score_per_hand = total_score / (episode_after_batch + 1) if episode_after_batch > 0 else 0.0
                        self._evaluate(total_episodes=episode_after_batch+1, total_fouls=total_fouls, 
                                      total_royalties=total_royalties, total_zero=total_zero,
                                      training_foul_rate=training_foul_rate,
                                      avg_score_per_hand=avg_score_per_hand)
                        sys.stdout.flush()
                        
                        # Step 3: Save checkpoint
                        checkpoint_path = f'value_net_checkpoint_ep{last_checkpoint}.pth'
                        torch.save(self.model.state_dict(), checkpoint_path)
                        print(f"\nCheckpoint saved: {checkpoint_path}\n")
                        sys.stdout.flush()
            
            if self.use_engine_policy:
                # Engine policy path: GPU-assisted selection
                seed = int(absolute_episode_start * 1000)
                enc2, offs, scores = self._engine_policy_generate_once(seed)
                # Update buffer and stats
                if scores.shape[0] > 0:
                    self.add_encoded_to_buffer(enc2, offs, scores)
                    for final_score in scores.tolist():
                        total_score += float(final_score)
                        if final_score > 0:
                            total_royalties += 1
                            royalty_scores.append(float(final_score))
                        elif final_score < 0:
                            total_fouls += 1
                        else:
                            total_zero += 1
                episodes_generated = int(scores.shape[0])
                # Update progress and training using episodes_generated
                for _ in range(episodes_generated):
                    episode_idx += 1
                    absolute_episode = start_episode + episode_idx - 1
                    since_last_update += 1
                    if since_last_update >= pbar_update_stride:
                        pbar.update(since_last_update)
                        since_last_update = 0
                    if len(self.replay_buffer) >= self.batch_size and episode_idx % episodes_per_train_step == 0:
                        # Only train here if NOT at a checkpoint (checkpoints handle training separately)
                        # Also check if we've already trained at this episode
                        if (absolute_episode not in trained_episodes and
                            not (absolute_episode >= eval_frequency and absolute_episode % eval_frequency == 0)):
                            trained_episodes.add(absolute_episode)
                            pbar.clear()
                            print(f"\n[Episode {absolute_episode:,}] Training cycle ({updates_per_cycle} gradient updates)")
                            train_pbar = tqdm(range(updates_per_cycle), desc="Training", unit="update", leave=False, mininterval=0.5)
                            for update_idx in train_pbar:
                                loss = self.train_step()
                                losses.append(loss)
                            train_pbar.close()
                            avg_loss = np.mean(losses[-updates_per_cycle:])
                            print(f"Training complete. Average loss: {avg_loss:.4f}\n")
        
        pbar.close()
        # Flush any remaining progress not reflected due to batching
        if since_last_update > 0:
            try:
                pbar.update(since_last_update)
            except Exception:
                pass
        
        # Final checkpoint handling (only if we didn't already process it)
        # Check if the target episode (num_episodes) was already evaluated as a checkpoint
        # If it was, skip - evaluation already done. If not, process it now.
        if num_episodes not in evaluated_checkpoints:
            # Final checkpoint wasn't processed during the loop, process it now
            if num_episodes >= eval_frequency and num_episodes % eval_frequency == 0:
                final_episode = num_episodes
                pbar.clear()
                print(f"\n{'='*60}")
                print(f"Checkpoint at episode {final_episode:,}")
                print(f"{'='*60}\n")
                import sys
                sys.stdout.flush()
                
                # Step 1: Training (if needed and not already done)
                if (len(self.replay_buffer) >= self.batch_size and 
                    final_episode % episodes_per_train_step == 0 and
                    final_episode not in trained_episodes):
                    trained_episodes.add(final_episode)
                    print(f"[Episode {final_episode:,}] Training cycle ({updates_per_cycle} gradient updates)")
                    train_pbar = tqdm(range(updates_per_cycle), desc="Training", unit="update", leave=False, mininterval=0.5)
                    for update_idx in train_pbar:
                        loss = self.train_step()
                        losses.append(loss)
                    train_pbar.close()
                    avg_loss = np.mean(losses[-updates_per_cycle:])
                    print(f"Training complete. Average loss: {avg_loss:.4f}\n")
                
                # Step 2: Evaluation
                print(f"--- Evaluation at episode {final_episode:,} ---\n")
                training_foul_rate = (total_fouls / (final_episode + 1)) * 100 if final_episode > 0 else 0
                avg_score_per_hand = total_score / (final_episode + 1) if final_episode > 0 else 0.0
                self._evaluate(total_episodes=final_episode+1, total_fouls=total_fouls, 
                              total_royalties=total_royalties, total_zero=total_zero,
                              training_foul_rate=training_foul_rate,
                              avg_score_per_hand=avg_score_per_hand)
                sys.stdout.flush()
                
                # Step 3: Save checkpoint
                checkpoint_path = f'value_net_checkpoint_ep{final_episode}.pth'
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}\n")
                sys.stdout.flush()
                evaluated_checkpoints.add(final_episode)
        # If num_episodes was already in evaluated_checkpoints, skip - evaluation already done
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        elapsed = time.time() - start_time
        print(f"Total time: {elapsed/3600:.2f} hours ({elapsed:.0f} seconds)")
        print(f"Episodes: {num_episodes:,}")
        print(f"Average time per episode: {elapsed/num_episodes*1000:.2f} ms")
        print(f"{'='*60}\n")
        
        # Final summary statistics (evaluation already done at final checkpoint)
        training_foul_rate = (total_fouls / num_episodes) * 100 if num_episodes > 0 else 0
        avg_score_per_hand = total_score / num_episodes if num_episodes > 0 else 0.0
        print("Final Training Statistics:")
        print(f"  Total Hands: {num_episodes:,}")
        print(f"  Hands Fouled: {total_fouls}/{num_episodes} ({training_foul_rate:.1f}%)")
        print(f"  Hands Scored 0: {total_zero}/{num_episodes} ({total_zero/num_episodes*100:.1f}%)")
        print(f"  Hands with Royalties: {total_royalties}/{num_episodes} ({total_royalties/num_episodes*100:.2f}%)")
        print(f"  Average Score Per Hand: {avg_score_per_hand:.2f}")
        print()
    
    def _evaluate(self, total_episodes: int = 0, total_fouls: int = 0, 
                  total_royalties: int = 0, total_zero: int = 0, 
                  training_foul_rate: float = 0.0, avg_score_per_hand: float = 0.0):
        """Evaluate model on test episodes."""
        self.model.eval()
        test_scores = []
        test_fouls = 0
        incomplete_boards = 0
        complete_boards = 0
        foul_by_round = [0, 0, 0, 0, 0]  # round index when done (0..4)
        royalties_top_ct = 0
        royalties_mid_ct = 0
        royalties_bot_ct = 0
        
        with torch.no_grad():
            eval_pbar = tqdm(range(500), desc="Evaluating", unit="hand", leave=False, mininterval=0.5)
            for eval_idx in eval_pbar:
                # For evaluation, use Python path to ensure reliable single-episode generation
                # This is more reliable than the C++ engine for single episodes
                from ofc_env import OfcEnv
                env = OfcEnv()
                state = env.reset()
                
                # Verify reset worked correctly
                if state.round != 0 or len(state.current_draw) != 5:
                    # Reset failed, skip this episode
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
                        episode_states.append(state)
                        break
                    
                    episode_states.append(state)
                    step_count += 1
                    
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
                    
                    state, reward, done = env.step(state, action)
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
            
            if total_episodes > 0:
                print(f"{'='*23}")
                print(f"Training Statistics:")
                print(f"    Total Hands: {total_episodes:,}")
                print(f"    Hands Fouled: {total_fouls:,}/{total_episodes:,} ({training_foul_rate:.1f}%)")
                print(f"    Hands Scored 0: {total_zero:,}/{total_episodes:,} ({total_zero/total_episodes*100:.1f}%)")
                print(f"    Hands with Royalties: {total_royalties:,}/{total_episodes:,} ({total_royalties/total_episodes*100:.2f}%)")
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
        # C++ engine handles its own cleanup, no Python multiprocessing pool to clean
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()


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

