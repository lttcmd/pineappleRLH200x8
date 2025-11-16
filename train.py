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

from ofc_env import OfcEnv, State, Action
from state_encoding import encode_state, encode_state_batch, get_input_dim
from value_net import ValueNet


# Worker function for parallel episode generation (must be at module level for pickling)
def _generate_episode_worker(seed: int) -> List[Tuple[State, float]]:
    """
    Worker function to generate one episode in a separate process.
    
    Args:
        seed: Random seed for reproducibility
    
    Returns:
        List of (state, final_score) pairs
    """
    # Set random seed for this worker
    random.seed(seed)
    np.random.seed(seed)
    
    # Create environment (each worker gets its own)
    env = OfcEnv()
    state = env.reset()
    episode_states = []
    
    while True:
        # Get legal actions
        legal_actions = env.legal_actions(state)
        
        if not legal_actions:
            episode_states.append(state)
            break
        
        # Save current state BEFORE stepping
        episode_states.append(state)
        
        # Use random action (for parallel generation, we only do random episodes)
        action = legal_actions[random.randint(0, len(legal_actions) - 1)]
        
        # Step environment
        state, reward, done = env.step(state, action)
        
        if done:
            episode_states.append(state)
            break
    
    # Compute final score
    final_score = env.score(state)
    
    # Return (state, final_score) pairs
    return [(s, final_score) for s in episode_states]


def _generate_episode_with_net_worker(args: Tuple[int, dict]) -> List[Tuple[State, float]]:
    """
    Placeholder for future designs; not used in the centralized action server path.
    """
    raise NotImplementedError("Net-based worker episodes are not used in this version.")


# Centralized GPU action serving components
def _env_worker_requesting_actions(args):
    """
    Env-only worker: simulates one episode and requests actions from a central GPU server.
    Communicates via request_queue (to send encoded batches) and response_queue (to receive best index).
    """
    worker_id, seed, request_queue, response_queue = args

    random.seed(seed)
    np.random.seed(seed)

    env = OfcEnv()
    state = env.reset()
    episode_states = []
    req_counter = 0

    while True:
        legal_actions = env.legal_actions(state)
        if not legal_actions:
            episode_states.append(state)
            break

        episode_states.append(state)

        # Build candidate next states and encode on CPU
        next_states = []
        for action in legal_actions:
            next_state, _, _ = env.step(state, action)
            next_states.append(next_state)

        encoded_batch = encode_state_batch(next_states).numpy()  # np.float32

        request_id = (worker_id, req_counter)
        req_counter += 1
        request_queue.put((request_id, len(legal_actions), encoded_batch))

        # Wait for best index
        resp_request_id, best_idx = response_queue.get()
        # Basic sanity: ensure response matches request
        if resp_request_id != request_id:
            while resp_request_id != request_id:
                resp_request_id, best_idx = response_queue.get()

        best_action = legal_actions[best_idx]
        state, _, done = env.step(state, best_action)
        if done:
            episode_states.append(state)
            break

    final_score = env.score(state)
    return [(s, final_score) for s in episode_states]


def _run_one_episode(wid, seed, request_q, response_q, out_q):
    """
    Top-level wrapper so it is picklable under 'spawn'.
    Runs a single episode using the env-only worker and returns data via out_q.
    """
    data = _env_worker_requesting_actions((wid, seed, request_q, response_q))
    out_q.put((wid, data))


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
            values = self.model(mega).squeeze()

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
        
        # Use multiple environments for parallel episode generation
        # This helps keep GPU busy while CPU generates episodes
        # Increased to better utilize many-core CPUs
        self.num_envs = 64  # Number of parallel environments
        self.envs = [OfcEnv() for _ in range(self.num_envs)]
        
        # Use pinned memory for faster CPU->GPU transfer
        self.pin_memory = True if torch.cuda.is_available() and use_cuda else False
        
        # Multiprocessing pool for parallel episode generation
        # Default: aggressively use available CPU cores
        if num_workers is None:
            num_workers = max(1, mp.cpu_count())  # Use all logical CPUs by default
        self.num_workers = num_workers
        self.pool = Pool(processes=num_workers)
        print(f"Multiprocessing: Using {num_workers} worker processes")
        # Centralized action serving infra (lazy init)
        self.request_queue: Optional[Queue] = None
        self.response_queues = None

        # Engine-policy settings (Phase 2)
        self.use_engine_policy = _USE_ENGINE_POLICY and (_CPP is not None)
        # Configure default engine parameters
        self.engine_num_envs = max(64, min(256, self.num_workers))  # more envs
        self.engine_max_candidates = 128  # explore more actions
        self.engine_cycles = 300  # push episodes further per run
        # Hybrid exploration schedule
        self.random_phase_episodes = 2_000_000        # 100% random for first 2M hands
        self.anneal_phase_episodes = 3_000_000        # linearly anneal over next 3M
        self.min_random_prob = 0.15                   # keep at least 15% random thereafter
    
    def generate_episode(self, use_random: bool = True, env_idx: int = 0) -> List[Tuple[State, float]]:
        """
        Generate one episode of self-play.
        Optimized for speed.
        
        Args:
            use_random: If True, use random actions. If False, use value network.
            env_idx: Which environment to use (for parallel generation)
        
        Returns:
            List of (state, final_score) pairs
        """
        env = self.envs[env_idx % len(self.envs)]
        state = env.reset()
        episode_states = []
        
        while True:
            # Get legal actions
            legal_actions = env.legal_actions(state)
            
            if not legal_actions:
                episode_states.append(state)
                break
            
            # Save current state BEFORE stepping
            episode_states.append(state)
            
            # Choose action (use faster random.choice for random)
            if use_random:
                action = legal_actions[random.randint(0, len(legal_actions) - 1)]  # Faster than random.choice
            else:
                action = self._choose_action_with_net(state, legal_actions, env_idx=env_idx)
            
            # Step environment
            state, reward, done = env.step(state, action)
            
            if done:
                episode_states.append(state)
                break
        
        # Compute final score
        final_score = env.score(state)
        
        # Return (state, final_score) pairs
        return [(s, final_score) for s in episode_states]
    
    def generate_episodes_parallel(self, num_episodes: int, use_random: bool = True, base_seed: int = 0) -> List[Tuple[State, float]]:
        """
        Generate multiple episodes in parallel using multiprocessing.
        This significantly speeds up episode generation by using multiple CPU cores.
        
        Args:
            num_episodes: Number of episodes to generate
            use_random: If True, use random actions (for parallel generation)
            base_seed: Base seed for reproducibility
        
        Returns:
            List of all (state, final_score) pairs from all episodes
        """
        # For now, only random-policy episodes use multiprocessing.
        # Net-based episodes fall back to the single-process path which keeps
        # the model on GPU in the main process for efficiency.
        if not use_random or num_episodes < 4:
            all_data = []
            for i in range(num_episodes):
                episode_data = self.generate_episode(use_random=use_random, env_idx=i)
                all_data.extend(episode_data)
            return all_data

        # Generate random episodes in parallel using multiprocessing (streamed)
        seeds = [base_seed + i for i in range(num_episodes)]
        all_data = []
        try:
            for episode_data in self.pool.imap_unordered(_generate_episode_worker, seeds, chunksize=1):
                all_data.extend(episode_data)
        except Exception:
            # Fallback to non-streaming map if imap_unordered unavailable
            episode_results = self.pool.map(_generate_episode_worker, seeds)
            for episode_data in episode_results:
                all_data.extend(episode_data)

        return all_data
    
    def generate_episodes_parallel_with_server(self, num_episodes: int, base_seed: int = 0) -> List[List[Tuple[State, float]]]:
        """
        Parallel net-based episodes using env-only workers and a central GPU action server.
        Spawns up to num_workers processes, each producing one episode.
        """
        if self.request_queue is None:
            # Use fast, process-shared Queue (avoid Manager proxies)
            self.request_queue = Queue(maxsize=8192)
        self.response_queues = {}
        
        # Run up to the full number of workers to better keep the server busy
        num_to_run = min(num_episodes, self.num_workers)
        # Create per-worker response queues
        for wid in range(num_to_run):
            self.response_queues[wid] = Queue(maxsize=1024)
        
        out_q = Queue(maxsize=num_to_run)
        workers: List[Process] = []
        for wid in range(num_to_run):
            p = Process(
                target=_run_one_episode,
                args=(wid, base_seed + wid, self.request_queue, self.response_queues[wid], out_q)
            )
            p.daemon = True
            workers.append(p)
        
        # Start action server in main process with larger max batch
        server = ActionServer(self.model, self.device, self.request_queue, self.response_queues, max_batch=16384)
        
        # Start workers
        for p in workers:
            p.start()
        
        # Serve until all workers finish
        finished = 0
        episodes: List[List[Tuple[State, float]]] = []
        while finished < num_to_run:
            served = server.serve_once(timeout=0.10)
            # Check for finished outputs
            try:
                wid, data = out_q.get_nowait()
                episodes.append(data)
                finished += 1
            except Exception:
                pass
        
        # Final quick drain
        for _ in range(5):
            if server.serve_once(timeout=0.0) == 0:
                break
        server.stop()
        
        # Cleanup workers
        for p in workers:
            p.join(timeout=0.1)
            if p.is_alive():
                p.terminate()
        
        return episodes
    
    def _choose_action_with_net(self, state: State, legal_actions: List[Action], env_idx: int = 0) -> Action:
        """Choose action using value network (greedy) - batched for GPU efficiency."""
        if not legal_actions:
            return None
        
        # Use one of the environments for simulation
        env = self.envs[env_idx % len(self.envs)]
        
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
            values = self.model(encoded_batch).squeeze()
            
            # Find best action
            best_idx = values.argmax().item()
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
                    vals = self.model(x).squeeze()  # [T]
                    vals_cpu = vals.detach().float().cpu().numpy()
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
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        
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
        
        # Forward pass
        self.optimizer.zero_grad()
        predictions = self.model(state_batch)
        # Weighted MSE: emphasize avoiding fouls (negative targets) and rewarding high scores
        with torch.no_grad():
            neg_mask = (targets < 0).float()
            high_mask = (targets > 5.0).float()
            weights = 1.0 + 0.5 * neg_mask + 0.25 * high_mask
        mse_per = (predictions - targets) ** 2
        loss = (mse_per * weights).mean()
        
        # Backward pass
        loss.backward()
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
        print(f"\n{'='*60}")
        print(f"Starting RL Training")
        print(f"{'='*60}")
        print(f"Total episodes (hands): {num_episodes:,}")
        print(f"Device: {self.device}")
        print(f"Buffer size: {self.buffer_size:,}")
        print(f"Batch size: {self.batch_size}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        losses = []
        total_royalties = 0
        total_fouls = 0
        total_zero = 0
        royalty_scores = []
        total_score = 0.0  # Track total score for average calculation
        start_episode = 0
        
        # Try to resume from checkpoint if requested
        if resume:
            checkpoint_info = self.find_latest_checkpoint()
            if checkpoint_info:
                checkpoint_path, checkpoint_episode = checkpoint_info
                self.load_checkpoint(checkpoint_path)
                start_episode = checkpoint_episode + 1
                print(f"Resuming training from episode {start_episode:,}")
                print(f"Will train for {num_episodes - start_episode:,} more episodes (total target: {num_episodes:,})")
        
        # Progress bar for episodes (starting from resume point)
        pbar = tqdm(
            range(start_episode, num_episodes),
            desc="Training",
            unit="hand",
            initial=start_episode,
            total=num_episodes,
            mininterval=0.5,
            smoothing=0.0,
            dynamic_ncols=False
        )
        
        # Generate episodes in batches sized to reduce blocking pauses
        episodes_per_batch = max(self.num_workers, 32)  # Smaller batches for smoother progress
        # Run long generation bursts, then do a large training burst
        episodes_per_train_step = 100_000  # Generate 100k hands, then train
        updates_per_cycle = 128  # Number of gradient updates after each 100k hands
        
        episode_idx = 0
        # Reduce UI overhead: batch progress bar updates
        pbar_update_stride = 512
        since_last_update = 0
        # Throughput-first: 100% random batches for maximum hands/sec
        while start_episode + episode_idx < num_episodes:
            # Calculate how many episodes to generate in this batch
            remaining = num_episodes - (start_episode + episode_idx)
            batch_size_current = min(episodes_per_batch, remaining)
            
            # Calculate absolute episode number
            absolute_episode = start_episode + episode_idx
            
            # Hybrid exploration schedule
            if self.use_engine_policy:
                # Determine random probability based on progress (absolute episode count)
                if absolute_episode < self.random_phase_episodes:
                    random_prob = 1.0
                elif absolute_episode < (self.random_phase_episodes + self.anneal_phase_episodes):
                    t = (absolute_episode - self.random_phase_episodes) / max(1, self.anneal_phase_episodes)
                    # 1.0 -> min_random_prob linearly
                    random_prob = 1.0 - t * (1.0 - self.min_random_prob)
                else:
                    random_prob = self.min_random_prob
                # Stochastic choice per batch
                use_random_for_batch = (np.random.rand() < random_prob)
            else:
                use_random_for_batch = True
                random_prob = 1.0
            
            # Generate batch of episodes
            if use_random_for_batch:
                episodes_generated = 0
                if _USE_CPP and _CPP is not None:
                    # Single C++ call generates many random episodes efficiently
                    seed = int(absolute_episode * 1000)
                    encoded, offsets, scores_np = _CPP.generate_random_episodes(np.uint64(seed), int(batch_size_current))
                    # Numpy arrays
                    encoded_np = np.array(encoded, copy=False)            # [S,838] float32
                    offsets_np = np.array(offsets, copy=False).astype(np.int32, copy=False)  # [E+1]
                    scores_np = np.array(scores_np, copy=False).astype(np.float32, copy=False)  # [E]
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
                    # Fallback: Python multiprocessing generator
                    all_episode_data = self.generate_episodes_parallel(
                        batch_size_current,
                        use_random=True,
                        base_seed=absolute_episode * 1000
                    )

                    # Split into individual episodes (each episode ~13 states)
                    episodes_list = []
                    current_ep = []
                    for state, score in all_episode_data:
                        current_ep.append((state, score))
                        if len(current_ep) >= 13:  # OFC typically has ~13 placements
                            episodes_list.append(current_ep)
                            current_ep = []
                    if current_ep:  # Add remaining
                        episodes_list.append(current_ep)

                    # Process each episode (Python path)
                    for episode_data in episodes_list:
                        if not episode_data:
                            continue
                        final_score = episode_data[-1][1]
                        total_score += final_score
                        if final_score > 0:
                            total_royalties += 1
                            royalty_scores.append(final_score)
                        elif final_score < 0:
                            total_fouls += 1
                        else:
                            total_zero += 1
                        self.add_to_buffer(episode_data)
                        episodes_generated += 1
            else:
                # Unused in 100% random mode
                episodes_generated = 0
            
            
            # Update progress and training using episodes_generated
            for _ in range(episodes_generated):
                episode_idx += 1
                absolute_episode = start_episode + episode_idx - 1
                since_last_update += 1
                if since_last_update >= pbar_update_stride:
                    pbar.update(since_last_update)
                    since_last_update = 0
                # Train in bursts after large generation windows
                if len(self.replay_buffer) >= self.batch_size and episode_idx % episodes_per_train_step == 0:
                    for _ in range(updates_per_cycle):
                        loss = self.train_step()
                        losses.append(loss)
                    # Update progress bar
                    avg_loss = np.mean(losses[-100:]) if losses else 0.0
                    royalty_rate = (total_royalties / (absolute_episode + 1)) * 100 if absolute_episode > 0 else 0
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'buffer': len(self.replay_buffer),
                        'random%': f'{random_prob*100:.1f}%',
                        'royalties': f'{total_royalties} ({royalty_rate:.2f}%)'
                    })
                # Periodic evaluation and checkpointing
                if absolute_episode > 0 and absolute_episode % eval_frequency == 0:
                    pbar.clear()
                    print(f"\n--- Evaluation at episode {absolute_episode:,} ---\n")
                    training_foul_rate = (total_fouls / (absolute_episode + 1)) * 100 if absolute_episode > 0 else 0
                    avg_score_per_hand = total_score / (absolute_episode + 1) if absolute_episode > 0 else 0.0
                    self._evaluate(total_episodes=absolute_episode+1, total_fouls=total_fouls, 
                                  total_royalties=total_royalties, total_zero=total_zero,
                                  training_foul_rate=training_foul_rate,
                                  avg_score_per_hand=avg_score_per_hand)
                    checkpoint_path = f'value_net_checkpoint_ep{absolute_episode}.pth'
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}\n")
            else:
                # Engine policy path: GPU-assisted selection
                seed = int(absolute_episode * 1000)
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
                        for _ in range(updates_per_cycle):
                            loss = self.train_step()
                            losses.append(loss)
                        avg_loss = np.mean(losses[-100:]) if losses else 0.0
                        royalty_rate = (total_royalties / (absolute_episode + 1)) * 100 if absolute_episode > 0 else 0
                        pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'buffer': len(self.replay_buffer),
                            'random%': f'{random_prob*100:.1f}%',
                            'royalties': f'{total_royalties} ({royalty_rate:.2f}%)'
                        })
                    if absolute_episode > 0 and absolute_episode % eval_frequency == 0:
                        pbar.clear()
                        print(f"\n--- Evaluation at episode {absolute_episode:,} ---\n")
                        training_foul_rate = (total_fouls / (absolute_episode + 1)) * 100 if absolute_episode > 0 else 0
                        avg_score_per_hand = total_score / (absolute_episode + 1) if absolute_episode > 0 else 0.0
                        self._evaluate(total_episodes=absolute_episode+1, total_fouls=total_fouls, 
                                      total_royalties=total_royalties, total_zero=total_zero,
                                      training_foul_rate=training_foul_rate,
                                      avg_score_per_hand=avg_score_per_hand)
                        checkpoint_path = f'value_net_checkpoint_ep{absolute_episode}.pth'
                        torch.save(self.model.state_dict(), checkpoint_path)
                        print(f"\nCheckpoint saved: {checkpoint_path}\n")
        
        pbar.close()
        # Flush any remaining progress not reflected due to batching
        if since_last_update > 0:
            try:
                pbar.update(since_last_update)
            except Exception:
                pass
        
        # Final evaluation
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        elapsed = time.time() - start_time
        print(f"Total time: {elapsed/3600:.2f} hours ({elapsed:.0f} seconds)")
        print(f"Episodes: {num_episodes:,}")
        print(f"Average time per episode: {elapsed/num_episodes*1000:.2f} ms")
        print(f"{'='*60}\n")
        
        # Final evaluation with training stats
        training_foul_rate = (total_fouls / num_episodes) * 100 if num_episodes > 0 else 0
        avg_score_per_hand = total_score / num_episodes if num_episodes > 0 else 0.0
        self._evaluate(total_episodes=num_episodes, total_fouls=total_fouls,
                      total_royalties=total_royalties, total_zero=total_zero,
                      training_foul_rate=training_foul_rate,
                      avg_score_per_hand=avg_score_per_hand)
    
    def _evaluate(self, total_episodes: int = 0, total_fouls: int = 0, 
                  total_royalties: int = 0, total_zero: int = 0, 
                  training_foul_rate: float = 0.0, avg_score_per_hand: float = 0.0):
        """Evaluate model on test episodes."""
        self.model.eval()
        test_scores = []
        test_fouls = 0
        incomplete_boards = 0
        complete_boards = 0
        
        with torch.no_grad():
            for _ in range(50):  # More episodes for better stats
                episode_data = self.generate_episode(use_random=False)
                if episode_data:
                    final_score = episode_data[-1][1]
                    test_scores.append(final_score)
                    
                    # Check if board was complete
                    if final_score == 0.0:
                        # Check the actual final state
                        state = episode_data[-1][0] if hasattr(episode_data[-1][0], 'board') else None
                        if state:
                            is_complete = all(slot is not None for slot in state.board)
                            if is_complete:
                                complete_boards += 1
                            else:
                                incomplete_boards += 1
                        else:
                            incomplete_boards += 1
                    else:
                        complete_boards += 1
                    
                    if final_score < 0:  # Foul penalty
                        test_fouls += 1
        
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
            print(f"Evaluation Statistics: (50 test hands)")
            print(f"  Avg score: {avg_score:.2f} ± {std_score:.2f}")
            print(f"  Range: [{min_score:.1f}, {max_score:.1f}]")
            print(f"  Foul rate: {foul_rate:.1f}%")
            print(f"  Board completion: {complete_boards}/{len(test_scores)} complete, {incomplete_boards} incomplete")
            
            # Show score distribution
            positive_scores = sum(1 for s in test_scores if s > 0)
            zero_scores = sum(1 for s in test_scores if s == 0)
            negative_scores = sum(1 for s in test_scores if s < 0)
            print(f"  Score breakdown: {positive_scores} positive, {zero_scores} zero, {negative_scores} negative")
            print()
        
        self.model.train()
    
    def cleanup(self):
        """Cleanup multiprocessing resources."""
        if hasattr(self, 'pool'):
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
        # Fresh 10,000,000-hand run
        num_episodes = 10_000_000
        
        trainer.train(
            num_episodes=num_episodes,
            episodes_per_update=10,
            eval_frequency=1000000,  # Evaluate and checkpoint every 1M hands
            resume=False  # start fresh: do not reload checkpoints
        )
        
        # Save final model
        torch.save(model.state_dict(), 'value_net.pth')
        print("Final model saved to value_net.pth")
    finally:
        # Cleanup multiprocessing resources
        trainer.cleanup()


if __name__ == '__main__':
    # Use 'spawn' start method so CUDA can be safely used in worker processes
    mp.set_start_method('spawn', force=True)
    main()

