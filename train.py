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
    def __init__(self, model: ValueNet, device: torch.device, request_queue: Queue, response_queues, max_batch: int = 2048):
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

        # Generate random episodes in parallel using multiprocessing
        seeds = [base_seed + i for i in range(num_episodes)]
        episode_results = self.pool.map(_generate_episode_worker, seeds)

        # Flatten results
        all_data = []
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
        
        # Start action server in main process
        server = ActionServer(self.model, self.device, self.request_queue, self.response_queues, max_batch=4096)
        
        # Start workers
        for p in workers:
            p.start()
        
        # Serve until all workers finish
        finished = 0
        episodes: List[List[Tuple[State, float]]] = []
        while finished < num_to_run:
            served = server.serve_once(timeout=0.02)
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
        
        # Encode states in batch (much faster than one-by-one)
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
        loss = self.criterion(predictions, targets)
        
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
        pbar = tqdm(range(start_episode, num_episodes), desc="Training", unit="hand", initial=start_episode, total=num_episodes)
        
        # Generate episodes in larger batches using multiprocessing for speed
        # Aggressively scale batch size with number of workers to push CPU harder
        episodes_per_batch = max(self.num_workers * 4, 32)  # Generate many episodes at once
        episodes_per_train_step = 8  # Train every N episodes
        
        episode_idx = 0
        while start_episode + episode_idx < num_episodes:
            # Calculate how many episodes to generate in this batch
            remaining = num_episodes - (start_episode + episode_idx)
            batch_size_current = min(episodes_per_batch, remaining)
            
            # Calculate absolute episode number
            absolute_episode = start_episode + episode_idx
            
            # Gradually transition from random to learned policy
            random_prob = max(0.0, 1.0 - (absolute_episode / (num_episodes * 0.8)))
            use_random_for_batch = random_prob > 0.5  # Use multiprocessing for random episodes
            
            # Generate batch of episodes
            if use_random_for_batch:
                # Parallel generation for random episodes
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
            else:
                # Parallel net-based episodes via centralized GPU action server
                episodes_parallel = self.generate_episodes_parallel_with_server(
                    num_episodes=batch_size_current,
                    base_seed=absolute_episode * 2000
                )
                episodes_list = [ep for ep in episodes_parallel if ep]
            
            # Process each episode
            for episode_data in episodes_list:
                if not episode_data:
                    continue
                
                # Track statistics
                final_score = episode_data[-1][1]
                total_score += final_score
                if final_score > 0:
                    total_royalties += 1
                    royalty_scores.append(final_score)
                elif final_score < 0:
                    total_fouls += 1
                else:
                    total_zero += 1
                
                # Add to buffer
                self.add_to_buffer(episode_data)
                
                # Update progress
                episode_idx += 1
                absolute_episode = start_episode + episode_idx - 1
                pbar.update(1)
                
                # Train less frequently but do more gradient steps per cycle
                if len(self.replay_buffer) >= self.batch_size and episode_idx % episodes_per_train_step == 0:
                    # Do 4 gradient updates per training cycle for better amortization
                    for _ in range(4):
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
                    # Clear progress bar and print clean evaluation
                    pbar.clear()
                    print(f"\n--- Evaluation at episode {absolute_episode:,} ---\n")
                    # Pass training stats to evaluation
                    training_foul_rate = (total_fouls / (absolute_episode + 1)) * 100 if absolute_episode > 0 else 0
                    avg_score_per_hand = total_score / (absolute_episode + 1) if absolute_episode > 0 else 0.0
                    self._evaluate(total_episodes=absolute_episode+1, total_fouls=total_fouls, 
                                  total_royalties=total_royalties, total_zero=total_zero,
                                  training_foul_rate=training_foul_rate,
                                  avg_score_per_hand=avg_score_per_hand)
                    
                    # Save checkpoint
                    checkpoint_path = f'value_net_checkpoint_ep{absolute_episode}.pth'
                    torch.save(self.model.state_dict(), checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}\n")
        
        pbar.close()
        
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
        buffer_size=6000,  # Back to original size
        batch_size=16,  # Back to original size
        learning_rate=1e-3,
        use_cuda=True  # Will use CUDA if available
    )
    
    try:
        # Train for millions of hands
        # Start with smaller number for testing, then scale up
        num_episodes = 1_000_000  # 1 million hands
        
        trainer.train(
            num_episodes=num_episodes,
            episodes_per_update=10,
            eval_frequency=10000  # Evaluate every 10k hands
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

