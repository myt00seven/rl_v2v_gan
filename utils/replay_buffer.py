import numpy as np
import tensorflow as tf
from collections import deque

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    (state, action, reward, next_state, true_action, position, domain)
    """
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # For priority sampling
        self.domain_counts = {'x': 0, 'y': 0, 'z': 0}  # Track samples per domain
        
    def add(self, state, action, reward, next_state, true_action, position, domain='x'):
        """
        Add a transition to the buffer
        Args:
            state: Current state (video frames)
            action: Action taken
            reward: Reward received
            next_state: Next state
            true_action: Ground truth action
            position: Position in video sequence
            domain: Domain of the transition ('x', 'y', or 'z')
        """
        # Store transition
        self.buffer.append((state, action, reward, next_state, true_action, position, domain))
        
        # Update priorities (using reward magnitude)
        priority = float(abs(reward))
        self.priorities.append(priority)
        
        # Update domain counts
        self.domain_counts[domain] += 1
        
    def sample(self, batch_size, prioritize=True):
        """
        Sample a batch of transitions
        Args:
            batch_size: Number of transitions to sample
            prioritize: Whether to use priority sampling
        Returns:
            List of (state, action, reward, next_state, true_action, position, domain) tuples
        """
        if len(self.buffer) < batch_size:
            return None
            
        if prioritize:
            # Convert priorities to sampling probabilities
            priorities = np.array(self.priorities)
            probs = priorities / np.sum(priorities)
            
            # Sample indices based on priorities
            indices = np.random.choice(
                len(self.buffer),
                batch_size,
                replace=False,
                p=probs
            )
        else:
            # Uniform random sampling
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            
        return [self.buffer[i] for i in indices]
    
    def sample_by_domain(self, batch_size, domain):
        """
        Sample transitions from a specific domain
        Args:
            batch_size: Number of transitions to sample
            domain: Domain to sample from ('x', 'y', or 'z')
        Returns:
            List of transitions from the specified domain
        """
        # Get indices of transitions from specified domain
        domain_indices = [
            i for i, (_, _, _, _, _, _, d) in enumerate(self.buffer)
            if d == domain
        ]
        
        if len(domain_indices) < batch_size:
            return None
            
        # Sample indices
        indices = np.random.choice(domain_indices, batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def sample_by_position(self, batch_size, position):
        """
        Sample transitions from a specific position in sequence
        Args:
            batch_size: Number of transitions to sample
            position: Position in sequence to sample from
        Returns:
            List of transitions from the specified position
        """
        # Get indices of transitions from specified position
        position_indices = [
            i for i, (_, _, _, _, _, p, _) in enumerate(self.buffer)
            if p == position
        ]
        
        if len(position_indices) < batch_size:
            return None
            
        # Sample indices
        indices = np.random.choice(position_indices, batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def get_domain_stats(self):
        """Get statistics about samples in each domain"""
        return self.domain_counts
    
    def __len__(self):
        """Return current size of buffer"""
        return len(self.buffer) 