import tensorflow as tf
import numpy as np

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions
    (state, action, reward, next_state, true_action, position)
    """
    def __init__(self, max_size):
        """
        Initialize replay buffer
        Args:
            max_size: Maximum number of transitions to store
        """
        self.max_size = max_size
        self.buffer = []
        self.position = 0
        
        # Initialize tensor placeholders for batch processing
        self.state_shape = None
        self.action_shape = None
        self.next_state_shape = None
        self.true_action_shape = None
        
        # Pre-allocate numpy arrays for efficient batch operations
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.true_actions = None
        self.positions = None
        
    def add(self, state, action, reward, next_state, true_action, position):
        """
        Add a transition to the buffer
        Args:
            state: Current state tensor of shape [batch, p, height, width, channels]
            action: Action tensor of shape [batch, height, width, channels]
            reward: Reward tensor of shape [batch]
            next_state: Next state tensor of shape [batch, p, height, width, channels]
            true_action: Ground truth action tensor of shape [batch, height, width, channels]
            position: Position tensor of shape [batch]
        """
        # Store shapes on first add
        if self.state_shape is None:
            self.state_shape = state.shape
            self.action_shape = action.shape
            self.next_state_shape = next_state.shape
            self.true_action_shape = true_action.shape
            
            # Pre-allocate numpy arrays
            self.states = np.zeros((self.max_size,) + self.state_shape[1:], dtype=np.float32)
            self.actions = np.zeros((self.max_size,) + self.action_shape[1:], dtype=np.float32)
            self.rewards = np.zeros((self.max_size,), dtype=np.float32)
            self.next_states = np.zeros((self.max_size,) + self.next_state_shape[1:], dtype=np.float32)
            self.true_actions = np.zeros((self.max_size,) + self.true_action_shape[1:], dtype=np.float32)
            self.positions = np.zeros((self.max_size,), dtype=np.int32)
        
        # Convert tensors to numpy arrays
        state_np = state.numpy() if isinstance(state, tf.Tensor) else state
        action_np = action.numpy() if isinstance(action, tf.Tensor) else action
        reward_np = reward.numpy() if isinstance(reward, tf.Tensor) else reward
        next_state_np = next_state.numpy() if isinstance(next_state, tf.Tensor) else next_state
        true_action_np = true_action.numpy() if isinstance(true_action, tf.Tensor) else true_action
        position_np = position.numpy() if isinstance(position, tf.Tensor) else position
        
        # Store in pre-allocated arrays
        idx = self.position % self.max_size
        self.states[idx] = state_np
        self.actions[idx] = action_np
        self.rewards[idx] = reward_np
        self.next_states[idx] = next_state_np
        self.true_actions[idx] = true_action_np
        self.positions[idx] = position_np
        
        self.position = (self.position + 1) % self.max_size
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions efficiently
        Args:
            batch_size: Number of transitions to sample
        Returns:
            Tuple of (states, actions, rewards, next_states, true_actions, positions)
        """
        if len(self) < batch_size:
            return None
            
        # Sample indices
        indices = np.random.choice(len(self), batch_size, replace=False)
        
        # Return batch using pre-allocated arrays
        return (
            tf.convert_to_tensor(self.states[indices]),
            tf.convert_to_tensor(self.actions[indices]),
            tf.convert_to_tensor(self.rewards[indices]),
            tf.convert_to_tensor(self.next_states[indices]),
            tf.convert_to_tensor(self.true_actions[indices]),
            tf.convert_to_tensor(self.positions[indices])
        )
        
    def __len__(self):
        """Return current size of buffer"""
        return min(self.position, self.max_size) 