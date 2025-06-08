import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from .models import Generator, Discriminator, QNetwork, Predictor
from .losses import (
    adversarial_loss, recurrent_loss, recycle_loss,
    video_loss, compute_reward, policy_gradient_loss
)
from .replay_buffer import ReplayBuffer

class RL_V2V_GAN_Trainer:
    def __init__(
        self,
        input_shape,
        learning_rate=0.0001,
        gamma=0.99,
        tau=0.001,
        lambda_v=1.0,
        lambda_rr=0.1,
        lambda_rc=0.1,
        sigma1=0.5,
        sigma2=0.5,
        buffer_size=10000
    ):
        # Initialize session
        self.sess = tf.Session()
        
        self.input_shape = input_shape
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Target network update rate
        self.lambda_v = lambda_v  # Video loss coefficient
        self.lambda_rr = lambda_rr  # Recurrent loss coefficient
        self.lambda_rc = lambda_rc  # Recycle loss coefficient
        self.sigma1 = sigma1  # Bernoulli parameter for replay buffer sampling
        self.sigma2 = sigma2  # Bernoulli parameter for action selection
        
        # Initialize networks
        with tf.variable_scope('networks'):
            self.generator_x = Generator(input_shape)
            self.generator_y = Generator(input_shape)
            self.predictor_x = Predictor(input_shape)
            self.predictor_y = Predictor(input_shape)
            self.discriminator_x = Discriminator(input_shape)
            self.discriminator_y = Discriminator(input_shape)
            self.q_network_x = QNetwork(input_shape)
            self.q_network_y = QNetwork(input_shape)
            
            # Target networks
            self.target_generator_x = Generator(input_shape)
            self.target_generator_y = Generator(input_shape)
            self.target_q_network_x = QNetwork(input_shape)
            self.target_q_network_y = QNetwork(input_shape)
        
        # Initialize target networks
        self._update_target_networks(tau=1.0)
        
        # Optimizers
        self.g_optimizer = tf.train.AdamOptimizer(learning_rate)
        self.d_optimizer = tf.train.AdamOptimizer(learning_rate)
        self.q_optimizer = tf.train.AdamOptimizer(learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize variables
        self.sess.run(tf.global_variables_initializer())
        
    def save_weights(self, filepath):
        """Save model weights"""
        saver = tf.train.Saver()
        saver.save(self.sess, filepath)
        print(f"Saved weights to {filepath}")
        
    def load_weights(self, filepath):
        """Load model weights"""
        saver = tf.train.Saver()
        saver.restore(self.sess, filepath)
        print(f"Loaded weights from {filepath}")
        
    def _update_target_networks(self, tau=None):
        """Soft update target networks"""
        if tau is None:
            tau = self.tau
            
        update_ops = []
        
        # Update target generator weights
        for target, source in zip(
            self.target_generator_x.variables,
            self.generator_x.variables
        ):
            update_ops.append(tf.assign(target, tau * source + (1 - tau) * target))
            
        for target, source in zip(
            self.target_generator_y.variables,
            self.generator_y.variables
        ):
            update_ops.append(tf.assign(target, tau * source + (1 - tau) * target))
            
        # Update target Q-network weights
        for target, source in zip(
            self.target_q_network_x.variables,
            self.q_network_x.variables
        ):
            update_ops.append(tf.assign(target, tau * source + (1 - tau) * target))
            
        for target, source in zip(
            self.target_q_network_y.variables,
            self.q_network_y.variables
        ):
            update_ops.append(tf.assign(target, tau * source + (1 - tau) * target))
            
        self.sess.run(update_ops)
            
    def gan_update_step(self, x_video, y_video):
        """Single GAN training step"""
        # Forward pass
        fake_y = self.generator_y(self.generator_x(x_video))
        fake_x = self.generator_x(self.generator_y(y_video))
        
        # Predictions
        pred_y = self.predictor_y(fake_y)
        pred_x = self.predictor_x(fake_x)
        
        # Discriminator outputs
        real_x_logits = self.discriminator_x(x_video)
        fake_x_logits = self.discriminator_x(fake_x)
        real_y_logits = self.discriminator_y(y_video)
        fake_y_logits = self.discriminator_y(fake_y)
        
        # Compute losses
        d_loss_x, g_loss_x = adversarial_loss(real_x_logits, fake_x_logits)
        d_loss_y, g_loss_y = adversarial_loss(real_y_logits, fake_y_logits)
        
        rec_loss_x = recurrent_loss(pred_x, x_video)
        rec_loss_y = recurrent_loss(pred_y, y_video)
        
        cyc_loss_x = recycle_loss(fake_x, x_video)
        cyc_loss_y = recycle_loss(fake_y, y_video)
        
        vid_loss_x = video_loss(fake_x, x_video)
        vid_loss_y = video_loss(fake_y, y_video)
        
        # Total losses
        g_loss = (
            g_loss_x + g_loss_y +
            self.lambda_rr * (rec_loss_x + rec_loss_y) +
            self.lambda_rc * (cyc_loss_x + cyc_loss_y) +
            self.lambda_v * (vid_loss_x + vid_loss_y)
        )
        
        d_loss = d_loss_x + d_loss_y
        
        # Update networks
        g_gradients = self.g_optimizer.compute_gradients(
            g_loss,
            var_list=(
                self.generator_x.trainable_variables +
                self.generator_y.trainable_variables +
                self.predictor_x.trainable_variables +
                self.predictor_y.trainable_variables
            )
        )
        
        d_gradients = self.d_optimizer.compute_gradients(
            d_loss,
            var_list=(
                self.discriminator_x.trainable_variables +
                self.discriminator_y.trainable_variables
            )
        )
        
        self.sess.run([
            self.g_optimizer.apply_gradients(g_gradients),
            self.d_optimizer.apply_gradients(d_gradients)
        ])
        
        return {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'fake_y': fake_y,
            'fake_x': fake_x
        }
        
    def _generate_candidate_actions(self, state, is_x=True):
        """
        Generate two candidate actions for a state
        Args:
            state: Current state
            is_x: Whether this is from domain X (True) or Y (False)
        Returns:
            Tuple of (a1, a2) where:
            a1 = P(s)
            a2 = G^(1)(P(G^(2)(s)))
        """
        if is_x:
            P = self.predictor_x
            G1 = self.generator_y
            G2 = self.generator_x
        else:
            P = self.predictor_y
            G1 = self.generator_x
            G2 = self.generator_y
            
        # Generate a1 = P(s)
        a1 = P(state)
        
        # Generate a2 = G^(1)(P(G^(2)(s)))
        g2_output = G2(state)
        p_output = P(g2_output)
        a2 = G1(p_output)
        
        return a1, a2
        
    def _select_action(self, state, a1, a2, is_x=True):
        """Select action using epsilon-greedy strategy according to the paper.
        
        Args:
            state: Current state tensor of shape [batch, p, height, width, channels]
            a1, a2: Candidate action tensors of shape [batch, height, width, channels]
            is_x: Whether this is from domain X (True) or Y (False)
            
        Returns:
            Selected action tensor of shape [batch, height, width, channels]
        """
        # Get Q-values for both actions
        if is_x:
            q_network = self.q_network_x
        else:
            q_network = self.q_network_y
            
        # Compute Q-values for both actions
        q1 = q_network(tf.concat([state, a1], axis=-1))
        q2 = q_network(tf.concat([state, a2], axis=-1))
        
        # Epsilon-greedy action selection
        epsilon = 0.1  # Exploration rate
        random_actions = tf.random.uniform([tf.shape(state)[0]]) < epsilon
        
        # Select actions based on Q-values (greedy selection)
        selected_actions = tf.where(
            tf.greater(q1, q2),
            a1,
            a2
        )
        
        # For random actions, select from candidate actions with equal probability
        random_selection = tf.random.uniform([tf.shape(state)[0]]) < 0.5
        random_actions = tf.logical_and(random_actions, random_selection)
        
        # Combine selected and random actions
        final_actions = tf.where(
            tf.expand_dims(random_actions, axis=1),
            tf.where(
                tf.expand_dims(random_selection, axis=1),
                a1,
                a2
            ),
            selected_actions
        )
        
        return final_actions
        
    def _compute_reward(self, state, action, next_state, position, is_x=True):
        """
        Compute reward based on position in sequence according to the paper:
        For p = T-1: r = -λv Lv
        For p < T-1: r = -(Lg + λrr Lrr + λrc Lrc)
        
        Args:
            state: Current state
            action: Selected action
            next_state: Next state
            position: Position in sequence
            is_x: Whether this is from domain X (True) or Y (False)
        Returns:
            Computed reward
        """
        if position == self.input_shape[0] - 1:  # Last position (T-1)
            # Video loss reward: r = -λv Lv
            if is_x:
                vid_loss = video_loss(action, next_state)
                return -self.lambda_v * vid_loss
            else:
                vid_loss = video_loss(action, next_state)
                return -self.lambda_v * vid_loss
        else:
            # Combined loss reward: r = -(Lg + λrr Lrr + λrc Lrc)
            if is_x:
                # Adversarial loss
                g_loss = adversarial_loss(
                    self.discriminator_x(state),
                    self.discriminator_x(action)
                )[1]
                
                # Recurrent loss
                rr_loss = recurrent_loss(self.predictor_x(action), next_state)
                
                # Recycle loss
                rc_loss = recycle_loss(action, state)
                
                # Total reward
                return -(g_loss + self.lambda_rr * rr_loss + self.lambda_rc * rc_loss)
            else:
                # Adversarial loss
                g_loss = adversarial_loss(
                    self.discriminator_y(state),
                    self.discriminator_y(action)
                )[1]
                
                # Recurrent loss
                rr_loss = recurrent_loss(self.predictor_y(action), next_state)
                
                # Recycle loss
                rc_loss = recycle_loss(action, state)
                
                # Total reward
                return -(g_loss + self.lambda_rr * rr_loss + self.lambda_rc * rc_loss)
            
    def _compute_q_target(self, state, action, reward, next_state, true_action, is_x=True):
        """
        Compute Q-target for Q-network update according to the paper's formula:
        r + γ Qx(s, μ(s|θ^μ')|θ^Q'_x) · I(a^true ∈ {x}) + γ Qy(s, μ(s|θ^μ')|θ^Q'_y) · I(a^true ∈ {y})
        """
        # Get next actions from target policy networks
        next_action_x = self.target_generator_x(next_state)
        next_action_y = self.target_generator_y(next_state)
        
        # Get Q-values from target Q-networks
        next_q_x = self.target_q_network_x(tf.concat([next_state, next_action_x], axis=-1))
        next_q_y = self.target_q_network_y(tf.concat([next_state, next_action_y], axis=-1))
        
        # Compute indicator functions
        # I(a^true ∈ {x}) = 1 if true_action is from domain X
        indicator_x = tf.cast(tf.reduce_all(
            tf.abs(true_action - self._get_domain_x_reference(true_action)) < 1e-6
        ), tf.float32)
        
        # I(a^true ∈ {y}) = 1 if true_action is from domain Y
        indicator_y = tf.cast(tf.reduce_all(
            tf.abs(true_action - self._get_domain_y_reference(true_action)) < 1e-6
        ), tf.float32)
        
        # Compute target according to paper's formula
        target = reward + self.gamma * (
            next_q_x * indicator_x + 
            next_q_y * indicator_y
        )
        
        return target
        
    def _get_domain_x_reference(self, action):
        """Get reference action from domain X for comparison"""
        # This should return a reference action from domain X
        # For now, we'll use a simple heuristic based on action statistics
        return tf.reduce_mean(action, axis=[1, 2, 3], keepdims=True)
        
    def _get_domain_y_reference(self, action):
        """Get reference action from domain Y for comparison"""
        # This should return a reference action from domain Y
        # For now, we'll use a simple heuristic based on action statistics
        return tf.reduce_std(action, axis=[1, 2, 3], keepdims=True)
        
    @tf.function
    def rl_update_step(self, x_video, y_video, fake_x, fake_y):
        """Single RL training step"""
        batch_size = tf.shape(x_video)[0]
        
        # Sample from replay buffer with probability sigma1
        u1 = tf.random.uniform([batch_size]) < self.sigma1
        buffer_samples = self.replay_buffer.sample(batch_size)
        
        # Process each video in batch
        for i in range(batch_size):
            if u1[i] and buffer_samples is not None:
                # Use sample from buffer
                state, action, reward, next_state, true_action, position = [
                    s[i] for s in buffer_samples
                ]
            else:
                # Generate new transition
                state = x_video[i] if i % 2 == 0 else y_video[i]
                is_x = (i % 2 == 0)
                
                # Generate candidate actions
                a1, a2 = self._generate_candidate_actions(state, is_x)
                
                # Select action
                action = self._select_action(state, a1, a2, is_x)
                
                # Get next state and position
                next_state = y_video[i] if is_x else x_video[i]
                position = tf.random.uniform([], 0, self.input_shape[0], dtype=tf.int32)
                
                # Compute reward
                reward = self._compute_reward(state, action, next_state, position, is_x)
                
                # Get true action
                true_action = fake_x[i] if is_x else fake_y[i]
                
                # Store in buffer
                self.replay_buffer.add(state, action, reward, next_state, true_action, position)
            
            # Compute Q-target
            q_target = self._compute_q_target(
                state, action, reward, next_state, true_action, is_x=(i % 2 == 0)
            )
            
            # Update Q-network
            with tf.GradientTape() as tape:
                if i % 2 == 0:
                    q_value = self.q_network_x(tf.concat([state, action], axis=-1))
                    q_loss = tf.reduce_mean(tf.square(q_target - q_value))
                    q_gradients = tape.gradient(q_loss, self.q_network_x.trainable_variables)
                    self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network_x.trainable_variables))
                else:
                    q_value = self.q_network_y(tf.concat([state, action], axis=-1))
                    q_loss = tf.reduce_mean(tf.square(q_target - q_value))
                    q_gradients = tape.gradient(q_loss, self.q_network_y.trainable_variables)
                    self.q_optimizer.apply_gradients(zip(q_gradients, self.q_network_y.trainable_variables))
        
        # Update target networks
        self._update_target_networks()
        
        return {
            'q_loss': q_loss,
            'reward': reward
        }
        
    def train(self, dataset, epochs, steps_per_epoch):
        """Full training loop with alternating GAN and RL updates"""
        # Initialize memory management
        memory_limit = 0.8  # Use 80% of available GPU memory
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        
        with tf.Session(config=config) as sess:
            self.sess = sess
            
            for epoch in range(epochs):
                print(f"\nEpoch {epoch + 1}/{epochs}")
                
                # Initialize epoch metrics
                epoch_g_loss = 0
                epoch_d_loss = 0
                epoch_q_loss = 0
                epoch_reward = 0
                
                for step in range(steps_per_epoch):
                    # Get batch from dataset
                    x_batch, y_batch, z_batch, z_images = next(iter(dataset))
                    
                    # Adjust batch size if needed
                    batch_size = self._adjust_batch_size(x_batch.shape[0])
                    if batch_size != x_batch.shape[0]:
                        x_batch = x_batch[:batch_size]
                        y_batch = y_batch[:batch_size]
                        z_batch = z_batch[:batch_size]
                        z_images = z_images[:batch_size]
                    
                    # Construct batch B̂ based on video membership
                    batch_B = []
                    for i in range(batch_size):
                        # Randomly select video type (x, y, or z)
                        video_type = np.random.choice(['x', 'y', 'z'])
                        if video_type == 'x':
                            video = x_batch[i]
                        elif video_type == 'y':
                            video = y_batch[i]
                        else:
                            video = z_batch[i]
                            
                        # Randomly select t
                        t = np.random.randint(1, self.input_shape[0] + 1)
                        
                        # Add to batch
                        batch_B.append((video, t, video_type))
                    
                    # Split batch into B̂1 (t = T) and B̂2 (t < T)
                    batch_B1 = [(v, t, vt) for v, t, vt in batch_B if t == self.input_shape[0]]
                    batch_B2 = [(v, t, vt) for v, t, vt in batch_B if t < self.input_shape[0]]
                    
                    # GAN update step with gradient checkpointing
                    with tf.GradientTape(persistent=True) as tape:
                        # Update discriminator for B̂1
                        if batch_B1:
                            d_loss_B1 = self._update_discriminator_batch1(batch_B1)
                        
                        # Update discriminator for B̂2
                        if batch_B2:
                            d_loss_B2 = self._update_discriminator_batch2(batch_B2)
                        
                        # Generate fake videos for RL update
                        gan_results = self.gan_update_step(x_batch, y_batch)
                    
                    # RL update step using generated videos
                    rl_results = self.rl_update_step(
                        x_batch, y_batch,
                        gan_results['fake_x'],
                        gan_results['fake_y']
                    )
                    
                    # Update metrics
                    epoch_g_loss += gan_results['g_loss']
                    epoch_d_loss += d_loss_B1 + d_loss_B2 if 'd_loss_B1' in locals() else 0
                    epoch_q_loss += rl_results['q_loss']
                    epoch_reward += rl_results['reward']
                    
                    # Log progress
                    if step % 100 == 0:
                        print(f"Step {step}/{steps_per_epoch}")
                        print(f"Generator Loss: {gan_results['g_loss']:.4f}")
                        print(f"Discriminator Loss: {epoch_d_loss:.4f}")
                        print(f"Q-Loss: {rl_results['q_loss']:.4f}")
                        print(f"Reward: {rl_results['reward']:.4f}")
                        print("------------------------")
                        
                        # Clear memory periodically
                        tf.keras.backend.clear_session()
                
                # Print epoch summary
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"Average Generator Loss: {epoch_g_loss/steps_per_epoch:.4f}")
                print(f"Average Discriminator Loss: {epoch_d_loss/steps_per_epoch:.4f}")
                print(f"Average Q-Loss: {epoch_q_loss/steps_per_epoch:.4f}")
                print(f"Average Reward: {epoch_reward/steps_per_epoch:.4f}")

    def _update_discriminator_batch1(self, batch_B1):
        """Update discriminator for batch B̂1 (t = T)"""
        d_loss = 0
        for video, t, video_type in batch_B1:
            if video_type in ['x', 'y']:
                # Get discriminator based on video type
                discriminator = self.discriminator_x if video_type == 'x' else self.discriminator_y
                
                # Compute discriminator loss
                real_logits = discriminator(video)
                d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(real_logits),
                    logits=real_logits
                ))
        
        # Apply gradients
        d_gradients = self.d_optimizer.compute_gradients(
            d_loss,
            var_list=(
                self.discriminator_x.trainable_variables +
                self.discriminator_y.trainable_variables
            )
        )
        self.sess.run(self.d_optimizer.apply_gradients(d_gradients))
        
        return d_loss

    def _update_discriminator_batch2(self, batch_B2):
        """Update discriminator for batch B̂2 (t < T)"""
        d_loss = 0
        for video, t, video_type in batch_B2:
            if video_type in ['x', 'y']:
                # Get discriminator based on video type
                discriminator = self.discriminator_x if video_type == 'x' else self.discriminator_y
                
                # Compute discriminator loss without video loss
                real_logits = discriminator(video)
                d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(real_logits),
                    logits=real_logits
                ))
                
                # Add adversarial loss
                fake_video = self.generator_y(video) if video_type == 'x' else self.generator_x(video)
                fake_logits = discriminator(fake_video)
                d_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(fake_logits),
                    logits=fake_logits
                ))
        
        # Apply gradients
        d_gradients = self.d_optimizer.compute_gradients(
            d_loss,
            var_list=(
                self.discriminator_x.trainable_variables +
                self.discriminator_y.trainable_variables
            )
        )
        self.sess.run(self.d_optimizer.apply_gradients(d_gradients))
        
        return d_loss

    def _adjust_batch_size(self, current_batch_size):
        """Dynamically adjust batch size based on memory usage"""
        try:
            # Get current memory usage
            stats = tf.contrib.memory_stats.MemoryStats()
            current_memory = stats['current_memory']
            peak_memory = stats['peak_memory']
            
            # If memory usage is high, reduce batch size
            if current_memory > peak_memory * 0.8:
                return max(1, current_batch_size // 2)
            return current_batch_size
        except:
            # If memory stats not available, return current batch size
            return current_batch_size

    def _get_next_position(self, current_position, sequence_length):
        """Get next position in sequence"""
        if current_position >= sequence_length - 1:
            return 0  # Reset to start
        return current_position + 1 

    def _construct_mini_batch(self, video, is_x=True):
        """Construct mini-batch of size m for a video"""
        mini_batch = []
        position = 0
        
        for _ in range(self.batch_size):
            # Get state (p consecutive frames)
            state = self._get_state_frames(video, position)
            
            # Generate and select action
            a1, a2 = self._generate_candidate_actions(state, is_x)
            action = self._select_action(state, a1, a2, is_x)
            
            # Get next state and compute reward
            next_state = self._get_state_frames(video, position + 1)
            reward = self._compute_reward(state, action, next_state, position, is_x)
            
            # Get true action
            true_action = self._get_true_action(video, position)
            
            mini_batch.append((state, action, reward, next_state, true_action, position))
            position = self._get_next_position(position, len(video))
        
        return mini_batch 

    def _get_state_frames(self, video, position):
        """Extract p consecutive frames starting from position.
        
        Args:
            video: Tensor of shape [batch, seq_len, height, width, channels]
            position: Starting position in sequence
            
        Returns:
            Tensor of shape [batch, p, height, width, channels]
            
        Raises:
            ValueError: If position is invalid or video shape is incorrect
        """
        # Validate input shapes
        if len(video.shape) != 5:
            raise ValueError(f"Expected video shape [batch, seq_len, height, width, channels], got {video.shape}")
            
        batch_size, seq_len, height, width, channels = video.shape
        
        # Validate position
        if position < 0 or position >= seq_len:
            raise ValueError(f"Position {position} out of range [0, {seq_len})")
            
        # Get p consecutive frames starting from position
        # If we don't have enough frames, pad with zeros
        p = 3  # Number of frames to use for state
        frames = []
        
        for i in range(p):
            pos = position + i
            if pos < seq_len:
                frame = video[:, pos:pos+1, :, :, :]
            else:
                # Pad with zeros if we reach the end
                frame = tf.zeros([batch_size, 1, height, width, channels])
            frames.append(frame)
            
        # Concatenate frames along time dimension
        state = tf.concat(frames, axis=1)
        
        return state
        
    def _get_true_action(self, video, position):
        """Get ground truth action frame at position.
        
        Args:
            video: Tensor of shape [batch, seq_len, height, width, channels]
            position: Position in sequence
            
        Returns:
            Tensor of shape [batch, height, width, channels]
            
        Raises:
            ValueError: If position is invalid or video shape is incorrect
        """
        # Validate input shapes
        if len(video.shape) != 5:
            raise ValueError(f"Expected video shape [batch, seq_len, height, width, channels], got {video.shape}")
            
        batch_size, seq_len, height, width, channels = video.shape
        
        # Validate position
        if position < 0 or position >= seq_len:
            raise ValueError(f"Position {position} out of range [0, {seq_len})")
            
        # Get frame at position
        action = video[:, position, :, :, :]
        
        return action 

    def _update_policy_networks(self, batch_B):
        """
        Update policy networks using DDPG as specified in the paper:
        ∇θμJ ≈ -1/m ∑(s̄,·)∈B ∇aQ(s,a|θQ)|s=s̄,a=μ(s̄) ∘ ∇θμμ(s|θμ)|s=s̄
        
        Args:
            batch_B: List of transitions (state, action, reward, next_state, true_action, position, domain)
        """
        # Initialize policy gradients
        g_x_gradients = []
        g_y_gradients = []
        
        for state, _, _, _, _, _, domain in batch_B:
            with tf.GradientTape() as tape:
                # Get current policy action
                if domain == 'x':
                    policy = self.generator_x
                    q_network = self.q_network_x
                else:
                    policy = self.generator_y
                    q_network = self.q_network_y
                
                # Get policy action
                policy_action = policy(state)
                
                # Get Q-value
                q_value = q_network(tf.concat([state, policy_action], axis=-1))
                
                # Compute policy gradient
                policy_gradients = tape.gradient(
                    -q_value,  # Negative because we want to maximize Q-value
                    policy.trainable_variables
                )
                
                # Accumulate gradients
                if domain == 'x':
                    g_x_gradients.append(policy_gradients)
                else:
                    g_y_gradients.append(policy_gradients)
        
        # Average gradients
        if g_x_gradients:
            avg_g_x_gradients = [
                tf.reduce_mean(grads, axis=0)
                for grads in zip(*g_x_gradients)
            ]
            self.g_optimizer.apply_gradients(
                zip(avg_g_x_gradients, self.generator_x.trainable_variables)
            )
            
        if g_y_gradients:
            avg_g_y_gradients = [
                tf.reduce_mean(grads, axis=0)
                for grads in zip(*g_y_gradients)
            ]
            self.g_optimizer.apply_gradients(
                zip(avg_g_y_gradients, self.generator_y.trainable_variables)
            ) 