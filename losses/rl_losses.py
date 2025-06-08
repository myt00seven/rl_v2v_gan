import tensorflow as tf

def compute_reward(generated_video, target_video, discriminator):
    """
    Compute reward for RL training
    R = w1 * D(G(x)) + w2 * similarity(G(x), y)
    """
    # Discriminator confidence
    d_score = discriminator(generated_video)
    
    # Similarity reward (SSIM)
    ssim = tf.image.ssim(
        generated_video,
        target_video,
        max_val=1.0
    )
    
    # Combine rewards
    reward = 0.5 * d_score + 0.5 * ssim
    return reward

def policy_gradient_loss(q_values, actions, rewards):
    """
    Policy gradient loss for RL training
    L_policy = -E[Q(s,a) * log π(a|s)]
    """
    # Compute advantage
    advantage = rewards - tf.reduce_mean(q_values)
    
    # Policy gradient loss
    policy_loss = -tf.reduce_mean(
        advantage * tf.math.log(actions + 1e-10)
    )
    
    return policy_loss 