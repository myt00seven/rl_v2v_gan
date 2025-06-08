import tensorflow as tf

def adversarial_loss(real_logits, fake_logits):
    """
    Adversarial loss for GAN training
    L_D = -E[log(D(x))] - E[log(1-D(G(z)))]
    L_G = -E[log(D(G(z)))]
    """
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real_logits),
            logits=real_logits
        )
    )
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake_logits),
            logits=fake_logits
        )
    )
    d_loss = d_loss_real + d_loss_fake
    
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake_logits),
            logits=fake_logits
        )
    )
    
    return d_loss, g_loss

def recurrent_loss(pred_sequence, target_sequence):
    """
    Recurrent loss for temporal consistency
    L_rec = ||P(y) - y||_1
    """
    return tf.reduce_mean(tf.abs(pred_sequence - target_sequence))

def recycle_loss(generated_video, target_video):
    """
    Recycle loss for cycle consistency
    L_cyc = ||G(G(x)) - x||_1
    """
    return tf.reduce_mean(tf.abs(generated_video - target_video))

def video_loss(generated_video, target_video, discriminator_x, discriminator_y, is_x_to_y=True):
    """
    Video loss using the model's discriminator to ensure temporal consistency
    L_v = L_vx(G_x, G_y, D_x) + L_vy(G_y, G_x, D_y)
    
    Args:
        generated_video: Generated video sequence
        target_video: Target video sequence
        discriminator_x: Discriminator for domain X
        discriminator_y: Discriminator for domain Y
        is_x_to_y: Boolean indicating if this is X->Y direction (True) or Y->X (False)
    """
    if is_x_to_y:
        # L_vx(G_x, G_y, D_x)
        # Real video loss
        real_logits = discriminator_x(target_video)
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_logits),
                logits=real_logits
            )
        )
        
        # Generated video loss
        fake_logits = discriminator_x(generated_video)
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_logits),
                logits=fake_logits
            )
        )
        
        return real_loss + fake_loss
    else:
        # L_vy(G_y, G_x, D_y)
        # Real video loss (both D_y,0 and D_y,1)
        real_logits_0 = discriminator_y(target_video, training=True, output_type=0)
        real_logits_1 = discriminator_y(target_video, training=True, output_type=1)
        
        real_loss_0 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_logits_0),
                logits=real_logits_0
            )
        )
        real_loss_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_logits_1),
                logits=real_logits_1
            )
        )
        
        # Generated video loss (both D_y,0 and D_y,1)
        fake_logits_0 = discriminator_y(generated_video, training=True, output_type=0)
        fake_logits_1 = discriminator_y(generated_video, training=True, output_type=1)
        
        fake_loss_0 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_logits_0),
                logits=fake_logits_0
            )
        )
        fake_loss_1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_logits_1),
                logits=fake_logits_1
            )
        )
        
        return real_loss_0 + real_loss_1 + fake_loss_0 + fake_loss_1 