import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import InceptionV3
from scipy import linalg
import tensorflow.keras.backend as K

class VideoEvaluator:
    """Evaluator for video generation quality metrics"""
    
    def __init__(self):
        # Load Inception model for FID
        self.inception_model = InceptionV3(
            include_top=False,
            pooling='avg',
            input_shape=(299, 299, 3)
        )
        
    def _preprocess_for_inception(self, video):
        """Preprocess video frames for Inception model"""
        # Resize frames to 299x299
        frames = tf.image.resize_images(video, (299, 299))
        # Convert to RGB if needed
        if frames.shape[-1] == 1:
            frames = tf.tile(frames, [1, 1, 1, 3])
        # Scale to [-1, 1]
        frames = (frames + 1) / 2
        return frames
    
    def _get_inception_features(self, video):
        """Extract features using Inception model"""
        frames = self._preprocess_for_inception(video)
        features = self.inception_model(frames)
        return features
    
    def compute_fid(self, real_videos, generated_videos):
        """
        Compute Fréchet Inception Distance between real and generated videos
        Args:
            real_videos: tensor of shape (batch, seq_len, height, width, channels)
            generated_videos: tensor of shape (batch, seq_len, height, width, channels)
        Returns:
            FID score
        """
        # Extract features
        real_features = self._get_inception_features(real_videos)
        gen_features = self._get_inception_features(generated_videos)
        
        # Calculate mean and covariance
        mu_real = tf.reduce_mean(real_features, axis=0)
        mu_gen = tf.reduce_mean(gen_features, axis=0)
        
        # Calculate covariance matrices
        sigma_real = tf.matmul(
            tf.transpose(real_features - mu_real),
            real_features - mu_real
        ) / tf.cast(tf.shape(real_features)[0], tf.float32)
        
        sigma_gen = tf.matmul(
            tf.transpose(gen_features - mu_gen),
            gen_features - mu_gen
        ) / tf.cast(tf.shape(gen_features)[0], tf.float32)
        
        # Calculate FID
        ssdiff = tf.reduce_sum(tf.square(mu_real - mu_gen))
        
        # Compute matrix square root
        covmean = tf.py_func(
            lambda x: linalg.sqrtm(x),
            [tf.matmul(sigma_real, sigma_gen)],
            tf.float32
        )
        
        if tf.reduce_any(tf.is_nan(covmean)):
            covmean = tf.zeros_like(covmean)
            
        fid = ssdiff + tf.trace(sigma_real + sigma_gen - 2 * covmean)
        return fid
    
    def compute_psnr(self, real_videos, generated_videos):
        """
        Compute Peak Signal-to-Noise Ratio
        Args:
            real_videos: tensor of shape (batch, seq_len, height, width, channels)
            generated_videos: tensor of shape (batch, seq_len, height, width, channels)
        Returns:
            PSNR score
        """
        # Scale to [0, 1]
        real = (real_videos + 1) / 2
        gen = (generated_videos + 1) / 2
        
        # Calculate MSE
        mse = tf.reduce_mean(tf.square(real - gen))
        
        # Calculate PSNR
        psnr = 20 * tf.log(1.0) / tf.log(10.0) - 10 * tf.log(mse) / tf.log(10.0)
        return psnr
    
    def compute_ssim(self, real_videos, generated_videos):
        """
        Compute Structural Similarity Index
        Args:
            real_videos: tensor of shape (batch, seq_len, height, width, channels)
            generated_videos: tensor of shape (batch, seq_len, height, width, channels)
        Returns:
            SSIM score
        """
        # Scale to [0, 1]
        real = (real_videos + 1) / 2
        gen = (generated_videos + 1) / 2
        
        # Calculate SSIM for each frame
        ssim_scores = []
        for t in range(real.shape[1]):
            frame_real = real[:, t]
            frame_gen = gen[:, t]
            ssim = tf.image.ssim(frame_real, frame_gen, 1.0)
            ssim_scores.append(ssim)
            
        # Average SSIM across frames
        return tf.reduce_mean(ssim_scores)
    
    def evaluate_videos(self, real_videos, generated_videos):
        """
        Compute all evaluation metrics
        Args:
            real_videos: tensor of shape (batch, seq_len, height, width, channels)
            generated_videos: tensor of shape (batch, seq_len, height, width, channels)
        Returns:
            Dictionary containing FID, PSNR, and SSIM scores
        """
        with tf.Session() as sess:
            fid = sess.run(self.compute_fid(real_videos, generated_videos))
            psnr = sess.run(self.compute_psnr(real_videos, generated_videos))
            ssim = sess.run(self.compute_ssim(real_videos, generated_videos))
        
        return {
            'fid': fid,
            'psnr': psnr,
            'ssim': ssim
        }

def evaluate_model(model, test_dataset, num_batches=10):
    """
    Evaluate model on test dataset
    Args:
        model: RL_V2V_GAN_Trainer instance
        test_dataset: tf.data.Dataset containing test data
        num_batches: number of batches to evaluate
    Returns:
        Dictionary containing average metrics
    """
    evaluator = VideoEvaluator()
    metrics = []
    
    with tf.Session() as sess:
        for i, (x_batch, y_batch, z_batch, z_images) in enumerate(test_dataset):
            if i >= num_batches:
                break
                
            # Generate videos
            fake_y = sess.run(model.generator_y(model.generator_x(x_batch)))
            fake_x = sess.run(model.generator_x(model.generator_y(y_batch)))
            
            # Compute metrics
            batch_metrics = evaluator.evaluate_videos(y_batch, fake_y)
            metrics.append(batch_metrics)
    
    # Average metrics across batches
    avg_metrics = {
        'fid': np.mean([m['fid'] for m in metrics]),
        'psnr': np.mean([m['psnr'] for m in metrics]),
        'ssim': np.mean([m['ssim'] for m in metrics])
    }
    
    return avg_metrics 