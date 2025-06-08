import tensorflow as tf
import numpy as np
from .metrics import compute_fid, compute_psnr, compute_ssim, extract_features

class VideoEvaluator:
    """Class for evaluating video generation quality"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.feature_extractor = None
        
    def _get_feature_extractor(self):
        """Lazy initialization of feature extractor"""
        if self.feature_extractor is None:
            self.feature_extractor = tf.keras.applications.InceptionV3(
                include_top=False,
                pooling='avg',
                input_shape=(299, 299, 3)
            )
        return self.feature_extractor
        
    def evaluate_videos(self, real_videos, fake_videos):
        """
        Evaluate generated videos against real videos
        Args:
            real_videos: Batch of real videos
            fake_videos: Batch of generated videos
        Returns:
            Dictionary of metrics (FID, PSNR, SSIM)
        """
        # Extract features
        real_features = extract_features(real_videos)
        fake_features = extract_features(fake_videos)
        
        # Compute metrics
        fid = compute_fid(real_features, fake_features)
        psnr = compute_psnr(real_videos, fake_videos)
        ssim = compute_ssim(real_videos, fake_videos)
        
        return {
            'fid': fid,
            'psnr': psnr,
            'ssim': ssim
        }
        
    def evaluate_model(self, model, test_dataset, num_batches=10):
        """
        Evaluate model on test dataset
        Args:
            model: RL_V2V_GAN_Trainer instance
            test_dataset: tf.data.Dataset containing test data
            num_batches: number of batches to evaluate
        Returns:
            Dictionary containing average metrics
        """
        metrics = []
        
        with tf.Session() as sess:
            for i, (x_batch, y_batch, z_batch, z_images) in enumerate(test_dataset):
                if i >= num_batches:
                    break
                    
                # Generate videos
                fake_y = sess.run(model.generator_y(model.generator_x(x_batch)))
                fake_x = sess.run(model.generator_x(model.generator_y(y_batch)))
                
                # Compute metrics
                batch_metrics = self.evaluate_videos(y_batch, fake_y)
                metrics.append(batch_metrics)
        
        # Average metrics across batches
        avg_metrics = {
            'fid': np.mean([m['fid'] for m in metrics]),
            'psnr': np.mean([m['psnr'] for m in metrics]),
            'ssim': np.mean([m['ssim'] for m in metrics])
        }
        
        return avg_metrics 