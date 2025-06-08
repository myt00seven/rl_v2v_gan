import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

def compute_fid(real_features, fake_features):
    """
    Compute Fréchet Inception Distance (FID)
    Args:
        real_features: Features from real videos
        fake_features: Features from generated videos
    Returns:
        FID score
    """
    # Compute mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Compute FID
    ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
    covmean = np.sqrt(sigma_real.dot(sigma_fake))
    
    fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return fid

def compute_psnr(real_video, fake_video):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR)
    Args:
        real_video: Real video frames
        fake_video: Generated video frames
    Returns:
        PSNR score
    """
    mse = tf.reduce_mean(tf.square(real_video - fake_video))
    psnr = 20 * tf.math.log(1.0 / tf.sqrt(mse)) / tf.math.log(10.0)
    return psnr

def compute_ssim(real_video, fake_video):
    """
    Compute Structural Similarity Index (SSIM)
    Args:
        real_video: Real video frames
        fake_video: Generated video frames
    Returns:
        SSIM score
    """
    return tf.reduce_mean(
        tf.image.ssim(
            real_video,
            fake_video,
            max_val=1.0
        )
    )

def extract_features(video_batch):
    """
    Extract features using InceptionV3
    Args:
        video_batch: Batch of video frames
    Returns:
        Extracted features
    """
    # Initialize InceptionV3
    inception = InceptionV3(
        include_top=False,
        pooling='avg',
        input_shape=(299, 299, 3)
    )
    
    # Preprocess frames
    frames = tf.image.resize(video_batch, (299, 299))
    frames = preprocess_input(frames)
    
    # Extract features
    features = inception(frames)
    return features 