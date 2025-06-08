import tensorflow as tf
import numpy as np
import cv2

class SyntheticVideoDataset:
    """Generate synthetic video datasets for testing RL-V2V-GAN"""
    
    def __init__(
        self,
        batch_size=4,
        seq_length=16,
        height=64,
        width=64,
        channels=3
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.height = height
        self.width = width
        self.channels = channels
        
    def _create_moving_shape(self, frame_idx, shape_type='circle'):
        """Create a single frame with a moving shape"""
        frame = np.zeros((self.height, self.width, self.channels))
        
        # Calculate position based on frame index
        center_x = int(self.width * (0.3 + 0.4 * np.sin(frame_idx * 0.2)))
        center_y = int(self.height * (0.3 + 0.4 * np.cos(frame_idx * 0.2)))
        
        # Create shape
        if shape_type == 'circle':
            cv2.circle(
                frame,
                (center_x, center_y),
                radius=10,
                color=(1.0, 0.5, 0.0),
                thickness=-1
            )
        elif shape_type == 'rectangle':
            cv2.rectangle(
                frame,
                (center_x-10, center_y-10),
                (center_x+10, center_y+10),
                color=(0.0, 0.5, 1.0),
                thickness=-1
            )
            
        return frame
    
    def _create_color_transition(self, frame_idx):
        """Create a frame with color transition"""
        frame = np.zeros((self.height, self.width, self.channels))
        
        # Create color gradient
        for c in range(self.channels):
            frame[:, :, c] = 0.5 + 0.5 * np.sin(
                frame_idx * 0.1 + c * np.pi/3
            )
            
        return frame
    
    def generate_source_video(self):
        """Generate source video with moving shapes"""
        video = np.zeros((self.seq_length, self.height, self.width, self.channels))
        
        for t in range(self.seq_length):
            if t % 2 == 0:
                video[t] = self._create_moving_shape(t, 'circle')
            else:
                video[t] = self._create_moving_shape(t, 'rectangle')
                
        return video
    
    def generate_target_video(self):
        """Generate target video with color transitions"""
        video = np.zeros((self.seq_length, self.height, self.width, self.channels))
        
        for t in range(self.seq_length):
            video[t] = self._create_color_transition(t)
            
        return video
    
    def generate_style_video(self):
        """Generate style video with different patterns"""
        video = np.zeros((self.seq_length, self.height, self.width, self.channels))
        
        for t in range(self.seq_length):
            # Create checkerboard pattern
            pattern = np.zeros((self.height, self.width))
            block_size = 8
            for i in range(0, self.height, block_size):
                for j in range(0, self.width, block_size):
                    if (i//block_size + j//block_size) % 2 == 0:
                        pattern[i:i+block_size, j:j+block_size] = 1.0
                        
            # Add color
            for c in range(self.channels):
                video[t, :, :, c] = pattern * (0.5 + 0.5 * np.sin(t * 0.1 + c * np.pi/3))
                
        return video
    
    def generate_style_image(self):
        """Generate a single style image"""
        image = np.zeros((self.height, self.width, self.channels))
        
        # Create radial gradient
        center_x, center_y = self.width // 2, self.height // 2
        for i in range(self.height):
            for j in range(self.width):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                for c in range(self.channels):
                    image[i, j, c] = 0.5 + 0.5 * np.sin(dist * 0.1 + c * np.pi/3)
                    
        return image
    
    def create_dataset(self):
        """Create synthetic video dataset with optimized batch processing"""
        # Generate synthetic data
        x_videos = self._generate_synthetic_videos()
        y_videos = self._generate_synthetic_videos()
        z_videos = self._generate_synthetic_videos()
        z_images = self._generate_synthetic_images()
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            x_videos, y_videos, z_videos, z_images
        ))
        
        # Shuffle and batch
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(
            self.batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        
        return dataset
        
    def _generate_synthetic_videos(self):
        """Generate synthetic video sequences efficiently"""
        # Pre-allocate numpy array
        videos = np.zeros(
            (1000, self.seq_length, self.height, self.width, self.channels),
            dtype=np.float32
        )
        
        # Generate videos in parallel
        for i in range(1000):
            videos[i] = self._generate_single_video()
            
        return videos
        
    def _generate_synthetic_images(self):
        """Generate synthetic images efficiently"""
        # Pre-allocate numpy array
        images = np.zeros(
            (1000, self.height, self.width, self.channels),
            dtype=np.float32
        )
        
        # Generate images in parallel
        for i in range(1000):
            images[i] = self._generate_single_image()
            
        return images
        
    def _generate_single_video(self):
        """Generate a single synthetic video sequence"""
        video = np.random.rand(
            self.seq_length,
            self.height,
            self.width,
            self.channels
        ).astype(np.float32)
        
        # Normalize to [-1, 1]
        video = video * 2 - 1
        return video
        
    def _generate_single_image(self):
        """Generate a single synthetic image"""
        image = np.random.rand(
            self.height,
            self.width,
            self.channels
        ).astype(np.float32)
        
        # Normalize to [-1, 1]
        image = image * 2 - 1
        return image

def create_synthetic_dataset(
    batch_size=4,
    seq_length=16,
    height=64,
    width=64,
    channels=3
):
    """Helper function to create synthetic dataset with optimized batch processing"""
    dataset_generator = SyntheticVideoDataset(
        batch_size=batch_size,
        seq_length=seq_length,
        height=height,
        width=width,
        channels=channels
    )
    
    # Create dataset with optimized settings
    dataset = dataset_generator.create_dataset()
    
    # Optimize dataset performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    
    # Enable parallel processing
    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)
    
    return dataset 