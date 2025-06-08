import tensorflow as tf
from .dataset import create_synthetic_dataset

class DatasetFactory:
    """Factory class for creating different types of datasets"""
    
    @staticmethod
    def get_dataset(dataset_type, **kwargs):
        """
        Get dataset based on type
        Args:
            dataset_type: Type of dataset ('synthetic' for now)
            **kwargs: Additional arguments for dataset creation
        Returns:
            tf.data.Dataset
        """
        if dataset_type == 'synthetic':
            return create_synthetic_dataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    @staticmethod
    def prepare_dataset(dataset_type, **kwargs):
        """
        Prepare dataset based on type
        Args:
            dataset_type: Type of dataset ('synthetic' for now)
            **kwargs: Additional arguments for dataset preparation
        Returns:
            None (preparation is done in-place)
        """
        if dataset_type == 'synthetic':
            # Synthetic dataset doesn't need preparation
            print("Synthetic dataset is generated on-the-fly, no preparation needed.")
            return
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}") 