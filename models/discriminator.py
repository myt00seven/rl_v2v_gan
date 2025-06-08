import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from .blocks import ResidualBlock

class Discriminator(Model):
    """Discriminator network with RPB layers and FC output"""
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        
        self.rpb_blocks = [
            ResidualBlock(64),
            ResidualBlock(128),
            ResidualBlock(256)
        ]
        
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x = inputs
        for block in self.rpb_blocks:
            x = block(x, training=training)
            
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x) 