import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from .blocks import ResidualBlock, ConvLSTMBlock

class ResidualBlock(layers.Layer):
    """Residual Prediction Block (RPB) for video processing"""
    def __init__(self, filters, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv3D(filters, kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv3D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        return tf.nn.relu(inputs + x)

class ConvLSTMBlock(layers.Layer):
    """ConvLSTM block for spatiotemporal feature extraction"""
    def __init__(self, filters, kernel_size=3):
        super(ConvLSTMBlock, self).__init__()
        self.conv_lstm = layers.ConvLSTM2D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            return_sequences=True
        )
        self.bn = layers.BatchNormalization()
        
    def call(self, inputs, training=False):
        x = self.conv_lstm(inputs)
        return self.bn(x, training=training)

class Generator(Model):
    """Generator network with ConvLSTM and encoder-decoder architecture"""
    def __init__(self, input_shape, output_channels=3):
        super(Generator, self).__init__()
        self.input_shape = input_shape
        
        # Encoder
        self.encoder_blocks = [
            ResidualBlock(64),
            ResidualBlock(128),
            ResidualBlock(256)
        ]
        
        # ConvLSTM for temporal modeling
        self.conv_lstm = ConvLSTMBlock(256)
        
        # Decoder
        self.decoder_blocks = [
            ResidualBlock(256),
            ResidualBlock(128),
            ResidualBlock(64)
        ]
        
        # Final output layer
        self.output_layer = layers.Conv3D(
            output_channels,
            kernel_size=3,
            padding='same',
            activation='tanh'
        )
        
    def call(self, inputs, training=False):
        x = inputs
        
        # Encoder path
        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block(x, training=training)
            encoder_outputs.append(x)
            
        # ConvLSTM processing
        x = self.conv_lstm(x, training=training)
        
        # Decoder path with skip connections
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, training=training)
            if i < len(encoder_outputs):
                x = x + encoder_outputs[-(i+1)]
                
        return self.output_layer(x)

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

class QNetwork(Model):
    """Q-network with same architecture as discriminator but ReLU output"""
    def __init__(self, input_shape):
        super(QNetwork, self).__init__()
        
        self.rpb_blocks = [
            ResidualBlock(64),
            ResidualBlock(128),
            ResidualBlock(256)
        ]
        
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(1, activation='relu')  # ReLU activation for Q-values
        
    def call(self, inputs, training=False):
        x = inputs
        for block in self.rpb_blocks:
            x = block(x, training=training)
            
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)

class Predictor(Model):
    """Predictor network with same architecture as generator"""
    def __init__(self, input_shape, output_channels=3):
        super(Predictor, self).__init__()
        self.generator = Generator(input_shape, output_channels)
        
    def call(self, inputs, training=False):
        return self.generator(inputs, training=training) 