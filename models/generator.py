import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from .blocks import ResidualBlock, ConvLSTMBlock

class Generator(models.Model):
    """Generator network with ConvLSTM and encoder-decoder architecture"""
    def __init__(self, input_shape, output_channels=3, name=None):
        super(Generator, self).__init__(name=name)
        self.input_shape = input_shape
        
        with tf.variable_scope(name or 'generator'):
            # Encoder
            self.encoder_blocks = [
                ResidualBlock(64, name='encoder_1'),
                ResidualBlock(128, name='encoder_2'),
                ResidualBlock(256, name='encoder_3')
            ]
            
            # ConvLSTM for temporal modeling
            self.conv_lstm = ConvLSTMBlock(256, name='conv_lstm')
            
            # Decoder
            self.decoder_blocks = [
                ResidualBlock(256, name='decoder_1'),
                ResidualBlock(128, name='decoder_2'),
                ResidualBlock(64, name='decoder_3')
            ]
            
            # Final output layer
            self.output_layer = layers.Conv3D(
                output_channels,
                kernel_size=3,
                padding='same',
                activation='tanh',
                name='output'
            )
        
    def call(self, inputs, training=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
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

class Predictor(models.Model):
    """Predictor network with same architecture as generator"""
    def __init__(self, input_shape, output_channels=3, name=None):
        super(Predictor, self).__init__(name=name)
        with tf.variable_scope(name or 'predictor'):
            self.generator = Generator(input_shape, output_channels, name='generator')
        
    def call(self, inputs, training=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            return self.generator(inputs, training=training) 