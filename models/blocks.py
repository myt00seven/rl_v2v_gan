import tensorflow as tf
from tensorflow.python.keras import layers

class ResidualBlock(layers.Layer):
    """Residual Prediction Block (RPB) for video processing"""
    def __init__(self, filters, kernel_size=3, name=None):
        super(ResidualBlock, self).__init__(name=name)
        with tf.variable_scope(name or 'residual_block'):
            self.conv1 = layers.Conv3D(filters, kernel_size, padding='same', name='conv1')
            self.bn1 = layers.BatchNormalization(name='bn1')
            self.conv2 = layers.Conv3D(filters, kernel_size, padding='same', name='conv2')
            self.bn2 = layers.BatchNormalization(name='bn2')
        
    def call(self, inputs, training=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.conv1(inputs)
            x = self.bn1(x, training=training)
            x = tf.nn.relu(x)
            x = self.conv2(x)
            x = self.bn2(x, training=training)
            return tf.nn.relu(inputs + x)

class ConvLSTMBlock(layers.Layer):
    """ConvLSTM block for spatiotemporal feature extraction"""
    def __init__(self, filters, kernel_size=3, name=None):
        super(ConvLSTMBlock, self).__init__(name=name)
        with tf.variable_scope(name or 'conv_lstm_block'):
            self.conv_lstm = layers.ConvLSTM2D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same',
                return_sequences=True,
                name='conv_lstm'
            )
            self.bn = layers.BatchNormalization(name='bn')
        
    def call(self, inputs, training=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            x = self.conv_lstm(inputs)
            return self.bn(x, training=training) 