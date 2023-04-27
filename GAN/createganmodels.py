import tensorflow as tf
from tensorflow.keras import layers


# These functions creates new generator/discriminator based on the size of the matrices in input ticks

def new_gan_generator(input_ticks):
    generator = tf.keras.Sequential()
    
    # Dense layer takes input noise and gives it enough dimensions to do transpose convolution
    generator.add(layers.Dense((input_ticks // 4) * 11 * 256, use_bias=False, input_shape=(100,)))
    
    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())
    
    # Reshapes to 3 axis
    generator.add(layers.Reshape(((input_ticks // 4), 11, 256)))
    # (ticks/4, 11, 256)

    # Changes to 128 channels
    generator.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # (ticks/4, 11, 128)

    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())

    # Changes to 64 channels, and stride is 2 for axis 1
    generator.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 1), padding='same', use_bias=False))
    # (ticks/2, 11, 64)

    generator.add(layers.BatchNormalization())
    generator.add(layers.LeakyReLU())

    # Changes to final 2 channels, and stride is 2 for both first 2 axis
    generator.add(layers.Conv2DTranspose(2, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # (ticks, 22, 2)
    
    return generator

def new_gan_discriminator(input_ticks):
    discriminator = tf.keras.Sequential()
    
    # Changes to 64 channels
    discriminator.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[input_ticks, 22, 2]))
    discriminator.add(layers.LeakyReLU())
    discriminator.add(layers.Dropout(0.3))

    # Changes to 128 channels
    discriminator.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    discriminator.add(layers.LeakyReLU())
    discriminator.add(layers.Dropout(0.3))

    # Flattens and linearizes to only one dimension for prediction
    discriminator.add(layers.Flatten())
    discriminator.add(layers.Dense(1))
    
    return discriminator

