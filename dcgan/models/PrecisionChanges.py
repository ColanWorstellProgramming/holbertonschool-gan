#!/usr/bin/env python3
"""
Imports
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
import wandb
import os

# Set up WandB for experiment tracking
wandb.init(project='gans', entity='colan_worstell', dir='../logs/')

# Define a main function
def main():
    """
    Main Function
    """

    # Load and preprocess the dataset
    (train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float16")
    train_images = (train_images - 127.5) / 127.5
    BUFFER_SIZE = 60000
    BATCH_SIZE = 64

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(100,)))
        model.add(layers.Dense(7 * 7 * 256, use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Reshape((7, 7, 256)))

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

        return model

    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(28, 28, 1)))
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    generator = make_generator_model()
    discriminator = make_discriminator_model()

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Set up WandB for experiment tracking
    wandb.run.name = 'DCGAN_Experiment'

    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

    def train(dataset, epochs):
        for epoch in range(epochs):
            start = time.time()
            gen_losses, disc_losses = [], []

            for image_batch in dataset:
                gen_loss, disc_loss = train_step(image_batch)
                gen_losses.append(gen_loss)
                disc_losses.append(disc_loss)

            avg_gen_loss = sum(gen_losses) / len(gen_losses)
            avg_disc_loss = sum(disc_losses) / len(disc_losses)

            print('Epoch {}: Generator Loss: {}, Discriminator Loss: {}, Time: {}'.format(epoch + 1, avg_gen_loss, avg_disc_loss, time.time() - start))

            # Produce images for WandB
            generate_and_log_images(generator, epoch + 1, seed)

            # Log generator and discriminator loss to WandB
            wandb.log({"generator_loss": gen_loss, "discriminator_loss": disc_loss, "Epoch ": epoch + 1, "Time ": time.time() - start})


    # Set up the folder for saving images
    log_folder = "../logs"  # Adjust this to your preferred log folder
    custom_folder_name = get_experiment_folder(log_folder)

    # Create the custom folder if it doesn't exist
    os.makedirs(custom_folder_name, exist_ok=True)

    # Modify the save path in the generate_and_log_images function
    def generate_and_log_images(model, epoch, test_input):
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        save_path = os.path.join(custom_folder_name, f'image_at_epoch_{epoch:04d}.png')
        plt.savefig(save_path)

        # Log images to WandB
        wandb.log({"generated_images": [wandb.Image(plt)]})

    train(train_dataset, EPOCHS)

def get_experiment_folder(base_folder, prefix="Experiment"):
    experiment_folder = f"{prefix}_0"
    index = 0
    while os.path.exists(os.path.join(base_folder, experiment_folder)):
        index += 1
        experiment_folder = f"{prefix}_{index}"
    return os.path.join(base_folder, experiment_folder)

if __name__ == "__main__":
    main()
