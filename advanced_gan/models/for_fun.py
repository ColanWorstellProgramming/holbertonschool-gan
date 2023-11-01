#!/usr/bin/env python3
"""
Imports
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
import wandb
import os
from PIL import Image

# Set up WandB for experiment tracking
wandb.init(project='gans', entity='colan_worstell', dir='../logs/')

# Define a main function
def main():
    """
    Main Function
    """

    path_arts = []
    train_path_arts = '/content/drive/My Drive/imgs/'
    for path in os.listdir(train_path_arts):
        if '.jpg' in path:
            path_arts.append(os.path.join(train_path_arts, path))

    new_path=path_arts

    images = [np.array((Image.open(path)).resize((128,128))) for path in new_path]

    for i in range(len(images)):
        images[i] = ((images[i] - images[i].min())/(255 - images[i].min()))

    images = np.array(images)

    train_data=images

    BUFFER_SIZE = 60000
    BATCH_SIZE = 128
    EPOCHS = 100
    noise_dim = 50
    num_examples_to_generate = 16

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    def make_generator_model():
        model = tf.keras.Sequential()

        model.add(layers.Dense(4*4*512,input_shape=[noise_dim]))
        model.add(layers.Reshape([4,4,512]))
        model.add(layers.Conv2DTranspose(2048, kernel_size=4, strides=2, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2DTranspose(1024, kernel_size=4, strides=2, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same",
                                        activation='sigmoid'))

        return model

    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(32, kernel_size=4, strides=2, padding="same",input_shape=[128,128, 3]))
        model.add(layers.Conv2D(64, kernel_size=4, strides=2, padding="same"))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1,activation='sigmoid'))

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

    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Set up WandB for experiment tracking
    wandb.run.name = 'Advanced_DCGAN_Experiment_For_Fun'

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
            generate_and_log_images(generator, seed)

            # Log generator and discriminator loss to WandB
            wandb.log({"generator_loss": avg_gen_loss, "discriminator_loss": avg_disc_loss, "Epoch ": epoch + 1, "Time ": time.time() - start})


    # Set up the folder for saving images
    log_folder = "../logs"  # Adjust this to your preferred log folder
    custom_folder_name = get_experiment_folder(log_folder)

    # Create the custom folder if it doesn't exist
    os.makedirs(custom_folder_name, exist_ok=True)

    # Modify the save path in the generate_and_log_images function
    def generate_and_log_images(model, test_input):
        predictions = model(test_input, training=False)

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, :])
            plt.axis('off')

        plt.savefig("generated_image.png")

        # Log images to WandB
        wandb.log({"generated_images": [wandb.Image("generated_image.png")]})


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
