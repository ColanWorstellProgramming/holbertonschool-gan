WandB Reports

NOTE :: All Logs are saved on collab, not locally, takes way too long to run locally

Baseline DCGAN

    Data
        https://api.wandb.ai/links/colan_worstell/ur0qgybo

    Notes
        Starting off with same changes from the first project.
            Batch Size: 256
            Epochs: 25
            Noise_dim: 100
            Kernal: 4
            Strides: 2
            Optimizer: adam
            Learning Rate: 1e-4

        I've had to make a LOT of tweaks to get the model working, regardless, I've managed to get some quite beautiful results.
        My model generates abstract art based on a dataset of about 3000 images, greyscale and colored. After about 10 epochs, unique
        images start to take shape with vibrant colors, there is a bit of weird coloration but that is expected.

        I can definently see the stride and kernal size effecting the image quality and the grid like output in the images, something
        I will change in future experiments, also I think I will double the epochs, back to 50 to see what I can generate with more time,
        as I am seeing better and better results as the epochs go.

        The Generator and Discriminator loss went as you would want them too, generator decreased as discriminator increased which is what
        we were wanting here, and the visual results validated the better quality of this model than the previous project in my opinion.


Architecture Variations

    Data


    Notes
        Batch Size: 256
        Epochs: 50
        Noise_dim: 100
        Kernal: 8
        Strides: 1
        Optimizer: adam
        Learning Rate: 1e-5

        My goal here was to change the stride to a smaller size so I have less square shapes and pixilation in the image, so I adjusted the archietecture.
        I've also decided on a smaller learning rate as well as a higher epoch count


Hyperparameter tuning

    Data


    Notes


Transfer of Knowledge

