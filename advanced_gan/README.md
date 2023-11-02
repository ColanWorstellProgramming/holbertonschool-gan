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
        https://api.wandb.ai/links/colan_worstell/8z4f97j4

    Notes
        Batch Size: 256
        Epochs: 50
        Noise_dim: 100
        Kernal: 8
        Strides: 1-2
        Optimizer: adam
        Learning Rate: 1e-5

        My goal here was to change the structure to use a bigger filter size so I adjusted the strides to 1 to compensate for the shapes.
        I've also decided on a smaller learning rate as well as a higher epoch count.

        The results look much better, I managed in one run to generate the shape of a person but wasn't able to save the results due to an error
        in my wandb code. A seperate run produced an interesting shadowy image. This run definently worked much better and had clearer image quality.


Hyperparameter tuning

    Data
        https://api.wandb.ai/links/colan_worstell/fv7t9zc5

    Notes
        Batch Size: 64
        Epochs: 100
        Noise_dim: 50
        Kernal: 4
        Strides: 2
        Optimizer: adam
        Learning Rate: 1e-4

        I started with the Architecture Variations experiment as a base since it preformed better. This experement was extremely interesting.
        Instead of the model following a specific path, it generated very randomly different images, quite a few of them looked like real abstract images,
        some of them were just colors, so it seemed to work better in some cases while being pretty jumpy. I think the smaller batch size is the cause for this
        and potentially the learning rate from what I can tell. I also think the kernal size has something to do with it. I'm happy I ran this experiment as it was
        pretty interesting to see big changes in my output. The images all were very "gridy" as well, I think because I used a much larger kernal size.

        Generator and discrimenator loss did very well as well and did as expected.



Mixed Experiment - For Fun Not Required

    Data


    Notes
        Batch Size: 128
        Epochs: 100
        Noise_dim: 100
        Kernal: 8
        Strides: 1-2
        Optimizer: adam
        Learning Rate: 1e-4

        I wanted to combine all three experiments to see if I could get pretty results!


Transfer of Knowledge

    The DCGAN MNIST was a good way to get into understanding gans models. I had a base model that I could just do small edits to to see how they effect the simple number
    images we were working with. Now, on my more complicated abstract art model, I had good ideas of what to tweak to get the specific results that I wanted. Which actually came
    into play, and I did try some of what worked out during the first project. I decided to use a similar archetecture, I feel like I liked using more layers than less as I saw
    better results. I didn't have the best results from my initial experiments, but my baseline was good. So I knew to start with that and what not to do from after that point on because
    of my other expeirments. It was nice to have a fall back while working on the advancded task to a simpler model.