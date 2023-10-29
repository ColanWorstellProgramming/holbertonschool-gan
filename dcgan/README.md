WandB Reports

Baseline DCGAN

    Data
        https://api.wandb.ai/links/colan_worstell/11j3bwwi

    Notes
        Worked relatively okay, started with some pretty fuzzy images but about halfway through its epochs it was creating some pretty clear images.
        I do feel as if the image quality peaked pretty quickly probably about 40-50% of the way through training and I saw no real improvements.
        My thoughts on this are what can I do to improve the quality throughout changing and will implement some ideas in the next expeirement.

        Training increased drastically near the end, however that is because I trained locally on a 3080ti GPU and had opened other applications on my PC.
        Generator and Discriminator loss was very speradic.

Architecture Variations

    Data
        https://api.wandb.ai/links/colan_worstell/ef2xfdyz

    Notes
        Changed the structure, added layers to both the generator and discriminator model. I also decreased the filter and slider sizes.

        Preformed worse than baseline DCGAN, I believe the slide size was too small, it actually created some very interesting effects in some of the early and later images.
        It almost created some tiling and white squares which I thought was pretty interesting and makes sense when you take into account the slider size.

        At about the 30 epoch timing training time dropped drastically, I feel as if this is because the model wasn't learning anything new because of the sliding window size.

Hyperparameter tuning

    Data
        https://api.wandb.ai/links/colan_worstell/16hm0pdi

    Notes

        Started off of Baseline DCGAN because it preformed better.

        Changed the batch size to 32 from 64.
        Changed the optimizer to RMSprop from Adam
        Changed the learning rate to 5e-4 from 1e-4

        Initial observations show that the new optimizer and learning rate is really struggling more than the other models.
        Generator loss is preforming very well however discriminator loss is pretty jumpy within the first 10 epochs.
        Training time stays at around 65-70 seconds.

        Training never got much better, some numbers came out a bit clearer but it definetly preformed much worse.

Precision Changes

    Data


    Notes

        Switching from Float32 to Float16
        Starting with Baseline parameters as it has preformed best visually so far.

        Preforming on par with base model at around epoch 8 if not a bit better.
        Preformance at end looks much better and I am seeing constant improvement throughout all 25 epochs

        Image quailty is much better at the end
        Training time stays consistant throughout training