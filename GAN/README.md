# File Overview

### createganmodels.py

This python file gives a couple functions to create GAN discriminator and GAN generator based on the tick size of the feature tensors. 

### ganutilities.py

This file contains many utility method for GAN training, data processing, and MIDI file creation. 

### epochanalysis.ipynb

This notebook walks through creating a couple graphs to analyze the discriminator performance (on the same random noise input) over the epochs 

### extractfeatures.ipynb

This notebook demos extracting features from the Groove dataset, and creating pickle files for training the GAN later.

### gantraining.ipynb

This notebook demos loading/creating GAN models and training them

### listenfrommodel.ipynb

This notebook demos loading a saved generator model, generating a sequence, and playing it with pygame. I recommed using epoch 11 of the bigepochs folder for decent output.

# Experiments
The only quantitative analysis I did was looking at the discriminator success rate over many epochs. I put these plots in the epochvisualizationplots folder.
The discriminator for the 3200 input fared much worse than the 1000 input one. It seemed like the discriminator won quickly in the tranining of the 1000
tick GAN and the generator never really came back, but the oscillations never stopped for the 3200 input GAN. I don't know if I can draw any conclusions from these graph specifically,
since GANs are so hard to train, but it might have to do with more complexity in 3200 tick inputs into the discriminator. 

The qualitiative experiments I did mostly revolved around messing with the way I extracted features and made the model layers. Here a few I did.

1. My first feature matrix was the shape (tick, velocity) -> velocity = 0 for no tone. This didn't work well because the model would never have a velocity of exactly zero,
since my output would always think there are very soft tones. 

2. I tried to cut off every rhythym after a certain amount of ticks (instead of splitting each one into equal sizes like I ended up doing). This led to lots of blank input that
messed with the model training.

3. I tried 10000 tick sequences, but it took far too long to process for the timeframe of the project.

4. I created a linear model at first, but it did not really function at all. Convolution is definitely the better approach since it allows you to capture patterns
across time and tones more easily. 

5. I attempted more convolution layers (5, instead of my current 3), but it again made computation way more complicated. 


# Other Important Information:
The output for both the 1000 tick and 3200 tick GANs were not super successful, but a few epochs can produce somewhat legible results. Epoch 11 for the 3200 tick GAN
gave me consistently ok output, and epoch 18 for 1000 tick GAN was my favorite. Overall, it is not clear if training the models for many more epochs would have helped,
but I think increasing the latent dimension size would also help. The fillsplit pickle files give the dataset for the 1000 and 3200 tick inputs. The epochs folder gives the saved
models for discriminators and generators by epoch for 1000 tick inputs, and the bigepochs folder does the same for 3200 tick inputs. 


