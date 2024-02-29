This project aims to illustrate sketches of Manga Characters.
The training set has sketches and their respective colored counterparts.


Objective: 

This repository's main aim is to compare qualitatively and quantitatively the suitability of using WGAN GP loss for the task of image to image translation
The performance is compared with the already existing Pix2Pix architecture which includes a PatchGAN.


The Pix2Pix implementation has been taken from the following github repository:

https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/Pix2Pix

A separate train function and Discriminator is defined for both the architectures.

For quantifying performance the FID metric is used.
