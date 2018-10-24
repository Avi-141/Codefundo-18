									Project Toofan.
Team Members:

Dwaraknath,
Nipun,
Abhinav

Goal : To create a robust and upgraded prediction system to predict cyclones, and build a framework that eliminates possible misuse of opaqueness of the relief fund collection system.

UPDATE :

We found that the archived cyclone images we were able to find are noisy, and less coherent than what we would have liked. The application of VAE did not increase our chances of building the best model we could, so we went with Convolutional Autoencoder(CAE). Judging by pure reconstruction loss we have decided to continue with CAE as our model. The dataset is a time series of images taken by satellite over southern Indian region, we have used the trained encoder to encode all images into latent vector form, to create the encoded image matrix. We then use Multivariate Time series forecasting using LSTM to forecast the encoded form of t+1 image. We then proceed to decode using the trined Convolutional Decoder to decode the forecast and reconstruct future cloud pattern, and overlay it with input image to give a sense of direction to the clouds. 

We have also created an interactive T-SNE demo which, given a collection of normal and cyclone images, uses dimensionality reduction to plot the high dimension data into 3D space. This allows us to draw useful inferences about clusters and classification decision boundaries between recent cyclones. We quantify the "closeness" between cyclones using T-SNE. We can also use this trick to analyse how similar or different a new cyclone is to previous ones. 

Abstract :

PHASE 1 : Deep learning has, in recent times,garnered enormous attention from Academia, Industry, Government and media for it's rapid success and developement in the fields of computer vision, etc. Neural networks are considered amazing functional approximators. 
Cyclones are an inherently complicated, stochastic, and noisy phenomenon. Predicting cyclones has been a problem of formidable complexity since it was sought-after. Though current techniques may involve the use of  Deep or machine learning, to predict cyclones using recorded data, our idea is to model cyclones as probabilistic phenomenon. The inherent chaos and stochasticity of cyclones makes it difficult for models to properly understand the underlying probability distribution of the data to develop a plausible model. We propose the use of Variational Autoencoders(VAE) for this purpose. VAE's have been proved to be an excellent repoitre of models to learn the underlying structure of data, and doing so well enough to generate realistic looking images from sampled points from the distribution. We use a mixture of convolutional VAE as powerful feature extractors to help us classify satellite image data into cyclones or not. The problem of predicting the path is taken care by another Generative model called Generative Adversarial Networks(GAN) to reconstruct traced data. This would enable to make the image dataset obtain from INSAT satellite, a time series. A prediction can be made on how the state of weather is going to look like, or the trajectory of cyclone by using recurrent GAN to "look into the future" by rendering image at t+1 time step. 
We also propose the use T-SNE  algorithm to properly visualise and infer charecteristics from the cyclone dataset. 


PHASE 2: This phase of our project toys with the idea of using decentralised Blockchain technology to  create and maintain a decentralised ledger of transactions that happen during relief fund transfer. We want to increase transparency while, providing anonymity. A publicly maintained ledger is important to manage the funds that have been collected to be used only for relief purposes. 

