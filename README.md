Deep Learning Experiments with Keras & TensorFlow 
=
This repository contains a collection of my experiments with deep learning using Keras and TensorFlow. The focus is on exploring various neural network architectures and techniques for tasks like:

- **Recurrent Neural Networks (RNNs):**
  - **rnn directory:** Different RNN architectures (LSTM, GRU) for tasks like text generation and mood classification. This directory also includes experiments with word embedding techniques to represent text data as dense vectors
  - **Datasets:**
    - **train_data.txt:** Contains a collection of positive sentences.
    - **train_data_bad.txt:** Contains a collection of negative sentences. 
- **Image Colorization:**
  - **colorization.py:** Experimenting with convolutional neural networks to colorize grayscale images.
- **Style Transfer:**
  - **styletransfer.py:** Implementing neural style transfer to apply the artistic style of one image to another.
- **Generative Adversarial Networks (GANs):**
  - **gan\_with\_vae.py:** Implementing a GAN with Variational Autoencoder (VAE).
- **Dropout and Batch Normalization:**
  - This repository also includes experiments exploring the effects of dropout and batch normalization techniques on model performance.

**Getting Started:**

1. Clone this repository:
```Bash
git clone https://github.com/mateusxap/Keras-Tensorflow.git
```
2. Install the necessary libraries:
```Bash
pip install tensorflow keras
```
3. Explore the different scripts and run the experiments.

**Disclaimer:**

This repository is meant for personal exploration and learning. The code is provided as-is, and some experiments might be in early stages of development.

