# Concepts-and-Frameworks-in-Deep-Learning

This is an extensive implemetation of some concepts and frameworks in Deep Learning from scratch. It is based off the fifth assignment (aka mini-project) of the class 16720A Computer Vision at Carnegie Mellon University. 

It contains 

1. An in-depth implementation of a neural network model from scratch without the use of libraries. It is trained and tested on the NIST36 dataset.

2. The same network is then trained to recognize handwritten letters with reasonable accuracy, to parse text in an image. 

3. Image compression is then done using an auto-encoder. The PSNR Value obtained is then compared with value obtained by implementing Principal Component Analaysis

4. Training and testing suitable fully connected neural network models as well as CNNs on a number of datasets such as MNIST, EMNIST, NIST36.

5. In depth analysis of skip and residual connections. Demonstrated efficacy of each of these connections by deploying them in a neural network and testing them on the afore mentioned datasets.

6. Implemented a basic version of the following paper - “Resnet in Resnet: Generalizing Residual Architectures,”
S. Targ, D. Almeida, and K. Lyman,  ArXiv:1603.08029 [cs.LG], March 2016. It is tested on the EMNIST dataset.

7. Experiments in fine tuning have been performed by initializing models with weights obtained from another deep network that was trained for a diﬀerent purpose. 

The detailed set of questions for this assignment can also be found in this repository







