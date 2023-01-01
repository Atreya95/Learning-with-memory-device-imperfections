# Learning-with-memory-device-imperfections
Repository to reproduce the results of the paper "Model of the Weak Reset Process in HfOx Resistive Memory for Deep Learning Frameworks"

To run the fully connected network for the MNIST task run the following script with the MLP files:

python main.py --gpu 0 -lr 100

To run the convolutional network for the CIFAR-10 task run the following script with the ConvNet files:

python main.py --gpu 0 -lr 100
