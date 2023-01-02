# Learning-with-memory-device-imperfections
Repository to reproduce the results of the paper "Model of the Weak Reset Process in HfOx Resistive Memory for Deep Learning Frameworks"

## Requirements
To set the environment run in your conda main environment:
```
conda config --add channels conda-forge  
conda create --name environment_name  
conda activate environment_name  
conda install pytorch==1.1.0 torchvision==0.3.0 -c pytorch
```

## Fully connected architecture for the MNIST task with all device simulations
```python main.py --gpu 0 -lr 100```

## Convolutional architecture for the CIFAR-10 task with all device simulations
```python main.py --gpu 0 -lr 100```
