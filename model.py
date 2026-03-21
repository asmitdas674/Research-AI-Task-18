import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    # CNN model: Input > Conv Layer 1 > ReLU > Max Pool > Conv Layer 2
    # > ReLU > Max Pool > FC > FC

    # Why this model architecture?
    # - Two convolutional layers to learn hierarchical features from the images
    # - Max pooling layers to reduce spatial dimensions and computational complexity
    # - Fully connected layers to perform classification based on the extracted features 

    def __init__(self):
        super(CNN, self).__init__() # Call the parent class constructor

        # First conv layer
        self.conv1= nn.Conv2d(in_channels= 1, out_channels= 32, kernel_size= 3,
                              stride= 1, padding= 1) # Output: 32 x 28 x 28
        
        # Why did I choose such parameters for conv1?
        # - in_channels=1 because MNIST images are grayscale (1 channel)
        # - out_channels=32 to learn 32 different features
        # - kernel_size=3 for a 3x3 filter
        # - stride=1 to move the filter one pixel at a time
        # - padding=1 to maintain the spatial dimensions (28x28) after convolution


        # Second conv layer
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 64,
        kernel_size= 3, stride= 1, padding= 1) # Output: 64 x 14 x 14

        # Why did I choose such parameters for conv2?
        # - in_channels=32 because it takes the output of conv1 as input
        # - out_channels=64 to learn more complex features
        # - kernel_size=3 for a 3x3 filter
        # - stride=1 to move the filter one pixel at a time
        # - padding=1 to maintain the spatial dimensions (14x14) after convolution

        # Max pooling layer
        self.pool= nn.MaxPool2d(kernel_size= 2, stride=2) # Output: 64 x 7 x 7

        # - kernel_size=2 to reduce the spatial dimensions by half
        # - stride=2 to move the filter two pixels at a time, effectively downsampling the feature maps

        # Fully connected layers (FC)
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # Output: 128
        self.fc2= nn.Linear(128, 10) # Output: 10 (number of classes)

    def forward(self, x):

        # Function for forward propagation through the CNN

        x= self.pool(F.relu(self.conv1(x))) # Conv1 > ReLU > Max Pool
        x= self.pool(F.relu(self.conv2(x))) # Conv2 > ReLU > Max Pool

        x= x.view(x.size(0), -1) # Flatten the output for the fully connected layers

        x= F.relu(self.fc1(x)) # FC1 > ReLU
        x= self.fc2(x) # FC2 (output layer)

        return x