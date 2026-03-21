import torch
import torch.nn as nn
import torch.optim as optim # Importing the optimizer module for training the model
from torchvision import datasets, transforms # Importing datasets and transforms for data loading and preprocessing
from model import CNN # Importing the CNN model defined in model.py
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
epochs= 10
lr= 0.001 # Learning rate for the optimizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # check if GPU is available and set device accordingly

transform= transforms.Compose([transforms.ToTensor()]) # define a transformation to convert images to tensors

# What's a transformation here?

#   A transformation is a preprocessing step that converts raw data (like images) 
#   into a format suitable for training the model.

#   In this case, transforms.ToTensor() converts the PIL images 
#   from the MNIST dataset into PyTorch tensors, which can be fed into the CNN model for training.

# Load the MNIST dataset from torchvision datasets, applying the defined transformation

path = "./data" # Path to store the MNIST dataset (it will be downloaded if not already present)

train_dataset = datasets.MNIST(root= path, train= True, transform= transform, download= True)
test_dataset = datasets.MNIST(root= path, train= False, transform= transform, download= True)

# Create data loaders for batching and shuffling the training and test datasets
train_loader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True)
test_loader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size= batch_size, shuffle= False)

model = CNN().to(device) # Instantiate the CNN model and move it to the specified device (GPU or CPU)

criterion = nn.CrossEntropyLoss() # Define the loss function for classification tasks

# Why cross-entropy loss?

# Cross-entropy loss is commonly used for multi-class classification problems 
# because it measures the difference between the predicted probability distribution (output of 
# the model) and the true distribution (actual labels). 

# It penalises incorrect predictions more heavily, which helps the 
# model learn to make accurate classifications.

optimizer = optim.Adam(model.parameters(), lr= lr) # Define the Adam optimizer for updating model parameters during training

# Adam optimizer: It's an adaptive learning rate optimization algorithm that combines the advantages of two other extensions 
# of stochastic gradient descent, namely AdaGrad and RMSProp. 

# It computes adaptive learning rates for each parameter, which helps in faster convergence and better 
# performance on a wide range of problems.

# Why didn't I choose other optimizers like SGD?
# Other optimizers like SGD (Stochastic Gradient Descent) can be effective, but Adam is often preferred for its 
# adaptive learning rates and faster convergence, especially in deep learning tasks.

# - - - - - - - - - - - - - - - - -

# Training the model

train_losses = [] # List to store training losses for each epoch
train_accuracies = [] # List to store training accuracies for each epoch

for epoch in range(epochs):

    model.train() # Set the model to training mode
    running_loss = 0.0 # Variable to accumulate the loss for the current epoch

    for images, labels in train_loader: # Iterate over batches of images and labels from the training data loader

        images= images.to(device) # Move images to the specified device (GPU or CPU)
        labels= labels.to(device) # Same for labels

        optimizer.zero_grad() # Clear the gradients from the previous step

        outputs = model(images) # Forward pass: compute the model's predictions for the current batch of images

        loss= criterion(outputs, labels) # Compute the loss between the predicted outputs and the true labels

        # What does criterion do here?
        # The criterion (cross-entropy loss) calculates the difference between the predicted probability distribution (outputs)
        # and the true distribution (labels). It returns a scalar value representing the loss, which
        # the model will try to minimise during training.

        loss.backward() # Backward pass (compute the gradients of the loss with respect to the model's parameters)

        optimizer.step() # Update the model's parameters based on the computed gradients

        running_loss += loss.item() # Cumulative sum of the loss for the current epoch

    avg_loss = running_loss / len(train_loader) # Average loss for the current epoch
    train_losses.append(avg_loss)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

torch.save(model.state_dict(), 'MNIST_CNN_Model.pth') # Save the trained model's parameters to a file

plt.plot(train_losses, label='Training Loss') # Plot the training loss over epochs
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()
