import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Download and prepare the MNIST dataset
# The `transform` argument ensures that images are converted to tensors
train = datasets.MNIST(
    "", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
)
test = datasets.MNIST(
    "", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
)

# Create DataLoader objects for both training and test datasets
# The DataLoader will split the data into batches and shuffle the training data
trainSet = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testSet = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

# Define a neural network class by inheriting from nn.Module
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Fully connected (linear) layers with the specified input and output dimensions
        # First layer: Input size 784 (28x28 image flattened), output size 64
        self.fc1 = nn.Linear(784, 64)
        # Second layer: Input size 64, output size 64
        self.fc2 = nn.Linear(64, 64)
        # Third layer: Input size 64, output size 64
        self.fc3 = nn.Linear(64, 64)
        # Fourth layer: Input size 64, output size 10 (one for each digit class 0-9)
        self.fc4 = nn.Linear(64, 10)
    
    # Define the forward pass for the network
    # This method defines how data flows through the network
    def forward(self, x):
        # Pass the input through the first layer followed by a ReLU activation function
        x = F.relu(self.fc1(x))
        
        # Activation Function (ReLU):
        # ReLU (Rectified Linear Unit) is a widely used activation function in neural networks.
        # ReLU simply outputs the input if it's positive, otherwise, it outputs zero.
        # It introduces non-linearity to the model, which is crucial for learning complex patterns.
        # Without an activation function, a neural network would behave like a linear regression model,
        # regardless of how many layers it has.
        
        # Pass through the second layer followed by ReLU
        x = F.relu(self.fc2(x))
        
        # Same as the first activation function, it keeps the output non-linear.
        # The key advantage of ReLU is that it's computationally efficient
        # and avoids the vanishing gradient problem faced by other activation functions like sigmoid.
        
        # Pass through the third layer followed by ReLU
        x = F.relu(self.fc3(x))
        
        # Final layer without an activation function since we want raw logits (before softmax)
        # We apply softmax after this step to normalize the output values into probabilities.
        x = self.fc4(x)
        
        # Apply log_softmax on the output of the final layer with dimension 1
        # Softmax squashes the output values into a range between 0 and 1 and ensures they sum up to 1.
        # log_softmax is used here because it improves numerical stability and is typically combined with
        # negative log-likelihood loss during training.
        return F.log_softmax(x, dim=1)

# Instantiate the neural network
net = Net()

import torch.optim as optim

# Initialize Adam optimizer
# Adam is an adaptive learning rate optimization algorithm that combines the advantages of two other extensions of
# stochastic gradient descent (SGD), namely RMSProp and AdaGrad.
# It adjusts the learning rate dynamically for each parameter based on the estimates of first and second moments.
optimizer = optim.Adam(net.parameters(), lr=0.001)  # Learning rate is set to 0.001

# Set the number of epochs for training
EPOCHS = 3

# Training loop for the specified number of epochs
for epoch in range(EPOCHS):
    for data in trainSet:
        # 'data' is a batch of images (features) and their corresponding labels
        X, y = data  # X: input features (images), y: labels (digits)
        
        # Zero the gradients before performing the backward pass (this clears old gradients from previous batches)
        net.zero_grad()
        
        # Forward pass: Flatten the input image and pass it through the network
        output = net(X.view(-1, 28 * 28))  # Reshape (flatten) the 28x28 image to 784 elements
        
        # Calculate the loss between the predicted output and actual labels
        # Using Negative Log-Likelihood Loss (nll_loss), typically paired with log_softmax in the model
        loss = F.nll_loss(output, y)
        
        # Backward pass: Calculate the gradients for each parameter
        loss.backward()
        
        # Update model parameters using the gradients and the optimizer
        optimizer.step()
    # Print the loss for the last batch in the current epoch
    print(loss)

# After training, we check the model's performance on the training set without calculating gradients
correct = 0  # Count of correct predictions
total = 0    # Total number of predictions

with torch.no_grad():  # Disables gradient calculation (no need for backpropagation during evaluation)
    for data in trainSet:
        X, y = data  # X: images, y: labels
        output = net(X.view(-1, 28 * 28))  # Forward pass, flatten the input
        
        # Iterate over the output of each sample in the batch
        for idx, i in enumerate(output):
            # Get the predicted class by finding the index with the highest probability (argmax)
            if torch.argmax(i) == y[idx]:  # Compare the predicted class with the true label
                correct += 1  # Increment if the prediction is correct
            total += 1  # Keep track of total samples

# Calculate and print accuracy
# Accuracy = correct predictions / total predictions
print("Accuracy: ", round(correct / total, 3))
# Get the first batch of data from testSet using an iterator
first_batch = next(iter(testSet))

# Unpack the batch to get the images and labels
x1, y1 = first_batch  # x1: batch of images, y1: batch of labels

# Visualize the first image in the batch (for confirmation)
plt.imshow(x1[0].view(28, 28))  # Display the first image in the batch
plt.show()

# Use the trained network to predict the label for the first image in the batch
# Flatten the image (from 28x28 to 784) and pass it through the network
output = net(x1[0].view(-1, 784))  # Forward pass with the flattened image
predicted_label = torch.argmax(output)  # Get the predicted class (digit) for the first image

# Print the predicted label
print(predicted_label)

