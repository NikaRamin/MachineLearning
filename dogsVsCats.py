import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.optim as optim

# Set this to True if you want to create new training data (reload images from scratch)
REBUILD_DATA = True

# Class to handle data processing (loading and preparing images of dogs and cats)
class DogsVSCats():
    IMG_SIZE = 50  # Size to which all images will be resized (50x50 pixels)
    CATS = "PetImages/Cat"  # Directory containing cat images
    DOGS = "PetImages/Dog"  # Directory containing dog images
    LABELS = {CATS: 0, DOGS: 1}  # Assigning labels: 0 for cats, 1 for dogs
    training_data = []  # List to store processed training data
    catcount = 0  # Counter for cat images
    dogcount = 0  # Counter for dog images

    # Function to prepare and store the training data
    def make_training_data(self):
        for label in self.LABELS:  # Loop through both cat and dog directories
            print(label)
            for f in tqdm(os.listdir(label)):  # Loop through each file in the folder
                try:
                    path = os.path.join(label, f)  # Get the full path of the image
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))  # Resize the image to 50x50 pixels
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  
                    # Append image and one-hot encoded label to training data
                    # np.eye(2)[0] -> [1, 0] for cat, np.eye(2)[1] -> [0, 1] for dog
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass  # If any error occurs (like unreadable images), skip it
        
        # Shuffle the data to ensure random distribution of cats and dogs
        np.random.shuffle(self.training_data)
        # Save the data as a numpy array for future use (so we don't need to rebuild every time)
        np.save("training_data.npy", np.array(self.training_data, dtype=object), allow_pickle=True)
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)

# Define the Neural Network architecture (Convolutional Neural Network)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolutional layer: input 1 channel (grayscale image), 32 output channels, 5x5 kernel size
        self.conv1 = nn.Conv2d(1, 32, 5)
        # Second convolutional layer: input 32 channels, 64 output channels, 5x5 kernel size
        self.conv2 = nn.Conv2d(32, 64, 5)
        # Third convolutional layer: input 64 channels, 128 output channels, 5x5 kernel size
        self.conv3 = nn.Conv2d(64, 128, 5)

        # Create a dummy input to determine the size of the flattened layer after convolutions
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)  # Pass the dummy input through the convolution layers to determine the output size
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._to_linear, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, 2)  # Output layer (2 classes: cat or dog)

    # Function to apply the convolutional layers
    def convs(self, x):
        # Apply conv1 + ReLU + Max pooling (2x2)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # Apply conv2 + ReLU + Max pooling (2x2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # Apply conv3 + ReLU + Max pooling (2x2)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        
        # Flatten the output for fully connected layers
        if self._to_linear is None:
            # Set _to_linear to be the size of the flattened output
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    # Function to define the forward pass
    def forward(self, x):
        x = self.convs(x)  # Apply convolutional layers
        x = x.view(-1, self._to_linear)  # Flatten the output
        x = F.relu(self.fc1(x))  # Apply first fully connected layer with ReLU activation
        x = self.fc2(x)  # Output layer
        return F.softmax(x, dim=1)  # Apply softmax to output probabilities

# Rebuild data if specified
if REBUILD_DATA:
    dogsvscats = DogsVSCats()
    dogsvscats.make_training_data()

# Load the preprocessed training data
training_data = np.load("training_data.npy", allow_pickle=True)

# Initialize the neural network
net = Net()

# Define the optimizer (Adam optimizer) and loss function (Mean Squared Error Loss)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X_np = np.array([i[0] for i in training_data])  # Create a single numpy array for images
y_np = np.array([i[1] for i in training_data])  # Create a single numpy array for labels

# Now convert these numpy arrays to torch tensors efficiently
X = torch.tensor(X_np, dtype=torch.float32).view(-1, 50, 50)  # Reshape to (n_samples, 50, 50)
X = X / 255.0  # Normalize pixel values to the range [0, 1]

y = torch.tensor(y_np, dtype=torch.float32)

# Split the dataset into training and validation sets
VAL_PCT = 0.1  # Percentage of data used for validation
val_size = int(len(X) * VAL_PCT)  # Validation size
train_X = X[:-val_size]  # Training data
train_y = y[:-val_size]  # Training labels

test_X = X[-val_size:]  # Validation data
test_y = y[-val_size:]  # Validation labels

# Hyperparameters
BATCH_SIZE = 100  # Number of samples per batch
EPOCHS = 10  # Number of times to go through the entire dataset

# Training loop
for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        # Get batches
        batch_X = train_X[i:i + BATCH_SIZE].view(-1, 1, 50, 50)  # Reshape images for input
        batch_y = train_y[i:i + BATCH_SIZE]  # Get corresponding labels

        optimizer.zero_grad()  # Zero the gradients before each step
        outputs = net(batch_X)  # Forward pass
        loss = loss_function(outputs, batch_y)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update the weights

# Testing loop
correct = 0
total = 0
with torch.no_grad():  # Disable gradient calculation for testing
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])  # Get true label
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]  # Get network prediction
        predicted_class = torch.argmax(net_out)  # Get predicted label
        if predicted_class == real_class:
            correct += 1
        total += 1

# Print the accuracy of the model
print("Accuracy:", round(correct / total, 3))

# Save the model weights for later use
torch.save(net.state_dict(), 'model_weights.pth')
