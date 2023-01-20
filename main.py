"""This code is training a simple convolutional neural network (CNN) on an image classification
 task. The model is defined as a class called SimpleModel which inherits from the nn.Module
 class in PyTorch. The SimpleModel class has two convolutional layers, two fully connected layers,
and a final output layer. The forward method of the class defines the forward pass of the model
and applies a series of activation functions and pooling layers to the input. The model is trained
on a dataset of images of apples and oranges using the Adam optimizer and cross-entropy loss function.
After training, the code applies the same image transformation to a single image and then forward the
image through the trained model, finally it prints the class label of the image.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from PIL import Image


# Define the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        """Creating kernels for the convolutional layers"""
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        """Creating the fully connected layers"""
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate the model
model = SimpleModel()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the data transform
transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load the training and test datasets
train_dataset = datasets.ImageFolder(root='./training_data', transform=transform)
test_dataset = datasets.ImageFolder(root='./test_data', transform=transform)

# Define the data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Train the model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1} Loss: {running_loss / (i + 1)}')

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

"""Testing the model on a single image"""
img = Image.open('./test1.jpg')
# Transform the image
transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img_tensor = transform(img)
img_tensor = img_tensor.unsqueeze(0)  # add batch dimension

# Forward the image through the model
outputs = model(img_tensor)

# Get the class label
_, predicted = torch.max(outputs.data, 1)

# Print the class label
if predicted.item() == 0:
    print("This image is an apple.")
else:
    print("This image is an orange.")
