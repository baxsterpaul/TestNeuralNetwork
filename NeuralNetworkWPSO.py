# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:47:02 2023

@author: baxst
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pyswarm import pso  # Install with: pip install pyswarm
from torchsummary import summary
import matplotlib.pyplot as plt
import scipy.io

mat_data = scipy.io.loadmat('C:/Users/baxst/Downloads/weather_channel.mat')



time_variable_drizT = mat_data['drizT'][:, 0]
channel1_drizT = mat_data['drizT'][:, 1]
channel2_drizT = mat_data['drizT'][:, 2]
channel3_drizT = mat_data['drizT'][:, 3]
other_channels_drizT = mat_data['drizT'][:, 4]

time_variable_medT = mat_data['medT'][:, 0]
channel1_medT = mat_data['medT'][:, 1]
channel2_medT = mat_data['medT'][:, 2]
channel3_medT = mat_data['medT'][:, 3]
other_channels_medT = mat_data['medT'][:, 4]

time_variable_sunT = mat_data['sunT'][:, 0]
channel1_sunT = mat_data['sunT'][:, 1]
channel2_sunT = mat_data['sunT'][:, 2]
channel3_sunT = mat_data['sunT'][:, 3]
other_channels_sunT = mat_data['sunT'][:, 4]

time_variable_rainT = mat_data['rainT'][:, 0]
channel1_rainT = mat_data['rainT'][:, 1]
channel2_rainT = mat_data['rainT'][:, 2]
channel3_rainT = mat_data['rainT'][:, 3]
other_channels_rainT = mat_data['rainT'][:, 4]

# Plotting for drizT
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(time_variable_drizT, channel1_drizT, label='Channel 1')
plt.plot(time_variable_drizT, channel2_drizT, label='Channel 2')
plt.plot(time_variable_drizT, channel3_drizT, label='Channel 3')
plt.plot(time_variable_drizT, other_channels_drizT, label='Other Channels')
plt.xlabel('Time')
plt.ylabel('Optical Signal')
plt.title('Optical Signals over Time (drizT)')
plt.legend()

# Plotting for medT
plt.subplot(2, 2, 2)
plt.plot(time_variable_medT, channel1_medT, label='Channel 1')
plt.plot(time_variable_medT, channel2_medT, label='Channel 2')
plt.plot(time_variable_medT, channel3_medT, label='Channel 3')
plt.plot(time_variable_medT, other_channels_medT, label='Other Channels')
plt.xlabel('Time')
plt.ylabel('Optical Signal')
plt.title('Optical Signals over Time (medT)')
plt.legend()

# Plotting for sunT
plt.subplot(2, 2, 3)
plt.plot(time_variable_sunT, channel1_sunT, label='Channel 1')
plt.plot(time_variable_sunT, channel2_sunT, label='Channel 2')
plt.plot(time_variable_sunT, channel3_sunT, label='Channel 3')
plt.plot(time_variable_sunT, other_channels_sunT, label='Other Channels')
plt.xlabel('Time')
plt.ylabel('Optical Signal')
plt.title('Optical Signals over Time (sunT)')
plt.legend()

# Plotting for rainT
plt.subplot(2, 2, 4)
plt.plot(time_variable_rainT, channel1_rainT, label='Channel 1')
plt.plot(time_variable_rainT, channel2_rainT, label='Channel 2')
plt.plot(time_variable_rainT, channel3_rainT, label='Channel 3')
plt.plot(time_variable_rainT, other_channels_rainT, label='Other Channels')
plt.xlabel('Time')
plt.ylabel('Optical Signal')
plt.title('Optical Signals over Time (rainT)')
plt.legend()

plt.tight_layout()
plt.show()




# Data
drizT_data = torch.rand((50, 4))  # Assuming you have 4 channels
medT_data = torch.rand((50, 4))
sunT_data = torch.rand((50, 4))
rainT_data = torch.rand((50, 4))

# Data set target
drizT_targets = torch.rand((50, 1))
medT_targets = torch.rand((50, 1))
sunT_targets = torch.rand((50, 1))
rainT_targets = torch.rand((50, 1))

print("drizT_data shape:", drizT_data.shape)
print("medT_data shape:", medT_data.shape)
print("sunT_data shape:", sunT_data.shape)
print("rainT_data shape:", rainT_data.shape)

print("drizT_targets shape:", drizT_targets.shape)
print("medT_targets shape:", medT_targets.shape)
print("sunT_targets shape:", sunT_targets.shape)
print("rainT_targets shape:", rainT_targets.shape)




# Combine data and targets for each weather condition
combined_data = torch.cat([drizT_data, medT_data, sunT_data, rainT_data], dim=0)
combined_targets = torch.cat([drizT_targets, medT_targets, sunT_targets, rainT_targets], dim=0)

print("combined_data shape:", combined_data.shape)
print("combined_targets shape:", combined_targets.shape)


# Define the neural network
class WeatherModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WeatherModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instance of model, loss function, and optimizer
input_size = 4  # Adjust based on the number of channels
hidden_size = 64
output_size = 1

model = WeatherModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# MODEL SUMMARY
summary(model, (input_size,))

# DataLoader for training
batch_size = 16
train_dataset = TensorDataset(combined_data, combined_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 10
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    losses.append(epoch_loss / len(train_loader))

# Print or log the losses
print("Training Losses:", losses)

# PSO optimization function
def optimize(model, criterion, inputs, targets, input_size, hidden_size):
    def objective(params):
        # Set the model parameters with the PSO values
        model.fc1.weight.data = torch.tensor(params[:input_size * hidden_size].reshape(hidden_size, input_size),
                                             dtype=torch.float)
        model.fc1.bias.data = torch.tensor(params[input_size * hidden_size:input_size * hidden_size + hidden_size],
                                           dtype=torch.float)
        model.fc2.weight.data = torch.tensor(
            params[input_size * hidden_size + hidden_size:-1].reshape(1, hidden_size), dtype=torch.float)
        model.fc2.bias.data = torch.tensor(params[-1], dtype=torch.float)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        return loss.item()

    # Define the bounds for the PSO
    lb = [-1.0] * (input_size * hidden_size + hidden_size + hidden_size + 1)  # Lower bounds
    ub = [1.0] * (input_size * hidden_size + hidden_size + hidden_size + 1)  # Upper bounds

    # PSO optimization
    best_params, _ = pso(objective, lb, ub)

    return best_params

# PSO optimization for each weather condition
for data, targets, condition in zip([drizT_data, medT_data, sunT_data, rainT_data],
                                    [drizT_targets, medT_targets, sunT_targets, rainT_targets],
                                    ['drizT', 'medT', 'sunT', 'rainT']):
    # PSO optimization
    best_params = optimize(model, criterion, data, targets, input_size, hidden_size)

    # Set the model parameters with the best PSO values
    model.fc1.weight.data = torch.tensor(best_params[:input_size * hidden_size].reshape(hidden_size, input_size),
                                         dtype=torch.float)
    model.fc1.bias.data = torch.tensor(best_params[input_size * hidden_size:input_size * hidden_size + hidden_size],
                                       dtype=torch.float)
    model.fc2.weight.data = torch.tensor(
        best_params[input_size * hidden_size + hidden_size:-1].reshape(1, hidden_size), dtype=torch.float)
    model.fc2.bias.data = torch.tensor(best_params[-1], dtype=torch.float)

    # Evaluate the model with the optimized parameters
    predictions = model(data)

    # Loss calculation
    mse_loss = criterion(predictions, targets)

    # Print or log the losses
    print(f'Mean Squared Error (MSE) for {condition}: {mse_loss.item()}')

# Assuming predictions and targets are tensors
# Adjust to the appropriate dataset
predictions = model(drizT_data)
targets = drizT_targets

# Loss calculation
mse_loss = criterion(predictions, targets)

# Print or log the losses
print(f'Mean Squared Error (MSE): {mse_loss.item()}')

# Plot training loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()

def visualize_predictions(data, targets, model, dataset_name):
    model.eval()
    with torch.no_grad():
        predictions = model(data)

    plt.scatter(targets.numpy(), predictions.numpy(), marker='o', label='Predictions')
    plt.scatter(targets.numpy(), targets.numpy(), marker='x', label='Random data')  # Use 'x' for ground truth
    plt.xlabel('Random Data')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs Random Data - {dataset_name}')
    plt.legend()
    plt.show()

# Visualize predictions for each dataset
visualize_predictions(drizT_data, drizT_targets, model, 'drizT')
visualize_predictions(medT_data, medT_targets, model, 'medT')
visualize_predictions(sunT_data, sunT_targets, model, 'sunT')
visualize_predictions(rainT_data, rainT_targets, model, 'rainT')
# What test set to test performance? help 
