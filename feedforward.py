import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pyswarm import pso
from torch.utils.data.dataset import random_split

# Load the data
mat_data = scipy.io.loadmat('C:/Users/baxst/Downloads/weather_channel.mat')

# Extract data for drizT
time_variable_drizT = mat_data['drizT'][:, 0]
channel1_drizT = mat_data['drizT'][:, 1]
channel2_drizT = mat_data['drizT'][:, 2]
channel3_drizT = mat_data['drizT'][:, 3]
other_channels_drizT = mat_data['drizT'][:, 4]

# Extract data for medT
time_variable_medT = mat_data['medT'][:, 0]
channel1_medT = mat_data['medT'][:, 1]
channel2_medT = mat_data['medT'][:, 2]
channel3_medT = mat_data['medT'][:, 3]
other_channels_medT = mat_data['medT'][:, 4]

# Extract data for sunT
time_variable_sunT = mat_data['sunT'][:, 0]
channel1_sunT = mat_data['sunT'][:, 1]
channel2_sunT = mat_data['sunT'][:, 2]
channel3_sunT = mat_data['sunT'][:, 3]
other_channels_sunT = mat_data['sunT'][:, 4]

# Extract data for rainT
time_variable_rainT = mat_data['rainT'][:, 0]
channel1_rainT = mat_data['rainT'][:, 1]
channel2_rainT = mat_data['rainT'][:, 2]
channel3_rainT = mat_data['rainT'][:, 3]
other_channels_rainT = mat_data['rainT'][:, 4]

from sklearn.preprocessing import MinMaxScaler

# Create a scaler instance
scaler = MinMaxScaler()




# Assuming 'actual_sunt_targets' is the column containing the target values in your 'sunT' dataset
actual_sunt_targets = mat_data['sunT'][:, 1:5]  # Assuming columns 1 to 4 are the target values

# Convert NumPy arrays to PyTorch tensors
sunT_data = torch.tensor([channel1_sunT, channel2_sunT, channel3_sunT, other_channels_sunT], dtype=torch.float).view(-1, 1, 4)
sunT_targets = torch.tensor(actual_sunt_targets, dtype=torch.float)

drizT_data = torch.tensor([channel1_drizT, channel2_drizT, channel3_drizT, other_channels_drizT], dtype=torch.float).view(-1, 1, 4)
medT_data = torch.tensor([channel1_medT, channel2_medT, channel3_medT, other_channels_medT], dtype=torch.float).view(-1, 1, 4)
rainT_data = torch.tensor([channel1_rainT, channel2_rainT, channel3_rainT, other_channels_rainT], dtype=torch.float).view(-1, 1, 4)

# Split data into training and testing sets
train_ratio = 0.8
train_size = int(train_ratio * len(sunT_data))
test_size = len(sunT_data) - train_size

# Randomly split the datasets
sunT_train, sunT_test = random_split(TensorDataset(sunT_data, sunT_targets), [train_size, test_size])
drizT_train, drizT_test = random_split(TensorDataset(drizT_data, sunT_targets), [train_size, test_size])
medT_train, medT_test = random_split(TensorDataset(medT_data, sunT_targets), [train_size, test_size])
rainT_train, rainT_test = random_split(TensorDataset(rainT_data, sunT_targets), [train_size, test_size])

# Combine data and targets for each weather condition (excluding sunT)
combined_train_data = torch.cat([drizT_train.dataset.tensors[0], medT_train.dataset.tensors[0], rainT_train.dataset.tensors[0]], dim=0)
combined_train_targets = torch.cat([drizT_train.dataset.tensors[1], medT_train.dataset.tensors[1], rainT_train.dataset.tensors[1]], dim=0)

combined_test_data = torch.cat([drizT_test.dataset.tensors[0], medT_test.dataset.tensors[0], rainT_test.dataset.tensors[0]], dim=0)
combined_test_targets = torch.cat([drizT_test.dataset.tensors[1], medT_test.dataset.tensors[1], rainT_test.dataset.tensors[1]], dim=0)

combined_train_data_normalized = scaler.fit_transform(combined_train_data.numpy().reshape(-1, 4))
combined_test_data_normalized = scaler.transform(combined_test_data.numpy().reshape(-1, 4))

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm2 = nn.LSTM(hidden_size, output_size, batch_first=True)

    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out1_relu = self.relu(lstm_out1)
        lstm_out1_dropout = self.dropout(lstm_out1_relu)
        lstm_out2, _ = self.lstm2(lstm_out1_dropout)
        # Squeeze the extra dimension
        return lstm_out2.squeeze(1)

# Instance of the LSTM model, loss function, and optimizer
input_size = 4
hidden_size = 64
output_size = 4

lstm_model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()

# DataLoader for training and testing
batch_size = 16
train_dataset = TensorDataset(combined_train_data, combined_train_targets)
test_dataset = TensorDataset(combined_test_data, combined_test_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

weight_ih_l0_shape = lstm_model.lstm1.weight_ih_l0.shape
weight_hh_l0_shape = lstm_model.lstm1.weight_hh_l0.shape
bias_ih_l0_shape = lstm_model.lstm1.bias_ih_l0.shape
bias_hh_l0_shape = lstm_model.lstm1.bias_hh_l0.shape
# PSO optimization function for the LSTM model
# PSO optimization function for the LSTM model
def lstm_optimize(model, criterion, inputs, targets, input_size, hidden_size):
    # Get the shapes for the LSTM weights and biases
    weight_ih_l0_shape = model.lstm1.weight_ih_l0.shape
    weight_hh_l0_shape = model.lstm1.weight_hh_l0.shape
    bias_ih_l0_shape = model.lstm1.bias_ih_l0.shape
    bias_hh_l0_shape = model.lstm1.bias_hh_l0.shape

    def objective(params):
        nonlocal weight_ih_l0_shape  # Ensure it's recognized as a nonlocal variable
        # Set the model parameters with the PSO values
        model.lstm1.weight_ih_l0.data = torch.tensor(params[:np.prod(weight_ih_l0_shape)].reshape(weight_ih_l0_shape),
                                                     dtype=torch.float)
        model.lstm1.weight_hh_l0.data = torch.tensor(params[np.prod(weight_ih_l0_shape):np.prod(weight_ih_l0_shape) + np.prod(weight_hh_l0_shape)].reshape(weight_hh_l0_shape),
                                                     dtype=torch.float)
        model.lstm1.bias_ih_l0.data = torch.tensor(params[np.prod(weight_ih_l0_shape) + np.prod(weight_hh_l0_shape):np.prod(weight_ih_l0_shape) + np.prod(weight_hh_l0_shape) + np.prod(bias_ih_l0_shape)],
                                                  dtype=torch.float)
        model.lstm1.bias_hh_l0.data = torch.tensor(params[np.prod(weight_ih_l0_shape) + np.prod(weight_hh_l0_shape) + np.prod(bias_ih_l0_shape):],
                                                  dtype=torch.float)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        return loss.item()

    # Define the bounds for the PSO
    lb = [-1.0] * (np.prod(weight_ih_l0_shape) + np.prod(weight_hh_l0_shape) + np.prod(bias_ih_l0_shape) + np.prod(bias_hh_l0_shape))  # Lower bounds
    ub = [1.0] * (np.prod(weight_ih_l0_shape) + np.prod(weight_hh_l0_shape) + np.prod(bias_ih_l0_shape) + np.prod(bias_hh_l0_shape))  # Upper bounds

    # PSO optimization
    best_params, _ = pso(objective, lb, ub)

    return best_params



lstm_best_params = lstm_optimize(lstm_model, criterion, combined_train_data, combined_train_targets, input_size, hidden_size)


# Set the LSTM model parameters with the best PSO values
lstm_model.lstm1.weight_ih_l0.data = torch.tensor(lstm_best_params[:np.prod(weight_ih_l0_shape)].reshape(weight_ih_l0_shape),
                                                  dtype=torch.float)
lstm_model.lstm1.weight_hh_l0.data = torch.tensor(lstm_best_params[np.prod(weight_ih_l0_shape):np.prod(weight_ih_l0_shape) + np.prod(weight_hh_l0_shape)].reshape(weight_hh_l0_shape),
                                                  dtype=torch.float)
lstm_model.lstm1.bias_ih_l0.data = torch.tensor(lstm_best_params[np.prod(weight_ih_l0_shape) + np.prod(weight_hh_l0_shape):np.prod(weight_ih_l0_shape) + np.prod(weight_hh_l0_shape) + np.prod(bias_ih_l0_shape)],
                                               dtype=torch.float)
lstm_model.lstm1.bias_hh_l0.data = torch.tensor(lstm_best_params[np.prod(weight_ih_l0_shape) + np.prod(weight_hh_l0_shape) + np.prod(bias_ih_l0_shape):],
                                               dtype=torch.float)

# Evaluate the LSTM model with the optimized parameters on the test set
lstm_model.eval()
with torch.no_grad():
    lstm_predictions = lstm_model(combined_test_data)


# Loss calculation on the test set
lstm_mse_loss = criterion(lstm_predictions, combined_test_targets)


# Print or log the LSTM losses
print(f'Mean Squared Error (MSE) for LSTM: {lstm_mse_loss.item()}')

def visualize_lstm_predictions(data, targets, model, dataset_name):
    model.eval()
    with torch.no_grad():
        predictions = model(data)

    # Ensure both targets and predictions have the same shape
    targets_flat = targets.numpy().flatten()
    predictions_flat = predictions.numpy().flatten()

    # Print shapes for debugging
    print("Targets shape:", targets_flat.shape)
    print("Predictions shape:", predictions_flat.shape)

    # Make sure the shapes are the same
    if targets_flat.shape != predictions_flat.shape:
        raise ValueError("Targets and predictions must have the same shape for plotting.")

    # Continue with scatter plot
    plt.scatter(targets_flat, predictions_flat, label='Predictions', marker='o')
    plt.scatter(targets_flat, targets_flat, label='Ground Truth', marker='x')  # Ground truth denoted with 'x'
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs Ground Truth - {dataset_name}')
    plt.legend()
    plt.show()

# Visualize predictions for the LSTM model
# Visualize predictions for the LSTM model for the entire training set
visualize_lstm_predictions(combined_train_data, combined_train_targets, lstm_model, 'LSTM')




def calculate_mse(model, data, targets, condition_name):
    model.eval()
    with torch.no_grad():
        predictions = model(data)

    mse_loss = criterion(predictions, targets)
    print(f'Mean Squared Error (MSE) for {condition_name}: {mse_loss.item()}')
    return mse_loss.item()

# Calculate and print MSE for each weather condition
# Calculate and print MSE for each weather condition using the test sets
mse_drizT = calculate_mse(lstm_model, drizT_test.dataset.tensors[0], drizT_test.dataset.tensors[1], 'drizT')
mse_medT = calculate_mse(lstm_model, medT_test.dataset.tensors[0], medT_test.dataset.tensors[1], 'medT')
mse_rainT = calculate_mse(lstm_model, rainT_test.dataset.tensors[0], rainT_test.dataset.tensors[1], 'rainT')
mse_sunT = calculate_mse(lstm_model, combined_test_data, combined_test_targets, 'sunT')



# Visualize predictions for the LSTM model for each weather condition
# Visualize predictions for the LSTM model for each weather condition using the test sets
visualize_lstm_predictions(drizT_test.dataset.tensors[0], drizT_test.dataset.tensors[1], lstm_model, 'drizT')
visualize_lstm_predictions(medT_test.dataset.tensors[0], medT_test.dataset.tensors[1], lstm_model, 'medT')
visualize_lstm_predictions(rainT_test.dataset.tensors[0], rainT_test.dataset.tensors[1], lstm_model, 'rainT')
visualize_lstm_predictions(combined_test_data, combined_test_targets, lstm_model, 'sunT')




