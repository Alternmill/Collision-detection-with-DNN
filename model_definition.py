import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import nn, optim

class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size, layers):
        super(RegressionModel, self).__init__()
        self.layers = nn.ModuleList()
        for i, layer_size in enumerate(layers):
            self.layers.append(nn.Linear(input_size, layer_size))
            self.layers.append(nn.ReLU())
            input_size = layer_size
        self.layers.append(nn.Linear(layers[-1], output_size))
        self.final_activation = nn.ReLU() # Adding a ReLU activation for the output

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_activation(x) # Applying ReLU to the final layer's output


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')



class EnhancedRegressionModel(nn.Module):
    def __init__(self, input_size, output_size, layers, dropout_probability=0.1):
        super(EnhancedRegressionModel, self).__init__()
        self.layers = nn.ModuleList()

        # Add the first layer to the module list
        self.layers.append(nn.Linear(input_size, layers[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_probability))  # Dropout after activation

        # Add subsequent layers
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i - 1], layers[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_probability))  # Dropout for each layer

        # Final output layer without dropout
        self.layers.append(nn.Linear(layers[-1], output_size))
        self.final_activation = nn.ReLU()  # Consider if ReLU is appropriate for your output; it constrains output to non-negative values

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_activation(x)
        return x

class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensuring the predictions and true values are positive + 1 to avoid log(0)
        y_pred_log = torch.log(y_pred + 1)
        y_true_log = torch.log(y_true + 1)
        # Calculating the mean squared logarithmic error
        loss = torch.mean((y_pred_log - y_true_log) ** 2)
        return loss


import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        """
        Custom Weighted MSE Loss.

        Parameters:
            epsilon (float): A small constant to avoid division by zero and to moderate the weight for large values.
        """
        super(WeightedMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        # Calculate the basic squared errors
        squared_errors = (predictions - targets) ** 2

        # Compute weights inversely proportional to targets, moderated by epsilon
        weights = 1 / (targets + self.epsilon)

        # Apply weights to the squared errors
        weighted_squared_errors = weights * squared_errors

        # Return the mean weighted squared error
        loss = torch.mean(weighted_squared_errors)
        return loss


import torch
import torch.nn as nn


class ExponentialWeightedMSELoss(nn.Module):
    def __init__(self, base=10, epsilon=1e-6):
        """
        Custom Exponential Weighted MSE Loss.

        Parameters:
        - base (float): The base of the exponential function to adjust weighting.
                        Error importance will increase by a factor of 'base' as the target approaches zero.
        - epsilon (float): Small constant to ensure numerical stability for zero targets.
        """
        super(ExponentialWeightedMSELoss, self).__init__()
        self.base = base
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        # Calculate the basic squared errors
        squared_errors = (predictions - targets) ** 2

        # Compute exponential weights based on the target values
        weights = torch.exp(-targets + self.epsilon)

        # Optionally, adjust the weight scale using a base value
        if self.base != 10:
            weights = torch.pow(weights, torch.log(torch.tensor(self.base)))

        # Apply weights to the squared errors
        weighted_squared_errors = weights * squared_errors

        # Return the mean weighted squared error
        loss = torch.mean(weighted_squared_errors)
        return loss
