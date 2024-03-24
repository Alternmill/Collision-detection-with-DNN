import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
import time

from simulation_utils import initialize_simulation, load_scene_and_robot
from model_definition import RegressionModel, EnhancedRegressionModel, MSLELoss, init_weights, WeightedMSELoss, \
    ExponentialWeightedMSELoss
from generate_load_datasets import load_dataset, feature_engineering

initialize_simulation()
scene_id, robot_id = load_scene_and_robot()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def train_model(model, train_loader, criterion, optimizer, scheduler, epochs, patience=100):
    """Trains the model with dynamic learning rate adjustment and early stopping."""
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        flag = False
        start_time = time.time()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if not flag:
                # print(outputs)
                flag = True
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        scheduler.step(avg_loss)  # Adjust learning rate based on the average loss

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.7f},"
              f" Time: {time.time() - start_time:.2f} seconds ,"
              f" Lr: {optimizer.param_groups[0]['lr']}")

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping triggered.")
                break


# Initialize model, criterion, optimizer, and scheduler
model = RegressionModel(input_size=12, output_size=1, layers=[256, 256, 256, 64]).to(device)
criterion = ExponentialWeightedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-5)

# Generate initial dataset and DataLoader
print("Loading dataset...")
inputs, targets = load_dataset("dataset_big_5000000_samples", feature_eng_func=feature_engineering)
print("Converting to tensors...:")
train_dataset = TensorDataset(inputs, targets)
print("Making dataloaders...")
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
print("Dataset loaded!")

# Train the model
train_model(model, train_loader, criterion, optimizer, scheduler, epochs=200)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f'models/model_{current_time}.pt'
torch.save(model, filename)
print(f'Model saved as {filename}')
