from simulation_utils import *
from generate_load_datasets import feature_engineering
import torch
import math

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

initialize_simulation()
scene_id, robot_id = load_scene_and_robot()

path = ("models/model_2_2024-04-08_20-18-06.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path, map_location=device)

bbox_x, bbox_y = calculate_overall_bounding_box()
print("Bounding box x: {}".format(bbox_x))
print("Bounding box y: {}".format(bbox_y))

def create_collision_grid(theta_degrees, grid_size=20):
    theta_radians = math.radians(theta_degrees)
    x_values = np.linspace(bbox_x[0], bbox_x[1], grid_size)
    y_values = np.linspace(bbox_y[0], bbox_y[1], grid_size)
    collision_grid = np.zeros((grid_size, grid_size))
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            collision_grid[i, j] = calculate_closest_distance(scene_id, robot_id, x, y, theta_radians)
    return x_values, y_values, collision_grid


def create_model_prediction_grid(theta_degrees, grid_size=20, model=model):
    theta_radians = math.radians(theta_degrees)
    x_values = np.linspace(bbox_x[0], bbox_x[1], grid_size)
    y_values = np.linspace(bbox_y[0], bbox_y[1], grid_size)

    xx, yy = np.meshgrid(x_values, y_values, indexing='ij')

    x_flat = xx.ravel()
    y_flat = yy.ravel()
    data = np.vstack((x_flat, y_flat, np.full_like(x_flat, theta_radians))).T
    inputs = feature_engineering(data)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(inputs_tensor).cpu().numpy().squeeze()

    prediction_grid = predictions.reshape(grid_size, grid_size)

    return prediction_grid


# Function to generate the collision and prediction grids
def generate_grids(model, theta_values, grid_size):
    collision_grids = []
    prediction_grids = []
    difference_grids = []

    for theta in theta_values:
        print(theta)
        _, _, collision_grid = create_collision_grid(theta, grid_size)
        prediction_grid = create_model_prediction_grid(theta, grid_size, model)
        difference_grid = np.abs(collision_grid - prediction_grid)
        collision_grids.append(collision_grid)
        prediction_grids.append(prediction_grid)
        difference_grids.append(difference_grid)

    return collision_grids, prediction_grids, difference_grids


theta_values = range(0, 91, 5)  # Define your theta values here
grid_size = 401  # Define your grid size

# Generate the grids
collision_grids, prediction_grids, difference_grids = generate_grids(model, theta_values, grid_size)

# Create subplots: 1 row, 3 columns
fig = make_subplots(rows=1, cols=3, subplot_titles=['Ground Truth', 'Model Prediction', 'Absolute Difference'],
                    horizontal_spacing=0.02)

# Create sliders
steps = []
for i, theta in enumerate(theta_values):
    # Calculate the start index for the current set of plots
    start_index = i * 3  # Since we have 3 plots (Ground Truth, Model Prediction, Absolute Difference) per theta value
    # Create a list of visibility states for all plots, setting only the relevant plots for the current theta to True
    visibility = [False] * len(theta_values) * 3  # Initialize all to False
    visibility[start_index] = True  # Ground Truth for current theta
    visibility[start_index + 1] = True  # Model Prediction for current theta
    visibility[start_index + 2] = True  # Absolute Difference for current theta

    # Create the step dict
    step = dict(
        method="update",
        args=[{"visible": visibility},
              {"title": f"Theta: {theta}°"}],  # layout attribute
    )
    steps.append(step)

# Add initial data to subplots for the first theta value
fig.add_trace(go.Heatmap(z=collision_grids[0], colorscale='Viridis', showscale=False), row=1, col=1)
fig.add_trace(go.Heatmap(z=prediction_grids[0], colorscale='Viridis', showscale=False), row=1, col=2)
fig.add_trace(go.Heatmap(z=difference_grids[0], colorscale='Viridis', showscale=False), row=1, col=3)

# Add rest of the data for other theta values but make them invisible
for i in range(1, len(theta_values)):
    fig.add_trace(go.Heatmap(z=collision_grids[i], colorscale='Viridis', showscale=False, visible=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=prediction_grids[i], colorscale='Viridis', showscale=False, visible=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=difference_grids[i], colorscale='Viridis', showscale=False, visible=False), row=1, col=3)

# Update layout with sliders and axis labels
fig.update_layout(
    sliders=[{
        "active": 0,
        "steps": steps,
        "currentvalue": {"prefix": "Theta: "}
    }],
    xaxis_title='X',
    yaxis_title='Y',
    xaxis2_title='X',
    yaxis2_title='Y',
    xaxis3_title='X',
    yaxis3_title='Y',
)

# Show the figure with sliders to control theta value
fig.show()
