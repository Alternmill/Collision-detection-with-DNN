from simulation_utils import *
from generate_load_datasets import feature_engineering
import torch
import math
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Initialize and load the model
initialize_simulation()
scene_id, robot_id = load_scene_and_robot()
path = "models/model_2_2024-04-08_20-18-06.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(path, map_location=device)
bbox_x, bbox_y = calculate_overall_bounding_box()


# Function to generate random samples and predict collisions
def ground_truth_and_predictions(size):
    x = np.random.uniform(bbox_x[0], bbox_x[1], size)
    y = np.random.uniform(bbox_y[0], bbox_y[1], size)
    theta = np.random.uniform(0, 2 * np.pi, size)
    distances = np.zeros(size)  # Preallocate array for distances

    for i in range(size):
        distances[i] = calculate_closest_distance(scene_id, robot_id, x[i], y[i], theta[i])

    data = np.vstack((x, y, theta)).T
    inputs = feature_engineering(data)
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(inputs_tensor).cpu().numpy().squeeze()

    return distances, predictions


# Function to calculate confusion matrix from predictions
def calculate_confusion_matrix(predictions, ground_truth):
    TP = np.sum((predictions == 0) & (ground_truth == 0))
    TN = np.sum((predictions > 0) & (ground_truth != 0))
    FP = np.sum((predictions == 0) & (ground_truth != 0))
    FN = np.sum((predictions > 0) & (ground_truth == 0))
    return TP, TN, FP, FN


# Sample size for each angle
sample_size = 1000000

# Define angles for evaluation
distances, predictions = ground_truth_and_predictions(sample_size)

# Calculate confusion matrix components
TP, TN, FP, FN = calculate_confusion_matrix(predictions, distances)

# Create heatmap data and labels
data = np.array([[TP, FP], [FN, TN]])  # Arranged in a 2x2 grid
x = ['Predicted: Collision', 'Predicted: No Collision']
y = ['Actual: Collision', 'Actual: No Collision']
z_text = [[str(TP), str(FP)], [str(FN), str(TN)]]  # Text annotations for each cell

# Use Plotly Figure Factory to create annotated heatmap
fig = ff.create_annotated_heatmap(data, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
fig['layout']['yaxis']['autorange'] = "reversed"
# Add titles and labels for clarity
fig.update_layout(title_text='<b>Confusion Matrix</b>',
                  xaxis=dict(title='Predicted Value'),
                  yaxis=dict(title='Actual Value'))

fig.show()
