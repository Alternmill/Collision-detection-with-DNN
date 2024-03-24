import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from simulation_utils import initialize_simulation, load_scene_and_robot, calculate_closest_distance, \
    calculate_overall_bounding_box


def generate_dataset(size, scene_id, robot_id):
    print("Generating dataset...")
    bbox_x, bbox_y = calculate_overall_bounding_box()
    print("Bounding box x: {}".format(bbox_x))
    print("Bounding box y: {}".format(bbox_y))

    x = np.random.uniform(bbox_x[0], bbox_x[1], size)
    y = np.random.uniform(bbox_y[0], bbox_y[1], size)
    theta = np.random.uniform(0, 2 * np.pi, size)
    distances = np.zeros(size)  # Preallocate array for distances

    print("Calculating distances...")
    for i in tqdm(range(size), desc="Calculating distances"):
        distances[i] = calculate_closest_distance(scene_id, robot_id, x[i], y[i], theta[i])

    data = np.stack((x, y, theta, distances), axis=1)  # Include distances in the dataset
    return data


def save_dataset(data, dataset_name):
    if not os.path.exists(f'datasets/{dataset_name}'):
        os.makedirs(f'datasets/{dataset_name}')
    df = pd.DataFrame(data, columns=['x', 'y', 'theta', 'distance'])
    df.to_csv(f'datasets/{dataset_name}/{dataset_name}.csv', index=False)
    print(f"Dataset saved to datasets/{dataset_name}/{dataset_name}.csv")


def feature_engineering(data):
    x, y, theta = data[:, 0], data[:, 1], data[:, 2]
    features = np.stack((x, np.sin(x), np.cos(x), y, np.sin(y), np.cos(y), np.sin(13 * x + 17 * y),
                         np.sin(19 * x - 15 * y),np.sin(23*x+y), np.sin(29*y-x), np.sin(theta), np.cos(theta)), axis=1)
    return features


def load_dataset(dataset_name, feature_eng_func=None):
    df = pd.read_csv(f'datasets/{dataset_name}/{dataset_name}.csv')
    inputs = df[['x', 'y', 'theta']].values
    distances = df['distance'].values  # Simplified for single column extraction

    if feature_eng_func:
        inputs = feature_eng_func(inputs)

    # Convert inputs and distances to PyTorch tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
    distances_tensor = torch.tensor(distances, dtype=torch.float32).view(-1, 1)  # Ensure distances is a column vector

    return inputs_tensor, distances_tensor


if __name__ == '__main__':
    initialize_simulation()
    scene_id, robot_id = load_scene_and_robot()

    dataset_size = 5000000
    dataset_name = f"dataset_big_{dataset_size}_samples"

    data = generate_dataset(dataset_size, scene_id, robot_id)
    save_dataset(data, dataset_name)
