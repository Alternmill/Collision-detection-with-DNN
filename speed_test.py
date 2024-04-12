import torch
from torch.utils.data import DataLoader, TensorDataset
import time
from generate_load_datasets import load_dataset, feature_engineering

from torch.profiler import profile, record_function, ProfilerActivity


def test_model_speed_with_profiling(dataloader, model):
    # Start profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with torch.no_grad():
            for inputs, targets in dataloader:
                with record_function("model_inference"):
                    outputs = model(inputs)

    # Print profiler results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")  # Export the trace to a file


def load_model_and_data(path, dataset_name, feature_eng_func, device):
    # Load the model
    model = torch.load(path, map_location=device)
    model = model.to(device)
    model.eval()

    # Load and prepare the dataset
    inputs, targets = load_dataset(dataset_name, feature_eng_func)
    inputs = inputs.to(device)
    targets = targets.to(device)
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=1000)

    return model, loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "models/model_2_2024-04-08_20-18-06.pt"
dataset_name = "dataset_big_5000000_samples_second_scene"

# Load the model and data outside the timing
model, loader = load_model_and_data(path, dataset_name, feature_engineering, device)
model1, loader1 = load_model_and_data(path, dataset_name, feature_engineering, device)
model2, loader2 = load_model_and_data(path, dataset_name, feature_engineering, "cpu")


# Test the model speed with profiling

def test_model_speed(dataloader, model):
    start_time = time.time()
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
    end_time = time.time()
    print(f"Total computation time for one epoch: {end_time - start_time:.5f} seconds")


# test_model_speed_with_profiling(loader, model)
test_model_speed(loader, model)
test_model_speed(loader1, model1)
test_model_speed(loader2, model2)
