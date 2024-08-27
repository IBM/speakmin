import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import struct
import os
import matplotlib.pyplot as plt

size = 12  # Set to 16 for 16x16 images, otherwise 28 for 28x28 images
crop_size = 3
batch_size = 1
use_validation = False  # Set this to True if you want to use a validation set

class ResizeToTensor(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        width, height = img.size
        img = img.crop((crop_size, crop_size, width - crop_size, height - crop_size))
        img = transforms.Resize(self.size)(img)
        return transforms.ToTensor()(img)

# Load and preprocess the MNIST dataset
if size == 12:
    transform = transforms.Compose([
        ResizeToTensor((size, size)),
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

if not use_validation:
    train_loader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
else:
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(mnist_dataset))
    val_size = len(mnist_dataset) - train_size
    train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def rate_coding(image, steps=4000):
    """Perform RATE coding on MNIST images."""
    num_bias = 10
    rate_bias = 1.25 * 0 # 1%

    image = image.numpy().flatten()
    spikes = np.zeros((image.size+num_bias, steps), dtype=np.uint8)

    for i, pixel_value in enumerate(image):
        rate = pixel_value / 255.0  # Normalize pixel values to [0, 1] range
        spikes[i] = np.random.binomial(1, rate, steps)

    for j in range(num_bias):
        rate = rate_bias / 255.0  # Normalize pixel values to [0, 1] range
        spikes[image.size+j] = np.random.binomial(1, rate, steps)

    return spikes

def save_spike_binary(filename, data, labels):
    """Save spike trains in the specified binary format."""
    with open(filename, 'wb') as f:
        for spikes, label in zip(data, labels):
            num_data_points = np.sum(spikes)
            f.write(struct.pack('B', label))                # Unsigned char (1 byte)
            f.write(struct.pack('I', num_data_points))      # Unsigned int (4 bytes)
            for neuron_index, times in enumerate(spikes):
                for time in np.where(times)[0]:
                    f.write(struct.pack('I', time))         # Unsigned int (4 bytes)
                    f.write(struct.pack('H', neuron_index)) # Unsigned short (2 bytes)

# Generate and save spike trains for the training data in chunks
train_spikes = []
train_labels = []
chunk_size = 10000  # Data points per chunk
chunk_index = 0

print("Generating training spikes:")
data_point_count = 0

for i, (images, labels) in enumerate(tqdm(train_loader)):
    for image, label in zip(images, labels):
        spikes = rate_coding(image)
        train_spikes.append(spikes)
        train_labels.append(label.item())
        data_point_count += 1
        
        if data_point_count == chunk_size:
            # train_path = f'./data/{size}x{size}/{batch_size}/train_spikes_chunk_{chunk_index}.bin'
            # train_path = f'./data/{size}x{size}_bias/{batch_size}/train_spikes_chunk_{chunk_index}.bin'
            train_path = f'./data/{size}x{size}_bias/{batch_size}/train_spikes_chunk_{chunk_index}.bin'
            os.makedirs(os.path.dirname(train_path), exist_ok=True)
            print(f"Saving training spikes to binary file: {train_path}")
            save_spike_binary(train_path, train_spikes, train_labels)
            train_spikes = []
            train_labels = []
            data_point_count = 0
            chunk_index += 1

# Save any remaining training data
if train_spikes:
    # train_path = f'./data/{size}x{size}/{batch_size}/train_spikes_chunk_{chunk_index}.bin'
    # train_path = f'./data/{size}x{size}_bias/{batch_size}/train_spikes_chunk_{chunk_index}.bin'
    train_path = f'./data/{size}x{size}_bias/{batch_size}/train_spikes_chunk_{chunk_index}.bin'
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    print(f"Saving remaining training spikes to binary file: {train_path}")
    save_spike_binary(train_path, train_spikes, train_labels)

# Generate and save spike trains for the test data
test_spikes = []
test_labels = []
print("Generating test spikes:")
for images, labels in tqdm(test_loader):
    for image, label in zip(images, labels):
        spikes = rate_coding(image)
        test_spikes.append(spikes)
        test_labels.append(label.item())

# test_path = f'./data/{size}x{size}/{batch_size}/test_spikes.bin'
# test_path = f'./data/{size}x{size}_bias/{batch_size}/test_spikes.bin'
test_path = f'./data/{size}x{size}_bias/{batch_size}/test_spikes.bin'
os.makedirs(os.path.dirname(test_path), exist_ok=True)
print(f"Saving test spikes to binary file: {test_path}")
save_spike_binary(test_path, test_spikes, test_labels)
