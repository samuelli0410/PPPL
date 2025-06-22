import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import readsav
import torch.optim as optim
import sys
from pathlib import Path
import csv
import ast
from torch.utils.data import random_split


class FrameInvertedDataset(Dataset):
    def __init__(self, data_path="/scratch/gpfs/sl4318/data.npz"):
        data = np.load(data_path)
        self.frames = data["X"]
        self.inverted = data["y"]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        inv = self.inverted[idx]

        frame = frame / frame.max() 
        inv = inv / inv.max()      

        frame = torch.tensor(frame, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
        inv = torch.tensor(inv, dtype=torch.float32).unsqueeze(0)

        frame = frame.repeat(3, 1, 1)  

        return frame, inv
    
    def validation_split(self, percent=0.2, seed=42):
        total_len = len(self)
        val_len = int(total_len * percent)
        train_len = total_len - val_len
        generator = torch.Generator().manual_seed(seed)
        return random_split(self, [train_len, val_len], generator=generator)


class DataHolder(FrameInvertedDataset):
    def __init__(self, frames, inverted):
        
        self.frames = frames
        self.inverted = inverted


class MLP(nn.Module):
    def __init__(self, input_size=518400, hidden_sizes=[256, 128], output_size=40401):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)
    
# input_size = 240 * 720*3
# hidden_sizes = [256, 128]
# output_size = 201*201
# Dataset = FrameInvertedDataset()
# batch_size = 16
# dataloader = DataLoader(Dataset, batch_size=batch_size, shuffle=True)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = MLP(input_size, hidden_sizes, output_size).to(device)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
# model.train()
# for epoch in range(30):
#         running_loss = 0.0

#         for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{30}"):
#             inputs = inputs.to(device)
#             targets = targets.to(device)
#             inputs = inputs.view(inputs.size(0), -1)   # [batch, 3*240*720]
#             targets = targets.view(targets.size(0), -1) # [batch, 201*201]

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         avg_loss = running_loss / len(dataloader)

#         print(f"Epoch {epoch+1}/{30}, Loss: {avg_loss:.6f}")


# torch.save(model.state_dict(), "MLP.pth")















# model = MLP()
# model.load_state_dict(torch.load("MLP.pth", map_location="cpu"))
# model.eval()

# # Load data
# data = np.load("data.npz")
# frames = data["X"]
# inverted = data["y"]

# # Choose a frame (e.g., first frame)
# test_input = frames[20]
# ground_truth = inverted[20]

# # Normalize input
# test_input = test_input / test_input.max()

# # Convert to tensor and reshape for MLP
# test_tensor = torch.tensor(test_input, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
# test_tensor = test_tensor.repeat(3, 1, 1)  # [3, H, W]
# input_flat = test_tensor.view(1, -1)       # [1, 3*H*W]

# # Generate output
# with torch.no_grad():
#     output_flat = model(input_flat)

# output_img = output_flat.view(201, 201).cpu().numpy()

# # Normalize output for visualization
# output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())

# # Plot
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.title("Input Camera Image")
# plt.imshow(test_input, cmap='inferno')
# plt.colorbar()

# plt.subplot(1, 3, 2)
# plt.title("Ground Truth Inversion")
# plt.imshow(ground_truth, cmap='inferno')
# plt.colorbar()

# plt.subplot(1, 3, 3)
# plt.title("MLP Generated Inversion")
# plt.imshow(output_img, cmap='inferno')
# plt.colorbar()
# project_root = Path(__file__).parent.parent
# plt.savefig(project_root / "MLP.png", dpi=300, bbox_inches='tight')