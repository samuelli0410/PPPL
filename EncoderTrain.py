import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.io import readsav
import sys
from pathlib import Path
import csv
import ast
from torch.utils.data import random_split

class ResNetEncoder(nn.Module):
    def __init__(self,backbone='resnet50',pretrained=True):
        super(ResNetEncoder,self).__init__()
        resnet = getattr(models,backbone)(pretrained=pretrained)
        self.features = nn.Sequential(resnet.conv1, 
                                      resnet.bn1,
                                      resnet.relu, 
                                      resnet.maxpool, 
                                      resnet.layer1, 
                                      resnet.layer2, 
                                      resnet.layer3,
                                      resnet.layer4)
    def forward(self,x):
        return self.features(x)
class Decoder(nn.Module):
    def __init__(self,in_channels):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=3,stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        x = F.interpolate(x, size=(201, 201), mode='bilinear', align_corners=False)
        return x




#SRCNN to refine output from Resnet

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),  
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2))
            
        
        
    def forward(self, x):
        return self.features(x)










#Resnet
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder()
        self.SRCNN = SRCNN()
        self.decoder = Decoder(in_channels=2048)
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.SRCNN(x)
        return x


#VAE
# class Autoencoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = ResNetEncoder()
        
#         self.latent_dim = 2  
#         self.encoder_output_shape = (2048, 8, 23)
#         encoder_flat_dim = 2048 * 8 * 23
        
#         self.mean_layer = nn.Linear(encoder_flat_dim, self.latent_dim)
#         self.logvar_layer = nn.Linear(encoder_flat_dim, self.latent_dim)

#         self.decoder_input_channels = 512  
#         self.decoder_h, self.decoder_w = 4, 4

#         self.latent_to_decoder = nn.Linear(self.latent_dim, self.decoder_input_channels * self.decoder_h * self.decoder_w)
        
#         self.decoder = Decoder(in_channels=self.decoder_input_channels)

#     def forward(self, x):
#         x = self.encoder(x)
#         # print("latent space dim")
#         # print(x.shape)
#         B = x.shape[0]
#         x_flat = x.view(B, -1)

#         mean = self.mean_layer(x_flat)
#         logvar = self.logvar_layer(x_flat)
#         z = self.reparameterization(mean, logvar)

#         z = self.latent_to_decoder(z)
#         z = z.view(B, self.decoder_input_channels, self.decoder_h, self.decoder_w)

#         x_recon = self.decoder(z)
#         return x_recon

#     def reparameterization(self, mean, var):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         epsilon = torch.randn_like(var).to(device)      
#         z = mean + var*epsilon
#         return z









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


def train_resNet(batch_size=16, percent=0.2, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    num_epochs = 50
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Get datasets
    dataset = FrameInvertedDataset()
    train_set, val_set = dataset.validation_split(percent, seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)


    model.train()
    errors = []
    print("Start Training")

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        errors.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    return model, errors, val_set






def load_csv_data(file_index):
    
    csv.field_size_limit(sys.maxsize)
    data_dir = Path(__file__).parent.parent / "data/FrameInvertedData"
    file_path = data_dir / f"data{file_index}.csv"

    reconstructed = []

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 2:
                continue  # skip malformed rows

            try:
                arr1 = np.array(ast.literal_eval(row[0]))
                arr2 = np.array(ast.literal_eval(row[1]))
                reconstructed.append((arr1, arr2))
            except Exception as e:
                print(f"Failed to parse row: {row}\nError: {e}")

    return reconstructed


def getData():
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    sys.path.append(str(project_root))
    data_dir = data_dir / "FrameInvertedData"
    data = []
    for i in range(26):
        out = load_csv_data(i)
        data.extend(out)
    return data
 
def getDataBetter():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    sys.path.append(str(project_root))
    sav_files_dir = data_dir / "all"
    from data.file_utils import GetEmission
    em = GetEmission(file_path=sav_files_dir)
    files = em.list_files(display=True)

    X = []  # vid frames
    y = []  # inverted frames

    def loadFramesAndInverted(i):
        inverted, radii, elevation, frames, times, vid_frames, vid_times, vid = em.load_all(files[i])

        count = 0
        for j in frames:
            j = int(j)
            X.append(vid[j])
            y.append(inverted[count])
            count += 1

    for i in range(26):
        loadFramesAndInverted(i)

    X = np.stack(X)  # shape: (N, H, W)
    y = np.stack(y)  # shape: (N, H, W)
    np.savez_compressed("data.npz", X=X, y=y)
    return X, y
    

def save_val_set(val_set, save_path="val_data.npz"):
    frames = []
    inverted = []

    for frame, inv in val_set:
        # Undo channel repeat and squeeze to get [H, W]
        frame = frame[0].numpy()  # or frame.mean(0).numpy() if 3-channel
        inv = inv[0].numpy()

        # Re-normalize if needed (depends on usage)
        frames.append(frame)
        inverted.append(inv)

    np.savez_compressed(save_path, X=np.array(frames), y=np.array(inverted))
    print(f"Saved validation set to {save_path}")


















if __name__ == "__main__":
    # data = np.load("data.npz")
    # frames = data["X"]
    # inverted = data["y"]
    # project_root = Path(__file__).parent.parent
    # print(project_root)

    ######################Training
    # generator,errors,val_set = train_resNet()
    # torch.save(generator.state_dict(), "ResNetSRCNN.pth")
    # data_dir = Path(__file__).parent.parent
    # output_file = data_dir/ "ResNetSRCNNerrors.csv"
    # with open(output_file, "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows([[e] for e in errors])
    # save_val_set(val_set)
    # print("done")



    ######################Validation

    model = Autoencoder()
    data = np.load("data.npz")
    frames = data["X"]
    inverted = data["y"]

    test_input = frames[0]
    ground_truth = inverted[0]

    test_input = test_input / test_input.max()
    test_tensor = torch.tensor(test_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    test_tensor = test_tensor.repeat(1, 3, 1, 1)  # [1, 3, H, W] for ResNet-style models


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    project_root = Path(__file__).parent.parent
    model.load_state_dict(torch.load(project_root / "ResNetSRCNN.pth", map_location=device))
    model.eval()


    with torch.no_grad():
        test_output = model(test_tensor.to(device))
        test_output = test_output.squeeze().cpu().numpy()
    print("starting")
    plt.imshow(ground_truth, cmap='inferno')
    plt.colorbar()
    plt.title("Ground Truth")
    plt.savefig(project_root / "GroundTruthSRCNN.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.imshow(test_output, cmap='inferno')
    plt.colorbar()
    plt.title("Model Output")
    plt.savefig(project_root / "OutputSRCNN.png", dpi=300, bbox_inches='tight')
    plt.close()






# cursor jail (for cat commands)
