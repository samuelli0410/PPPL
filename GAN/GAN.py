import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from scipy.interpolate import RegularGridInterpolator
import pickle
from scipy.io import readsav
from scipy.interpolate import interp1d
import torch.nn.functional as F

#Generator class - UNET (maybe try another image gen?)

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),  # same padding
        nn.ReLU(inplace=True)
    )

#autoencoder

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = self._block(1, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))  # [4, 64, 240, 360]
        
        self.enc2 = self._block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))  # [4, 128, 120, 180]
        
        self.enc3 = self._block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))  # [4, 256, 60, 90]
        
        self.enc4 = self._block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))  # [4, 512, 30, 45]

        self.bottleneck = self._block(512, 512)  # [4, 512, 30, 45]

        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2,padding=0, output_padding=0)  # [4, 512, 60, 90]
        self.dec1 = self._block(1024, 256)  # After cat with enc4

        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, output_padding=0)  # [4, 256, 120, 180]
        self.dec2 = self._block(512, 128)  # After cat with enc3

        self.up3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0,output_padding=0)  # [4, 128, 240, 360]
        self.dec3 = self._block(256, 64)  # After cat with enc2

        # Final upsampling to 201x201
        self.final_upsample = nn.Sequential(
            nn.Upsample(size=(201, 201), mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # [4, 64, 240, 720]
        p1 = self.pool1(e1)      # [4, 64, 240, 360]
        
        e2 = self.enc2(p1)       # [4, 128, 240, 360]
        p2 = self.pool2(e2)      # [4, 128, 120, 180]
        
        e3 = self.enc3(p2)       # [4, 256, 120, 180]
        p3 = self.pool3(e3)      # [4, 256, 60, 90]
        
        e4 = self.enc4(p3)       # [4, 512, 60, 90]
        p4 = self.pool4(e4)      # [4, 512, 30, 45]

        # Bottleneck
        bn = self.bottleneck(p4) # [4, 512, 30, 45]

        # Decoder
        d1 = self.up1(bn)        # [4, 512, 60, 90]
        if d1.shape[2:] != e4.shape[2:]:
            d1 = F.interpolate(d1, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e4], dim=1)  # [4, 1024, 60, 90]
        d1 = self.dec1(d1)       # [4, 256, 60, 90]
        
        d2 = self.up2(d1)        # [4, 256, 120, 180]
        if d2.shape[2:] != e3.shape[2:]:
            d2 = F.interpolate(d2, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e3], dim=1)  # [4, 512, 120, 180]
        d2 = self.dec2(d2)       # [4, 128, 120, 180]
        
        d3 = self.up3(d2)        # [4, 128, 240, 360]
        if d3.shape[2:] != e2.shape[2:]:
            d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1) 
        d3 = self.dec3(d3)       # [4, 64, 240, 360]
        
        # Final resize to 201x201
        d4 = self.final_upsample(d3)  # [4, 64, 201, 201]
        
        return torch.tanh(self.final(d4))  

class PatchGAN(nn.Module):
    def __init__(self):
        super(PatchGAN, self).__init__()
        
        self.x_processor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=(1,2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),  
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        self.y_processor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1),
        )
    
    def forward(self, x, y):
        x_feat = self.x_processor(x)
        y_feat = self.y_processor(y)
        x_feat = F.interpolate(x_feat, size=(13,13), mode='bilinear', align_corners=False)
        y_feat = F.interpolate(y_feat, size=(13,13), mode='bilinear', align_corners=False)
        xy = torch.cat([x_feat, y_feat], dim=1)
        return self.final(xy)


#Suggestions for better loss func? 
def forwardProject(cam_inver, camgeo):
    cam_image = np.zeros_like(camgeo['cam_x'])
    # print(cam_image)
    inv_x = camgeo['inv_x']
    inv_y = camgeo['inv_y']
    dx = inv_x[1] - inv_x[0]
    dy = inv_y[1] - inv_y[0]
    inv_x_min = inv_x[0]
    inv_y_min = inv_y[0]

    num_samples = 100

    for ih in range(camgeo['cam_x'].shape[0]):
        for iw in range(camgeo['cam_x'].shape[1]):

            # Skip if invalid pixel
            if camgeo['tar_x'][ih, iw] == 0.0:
                continue

            # Use precomputed camera position and vector components
            cam_x = camgeo['cam_x'][ih, iw]
            cam_y = camgeo['cam_y'][ih, iw]
            cam_r = np.sqrt(cam_x**2 + cam_y**2)
            cam_z = camgeo['cam_z'][ih, iw]

            vec_x = camgeo['vec_x'][ih, iw]
            vec_y = camgeo['vec_y'][ih, iw]
            vec_z = camgeo['vec_z'][ih, iw]
            vec_r = np.sqrt(vec_x**2 + vec_y**2)

            dl = np.sqrt(vec_r**2 + vec_z**2) / num_samples

            for alpha in np.linspace(0, 1, num_samples):
                r = cam_r + alpha * vec_r
                z = cam_z + alpha * vec_z

                # Fast index computation (instead of np.argmin)
                r_idx = int((r - inv_x_min) / dx)
                z_idx = int((z - inv_y_min) / dy)

                if 0 <= z_idx < cam_inver.shape[0] and 0 <= r_idx < cam_inver.shape[1]:
                    intensity = cam_inver[z_idx, r_idx]
                else:
                    intensity = 0.0

                cam_image[ih, iw] += intensity * dl

    return cam_image


def LossFunc(y_fake, x, camgeo):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proj_fake = forwardProject(y_fake, camgeo)
    proj_fake = torch.from_numpy(proj_fake).unsqueeze(0).unsqueeze(0).repeat(8, 1, 1, 1).to(device)
    x = x.to(device) 
    loss_i = F.l1_loss(proj_fake, x)

    return loss_i
    

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





def train_cgan(batch_size=16, percent=0.2, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.02)
            else:
                print(f"Skipped Conv2d: {m}")
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            else:
                print(f"Skipped Norm (no weight): {m}")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            else:
                print(f"Skipped Norm (no bias): {m}")

    generator = UNet().to(device)
    generator.apply(weights_init)
    
    discriminator = PatchGAN().to(device)
    discriminator.apply(weights_init)

    criterion_gan = nn.MSELoss()  
    criterion_l1 = nn.L1Loss()
    criterion_proj = nn.MSELoss() 

    opt_g = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(opt_g, 'min', patience=5, factor=0.5)
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(opt_d, 'min', patience=5, factor=0.5)

    dataset = FrameInvertedDataset()
    train_set, val_set = dataset.validation_split(percent, seed)

    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    for epoch in range(30):
        loss_d_cumul = 0
        loss_g_cumul = 0

        for x,y_real in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # print(x.shape)
            # print(y_real.shape)
            x, y_real = x.to(device), y_real.to(device)

            opt_d.zero_grad()
            
            with torch.no_grad():
                y_fake = generator(x)
            
            d_real = discriminator(x, y_real)
            d_fake = discriminator(x, y_fake.detach())
            
            loss_d_real = criterion_gan(d_real, torch.ones_like(d_real))
            loss_d_fake = criterion_gan(d_fake, torch.zeros_like(d_fake))
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d_cumul+= loss_d.item()
            loss_d.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            y_fake = generator(x)
            
            # Multi-component loss
            d_fake = discriminator(x, y_fake)
            loss_g_gan = criterion_gan(d_fake, torch.ones_like(d_fake))
            loss_g_l1 = criterion_l1(y_fake, y_real) * 100  # Weighted more heavily
            
            loss_g = loss_g_gan + loss_g_l1
            loss_g_cumul+= loss_g.item()
            loss_g.backward()
            opt_g.step()

        scheduler_g.step(loss_g)
        scheduler_d.step(loss_d)
        print(f"{epoch} AVG Gen Loss:{loss_g_cumul/len(dataloader)} | AVG Disc Loss: {loss_d_cumul/len(dataloader)}")

    return generator




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

if __name__ == "__main__":
    generator = train_cgan()
    torch.save(generator.state_dict(), "GAN.pth")

