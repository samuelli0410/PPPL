import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
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

        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0, output_padding=0)  # [4, 512, 60, 90]
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
        d1 = torch.cat([d1, e4], dim=1)  # [4, 1024, 60, 90]
        d1 = self.dec1(d1)       # [4, 256, 60, 90]
        
        d2 = self.up2(d1)        # [4, 256, 120, 180]
        d2 = torch.cat([d2, e3], dim=1)  # [4, 512, 120, 180]
        d2 = self.dec2(d2)       # [4, 128, 120, 180]
        
        d3 = self.up3(d2)        # [4, 128, 240, 360]
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
    



def train_cgan():
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

    dataset = SynthTrainDataset(num_samples=240, data_dir="C:/Users/samue/Downloads/Research/Plasma/checkpoints")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True)

    for epoch in range(30):
        loss_d_cumul = 0
        loss_g_cumul = 0

        for y_real,x in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # print(x.shape)
            print(y_real.shape)
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
        print(f"AVG Gen Loss:{loss_g_cumul/len(dataloader)} | AVG Disc Loss: {loss_d_cumul/len(dataloader)}")

        if epoch % 10 == 5:
            print("saving")
            torch.save(generator.state_dict(), f"C:/Users/samue/Downloads/Research/Plasma/checkpoints/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"C:/Users/samue/Downloads/Research/Plasma/checkpoints/discriminator_epoch_{epoch}.pth")

    return generator

#Synthetic dataset because i dont have enough actual data LOL
def _make_samples():
    
    # Make R0s, Z0s, A0s, M0s
    nsp = 1
    numpoints = np.random.randint(1, 10)
    R0s = np.random.uniform(low=1.4, high=1.7, size=(nsp, numpoints))
    Z0s = np.random.uniform(low=-1.3, high=-0.9, size=(nsp, numpoints))
    
    nsample = Z0s.shape[0]
    
    A0s = np.ones((nsample, numpoints))
    M0s = np.ones((nsample, numpoints)) * 0.015
    
    return R0s, Z0s, A0s, M0s, nsample

def _make_setup():

#----- Sample input
    R0s      = [+1.396,+1.485];   #R of radiation point R1,R2
    Z0s      = [-1.084,-1.249];   #Z of radiation point Z1,Z2
    A0s      = [+1.000,+1.000];   #Amplitude
    M0s      = [+0.015,+0.015];   #Margins for
    do_plot  = True               #Show image plot
    save_name= 'synthetic_outs_2pnt.pl'
#-------

    #Run info
    Rinfo = {} 
    
    #Add radiation geometry info
    Rinfo['nsample'] = 0
    for key in ['R0s','Z0s','A0s','M0s']:
        Rinfo[key] = [];
    
#--- Append R-info here to do the scan 
    Rinfo['R0s'], Rinfo['Z0s'], Rinfo['A0s'], Rinfo['M0s'], Rinfo['nsample'] = _make_samples()
    # Rinfo['R0s'].append(R0s)
    # Rinfo['Z0s'].append(Z0s)
    # Rinfo['A0s'].append(A0s)
    # Rinfo['M0s'].append(M0s)
    # Rinfo['nsample'] += 1

    #Do plot and out(save) file name
    Rinfo['doplot']  = do_plot 
    Rinfo['outfile'] = save_name    

    return Rinfo

def _draw(cam_image=[],cam_inver=[],camgeo={}):

    #Draw synthetic/inverted images

    #cam_image: Synthetic image
    #cam_inver: Inverted image
    #cam_geo:   Camera geometry

    fig = plt.figure(1)
    
    #Draw inverted image
    plt.subplot(1,3,1)
    plt.title('Inverted')
    plt.pcolormesh(camgeo['inv_x'],camgeo['inv_y'],cam_inver)       
    plt.xlabel('R[m]')
    plt.ylabel('Z[m]')

    #Draw synthetic image
    plt.subplot(1,3,2)
    plt.title('Synthetic raw')
    plt.pcolormesh(cam_image)
    plt.xlabel('X[#]')
    plt.ylabel('Y[#]')

    #Draw synthetic image with wall picture
    xx = []; yy = [];
    plt.subplot(1,3,3)
    plt.title('Overlay')
    for ih in range(camgeo['cam_x'].shape[0]):
        for iw in range(camgeo['cam_x'].shape[1]):
            if cam_image[ih,iw]>0.01:
                xx.append(iw); yy.append(ih)
    plt.pcolormesh(camgeo['tar_r'])
    plt.scatter(xx,yy,marker='x',color='r',s=0.1)
    plt.xlabel('X[#]')
    plt.ylabel('Y[#]')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.3, 
                        hspace=0.1)
    plt.show()

def _load_camera(camera_save='Camera_geo.pl',filename1="C:/Users/samue/Downloads/Research/Plasma/Reconstruction Matrix/geom_240perp_unwarp_2022fwd.sav",filename2="C:/Users/samue/Downloads/Research/Plasma/Reconstruction Matrix/cam240perp_geometry_2022.sav"):

    #Load camera geometry

    #camera_save: Post-process camera info
    #filename1:   Camera target RZPhi info
    #filename2:   Camera CCD vertex info    

    camgeo={}    #geometry variables

    if not os.path.isfile(camera_save):

        target = readsav(filename1)
        camgeo['tar_r'] = target['newR'] / 1.e2
        camgeo['tar_z'] = target['newZ'] / 1.e2
        camgeo['tar_p'] = target['newPhi'] / 180*np.pi    

        vertex = readsav(filename2)
        location = vertex.Geom.povray[0][0][0] / 1.e2

        with open(camera_save,'wb') as f: 
            pickle.dump([camgeo['tar_r'],camgeo['tar_z'],camgeo['tar_p'],location],f)
    else:
        with open(camera_save,'rb') as f: 
            [camgeo['tar_r'],camgeo['tar_z'],camgeo['tar_p'],location] = pickle.load(f)

    camgeo['tar_x'] = camgeo['tar_r']  * np.cos(camgeo['tar_p'])
    camgeo['tar_y'] = camgeo['tar_r']  * np.sin(camgeo['tar_p'])

    [camgeo['nh'], camgeo['nw']] = camgeo['tar_x'].shape
    
    pre_ih = []
    new_ih = []
    for ih in range(camgeo['tar_x'].shape[0]):
        pre_ih.append(ih)
        new_ih.append(_calibrating_indexes(ih,camgeo))

    for iw in range(camgeo['tar_x'].shape[1]):

        camgeo['tar_x'][:,iw] = interp1d(pre_ih,camgeo['tar_x'][:,iw])(new_ih)
        camgeo['tar_y'][:,iw] = interp1d(pre_ih,camgeo['tar_y'][:,iw])(new_ih)
        camgeo['tar_z'][:,iw] = interp1d(pre_ih,camgeo['tar_z'][:,iw])(new_ih)
        camgeo['tar_r'][:,iw] = interp1d(pre_ih,camgeo['tar_r'][:,iw])(new_ih)

    camgeo['cam_x']  = np.ones((camgeo['nh'],camgeo['nw'])) * location[0]
    camgeo['cam_y']  = np.ones((camgeo['nh'],camgeo['nw'])) * location[1]
    camgeo['cam_z']  = np.ones((camgeo['nh'],camgeo['nw'])) * location[2]

    camgeo['vec_x'] = camgeo['tar_x']-camgeo['cam_x']
    camgeo['vec_y'] = camgeo['tar_y']-camgeo['cam_y']
    camgeo['vec_z'] = camgeo['tar_z']-camgeo['cam_z']
    camgeo['vec_s'] = np.sqrt(camgeo['vec_x']**2+camgeo['vec_y']**2+camgeo['vec_z']**2)

    camgeo['cam_c'] = np.zeros((camgeo['nh'],camgeo['nw']))

    ih2 = int(camgeo['nh']/2)
    iw2 = int(camgeo['nw']/2)

    for ih in range(camgeo['nh']):
        for iw in range(camgeo['nw']):
            sum0 = 0; sum1 = 0; sum2 = 0;
            for d in ['vec_x','vec_y','vec_z']:
                sum0+= camgeo[d][ih,iw]  *camgeo[d][ih,iw]
                sum1+= camgeo[d][ih2,iw2]*camgeo[d][ih2,iw2]
                sum2+= camgeo[d][ih,iw]  *camgeo[d][ih2,iw2]

            camgeo['cam_c'][ih,iw] = abs(sum2)/np.sqrt(sum0)/np.sqrt(sum1)

    camgeo['inv_x'] = np.linspace(+1.0,+2.0,201)
    camgeo['inv_y'] = np.linspace(-1.4,-0.4,201)
    
    # print('>>> Synthetic Camera dim.',camgeo['tar_r'].shape)
    # print('>>> Inverted  Camera dim. (%i, %i)'%(camgeo['inv_y'].shape[0],camgeo['inv_x'].shape[0]))
    return camgeo

def _integrate_image(Rinfo={},info_ind=0,camgeo={}):

    # Integrate images of different radiating rings in info_ind-th Rinfo

    R0s   = Rinfo['R0s'][info_ind]
    Z0s   = Rinfo['Z0s'][info_ind]
    A0s   = Rinfo['A0s'][info_ind]
    M0s   = Rinfo['M0s'][info_ind]

    cam_image = np.zeros(camgeo['cam_x'].shape)
    cam_inver = np.zeros((camgeo['inv_y'].shape[0],camgeo['inv_x'].shape[0]))

    if not (len(R0s)==len(Z0s)==len(A0s)==len(M0s)):
        print('>>> Given emission info is wrong!')
        exit()


    for i,R0 in enumerate(R0s):
        image, inver = _generate_image(R0s[i],Z0s[i],A0s[i],M0s[i],cam_image,cam_inver,camgeo)

        cam_image += image
        cam_inver += inver

    return cam_image, cam_inver

def _generate_image(R0=0.,Z0=0.,A0=0.,M0=0.,cam_image=[],cam_inver=[],camgeo={}):

    # Make images by emission ring at (R0,Z0)[m] with A0 amplitude, M0 [m] thickness
    # with camgeo info

    for iw in range(camgeo['cam_x'].shape[1]):
        for ih in range(camgeo['cam_x'].shape[0]):

            # Skip 0.0.0 pixels
            if (camgeo['tar_x'][ih,iw]==0.): continue

            # Location of emission ring along the line of sight (LOS)
            tt = (Z0-camgeo['cam_z'][ih,iw])/camgeo['vec_z'][ih,iw]
            dt =  M0/camgeo['vec_z'][ih,iw]
            # Skip if not on the LOS
            if (tt<0 or tt>1): continue       

            # Find the intersection of line of sight and emission ring
            tot_emission = 0.
            tot_emission += _get_emission(camgeo,M0,R0,Z0,ih,iw,tt+0.5*dt)
            tot_emission += _get_emission(camgeo,M0,R0,Z0,ih,iw,tt)
            tot_emission += _get_emission(camgeo,M0,R0,Z0,ih,iw,tt-0.5*dt)

            ssl  = camgeo['vec_s'][ih,iw] * abs(dt)

            tot_emission = A0 * tot_emission * ssl / 3

            # Accumulate emission on the pixel
            cam_image[ih,iw] += tot_emission

    # Generate inverted image
    for iw in range(camgeo['inv_x'].shape[0]):
        xx= camgeo['inv_x'][iw]
        for ih in range(camgeo['inv_y'].shape[0]):
            yy= camgeo['inv_y'][ih]
            dd = np.sqrt((Z0-yy)**2+(R0-xx)**2)
            cam_inver[ih,iw] += A0 * np.exp(-(dd/M0)**3)

    return cam_image, cam_inver

def _get_emission(camgeo={},M0=0.,R0=0.,Z0=0.,ih=0,iw=0,tt=0.):

    # Make emission from radiating point at tt of LOS to [ih,iw] pixel of camgeo

    xx = camgeo['vec_x'][ih,iw] * tt + camgeo['cam_x'][ih,iw]
    yy = camgeo['vec_y'][ih,iw] * tt + camgeo['cam_y'][ih,iw]
    zz = camgeo['vec_z'][ih,iw] * tt + camgeo['cam_z'][ih,iw]

    rr = np.sqrt(xx**2+yy**2)
    dd = np.sqrt((Z0-zz)**2+(R0-rr)**2)

    return np.exp(-(dd/M0)**3)

def _calibrating_indexes(ih=0,camgeo={}):

    # Re-adjustment of vertical pixel index to match the real-image with synthetic image
    coefs= [-5.94*1.e-6,
           +1.87*1.e-3,
           -2.49*1.e-1,
           +2.70*1.e+1]
    y   = 0.
    for k in range(4): y += coefs[k] * (ih ** (3-k))
    y = max(y,15)
    y+= ih
    y = min(y,camgeo['nh']-1)
    return y


#Synthetic dataset because i dont have enough actual data LOL
class SynthTrainDataset(Dataset):
    def __init__(self, num_samples=1000, camera_res=(256, 256), inversion_res=(256, 256),
                 data_dir="synthetic_data", force_regenerate=False, Generate = "Data"):
        self.num_samples = num_samples
        self.camera_res = camera_res
        self.inversion_res = inversion_res
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.data_file = os.path.join(data_dir, f"synth_data_{num_samples}samples_{camera_res[0]}x{camera_res[1]}.pkl")
        
        if not force_regenerate and os.path.exists(self.data_file):
            print("Loading pre-generated synthetic data...")
            self._load_data()
        else:
            print("Generating new synthetic data...")
            self._generate_data()
            self._save_data()
    
    def _save_data(self):
        """Save both samples and camera geometry"""
        with open(self.data_file, 'wb') as f:
            pickle.dump({
                'samples': self.samples,
                'camgeo': self.camgeo
            }, f)
        print("data Saved")
    
    def _load_data(self):
        """Load saved data from disk"""
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
            self.samples = data['samples']
            self.camgeo = data['camgeo']
    
    def _generate_data(self):
        self.camgeo = _load_camera(camera_save='Camera_geo.pl',
                                    filename1="C:/Users/samue/Downloads/Research/Plasma/Reconstruction Matrix/geom_240perp_unwarp_2022fwd.sav",
                                    filename2="C:/Users/samue/Downloads/Research/Plasma/Reconstruction Matrix/cam240perp_geometry_2022.sav")
        self.samples = []
        for i in tqdm(range(self.num_samples), desc="Generating synthetic data"):
            self.Rinfo  = _make_setup()
            

            # Output of rnd
            output = {};
            # Number of synthetic images
            output['run_setup'] = self.Rinfo
            # Synthetic image  dimension
            output['image_size']= self.camgeo['tar_x'].shape
            # Inverged image  dimension
            output['inver_size']= self.camgeo['inv_x'].shape    
            output['inver_R']   = self.camgeo['inv_x']
            output['inver_Z']   = self.camgeo['inv_y']

            output['image']     = {}
            output['inver']     = {}

            
            
            for rind in range(self.Rinfo['nsample']):
                camimg, inver = _integrate_image(self.Rinfo,rind,self.camgeo)
                self.samples.append((inver, camimg))
        
    
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        camera_img, cross_section = self.samples[idx]
        
        x = torch.FloatTensor(camera_img).unsqueeze(0)  # (1, H, W)
        y = torch.FloatTensor(cross_section).unsqueeze(0)  # (1, H, W)
        
        return x, y

if __name__ == "__main__":
    generator = train_cgan()
    torch.save(generator.state_dict(), "C:/Users/samue/Downloads/Research/Plasma/checkpoints/plasma_reconstructor.pth")

    # model = UNet()
    # test_input = torch.randn(1, 1, 256, 256)  # Batch of 1, 1 channel, 256x256
    # output = model(test_input)
    # print(f"Input shape: {test_input.shape}")
    # print(f"Output shape: {output.shape}")  # Should match input shape


    # generator = UNet()
    # generator.load_state_dict(torch.load("C:/Users/samue/Downloads/Research/Plasma/inversion/GAN/plasma_reconstructor.pth"))
    # generator.eval()