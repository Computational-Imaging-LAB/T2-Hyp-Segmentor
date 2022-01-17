
import torchio as tio

import torch
import torch.nn as nn
import nibabel as nib
import glob
import cv2
from models.model import UNet, UNet3D
import src.preprocess as preprocess
import numpy as np
#import matplotlib.pyplot as plt
import time
from dataloader.dataloader_gliom import DataLoader_Gliom
from tqdm import tqdm
import src.losses.loss as loss
import wandb
from inference import load_model
from torchvision.utils import save_image
scaler = torch.cuda.amp.GradScaler()
device0= 'cuda:0' if torch.cuda.is_available() else 'cpu'
device1= 'cuda:1' if torch.cuda.is_available() else 'cpu'
config = dict(
    epochs=100,
    classes=2,
    kernels=[16, 32],
    batch_size=5,
    slices=10,
    device10=device0,
    device11=device1,
    patch_size=256,
    normalize='unit-variance',
    learning_rate=0.0001,
    weight_decay=0.0001,
    experiment='t2-hyp-segment',
    dataset="GLIOM-T2",
    architecture="UNET+Classifier")
wandb.login()

ls = []
prg = preprocess.preGlioma()
# nib.load('/cta/users/abas/Desktop/Meningiom/MeningiomData/gliom_data/Gliom/nii_gliom_boun/nii_gliom_directory/G0001/T0001/Segmentations/T0001_T2_HYP.nii')
"""
for t2_hyp in T2_HYP_segs:

    t2_root=('/').join(t2_hyp.split('/')[:-2])
    t2=glob.glob(t2_root+'/Anatomic/T2_TSE_TRA*/*.nii')[0]
    image=nib.load(t2).get_fdata().astype('float32').shape
    seg=nib.load(t2_hyp).get_fdata().astype('float32').shape
    if seg!=image:
        ls.append(t2_root)

"""
def save_pl(image,seg,output):
    cx=torch.cat((seg,image,output),dim=2)
    save_image(cx,f'test_image.png')
png = False
# datasets = DataLoader_Gliom(
#  '/cta/users/abas/Desktop/Meningiom/MeningiomData/gliom_data/Gliom/nii_gliom_boun/nii_gliom_directory/*/*/Segmentations/*T2_HYP*.nii',save=False,png=True)
get_foreground = tio.ZNormalization.mean


training_transform = tio.Compose([
          # to MNI space (which is RAS+)
    tio.RandomAnisotropy(p=0.25),              # make images look anisotropic 25% of times
                # tight crop around brain,        # standardize histogram of foreground
            # zero mean, unit variance of foreground
    tio.RandomBlur(p=0.25),                    # blur 25% of times
    tio.RandomNoise(p=0.25),                   # Gaussian noise 25% of times
    tio.OneOf({                                # either
                      # random affine
        tio.RandomElasticDeformation(): 0.12,   # or random elastic deformation
    }, p=0.1),                                 # applied to 80% of images
    tio.RandomBiasField(p=0.3),                # magnetic field inhomogeneity 30% of times
    tio.OneOf({                                # either
        tio.RandomMotion(): 1,                 # random motion artifact
        tio.RandomSpike(): 2,                  # or spikes
        tio.RandomGhosting(): 2,               # or ghosts
    }, p=0.1)                                # applied to 50% of images
])


datasets = DataLoader_Gliom(
    root_path='/cta/users/abas/Desktop/segmentation/t2_hyp_segment/data/train/cta/users/abas/Desktop/segmentation/t2_hyp_segment/data/raw/Gliom/nii_gliom_boun/nii_gliom_directory/*/*', save=False, png=False, 
    patch_size=512, slices=4, normalize='min-max',transform=training_transform)
data_loader = torch.utils.data.DataLoader(
    datasets, batch_size=1, num_workers=0)

datasets_validation = DataLoader_Gliom(
    root_path='/cta/users/abas/Desktop/segmentation/t2_hyp_segment/data/valid/*/*', save=False, png=False, 
    patch_size=512, slices=4, normalize='min-max')

data_loader_validation = torch.utils.data.DataLoader(
    datasets_validation, batch_size=1, num_workers=0)

bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
last = 0
first=False
test=True
if first:
    model = UNet(in_channels=3)

    checkpoint = "https://github.com/mateuszbuda/brain-segmentation-pytorch/releases/download/v1.0/unet-e012d006.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        checkpoint, progress=False, map_location='cpu')
    model.load_state_dict(state_dict)
    model.encoder1.enc1conv1.in_channels = 1
    model.encoder1.enc1conv1.weight = torch.nn.Parameter(
        torch.mean(model.encoder1.enc1conv1.weight, dim=1).unsqueeze(1))
    

    model.convClassifier = nn.Conv2d(
        in_channels=256, out_channels=128, kernel_size=3, stride=1)
    model.convClassifier2 = nn.Conv2d(
        in_channels=128, out_channels=64, kernel_size=5, stride=1)
    model.convClassifier3 = nn.Conv2d(
        in_channels=64, out_channels=32, kernel_size=3, stride=1)    
    model.classifier = nn.Linear(in_features=2048, out_features=1024)
    model.classifier2 = nn.Linear(in_features=1024, out_features=256)
    model.classifier3 = nn.Linear(in_features=256, out_features=1)

    

elif test:
    model=UNet()
    model.convClassifier = nn.Conv2d(
        in_channels=256, out_channels=128, kernel_size=3, stride=1)
    model.convClassifier2 = nn.Conv2d(
        in_channels=128, out_channels=64, kernel_size=5, stride=1)
    model.convClassifier3 = nn.Conv2d(
        in_channels=64, out_channels=32, kernel_size=3, stride=1)    
    model.classifier = nn.Linear(in_features=2048, out_features=1024)
    model.classifier2 = nn.Linear(in_features=1024, out_features=256)
    model.classifier3 = nn.Linear(in_features=256, out_features=1)
    model=load_model('/cta/users/abas/Desktop/segmentation/t2_hyp_segment/checkpoints/INF_model_256_0.191_best.pt',model)

else:
    model = UNet(phase='test')
    #model = load_model('/cta/users/abas/Desktop/Meningiom/MeningiomData/model_0.654_best.pt',model)

model = model.to(device1)
model = model.train()
epochs = config['epochs']

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

pbar = tqdm(range(0, epochs))
ls_epoch = 0
loss_hs = []
best_val_loss=99
bce_epoch = 0
try:
    suspect_data=np.load('/cta/users/abas/Desktop/segmentation/t2_hyp_segment/suspect.npy')
except:
    suspect_data=['None']
with wandb.init(project=config['experiment'], config=config):
    for idx, epoch in enumerate(pbar):
        ls_datas = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)

        model.train()
        pbar2 = tqdm(data_loader)
        nk=0
        
        for dt_idx, data in enumerate(pbar2):
                images, segs, img, seg_o, name = data
                pbar2.set_description('Epoch: %d' % epoch)
                ls_data = 0
                nk+=1
                n=1e-6
                if name[0] in suspect_data:
                    continue
                for idc, (image, seg) in enumerate(zip(images, segs)):
                    

                    #image=prg.normalize(image,typx='min-max')
                    #image=torch.tensor(image)
                    if seg.sum()<13000:
                        continue
                    n += 1
                    image=image.permute(3,0,1,2).to(device1)
                    output= model(image)
                    optimizer.zero_grad()
                    seg=seg.permute(
                        3, 0, 1, 2).to(device1)
                    loss_value = loss.dice_loss(output, seg.to(torch.long))
                    if loss_value>0.8:
                        save_pl(image,seg,output)
                        print(name)
                        with open ('suspected.txt','a') as rb:
                            rb.write(name[0] +'\n')
                        
                    if loss_value.item() > 1.5:
                        save_image(torch.stack(((image[0, :, :, 0].squeeze().to(device1) > 0).float(), (output[0, 0].squeeze() > 0.95).float(), (seg[0, :, :, 0].squeeze().to(
                            device1) > 0).float()), dim=0), f'/cta/users/abas/Desktop/Meningiom/MeningiomData/model_images/{name}_{idx}_{dt_idx}_{idc}.png')

                    # scaler.scale(loss_value).backward()
                    # scaler.step(optimizer)
                    # scaler.update()
                    loss_value.backward()
                    optimizer.step()
                    scheduler.step()
                    ls = loss_value.item()
                    ls_data += ls

                    pbar.set_description(
                        f'dt_idx:{dt_idx},data: {name} ,lossval: {ls:.3f}, Per Data Loss:{(ls_data/(n)):.3f}')
                    wandb.log({"Dice loss Patch": ls,
                           "Subject":name,"id":idc})
                ls_datas += (ls_data/n)
                wandb.log({"Dice loss Data": ls_data/n
                           ,"Subject":name})

            # loss_hs.append(ls_datas/(dt_idx+1))
        ls_epoch = (ls_datas/nk)
        
    
        #wandb.watch(model, bce_loss, log="all", log_freq=15)
        #wandb.watch(model, loss.dice_loss, log="all", log_freq=15)

        wandb.log({"epoch": idx, "Dice loss Epoch": ls_epoch,
                   })
        

       
        model.eval()
        with torch.no_grad():
            pbar2 = tqdm(data_loader_validation)
            ls_datas_val=0
            for dt_idx, data in enumerate(pbar2):
                    images, segs, img, seg_o, name = data
                    if images == 1:
                        continue
                    pbar2.set_description('Epoch: %d' % epoch)
                    ls_data_val = 0
                    nk = 0
                    n=1e-6
                    for idc, (image, seg) in enumerate(zip(images, segs)):
                       
                        # print(lbl,'*'*100)
                        
                        # seg[:,:,:,0]=seg[:,:,:,0]*1
                        # seg[:,:,:,1]=seg[:,:,:,1]*2
                        # seg[:,:,:,2]=seg[:,:,:,2]*3
                        # seg=torch.sum(seg,dim=3)

                        #cls_out = efficient_model_pretrained(
                        #    image.permute(3, 0, 1, 2).to(device0))
                        #cls_out = classifier_model(cls_out)
                        #bce_loss_value_eff = bce_loss(
                        #    cls_out, lbl.to(device0).unsqueeze(1))
                        #bce_loss_value_eff_data_val += bce_loss_value_eff.item()

                        #output = model(image.permute(3, 0, 1, 2).to(device1))
                        if seg.sum()<13000:
                            continue
                        n += 1
                        #image=prg.normalize(image,typx='min-max')
                        #image=torch.tensor(image)
                        image=image.permute(3,0,1,2).to(device1)
                        seg=seg.permute(3,0,1,2).to(device1)

                        output = model(image)
                       
                        loss_value = loss.dice_loss(output, seg.to(torch.long))
                        if loss_value>0.5:
                            save_pl(image,seg,output)
                            print(name)
                        ls = loss_value.item()
                        ls_data_val += ls

                        pbar.set_description(
                            f'Val dt_idx:{dt_idx},data: {name} ,lossval: {ls:.3f}, Per Data Loss:{(ls_data_val/(n)):.3f}')
                    ls_datas_val += ls_data_val/n
                    
                    wandb.log({"Dice Val loss Per Data": ls_data_val/n
                            ,"Subject":name})
            # loss_hs.append(ls_datas/(dt_idx+1))
        ls_epoch_val = (ls_datas_val/len(data_loader_validation))
       
        
        print(
            f'Epoch: {idx}, Val Dice Loss:{ls_epoch_val:.3f} ')
        #wandb.watch(model, bce_loss, log="all", log_freq=15)
        #wandb.watch(model, loss.dice_loss, log="all", log_freq=15)

        wandb.log({ "Val Dice loss Epoch": ls_epoch_val,
                   })
        if ls_epoch_val<best_val_loss:
            best_val_loss=ls_epoch_val
            torch.save(model.state_dict(), f"/cta/users/abas/Desktop/segmentation/t2_hyp_segment/checkpoints/model_{config['patch_size']}_{ls_epoch_val:.3f}_best.pt")
        
    """
    ls=[]
    unique_dims=[]
    for t2_hyp in T2_HYP_segs:

        t2_root=('/').join(t2_hyp.split('/')[:-2])
        t2=glob.glob(t2_root+'/Anatomic/T2_TSE_TRA*/*.nii')[0]
        image=nib.load(t2).get_fdata().astype('float32')
        seg=nib.load(t2_hyp).get_fdata().astype('float32')
        inplane,inplane2,slices=seg.shape

        if seg.shape not in unique_dims:
            unique_dims.append(seg.shape)
        if seg.shape!=image.shape:
            ls.append(t2_root.split('/')[-1])
        else:    
            for idx,slice in enumerate(range(0,slices)):
                # time.sleep(0.5)
                dst = cv2.addWeighted(prg.normalize(image[:,:,slice]),
                0.8,prg.normalize(seg[:,:,slice]),0.4,0)
                cv2.imshow('Normal Image',dst)
            
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
    """
