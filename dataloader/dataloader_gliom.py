import torch
import nibabel as nib
import glob
import cv2
from src import preprocess as preprocess
import numpy as np
#import matplotlib.pyplot as plt
import time
import random
from torch.utils.data import Dataset


class DataLoader_Gliom(Dataset):

    def __init__(self, root_path,modality="T2",png=False,save=False,bet=True,phase='train',patch_size=256,slices=5,normalize='unit-variance',transform=None):
      
        ls = []
        self.prg = preprocess.preGlioma()
        self.T2_HYP_segs = glob.glob(root_path+f"/Segmentations/*{modality}*.nii*")

        self.save = save
        self.patch_size = patch_size
        self.slices=slices
        self.transform = transform
        self.normalize = normalize
        self.png=png
        self.bet=bet
        self.phase=phase
        #self.T2_HYP_segs=np.load('/cta/users/abas/Desktop/Meningiom/MeningiomData/not_used.npy')
    def __len__(self):
        return len(self.T2_HYP_segs)-1

    def __getitem__(self, idx):







            t2_hyp = self.T2_HYP_segs[idx]
            #t2_root = ('/').join(t2_hyp.split('/')[:-2])
            #try:
                #t2 = glob.glob(t2_root+'/Anatomic/T2_TSE_TRA*/*.nii')[0]
            #except:
                #return [1, 1, 1, 1]
            t2_anat=t2_hyp.replace('Segmentations','Anatomic').split('/')[:-1]
            t2_anat='/'.join(t2_anat)
            t2_anat=glob.glob(t2_anat+"/*T2_TSE_TRA*/*.nii")[0]
            #t2_root = ('/').join(t2_hyp.split('/')[:-1]) 

            image = nib.load(t2_anat).get_fdata().astype('float32')
            seg = nib.load(t2_hyp).get_fdata().astype('float32')
            image,seg=self.prg.trim_black_ends(image,seg)
            slices=self.prg.cut_tumor_image(seg)
            image=image[:,:,slices[0]:slices[1]+1]
            seg=seg[:,:,slices[0]:slices[1]+1]

            if not image.shape==seg.shape:
                testfile =  open("/cta/users/abas/Desktop/Meningiom/MeningiomData/shape_mismatch.txt", "a")

                testfile.write(f"{t2_hyp}")

                testfile.close()
                self.__getitem__(idx+1)

            #image=self.prg.betfsl(image)
            image = self.prg.normalize(image,typx=self.normalize)
            seg = self.prg.normalize(seg,typx='min-max')

            if self.transform is not None:
                image = self.transform(np.expand_dims(image,0))
            
                image=image.squeeze(0)
                
            
        


            images=self.prg.slice_chopper(image,seg,phase=self.phase,slices=self.slices)
            images = self.prg.patch_chopper(self.prg.patch_chopper(
                    images, dim=1,patch_size=self.patch_size), dim=0,patch_size=self.patch_size)
            
            segs=self.prg.slice_chopper(seg,seg,phase=self.phase,slices=self.slices)
            segs = self.prg.patch_chopper(self.prg.patch_chopper(
                    segs, dim=1,patch_size=self.patch_size), dim=0,patch_size=self.patch_size)
            







            if self.save:
                self.prg.save_image(images, segs, image, seg, t2_hyp.split('/')[-4])
            
            return [images, segs, image,seg,t2_hyp.split('/')[-4]]
            
